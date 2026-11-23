# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-side capture of live hybrid state into the PIC physical pool.

Stage 7-A is capture-only for the model execution path.  Stage 9-A-1 adds a
worker-local reference implementation of composable recurrent transitions:
the structured operator is kept beside its opaque pool handle and can be
applied by a later execution stage.  This module still does not change the
ordinary model forward path by itself.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.pic.cache import PICSegmentEntry
from vllm.v1.pic.gdn_transition import build_gdn_transition_operator
from vllm.v1.pic.lifecycle import PICLeaseRegistry, PICLeaseToken
from vllm.v1.pic.env import pic_debug_enabled
from vllm.v1.pic.materialization import PICMaterialization
from vllm.v1.pic.metrics import PICMetrics
from vllm.v1.pic.native_kv import (
    wait_pending_upload,
    PICKVPositionMode,
    PICNativeKVKind,
    PICNativeKVReference,
    get_covering_local_block_span,
    get_full_local_block_span,
    is_pic_exact_position_aliasable,
)
from vllm.v1.pic.pool import PICPhysicalPool
from vllm.v1.pic.profile_level import pic_profile_enabled
from vllm.v1.pic.segmenter import PICSegment
from vllm.v1.pic.snapshot import PICPhysicalSnapshot, PICSnapshotStore
from vllm.v1.pic.state import (
    PICAffineTransition,
    PICConvTransition,
    PICGDNTransition,
    PICLinearTransition,
    PICStateLayout,
    PICTransitionOperator,
)
from vllm.v1.pic.worker_plan import PICWorkerUnsupported

logger = init_logger(__name__)

def _pic_detail_profile_enabled() -> bool:
    return pic_profile_enabled()


if TYPE_CHECKING:
    import torch

    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


def ordered_gdn_layer_names(
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
) -> tuple[str, ...]:
    """Return GDN layers in the same order as live state collection.

    Live recurrent and convolution state is collected by walking KV cache
    groups and then each group's layer names. Transition operators are
    positional, so they must use that exact group-major order rather than the
    model traversal order of ``static_forward_context``.
    """
    names: list[str] = []
    seen: set[str] = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            if layer_name in seen:
                continue
            layer = forward_context.get(layer_name)
            if layer is None:
                raise ValueError(
                    f"PIC transition cannot find forward layer {layer_name}"
                )
            if getattr(getattr(layer, "mamba_type", None), "name", None) != (
                "GDN_ATTN"
            ):
                continue
            names.append(layer_name)
            seen.add(layer_name)
    return tuple(names)


@dataclass
class PICGDNTransitionCapture:
    """Collect real GDN transitions during one single-request prefill.

    The GDN layer supplies post-convolution ``key/value/log_decay/beta``
    tensors.  Once every GDN layer in the request has reported a segment, the
    per-layer operators are assembled in the same order as vLLM's recurrent
    state blocks and attached to ``CachedRequestState``.
    """

    request_state: CachedRequestState
    segments: tuple[PICSegment, ...]
    layer_names: tuple[str, ...]
    debug: bool = False
    _operators: dict[int, dict[str, PICTransitionOperator]] = field(
        default_factory=dict, init=False
    )
    _zero_conv_tails: dict[int, dict[str, torch.Tensor]] = field(
        default_factory=dict, init=False
    )
    _conv_transitions: dict[int, dict[str, PICConvTransition]] = field(
        default_factory=dict, init=False
    )
    # How many tokens this step actually forwards. The attention metadata reports the padded batch
    # size instead, so a bound taken from it let a window reach uninitialised rows.
    max_tokens: int | None = None
    _input_logged: bool = field(default=False, init=False)

    def _debug(self, message: str, *args: object) -> None:
        if self.debug:
            if args:
                message = message % args
            logger.warning("[PIC-DEBUG] %s", message)

    def record_gdn_layer(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        *,
        state_dtype: torch.dtype,
        conv_input: torch.Tensor,
        conv_kernel_size: int,
        backend: str | None = None,
        num_actual_tokens: int | None = None,
    ) -> None:
        if layer_name not in self.layer_names:
            return
        if backend not in (None, "triton", "flashinfer"):
            self._debug(
                "transition capture skipped layer=%s unsupported backend=%s",
                layer_name,
                backend,
            )
            return
        if key.ndim != 3 or value.ndim != 3:
            raise ValueError("PIC GDN capture expects unbatched post-conv tensors")

        # The tensors are a buffer, not a length: a step that computed only the head of the prompt
        # leaves the rest of it uninitialised, and describing a transition over those rows produced a
        # non-finite zero-start state that took the engine down. `num_actual_tokens` is how much of
        # the buffer this step filled, so a segment that reaches past it is simply not captured now.
        available = int(key.shape[0])
        if num_actual_tokens is not None:
            available = min(available, int(num_actual_tokens))
        if self.max_tokens is not None:
            available = min(available, int(self.max_tokens))
        if pic_debug_enabled() and not self._input_logged:
            self._input_logged = True
            logger.warning(
                "[PIC-TRACE] capture input layer=%s buffer=%d num_actual_tokens=%s "
                "available=%d segments=%s",
                layer_name,
                int(key.shape[0]),
                num_actual_tokens,
                available,
                [(s.start, s.end) for s in self.segments],
            )
        for segment_index, segment in enumerate(self.segments):
            if segment.start < 0 or segment.end > available:
                # Deliberately not gated by the capture's own debug flag: this is the line that says
                # whether the bound works, and a silent bound is indistinguishable from a missing one.
                if pic_debug_enabled():
                    logger.warning(
                        "[PIC-TRACE] transition capture deferred layer=%s segment=%d "
                        "needs=%d available=%d num_actual_tokens=%s",
                        layer_name,
                        segment_index,
                        segment.end,
                        available,
                        num_actual_tokens,
                    )
                continue
            # No window inspection and no finiteness fallback here, on purpose. Counting non-finite
            # rows meant two host synchronisations for each of the eighteen linear layers that record
            # one, and that inflated every debug run's timing - a measurement that includes its own
            # probe cannot judge anything. A bound taken from the wrong length has to fail loudly
            # instead: silencing it turns a wrong length into a silently abandoned capture, which is
            # indistinguishable from a working one until the reuse numbers come out wrong.
            operator = build_gdn_transition_operator(
                key[segment.start : segment.end],
                value[segment.start : segment.end],
                log_decay[segment.start : segment.end],
                beta[segment.start : segment.end],
                token_start=0,
                token_end=segment.length,
                state_dtype=state_dtype,
                backend=backend,
            )
            conv_tokens = conv_input[segment.start : segment.end]
            state_len = max(conv_kernel_size - 1, 0)
            if state_len == 0:
                zero_conv_tail = conv_tokens.new_empty((conv_tokens.shape[1], 0))
            elif conv_tokens.shape[0] >= state_len:
                zero_conv_tail = conv_tokens[-state_len:].transpose(0, 1)
            else:
                zero_padded = conv_tokens.new_zeros(
                    (state_len - conv_tokens.shape[0], conv_tokens.shape[1])
                )
                zero_conv_tail = conv_tokens.new_zeros(
                    (state_len, conv_tokens.shape[1])
                )
                zero_conv_tail[: zero_padded.shape[0]] = zero_padded
                zero_conv_tail[zero_padded.shape[0] :] = conv_tokens
                zero_conv_tail = zero_conv_tail.transpose(0, 1)
            zero_conv_tail = zero_conv_tail.contiguous()
            conv_transition = PICConvTransition(
                inputs=conv_tokens.contiguous(),
                state_length=state_len,
            )
            layer_operators = self._operators.setdefault(segment_index, {})
            layer_operators[layer_name] = operator
            self._conv_transitions.setdefault(segment_index, {})[layer_name] = (
                conv_transition
            )
            self._zero_conv_tails.setdefault(segment_index, {})[layer_name] = (
                zero_conv_tail
            )
            if all(name in layer_operators for name in self.layer_names):
                # One finiteness check for the whole segment. Each layer used to read its own
                # zero-start state back at build time, which cost a synchronisation per layer
                # and cut the prefill's layer pipeline into eighteen serial pieces. The
                # zero-start state lives on each transition, not on the operator.
                checked = [
                    (name, transition)
                    for name in self.layer_names
                    for transition in layer_operators[name].transitions
                ]
                flags = torch.stack(
                    [
                        torch.isfinite(transition.zero_state).all()
                        for _, transition in checked
                    ]
                )
                if not bool(flags.all().item()):
                    bad = sorted(
                        {name for (name, _), ok in zip(checked, flags.tolist()) if not ok}
                    )
                    raise PICWorkerUnsupported(
                        "PIC transition produced a non-finite zero-start state: "
                        f"segment={segment_index} layers={bad}"
                    )
                transitions = tuple(
                    transition
                    for name in self.layer_names
                    for transition in layer_operators[name].transitions
                )
                conv_transitions = tuple(
                    self._conv_transitions[segment_index][name]
                    for name in self.layer_names
                )
                self.request_state.pic_transition_operators[segment_index] = (
                    PICTransitionOperator(
                        transitions=transitions,
                        conv_transitions=conv_transitions,
                        token_start=0,
                        token_end=segment.length,
                    )
                )
                if pic_debug_enabled():
                    logger.warning(
                        "[PIC-TRACE] transition-recorded segment=%d range=[%d,%d) "
                        "layers=%d step_tokens=%d",
                        segment_index,
                        segment.start,
                        segment.end,
                        len(self.layer_names),
                        int(key.shape[0]),
                    )
                self.request_state.pic_transition_conv_tails[segment_index] = tuple(
                    self._zero_conv_tails[segment_index][name]
                    for name in self.layer_names
                )

@dataclass
class PICLiveCaptureManager:
    """Own worker-local PIC snapshots and pending scheduler metadata."""

    pool: PICPhysicalPool
    retain_published: bool = False
    debug: bool = False
    lease_registry: PICLeaseRegistry | None = None
    metrics: PICMetrics | None = None
    on_evict: Callable[[bytes], None] | None = None
    store: PICSnapshotStore = field(init=False)
    _snapshots: dict[tuple[str, int], PICPhysicalSnapshot] = field(
        default_factory=dict, init=False
    )
    _transitions: dict[int, PICTransitionOperator] = field(
        default_factory=dict, init=False
    )
    _pending: list[PICMaterialization] = field(default_factory=list, init=False)
    _published: set[tuple[str, int]] = field(default_factory=set, init=False)
    # Transition/recurrent state is position-independent for a segment.  Keep
    # one worker-owned snapshot per segment hash while the scheduler-side PIC
    # entry retains it; native KV references remain request-specific and are
    # carried by each materialization separately.
    _published_segment_owners: dict[bytes, tuple[str, int]] = field(
        default_factory=dict, init=False
    )
    # Requests that are currently consuming a published snapshot.  The cache
    # lease is separate from these request leases: a completed request may
    # release its lease while the segment remains reusable by later requests.
    _active_published_segments: dict[str, set[bytes]] = field(
        default_factory=dict, init=False
    )
    # Oldest entries are evicted first when the physical pool cannot satisfy a
    # new capture.  This bounds worker-local state even if the scheduler cache
    # receives many distinct segment hashes.
    _published_lru: OrderedDict[bytes, None] = field(
        default_factory=OrderedDict, init=False
    )
    _published_entry_leases: dict[bytes, PICLeaseToken] = field(
        default_factory=dict, init=False
    )
    _active_request_leases: dict[str, dict[bytes, PICLeaseToken]] = field(
        default_factory=dict, init=False
    )
    # Remote hybrid operators are imported into a worker-local handle table.
    # They are keyed by the Mooncake object key rather than by a request
    # segment index, because the same published segment may be consumed by
    # several requests concurrently.
    _external_transition_handles: dict[str, int] = field(
        default_factory=dict, init=False
    )
    _external_transition_requests: dict[str, set[str]] = field(
        default_factory=dict, init=False
    )
    _next_external_transition_handle: int = field(default=-1, init=False)
    # Keep the raw operator for partial reuse, but use a compact operator for
    # complete-segment reuse when that representation is available.
    _compact_transitions: dict[int, PICTransitionOperator] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self.store = PICSnapshotStore(self.pool)
        if self.lease_registry is None:
            self.lease_registry = PICLeaseRegistry()

    def _debug(self, message: str, *args: object) -> None:
        if self.debug:
            if args:
                message = message % args
            logger.warning("[PIC-DEBUG] %s", message)

    def _remember_transition_operator(
        self, handle: int, operator: PICTransitionOperator
    ) -> None:
        """Retain raw and, when possible, compact full-reuse operators.

        Every operator is addressed in segment-local coordinates: a captured one
        covers the whole segment from zero, a loaded one covers the published
        span, whose start is the seam.  The captured operator is built local, so
        no conversion happens here.
        """
        stored = operator.detached_clone()
        self._transitions[handle] = stored
        self._compact_transitions.pop(handle, None)
        compact = stored.compact_for_full_reuse()
        if compact is None:
            return
        if compact is stored:
            # Already the length-independent form: alias it instead of cloning,
            # so a long segment does not keep two copies of the same operator.
            self._compact_transitions[handle] = stored
        else:
            self._compact_transitions[handle] = compact.detached_clone()

    def _operator_for_range(
        self,
        handle: int,
        local_start: int | None,
        local_end: int | None,
    ) -> PICTransitionOperator:
        operator = self._transitions.get(handle)
        if operator is None:
            raise KeyError(f"PIC transition handle {handle} is not live on worker")
        if local_start is None and local_end is None:
            return self._compact_transitions.get(handle, operator)
        if local_start is None or local_end is None:
            raise ValueError("PIC transition slice needs both bounds")
        # Callers pass segment-local coordinates while an operator may start at
        # a non-zero offset (the span published for the reused range), so both
        # bounds are shifted before they are compared with the operator's range.
        length = operator.token_end - operator.token_start
        relative_start = int(local_start) - int(operator.token_start)
        relative_end = int(local_end) - int(operator.token_start)
        if relative_start < 0 or relative_end > length or relative_end <= relative_start:
            # The caller may already have expressed the range against the operator instead of
            # against the segment. Accepting both costs one comparison and removes a whole class of
            # mismatch between a published window and the range a request asks for.
            if 0 <= int(local_start) and int(local_end) <= length and int(local_end) > int(local_start):
                relative_start = int(local_start)
                relative_end = int(local_end)
            else:
                raise PICWorkerUnsupported(
                    "PIC transition range is outside the published span: "
                    f"operator=[{operator.token_start},{operator.token_end}) "
                    f"requested=[{local_start},{local_end})"
                )
        if relative_start == 0 and relative_end == length:
            return self._compact_transitions.get(handle, operator)
        if relative_start < 0 or relative_end > length or relative_end <= relative_start:
            raise PICWorkerUnsupported(
                "PIC transition range is outside the published span: "
                f"operator=[{operator.token_start},{operator.token_end}) "
                f"requested=[{local_start},{local_end})"
            )
        return operator.slice(relative_start, relative_end)

    def transition_representation(
        self,
        handle: int,
        local_start: int | None = None,
        local_end: int | None = None,
    ) -> str:
        """Describe the operator selected for the requested reuse range."""
        raw = self._transitions.get(handle)
        if raw is None:
            raise KeyError(f"PIC transition handle {handle} is not live on worker")
        length = raw.token_end - raw.token_start
        full_range = (
            local_start is None
            or local_end is None
            or (local_start == 0 and local_end == length)
        )
        selected = (
            self._compact_transitions.get(handle, raw)
            if full_range
            else raw
        )
        if handle in self._compact_transitions and selected is not raw:
            return "compact"
        if all(
            isinstance(item, (PICAffineTransition, PICLinearTransition))
            for item in selected.transitions
        ) and all(item.source_token_count is not None for item in selected.conv_transitions):
            return "compact"
        if any(isinstance(item, PICGDNTransition) for item in selected.transitions):
            return "raw"
        return "structured"

    def acquire_segment(self, request_id: str, seg_hash: bytes) -> bool:
        """Hold a request lease while a published snapshot is being used."""
        if seg_hash not in self._published_segment_owners:
            return False
        assert self.lease_registry is not None
        request_leases = self._active_request_leases.setdefault(request_id, {})
        if seg_hash not in request_leases:
            request_leases[seg_hash] = self.lease_registry.acquire(
                ("snapshot", seg_hash),
                owner=f"request:{request_id}",
                kind="active_request",
            )
        self._active_published_segments.setdefault(request_id, set()).add(seg_hash)
        self._published_lru.move_to_end(seg_hash, last=True)
        return True

    def _evict_published_segment(self, seg_hash: bytes) -> bool:
        """Evict an unleased published snapshot and notify the scheduler."""
        if any(
            seg_hash in segment_hashes
            for segment_hashes in self._active_published_segments.values()
        ):
            return False
        assert self.lease_registry is not None
        entry_lease = self._published_entry_leases.pop(seg_hash, None)
        if entry_lease is not None:
            self.lease_registry.release(entry_lease)
        self.lease_registry.retire(("snapshot", seg_hash))
        owner_key = self._published_segment_owners.pop(seg_hash, None)
        if owner_key is None:
            self._published_lru.pop(seg_hash, None)
            return False
        snapshot = self._snapshots.pop(owner_key, None)
        self._published.discard(owner_key)
        self._published_lru.pop(seg_hash, None)
        if snapshot is not None:
            if snapshot.transition_state_handle is not None:
                self._transitions.pop(snapshot.transition_state_handle, None)
                self._compact_transitions.pop(snapshot.transition_state_handle, None)
            self.store.release(snapshot)
        if self.on_evict is not None:
            try:
                self.on_evict(seg_hash)
            except Exception:
                logger.exception(
                    "PIC eviction cleanup callback failed for segment=%s",
                    seg_hash[:12],
                )
        self._pending.append(
            PICMaterialization(
                request_id=owner_key[0],
                segment_index=-1,
                seg_hash=seg_hash,
                evicted=True,
            )
        )
        if self.metrics is not None:
            self.metrics.record_eviction()
        self._debug(
            "published snapshot evicted owner=%s segment=%s free_bytes=%d "
            "capacity_bytes=%d used_bytes=%d free_ranges=%d largest_free=%d",
            owner_key[0],
            seg_hash[:12],
            self.pool.free_bytes,
            self.pool.capacity_bytes,
            self.pool.capacity_bytes - self.pool.free_bytes,
            self.pool.free_range_count,
            self.pool.largest_free_bytes,
        )
        return True

    def _evict_one_unleased_published(self) -> bool:
        for seg_hash in tuple(self._published_lru):
            if self._evict_published_segment(seg_hash):
                return True
        return False

    def _capture_with_eviction(
        self,
        segment: PICSegment,
        *,
        recurrent_state: Sequence[torch.Tensor],
        conv_tail: Sequence[torch.Tensor],
        full_kv: Sequence[torch.Tensor],
        native_kv_refs: Sequence[PICNativeKVReference],
        transition_operator: PICTransitionOperator | None,
    ) -> PICPhysicalSnapshot:
        """Capture, evicting only unleased public snapshots on pool pressure."""
        while True:
            try:
                if transition_operator is not None:
                    return self.store.capture_transition_operator(
                        segment,
                        transition_operator,
                        full_kv=full_kv,
                        recurrent_state=recurrent_state,
                        conv_tail=conv_tail,
                        native_kv_refs=native_kv_refs,
                    )
                return self.store.capture(
                    segment,
                    full_kv=full_kv,
                    recurrent_state=recurrent_state,
                    conv_tail=conv_tail,
                    native_kv_refs=native_kv_refs,
                )
            except MemoryError as exc:
                if pic_debug_enabled():
                    # The pool message names the size that failed to fit, which is
                    # the only way to tell a full pool from an oversized request.
                    logger.warning(
                        "[PIC-TRACE] pool-exhausted segment=%s length=%d "
                        "error=%s free_bytes=%d free_ranges=%d largest_free=%d",
                        segment.seg_hash[:12],
                        segment.length,
                        exc,
                        self.pool.free_bytes,
                        self.pool.free_range_count,
                        self.pool.largest_free_bytes,
                    )
                if not self._evict_one_unleased_published():
                    raise

    def _materialization_from_snapshot(
        self,
        request_id: str,
        segment_index: int,
        segment: PICSegment,
        snapshot: PICPhysicalSnapshot,
        native_kv_refs: Sequence[PICNativeKVReference],
        external_transition_key: str | None = None,
    ) -> PICMaterialization:
        """Queue scheduler metadata for an existing worker snapshot."""
        materialization = PICMaterialization(
            request_id=request_id,
            segment_index=segment_index,
            seg_hash=segment.seg_hash,
            full_kv_handles=snapshot.full_kv_handles,
            recurrent_state_handle=snapshot.recurrent_state_handle,
            transition_state_handle=snapshot.transition_state_handle,
            conv_tail_handle=snapshot.conv_tail_handle,
            native_kv_refs=tuple(native_kv_refs),
            external_transition_key=external_transition_key,
        )
        self._pending.append(materialization)
        return materialization

    def capture_segment(
        self,
        request_id: str,
        segment_index: int,
        segment: PICSegment,
        *,
        recurrent_state: Sequence[torch.Tensor] = (),
        conv_tail: Sequence[torch.Tensor] = (),
        full_kv: Sequence[torch.Tensor] = (),
        native_kv_refs: Sequence[PICNativeKVReference] = (),
        transition_operator: PICTransitionOperator | None = None,
        external_transition_key: str | None = None,
    ) -> PICMaterialization | None:
        """Capture one worker-owned segment and queue its metadata.

        The input tensors are live views supplied by the worker.  The pool
        immediately copies their contents, so later model execution cannot
        mutate the snapshot.  Existing snapshots for the same request/segment
        are released only after the replacement has been captured.
        """
        self._debug(
            "capture_segment entry request=%s segment=%d range=[%d,%d) "
            "recurrent=%d conv_tail=%d full_kv=%d transition=%s",
            request_id,
            segment_index,
            segment.start,
            segment.end,
            len(recurrent_state),
            len(conv_tail),
            len(full_kv),
            transition_operator is not None,
        )
        if not (
            recurrent_state
            or conv_tail
            or full_kv
            or native_kv_refs
            or transition_operator is not None
        ):
            self._debug(
                "capture_segment returned None "
                "request=%s segment=%d reason=empty_state",
                request_id,
                segment_index,
            )
            return None

        # A published transition operator and its zero-start state are keyed by
        # segment content, not by the request that happened to warm it.  Avoid
        # allocating another copy when the same segment is warmed repeatedly.
        # Native KV references are deliberately not reused here: the scheduler
        # may have reserved different public native blocks for this request.
        key = (request_id, segment_index)
        if self.retain_published and transition_operator is not None:
            owner_key = self._published_segment_owners.get(segment.seg_hash)
            if owner_key is not None and owner_key != key:
                published = self._snapshots.get(owner_key)
                if (
                    published is not None
                    and published.transition_state_handle is not None
                ):
                    materialization = self._materialization_from_snapshot(
                        request_id,
                        segment_index,
                        segment,
                        published,
                        native_kv_refs,
                        external_transition_key,
                    )
                    self._debug(
                        "capture deduplicated request=%s segment=%d owner=%s "
                        "transition_handle=%s",
                        request_id,
                        segment_index,
                        owner_key[0],
                        published.transition_state_handle,
                    )
                    if self.metrics is not None:
                        self.metrics.record_capture(deduplicated=True)
                    self.acquire_segment(request_id, segment.seg_hash)
                    return materialization

        if transition_operator is not None:
            # Only the segment-level form is stored: the per-token form costs
            # hundreds of kilobytes per token, so a few documents exhaust the
            # pool on their own.  Callers hand over an operator that already
            # covers exactly the reusable span, so nothing has to be sliced out
            # of it afterwards.
            compacted = transition_operator.compact_for_full_reuse()
            if compacted is None:
                logger.warning(
                    "PIC transition compaction unavailable request=%s segment=%d "
                    "tokens=%d; storing the per-token operator",
                    request_id,
                    segment_index,
                    transition_operator.token_end - transition_operator.token_start,
                )
            else:
                transition_operator = compacted
        try:
            snapshot = self._capture_with_eviction(
                segment,
                recurrent_state=recurrent_state,
                conv_tail=conv_tail,
                full_kv=full_kv,
                native_kv_refs=native_kv_refs,
                transition_operator=transition_operator,
            )
        except Exception:
            logger.warning(
                "[PIC-DEBUG] capture_segment failed request=%s segment=%d",
                request_id,
                segment_index,
                exc_info=True,
            )
            raise
        previous = self._snapshots.get(key)
        self._snapshots[key] = snapshot
        if snapshot.transition_state_handle is not None:
            if transition_operator is None:
                self.store.release(snapshot)
                self._snapshots.pop(key, None)
                raise ValueError(
                    "PIC transition snapshot requires a structured operator"
                )
            self._remember_transition_operator(
                snapshot.transition_state_handle, transition_operator
            )
        if previous is not None:
            if previous.transition_state_handle is not None:
                self._transitions.pop(previous.transition_state_handle, None)
                self._compact_transitions.pop(previous.transition_state_handle, None)
            self.store.release(previous)
        if self.retain_published:
            self._published.add(key)
            self._published_segment_owners[segment.seg_hash] = key
            self._published_lru[segment.seg_hash] = None
            assert self.lease_registry is not None
            if segment.seg_hash not in self._published_entry_leases:
                self._published_entry_leases[segment.seg_hash] = (
                    self.lease_registry.acquire(
                        ("snapshot", segment.seg_hash),
                        owner=f"entry:{segment.seg_hash.hex()}",
                        kind="cache_entry",
                    )
                )

        materialization = self._materialization_from_snapshot(
            request_id,
            segment_index,
            segment,
            snapshot,
            native_kv_refs,
            external_transition_key,
        )
        if self.metrics is not None:
            self.metrics.record_capture()
        self._debug(
            "capture_segment success request=%s segment=%d "
            "recurrent_handle=%s transition_handle=%s conv_handle=%s "
            "full_kv_handles=%s",
            request_id,
            segment_index,
            materialization.recurrent_state_handle,
            materialization.transition_state_handle,
            materialization.conv_tail_handle,
            materialization.full_kv_handles,
        )
        return materialization

    def take_pending(self) -> list[PICMaterialization]:
        """Return materializations once, for inclusion in ModelRunnerOutput."""
        pending = self._pending
        self._pending = []
        return pending

    def import_transition_operator(
        self,
        *,
        request_id: str,
        external_key: str,
        operator: PICTransitionOperator,
    ) -> int:
        """Install a Mooncake transition in the worker-local handle table."""
        key = str(external_key)
        handle = self._external_transition_handles.get(key)
        if handle is None:
            # Snapshot handles are allocated by PICSnapshotStore and are
            # non-negative.  Negative handles keep the remote namespace
            # unambiguous without changing the physical snapshot format.
            handle = self._next_external_transition_handle
            self._next_external_transition_handle -= 1
            self._external_transition_handles[key] = handle
            self._remember_transition_operator(handle, operator)
        self._external_transition_requests.setdefault(key, set()).add(request_id)
        return handle

    def has_transition(self, transition_handle: int | None) -> bool:
        """Return whether a transition handle is still live on this worker."""
        return transition_handle is not None and transition_handle in self._transitions

    def release_request(self, request_id: str) -> None:
        """Release snapshots not retained by the scheduler-side PIC cache."""
        self._active_published_segments.pop(request_id, None)
        assert self.lease_registry is not None
        for token in self._active_request_leases.pop(request_id, {}).values():
            self.lease_registry.release(token)
        for key, request_ids in tuple(self._external_transition_requests.items()):
            request_ids.discard(request_id)
            if request_ids:
                continue
            self._external_transition_requests.pop(key, None)
            handle = self._external_transition_handles.pop(key, None)
            if handle is not None:
                self._transitions.pop(handle, None)
                self._compact_transitions.pop(handle, None)
        keys = [key for key in self._snapshots if key[0] == request_id]
        for key in keys:
            if self.retain_published and key in self._published:
                continue
            snapshot = self._snapshots.pop(key)
            if snapshot.transition_state_handle is not None:
                self._transitions.pop(snapshot.transition_state_handle, None)
                self._compact_transitions.pop(snapshot.transition_state_handle, None)
            self.store.release(snapshot)
            self._published.discard(key)

    def apply_transition(
        self,
        transition_handle: int,
        recurrent_state: Sequence[torch.Tensor],
        *,
        local_start: int | None = None,
        local_end: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Apply a worker-local transition handle to the current state.

        The handle is deliberately worker-local.  Operators captured in this
        process and operators imported from Mooncake both enter the same
        table, while only the external object key crosses the scheduler
        boundary.
        """
        operator = self._operator_for_range(
            transition_handle, local_start, local_end
        )
        # The tensors of an imported operator arrive on the copy engine while the prompt computes, so
        # this is where the wait belongs: only now is the data about to be read.
        wait_pending_upload(operator)
        return operator.apply(recurrent_state)

    def apply_conv_transition(
        self,
        transition_handle: int,
        conv_state: Sequence[torch.Tensor],
        *,
        local_start: int | None = None,
        local_end: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Apply the rolling conv-state part of a worker-local transition."""
        operator = self._operator_for_range(
            transition_handle, local_start, local_end
        )
        if not operator.conv_transitions:
            raise KeyError(
                f"PIC transition handle {transition_handle} has no conv operator"
            )
        wait_pending_upload(operator)
        return operator.apply_conv(conv_state)

    def has_conv_transition(self, transition_handle: int) -> bool:
        """Return whether a live operator can advance a conv state."""
        operator = self._transitions.get(transition_handle)
        return operator is not None and bool(operator.conv_transitions)

    def restore_conv_tail(
        self,
        conv_tail_handle: int | None,
        conv_tail: Sequence[torch.Tensor],
    ) -> None:
        """Restore only the cached conv tail without touching transition state."""
        if conv_tail_handle is None:
            return
        self.store.pool.restore_tensor_snapshot(conv_tail_handle, conv_tail)

    def restore_materialized_segment(
        self,
        segment: PICSegment,
        entry: PICSegmentEntry,
        *,
        request_id: str,
        recurrent_state: Sequence[torch.Tensor] = (),
        transition_state: Sequence[torch.Tensor] = (),
        conv_tail: Sequence[torch.Tensor] = (),
    ) -> None:
        """Restore a scheduler materialization into caller-owned state tensors."""
        self._debug(
            "restore_segment entry hash=%s range=[%d,%d) recurrent=%d "
            "conv_tail=%d transition=%d full_kv_handles=%s",
            segment.seg_hash[:12],
            segment.start,
            segment.end,
            len(recurrent_state),
            len(conv_tail),
            len(transition_state),
            entry.full_kv_handles,
        )
        if entry.seg_hash != segment.seg_hash:
            raise ValueError("PIC materialization hash does not match segment")
        self.acquire_segment(request_id, segment.seg_hash)
        snapshot = PICPhysicalSnapshot(
            seg_hash=entry.seg_hash,
            token_start=0,
            token_end=segment.length,
            full_kv_handles=entry.full_kv_handles,
            recurrent_state_handle=entry.recurrent_state_handle,
            transition_state_handle=entry.transition_state_handle,
            conv_tail_handle=entry.conv_tail_handle,
        )
        try:
            self.store.restore(
                snapshot,
                recurrent_state=recurrent_state,
                transition_state=transition_state,
                conv_tail=conv_tail,
            )
        except Exception:
            logger.warning(
                "[PIC-DEBUG] restore_segment failed hash=%s range=[%d,%d)",
                segment.seg_hash[:12],
                segment.start,
                segment.end,
                exc_info=True,
            )
            raise
        self._debug(
            "restore_segment success hash=%s range=[%d,%d)",
            segment.seg_hash[:12],
            segment.start,
            segment.end,
        )

    def clear(self) -> None:
        """Release all worker-local snapshots and queued metadata."""
        assert self.lease_registry is not None
        if self.on_evict is not None:
            for seg_hash in tuple(self._published_segment_owners):
                try:
                    self.on_evict(seg_hash)
                except Exception:
                    logger.exception(
                        "PIC clear cleanup callback failed for segment=%s",
                        seg_hash[:12],
                    )
        for request_leases in self._active_request_leases.values():
            for token in request_leases.values():
                self.lease_registry.release(token)
        for token in self._published_entry_leases.values():
            self.lease_registry.release(token)
        for seg_hash in self._published_entry_leases:
            self.lease_registry.retire(("snapshot", seg_hash))
        self._active_request_leases.clear()
        self._published_entry_leases.clear()
        for snapshot in self._snapshots.values():
            self.store.release(snapshot)
        self._snapshots.clear()
        self._transitions.clear()
        self._compact_transitions.clear()
        self._external_transition_handles.clear()
        self._external_transition_requests.clear()
        self._next_external_transition_handle = -1
        self._published.clear()
        self._published_segment_owners.clear()
        self._active_published_segments.clear()
        self._published_lru.clear()
        self._pending.clear()
        self.pool.clear()


def _collect_mamba_state_tensors(
    *,
    request_id: str,
    segment: PICSegment,
    request_state: CachedRequestState,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    state_position: int | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Collect state tensors for one request and one exact segment boundary."""
    layout = PICStateLayout.from_kv_cache_config(kv_cache_config)
    recurrent: list[torch.Tensor] = []
    conv_tail: list[torch.Tensor] = []

    for group_id, group_layout in enumerate(layout.groups):
        if not group_layout:
            continue
        group = kv_cache_config.kv_cache_groups[group_id]
        block_size = int(group.kv_cache_spec.block_size)
        position = segment.end - 1 if state_position is None else state_position
        if position < 0:
            raise ValueError("PIC state position must be non-negative")
        block_index = position // block_size
        block_ids = request_state.block_ids[group_id]
        if block_index >= len(block_ids):
            raise ValueError(
                "PIC state restore block is not allocated: "
                f"request={request_id}, group={group_id}, block={block_index}"
            )
        block_id = block_ids[block_index]

        for layer_name in group.layer_names:
            attention = forward_context.get(layer_name)
            if attention is None or not hasattr(attention, "kv_cache"):
                raise ValueError(
                    f"PIC state restore cannot find Mamba state for {layer_name}"
                )
            state_tensors = attention.kv_cache
            if len(state_tensors) != len(group_layout):
                raise ValueError(
                    "PIC state restore count does not match layout: "
                    f"layer={layer_name}, expected={len(group_layout)}, "
                    f"actual={len(state_tensors)}"
                )

            for state_tensor, state_spec in zip(state_tensors, group_layout):
                tensor = state_tensor[block_id]
                if tuple(tensor.shape) != state_spec.shape:
                    raise ValueError(
                        "PIC state restore shape does not match layout: "
                        f"layer={layer_name}, group={group_id}, "
                        f"index={state_spec.state_index}"
                    )
                if tensor.dtype != state_spec.dtype:
                    raise ValueError(
                        "PIC state restore dtype does not match layout: "
                        f"layer={layer_name}, group={group_id}, "
                        f"index={state_spec.state_index}"
                    )
                if state_spec.kind.name == "CONV_TAIL":
                    conv_tail.append(tensor)
                elif state_spec.kind.name == "RECURRENT_STATE":
                    recurrent.append(tensor)
                else:
                    raise ValueError(
                        "PIC state restore encountered unsupported state kind: "
                        f"{state_spec.kind!r}"
                    )

    return tuple(recurrent), tuple(conv_tail)


def apply_matched_mamba_transition(
    manager: PICLiveCaptureManager,
    *,
    request_id: str,
    segment_index: int,
    segment: PICSegment,
    entry: PICSegmentEntry,
    request_state: CachedRequestState,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    state_position: int,
    local_start: int = 0,
    local_end: int | None = None,
) -> None:
    """Apply a cached segment transition to the target request's live state."""
    detail_profile = _pic_detail_profile_enabled()
    total_start = time.perf_counter() if detail_profile else 0.0
    acquire_start = time.perf_counter() if detail_profile else 0.0
    if entry.transition_state_handle is None:
        raise ValueError(
            "PIC hybrid runtime requires a transition handle for reused segment"
        )
    manager.acquire_segment(request_id, segment.seg_hash)
    acquire_ms = (
        (time.perf_counter() - acquire_start) * 1000.0 if detail_profile else 0.0
    )
    if local_end is None:
        local_end = segment.length
    collect_start = time.perf_counter() if detail_profile else 0.0
    recurrent, conv_tail = _collect_mamba_state_tensors(
        request_id=request_id,
        segment=segment,
        request_state=request_state,
        kv_cache_config=kv_cache_config,
        forward_context=forward_context,
        state_position=state_position,
    )
    collect_ms = (
        (time.perf_counter() - collect_start) * 1000.0 if detail_profile else 0.0
    )
    # A full reuse of the first segment starts from the model's initial
    # recurrent state, not from whatever allocator block happened to back the
    # new request.  Native vLLM normally establishes this through ordinary
    # prefill writes; PIC skips that body, so explicitly clear both GDN and
    # causal-conv state before applying the imported/cached transition.
    # Partial reuse (or a later segment) must retain the state produced by the
    # preceding recompute/reuse range.
    reset_ms = 0.0
    if segment.start == 0 and local_start == 0:
        reset_start = time.perf_counter() if detail_profile else 0.0
        for state in recurrent:
            state.zero_()
        for state in conv_tail:
            state.zero_()
        reset_ms = (
            (time.perf_counter() - reset_start) * 1000.0
            if detail_profile
            else 0.0
        )
        manager._debug(
            "transition initial state reset request=%s segment=%d",
            request_id,
            segment_index,
        )
    if (
        conv_tail
        and local_end < segment.length
        and not manager.has_conv_transition(entry.transition_state_handle)
    ):
        raise PICWorkerUnsupported(
            "PIC partial reuse has no conv transition for the live state"
        )
    apply_start = time.perf_counter() if detail_profile else 0.0
    updated = manager.apply_transition(
        entry.transition_state_handle,
        recurrent,
        local_start=local_start,
        local_end=local_end,
    )
    recurrent_apply_ms = (
        (time.perf_counter() - apply_start) * 1000.0 if detail_profile else 0.0
    )
    if len(updated) != len(recurrent):
        raise ValueError("PIC transition returned an unexpected state count")
    copy_start = time.perf_counter() if detail_profile else 0.0
    for target, source in zip(recurrent, updated):
        target.copy_(source)
    recurrent_copy_ms = (
        (time.perf_counter() - copy_start) * 1000.0 if detail_profile else 0.0
    )
    conv_apply_ms = 0.0
    conv_copy_ms = 0.0
    conv_restore_ms = 0.0
    if conv_tail:
        try:
            conv_apply_start = time.perf_counter() if detail_profile else 0.0
            updated_conv = manager.apply_conv_transition(
                entry.transition_state_handle,
                conv_tail,
                local_start=local_start,
                local_end=local_end,
            )
            conv_apply_ms = (
                (time.perf_counter() - conv_apply_start) * 1000.0
                if detail_profile
                else 0.0
            )
        except KeyError as err:
            if local_end < segment.length:
                raise PICWorkerUnsupported(
                    "PIC partial reuse has no conv transition for the live state"
                ) from err
            restore_start = time.perf_counter() if detail_profile else 0.0
            manager.restore_conv_tail(entry.conv_tail_handle, conv_tail)
            conv_restore_ms = (
                (time.perf_counter() - restore_start) * 1000.0
                if detail_profile
                else 0.0
            )
        else:
            if len(updated_conv) != len(conv_tail):
                raise ValueError("PIC conv transition returned unexpected state count")
            conv_copy_start = time.perf_counter() if detail_profile else 0.0
            for target, source in zip(conv_tail, updated_conv):
                target.copy_(source)
            conv_copy_ms = (
                (time.perf_counter() - conv_copy_start) * 1000.0
                if detail_profile
                else 0.0
            )
    manager._debug(
        "transition applied request=%s segment=%d range=[%d,%d) state_position=%d",
        request_id,
        segment_index,
        segment.start + local_start,
        segment.start + local_end,
        state_position,
    )
    if detail_profile:
        representation = manager.transition_representation(
            entry.transition_state_handle,
            local_start=local_start,
            local_end=local_end,
        )
        logger.warning(
            "[PIC-PROFILE] transition-detail request=%s segment=%d "
            "representation=%s acquire_ms=%.3f collect_ms=%.3f reset_ms=%.3f "
            "recurrent_apply_ms=%.3f recurrent_copy_ms=%.3f "
            "conv_apply_ms=%.3f conv_copy_ms=%.3f conv_restore_ms=%.3f "
            "total_ms=%.3f",
            request_id,
            segment_index,
            representation,
            acquire_ms,
            collect_ms,
            reset_ms,
            recurrent_apply_ms,
            recurrent_copy_ms,
            conv_apply_ms,
            conv_copy_ms,
            conv_restore_ms,
            (time.perf_counter() - total_start) * 1000.0,
        )


def capture_completed_mamba_segment(
    manager: PICLiveCaptureManager,
    *,
    request_id: str,
    segment_index: int,
    segment: PICSegment,
    completed_end: int,
    request_state: CachedRequestState,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    transition_operator: PICTransitionOperator | None = None,
) -> PICMaterialization | None:
    """Capture Mamba/GDN state at a completed segment boundary.

    vLLM stores one state page per Mamba layer.  We collect all layers for a
    state kind into one PIC allocation, preserving the authoritative ordering
    from ``PICStateLayout``.  A segment is captured only when the current
    worker step ends exactly at that segment; this prevents accidentally
    capturing a state that already includes later prompt tokens.
    """
    if segment.end != completed_end or segment.end <= segment.start:
        return None

    recurrent, conv_tail = _collect_mamba_state_tensors(
        request_id=request_id,
        segment=segment,
        request_state=request_state,
        kv_cache_config=kv_cache_config,
        forward_context=forward_context,
    )

    native_kv_refs = _collect_native_kv_references(
        request_id=request_id,
        segment_index=segment_index,
        segment=segment,
        request_state=request_state,
        kv_cache_config=kv_cache_config,
    )

    return manager.capture_segment(
        request_id,
        segment_index,
        segment,
        recurrent_state=tuple(recurrent),
        conv_tail=tuple(conv_tail),
        native_kv_refs=native_kv_refs,
        transition_operator=transition_operator,
        external_transition_key=getattr(
            request_state, "pic_mooncake_transition_keys", {}
        ).get(segment_index),
    )


def _pic_public_allocation(
    request_state: CachedRequestState, group_id: int
) -> Any | None:
    """Return the public allocation the worker published for this group, if any."""
    allocations = getattr(request_state, "pic_native_kv_allocations", {})
    for (candidate_group, _start, _end), allocation in allocations.items():
        if int(candidate_group) == int(group_id) and allocation.public_block_ids:
            return allocation
    return None


def _collect_native_kv_references(
    *,
    request_id: str,
    segment_index: int,
    segment: PICSegment,
    request_state: CachedRequestState,
    kv_cache_config: KVCacheConfig,
) -> tuple[PICNativeKVReference, ...]:
    """Describe complete *segment-local* attention blocks.

    The PIC segment is an independent object whose logical origin is zero.
    Therefore the reusable span is computed from ``len(segment)`` rather than
    requiring ``segment.start``/``segment.end`` to align with the source
    request's global block table.  The source coordinates are retained on the
    reference so a later token-level bridge or request-private copy can read
    the exact source slots without exposing neighbouring tokens.

    A reference is emitted when the worker either filled an independent
    canonical public allocation or explicitly recorded an exact-position
    request-block alias.  Shifted request-absolute source blocks are never
    mislabeled as reusable canonical KV.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

    references: list[PICNativeKVReference] = []
    public_blocks = {
        (int(segment_index), int(group_id)): tuple(
            int(block_id) for block_id in block_ids
        )
        for segment_index, group_id, block_ids in getattr(
            request_state, "pic_public_block_ids", ()
        )
    }
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not isinstance(spec, AttentionSpec) or isinstance(spec, CrossAttentionSpec):
            continue
        block_size = int(spec.block_size)
        local_span = get_covering_local_block_span(
            segment.end - segment.start, block_size
        )
        if local_span.block_count == 0:
            continue

        block_ids = public_blocks.get((segment_index, group_id))
        if block_ids is None:
            # A public allocation is required before a canonical reference can
            # be published.  Never label request-absolute KV as canonical: it
            # would be incorrect for the next request at another position.
            continue
        # The worker has already described this span when it published the KV: an allocation names the
        # blocks it owns, the unit they are counted in, and the tokens they reach. Recomputing any of
        # those here means counting the same window a second time in whatever unit this loop happens
        # to see, and the two counts disagree the moment the units do - one place said nine pages and
        # another a hundred and fifty-two kernel blocks for the same four thousand eight hundred and
        # thirty-nine tokens.
        allocation = _pic_public_allocation(request_state, group_id)
        if allocation is not None:
            block_ids = tuple(int(block_id) for block_id in allocation.public_block_ids)
            block_size = int(allocation.block_size)
            local_span = get_covering_local_block_span(
                int(allocation.target_end) - int(allocation.target_start), block_size
            )
        if len(block_ids) < local_span.block_count:
            raise ValueError(
                "PIC public KV allocation does not cover the segment: "
                f"request={request_id}, group={group_id}, "
                f"range=[{segment.start},{segment.end}) "
                f"blocks={len(block_ids)} expected={local_span.block_count} "
                f"block_size={block_size}"
            )
        external_key = getattr(request_state, "pic_mooncake_external_keys", {}).get(
            (segment_index, group_id)
        )
        reference_block_size = (
            getattr(request_state, "pic_mooncake_external_block_sizes", {}).get(
                (segment_index, group_id), block_size
            )
            if external_key is not None
            else block_size
        )
        references.append(
            PICNativeKVReference(
                kv_cache_group_id=group_id,
                block_ids=() if external_key is not None else block_ids,
                block_size=reference_block_size,
                token_count=local_span.reusable_token_count,
                kind=(
                    PICNativeKVKind.SHARED
                    if is_pic_exact_position_aliasable(segment.start)
                    else PICNativeKVKind.CANONICAL_PUBLIC
                ),
                position_mode=PICKVPositionMode.CANONICAL_LOCAL,
                canonical_start=0,
                source_token_start=0,
                source_block_start=0,
                dtype=str(getattr(spec, "dtype", "")),
                layout=(reference_block_size,),
                external_provider=("mooncake" if external_key is not None else None),
                external_key=external_key,
            )
        )
    return tuple(references)


def capture_transition_operator_segment(
    manager: PICLiveCaptureManager,
    *,
    request_id: str,
    segment_index: int,
    segment: PICSegment,
    request_state: CachedRequestState,
    kv_cache_config: KVCacheConfig,
    operator: PICTransitionOperator,
    conv_tail: Sequence[torch.Tensor] = (),
) -> PICMaterialization | None:
    """Publish a captured operator when no live end-state page is available."""
    native_kv_refs = _collect_native_kv_references(
        request_id=request_id,
        segment_index=segment_index,
        segment=segment,
        request_state=request_state,
        kv_cache_config=kv_cache_config,
    )
    return manager.capture_segment(
        request_id,
        segment_index,
        segment,
        conv_tail=conv_tail,
        native_kv_refs=native_kv_refs,
        transition_operator=operator,
        external_transition_key=getattr(
            request_state, "pic_mooncake_transition_keys", {}
        ).get(segment_index),
    )


def restore_matched_mamba_segment(
    manager: PICLiveCaptureManager,
    *,
    request_id: str,
    segment_index: int,
    segment: PICSegment,
    entry: PICSegmentEntry,
    request_state: CachedRequestState,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
) -> None:
    """Restore a matched scheduler entry into the target request's state block."""
    manager.acquire_segment(request_id, segment.seg_hash)
    if entry.recurrent_state_handle is None and entry.conv_tail_handle is None:
        return
    if entry.transition_state_handle is not None:
        # Stage 9-A-1 publishes the structured transition operator, but the
        # worker does not yet have the Stage 9-A-2 runtime inputs needed to
        # apply it to the current live state.  Restoring the old recurrent or
        # conv end-state directly would silently corrupt the new prefix, so
        # skip the whole legacy restore until the transition execution path is
        # wired.  This is safer than calling PICSnapshotStore.restore with an
        # empty transition tensor list and then partially mutating state.
        manager._debug(
            "restore skipped request=%s segment=%d reason="
            "transition_operator_requires_stage9A2 handle=%s",
            request_id,
            segment_index,
            entry.transition_state_handle,
        )
        return
    recurrent, conv_tail = _collect_mamba_state_tensors(
        request_id=request_id,
        segment=segment,
        request_state=request_state,
        kv_cache_config=kv_cache_config,
        forward_context=forward_context,
    )
    manager.restore_materialized_segment(
        segment,
        entry,
        request_id=request_id,
        recurrent_state=recurrent,
        conv_tail=conv_tail,
    )
