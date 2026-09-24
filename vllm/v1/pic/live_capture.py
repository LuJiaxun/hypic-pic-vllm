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

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import torch

from vllm.logger import init_logger
from vllm.v1.pic.cache import PICSegmentEntry
from vllm.v1.pic.materialization import PICMaterialization
from vllm.v1.pic.metrics import PICMetrics
from vllm.v1.pic.native_kv import (
    PICNativeKVKind,
    PICNativeKVReference,
    PICKVPositionMode,
    get_full_local_block_span,
)
from vllm.v1.pic.pool import PICPhysicalPool
from vllm.v1.pic.segmenter import PICSegment
from vllm.v1.pic.snapshot import PICPhysicalSnapshot, PICSnapshotStore
from vllm.v1.pic.gdn_transition import build_gdn_transition_operator
from vllm.v1.pic.lifecycle import PICLeaseRegistry, PICLeaseToken
from vllm.v1.pic.worker_plan import PICWorkerUnsupported
from vllm.v1.pic.state import (
    PICConvTransition,
    PICStateLayout,
    PICTransitionOperator,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    import torch

    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


def ordered_gdn_layer_names(
    kv_cache_config: "KVCacheConfig",
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

    request_state: "CachedRequestState"
    segments: tuple[PICSegment, ...]
    layer_names: tuple[str, ...]
    debug: bool = False
    _operators: dict[int, dict[str, PICTransitionOperator]] = field(
        default_factory=dict, init=False
    )
    _zero_conv_tails: dict[int, dict[str, "torch.Tensor"]] = field(
        default_factory=dict, init=False
    )
    _conv_transitions: dict[int, dict[str, PICConvTransition]] = field(
        default_factory=dict, init=False
    )

    def _debug(self, message: str, *args: object) -> None:
        if self.debug:
            logger.warning("[PIC-DEBUG] " + message, *args)

    def record_gdn_layer(
        self,
        layer_name: str,
        key: "torch.Tensor",
        value: "torch.Tensor",
        log_decay: "torch.Tensor",
        beta: "torch.Tensor",
        *,
        state_dtype: "torch.dtype",
        conv_input: "torch.Tensor",
        conv_kernel_size: int,
        backend: str | None = None,
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

        for segment_index, segment in enumerate(self.segments):
            if segment.start < 0 or segment.end > key.shape[0]:
                continue
            operator = build_gdn_transition_operator(
                key[segment.start : segment.end],
                value[segment.start : segment.end],
                log_decay[segment.start : segment.end],
                beta[segment.start : segment.end],
                token_start=segment.start,
                token_end=segment.end,
                state_dtype=state_dtype,
                backend=backend,
            )
            conv_tokens = conv_input[segment.start : segment.end]
            state_len = max(conv_kernel_size - 1, 0)
            if state_len == 0:
                zero_conv_tail = conv_tokens.new_empty(
                    (conv_tokens.shape[1], 0)
                )
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
            self._conv_transitions.setdefault(segment_index, {})[
                layer_name
            ] = conv_transition
            self._zero_conv_tails.setdefault(segment_index, {})[
                layer_name
            ] = zero_conv_tail
            if all(name in layer_operators for name in self.layer_names):
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
                        token_start=segment.start,
                        token_end=segment.end,
                    )
                )
                self.request_state.pic_transition_conv_tails[segment_index] = tuple(
                    self._zero_conv_tails[segment_index][name]
                    for name in self.layer_names
                )

    def validate_gdn_layer(
        self,
        layer_name: str,
        initial_state: "torch.Tensor",
        final_state: "torch.Tensor",
        *,
        token_start: int,
        token_end: int,
    ) -> bool | None:
        """Compare a captured operator with the fused model final state.

        The first strict validation target is a single-segment prefill. In
        that case the state entering the fused GDN kernel and the state it
        writes are both available, so the Python transition can be compared
        directly to the real model result. Multi-segment requests still keep
        their operators, but their per-segment intermediate states are left
        for the execution stage to validate at the exact seam.
        """
        if len(self.segments) != 1:
            return None
        segment = self.segments[0]
        if (segment.start, segment.end) != (token_start, token_end):
            return None
        operator = self._operators.get(0, {}).get(layer_name)
        if operator is None:
            return None
        transition = operator.transitions[0]
        if initial_state.ndim == transition.zero_state.ndim + 1:
            initial_state = initial_state[0]
        if final_state.ndim == transition.zero_state.ndim + 1:
            final_state = final_state[0]
        try:
            expected = transition.apply(initial_state)
        except PICWorkerUnsupported as exc:
            self._debug(
                "transition validation skipped layer=%s reason=%s",
                layer_name,
                exc,
            )
            return None
        actual = final_state.to(expected.dtype)
        expected_f = expected.float()
        actual_f = actual.float()
        error = (expected_f - actual_f).abs()
        max_abs = float(error.max().item()) if error.numel() else 0.0
        max_rel = float(
            (error / actual_f.abs().clamp_min(1e-6)).max().item()
        ) if error.numel() else 0.0
        relative_l2 = float(
            error.norm().item() / actual_f.norm().clamp_min(1e-6).item()
        ) if error.numel() else 0.0
        passed = bool(torch.allclose(expected_f, actual_f, rtol=5e-2, atol=5e-2))
        self.request_state.pic_transition_validation[layer_name] = passed
        self._debug(
            "transition validation layer=%s range=[%d,%d) passed=%s "
            "max_abs=%.6g max_rel=%.6g relative_l2=%.6g "
            "expected_dtype=%s actual_dtype=%s",
            layer_name,
            token_start,
            token_end,
            passed,
            max_abs,
            max_rel,
            relative_l2,
            expected.dtype,
            final_state.dtype,
        )
        if not passed:
            logger.warning(
                "PIC GDN transition validation failed for layer %s range [%d,%d)",
                layer_name,
                token_start,
                token_end,
            )
        return passed


@dataclass
class PICLiveCaptureManager:
    """Own worker-local PIC snapshots and pending scheduler metadata."""

    pool: PICPhysicalPool
    retain_published: bool = False
    debug: bool = False
    lease_registry: PICLeaseRegistry | None = None
    metrics: PICMetrics | None = None
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

    def __post_init__(self) -> None:
        self.store = PICSnapshotStore(self.pool)
        if self.lease_registry is None:
            self.lease_registry = PICLeaseRegistry()

    def _debug(self, message: str, *args: object) -> None:
        if self.debug:
            logger.warning("[PIC-DEBUG] " + message, *args)

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
            self.store.release(snapshot)
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
            "published snapshot evicted owner=%s segment=%s free_bytes=%d",
            owner_key[0],
            seg_hash[:12],
            self.pool.free_bytes,
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
        recurrent_state: Sequence["torch.Tensor"],
        conv_tail: Sequence["torch.Tensor"],
        full_kv: Sequence["torch.Tensor"],
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
            except MemoryError:
                if not self._evict_one_unleased_published():
                    raise

    def _materialization_from_snapshot(
        self,
        request_id: str,
        segment_index: int,
        segment: PICSegment,
        snapshot: PICPhysicalSnapshot,
        native_kv_refs: Sequence[PICNativeKVReference],
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
        )
        self._pending.append(materialization)
        return materialization

    def capture_segment(
        self,
        request_id: str,
        segment_index: int,
        segment: PICSegment,
        *,
        recurrent_state: Sequence["torch.Tensor"] = (),
        conv_tail: Sequence["torch.Tensor"] = (),
        full_kv: Sequence["torch.Tensor"] = (),
        native_kv_refs: Sequence[PICNativeKVReference] = (),
        transition_operator: PICTransitionOperator | None = None,
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
                "capture_segment returned None request=%s segment=%d reason=empty_state",
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
                if published is not None and published.transition_state_handle is not None:
                    materialization = self._materialization_from_snapshot(
                        request_id,
                        segment_index,
                        segment,
                        published,
                        native_kv_refs,
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
            self._transitions[snapshot.transition_state_handle] = (
                transition_operator.detached_clone()
            )
        if previous is not None:
            if previous.transition_state_handle is not None:
                self._transitions.pop(previous.transition_state_handle, None)
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

    def release_request(self, request_id: str) -> None:
        """Release snapshots not retained by the scheduler-side PIC cache."""
        self._active_published_segments.pop(request_id, None)
        assert self.lease_registry is not None
        for token in self._active_request_leases.pop(request_id, {}).values():
            self.lease_registry.release(token)
        keys = [key for key in self._snapshots if key[0] == request_id]
        for key in keys:
            if self.retain_published and key in self._published:
                continue
            snapshot = self._snapshots.pop(key)
            if snapshot.transition_state_handle is not None:
                self._transitions.pop(snapshot.transition_state_handle, None)
            self.store.release(snapshot)
            self._published.discard(key)

    def apply_transition(
        self,
        transition_handle: int,
        recurrent_state: Sequence["torch.Tensor"],
        *,
        local_start: int | None = None,
        local_end: int | None = None,
    ) -> tuple["torch.Tensor", ...]:
        """Apply a worker-local transition handle to the current state.

        The handle is deliberately not sufficient to reconstruct the operator
        across processes.  The structured operator remains worker-local until
        a later stage defines scheduler/worker transport for it.
        """
        operator = self._transitions.get(transition_handle)
        if operator is None:
            raise KeyError(
                f"PIC transition handle {transition_handle} is not live on worker"
            )
        if local_start is not None or local_end is not None:
            if local_start is None or local_end is None:
                raise ValueError("PIC transition slice needs both bounds")
            operator = operator.slice(local_start, local_end)
        return operator.apply(recurrent_state)

    def apply_conv_transition(
        self,
        transition_handle: int,
        conv_state: Sequence["torch.Tensor"],
        *,
        local_start: int | None = None,
        local_end: int | None = None,
    ) -> tuple["torch.Tensor", ...]:
        """Apply the rolling conv-state part of a worker-local transition."""
        operator = self._transitions.get(transition_handle)
        if operator is None:
            raise KeyError(
                f"PIC transition handle {transition_handle} is not live on worker"
            )
        if not operator.conv_transitions:
            raise KeyError(
                f"PIC transition handle {transition_handle} has no conv operator"
            )
        if local_start is not None or local_end is not None:
            if local_start is None or local_end is None:
                raise ValueError("PIC conv transition slice needs both bounds")
            operator = operator.slice(local_start, local_end)
        return operator.apply_conv(conv_state)

    def has_conv_transition(self, transition_handle: int) -> bool:
        """Return whether a live operator can advance a conv state."""
        operator = self._transitions.get(transition_handle)
        return operator is not None and bool(operator.conv_transitions)

    def restore_conv_tail(
        self,
        conv_tail_handle: int | None,
        conv_tail: Sequence["torch.Tensor"],
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
        recurrent_state: Sequence["torch.Tensor"] = (),
        transition_state: Sequence["torch.Tensor"] = (),
        conv_tail: Sequence["torch.Tensor"] = (),
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
            token_start=segment.start,
            token_end=segment.end,
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
    request_state: "CachedRequestState",
    kv_cache_config: "KVCacheConfig",
    forward_context: dict[str, Any],
    state_position: int | None = None,
) -> tuple[tuple["torch.Tensor", ...], tuple["torch.Tensor", ...]]:
    """Collect state tensors for one request and one exact segment boundary."""
    layout = PICStateLayout.from_kv_cache_config(kv_cache_config)
    recurrent: list["torch.Tensor"] = []
    conv_tail: list["torch.Tensor"] = []

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
    request_state: "CachedRequestState",
    kv_cache_config: "KVCacheConfig",
    forward_context: dict[str, Any],
    state_position: int,
    local_start: int = 0,
    local_end: int | None = None,
) -> None:
    """Apply a cached segment transition to the target request's live state."""
    if entry.transition_state_handle is None:
        raise ValueError(
            "PIC hybrid runtime requires a transition handle for reused segment"
        )
    manager.acquire_segment(request_id, segment.seg_hash)
    if local_end is None:
        local_end = segment.length
    recurrent, conv_tail = _collect_mamba_state_tensors(
        request_id=request_id,
        segment=segment,
        request_state=request_state,
        kv_cache_config=kv_cache_config,
        forward_context=forward_context,
        state_position=state_position,
    )
    if (
        conv_tail
        and local_end < segment.length
        and not manager.has_conv_transition(entry.transition_state_handle)
    ):
        raise PICWorkerUnsupported(
            "PIC partial reuse has no conv transition for the live state"
        )
    updated = manager.apply_transition(
        entry.transition_state_handle,
        recurrent,
        local_start=local_start,
        local_end=local_end,
    )
    if len(updated) != len(recurrent):
        raise ValueError("PIC transition returned an unexpected state count")
    for target, source in zip(recurrent, updated):
        target.copy_(source)
    if conv_tail:
        try:
            updated_conv = manager.apply_conv_transition(
                entry.transition_state_handle,
                conv_tail,
                local_start=local_start,
                local_end=local_end,
            )
        except KeyError:
            if local_end < segment.length:
                raise PICWorkerUnsupported(
                    "PIC partial reuse has no conv transition for the live state"
                )
            manager.restore_conv_tail(entry.conv_tail_handle, conv_tail)
        else:
            if len(updated_conv) != len(conv_tail):
                raise ValueError("PIC conv transition returned unexpected state count")
            for target, source in zip(conv_tail, updated_conv):
                target.copy_(source)
    manager._debug(
        "transition applied request=%s segment=%d range=[%d,%d) "
        "state_position=%d",
        request_id,
        segment_index,
        segment.start + local_start,
        segment.start + local_end,
        state_position,
    )


def capture_completed_mamba_segment(
    manager: PICLiveCaptureManager,
    *,
    request_id: str,
    segment_index: int,
    segment: PICSegment,
    completed_end: int,
    request_state: "CachedRequestState",
    kv_cache_config: "KVCacheConfig",
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
    )


def _collect_native_kv_references(
    *,
    request_id: str,
    segment_index: int,
    segment: PICSegment,
    request_state: "CachedRequestState",
    kv_cache_config: "KVCacheConfig",
) -> tuple[PICNativeKVReference, ...]:
    """Describe complete *segment-local* attention blocks.

    The PIC segment is an independent object whose logical origin is zero.
    Therefore the reusable span is computed from ``len(segment)`` rather than
    requiring ``segment.start``/``segment.end`` to align with the source
    request's global block table.  The source coordinates are retained on the
    reference so a later token-level bridge or request-private copy can read
    the exact source slots without exposing neighbouring tokens.

    A reference is emitted only when the scheduler reserved an independent
    public allocation for this segment.  The worker fills that allocation at
    canonical position zero before this metadata is published; request
    absolute source blocks are never mislabeled as reusable canonical KV.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

    references: list[PICNativeKVReference] = []
    public_blocks = {
        (int(segment_index), int(group_id)): tuple(int(block_id) for block_id in block_ids)
        for segment_index, group_id, block_ids in getattr(
            request_state, "pic_public_block_ids", ()
        )
    }
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not isinstance(spec, AttentionSpec) or isinstance(
            spec, CrossAttentionSpec
        ):
            continue
        block_size = int(spec.block_size)
        local_span = get_full_local_block_span(
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
        if len(block_ids) != local_span.block_count:
            raise ValueError(
                "PIC public KV allocation does not cover the segment: "
                f"request={request_id}, group={group_id}, "
                f"range=[{segment.start},{segment.end})"
            )
        references.append(
            PICNativeKVReference(
                kv_cache_group_id=group_id,
                block_ids=block_ids,
                block_size=block_size,
                token_count=local_span.reusable_token_count,
                kind=PICNativeKVKind.CANONICAL_PUBLIC,
                position_mode=PICKVPositionMode.CANONICAL_LOCAL,
                canonical_start=0,
                source_token_start=0,
                source_block_start=0,
                dtype=str(getattr(spec, "dtype", "")),
            )
        )
    return tuple(references)


def capture_transition_operator_segment(
    manager: PICLiveCaptureManager,
    *,
    request_id: str,
    segment_index: int,
    segment: PICSegment,
    request_state: "CachedRequestState",
    kv_cache_config: "KVCacheConfig",
    operator: PICTransitionOperator,
    conv_tail: Sequence["torch.Tensor"] = (),
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
    )


def restore_matched_mamba_segment(
    manager: PICLiveCaptureManager,
    *,
    request_id: str,
    segment_index: int,
    segment: PICSegment,
    entry: PICSegmentEntry,
    request_state: "CachedRequestState",
    kv_cache_config: "KVCacheConfig",
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
