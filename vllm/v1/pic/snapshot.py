# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Physical PIC snapshot/restore adapter.

This module composes the Stage 2 state layout, Stage 4 physical pool, and
Stage 1 cache metadata.  Callers provide already materialized tensors; this
adapter does not inspect vLLM's live KV cache or alter model execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from vllm.v1.pic.cache import PICSegmentCache, PICSegmentEntry
from vllm.v1.pic.handles import PICHandleKind
from vllm.v1.pic.native_kv import PICNativeKVReference
from vllm.v1.pic.pool import PICPhysicalPool
from vllm.v1.pic.segmenter import PICSegment
from vllm.v1.pic.state import (
    PICStateLayout,
    PICStateSpec,
    PICTransitionOperator,
)

if TYPE_CHECKING:
    import torch

PICStateKey = tuple[int, int]


@dataclass(frozen=True)
class PICPhysicalSnapshot:
    """Handle bundle for one position-independent prompt segment.

    Native attention KV is represented by ``native_kv_refs`` and is not copied
    into ``PICPhysicalPool``. The pool handles below are reserved for hybrid
    recurrent/transition/conv state and legacy explicit tensor adapters.
    """

    seg_hash: bytes
    token_start: int
    token_end: int
    full_kv_handles: tuple[int, ...] = ()
    recurrent_state_handle: int | None = None
    transition_state_handle: int | None = None
    conv_tail_handle: int | None = None
    native_kv_refs: tuple[PICNativeKVReference, ...] = ()

    @property
    def handle_ids(self) -> tuple[int, ...]:
        return tuple(
            handle_id
            for handles in (
                self.full_kv_handles,
                (self.recurrent_state_handle,),
                (self.transition_state_handle,),
                (self.conv_tail_handle,),
            )
            for handle_id in handles
            if handle_id is not None
        )

    @property
    def has_handles(self) -> bool:
        return bool(self.handle_ids or self.native_kv_refs)


class PICSnapshotStore:
    """Capture and restore explicit tensor snapshots in a PIC pool."""

    def __init__(self, pool: PICPhysicalPool) -> None:
        self.pool = pool

    def capture(
        self,
        segment: PICSegment,
        *,
        full_kv: Sequence["torch.Tensor"] = (),
        recurrent_state: Sequence["torch.Tensor"] = (),
        transition_state: Sequence["torch.Tensor"] = (),
        conv_tail: Sequence["torch.Tensor"] = (),
        native_kv_refs: Sequence[PICNativeKVReference] = (),
    ) -> PICPhysicalSnapshot:
        """Copy supplied tensors into independent physical allocations."""
        allocated: list[int] = []
        try:
            full_kv_handles = self._capture_one(
                full_kv,
                kind=PICHandleKind.FULL_KV,
                token_start=segment.start,
                token_end=segment.end,
                allocated=allocated,
            )
            recurrent_handle = self._capture_one(
                recurrent_state,
                kind=PICHandleKind.RECURRENT_STATE,
                token_start=segment.start,
                token_end=segment.end,
                allocated=allocated,
            )
            transition_handle = self._capture_one(
                transition_state,
                kind=PICHandleKind.TRANSITION_STATE,
                token_start=segment.start,
                token_end=segment.end,
                allocated=allocated,
            )
            conv_tail_handle = self._capture_one(
                conv_tail,
                kind=PICHandleKind.CONV_TAIL,
                token_start=segment.start,
                token_end=segment.end,
                allocated=allocated,
            )
        except Exception:
            for handle_id in reversed(allocated):
                self.pool.release(handle_id)
            raise

        return PICPhysicalSnapshot(
            seg_hash=segment.seg_hash,
            token_start=segment.start,
            token_end=segment.end,
            full_kv_handles=(() if full_kv_handles is None else (full_kv_handles,)),
            recurrent_state_handle=recurrent_handle,
            transition_state_handle=transition_handle,
            conv_tail_handle=conv_tail_handle,
            native_kv_refs=tuple(native_kv_refs),
        )

    def materialize_cache_entry(
        self,
        cache: PICSegmentCache,
        segment: PICSegment,
        snapshot: PICPhysicalSnapshot,
    ) -> PICSegmentEntry | None:
        """Publish a snapshot's opaque handles to the segment metadata cache."""
        if snapshot.seg_hash != segment.seg_hash:
            raise ValueError("PIC snapshot hash does not match segment")
        if (snapshot.token_start, snapshot.token_end) != (
            segment.start,
            segment.end,
        ):
            raise ValueError("PIC snapshot range does not match segment")
        return cache.insert(
            segment,
            full_kv_handles=snapshot.full_kv_handles,
            recurrent_state_handle=snapshot.recurrent_state_handle,
            transition_state_handle=snapshot.transition_state_handle,
            conv_tail_handle=snapshot.conv_tail_handle,
            native_kv_refs=snapshot.native_kv_refs,
        )

    def capture_native_kv(
        self,
        segment: PICSegment,
        native_kv_refs: Sequence[PICNativeKVReference],
    ) -> PICPhysicalSnapshot:
        """Describe native KV residency without copying it into PIC pool."""
        references = tuple(native_kv_refs)
        if not references:
            raise ValueError("PIC native KV capture requires at least one reference")
        expected_groups = {reference.kv_cache_group_id for reference in references}
        if len(expected_groups) != len(references):
            raise ValueError("PIC native KV capture has duplicate cache groups")
        if any(reference.token_count != len(segment.token_ids) for reference in references):
            raise ValueError("PIC native KV reference length does not match segment")
        return PICPhysicalSnapshot(
            seg_hash=segment.seg_hash,
            token_start=segment.start,
            token_end=segment.end,
            native_kv_refs=references,
        )

    def capture_transition_operator(
        self,
        segment: PICSegment,
        operator: PICTransitionOperator,
        *,
        full_kv: Sequence["torch.Tensor"] = (),
        recurrent_state: Sequence["torch.Tensor"] = (),
        conv_tail: Sequence["torch.Tensor"] = (),
        native_kv_refs: Sequence[PICNativeKVReference] = (),
    ) -> PICPhysicalSnapshot:
        """Persist a compact transition operator with a PIC segment.

        The operator tensors are stored in the transition allocation for
        accounting and lifetime tracking.  The worker keeps the structured
        operator for direct application; the opaque handle alone is not an
        end-state snapshot and must not be interpreted as one.
        """
        if (operator.token_start, operator.token_end) != (
            segment.start,
            segment.end,
        ):
            raise ValueError("PIC transition range does not match segment")
        return self.capture(
            segment,
            full_kv=full_kv,
            recurrent_state=recurrent_state,
            transition_state=operator.storage_tensors(),
            conv_tail=conv_tail,
            native_kv_refs=native_kv_refs,
        )

    def capture_hybrid_state(
        self,
        segment: PICSegment,
        layout: PICStateLayout,
        state_tensors: Mapping[PICStateKey, "torch.Tensor"],
    ) -> PICPhysicalSnapshot:
        """Capture state tensors in the authoritative layout order.

        The mapping key is ``(group_id, state_index)``.  Requiring explicit
        keys prevents callers from relying on an implicit Mamba/GDN ordering.
        """
        recurrent, conv_tail = self._ordered_state_tensors(layout, state_tensors)
        return self.capture(
            segment,
            recurrent_state=recurrent,
            conv_tail=conv_tail,
        )

    def restore_hybrid_state(
        self,
        snapshot: PICPhysicalSnapshot,
        layout: PICStateLayout,
        state_tensors: Mapping[PICStateKey, "torch.Tensor"],
    ) -> None:
        """Restore recurrent/conv tensors using the same layout validation."""
        recurrent, conv_tail = self._ordered_state_tensors(layout, state_tensors)
        self.restore(
            snapshot,
            recurrent_state=recurrent,
            conv_tail=conv_tail,
        )

    def restore(
        self,
        snapshot: PICPhysicalSnapshot,
        *,
        full_kv: Sequence["torch.Tensor"] = (),
        recurrent_state: Sequence["torch.Tensor"] = (),
        transition_state: Sequence["torch.Tensor"] = (),
        conv_tail: Sequence["torch.Tensor"] = (),
    ) -> None:
        """Restore a snapshot into caller-owned tensors."""
        self._restore_one(snapshot.full_kv_handles, full_kv, "full_kv")
        self._restore_one(
            (snapshot.recurrent_state_handle,)
            if snapshot.recurrent_state_handle is not None
            else (),
            recurrent_state,
            "recurrent_state",
        )
        self._restore_one(
            (snapshot.transition_state_handle,)
            if snapshot.transition_state_handle is not None
            else (),
            transition_state,
            "transition_state",
        )
        self._restore_one(
            (snapshot.conv_tail_handle,)
            if snapshot.conv_tail_handle is not None
            else (),
            conv_tail,
            "conv_tail",
        )

    def release(self, snapshot: PICPhysicalSnapshot) -> None:
        """Release all pool allocations owned by a snapshot."""
        for handle_id in snapshot.handle_ids:
            self.pool.release(handle_id)

    def _capture_one(
        self,
        tensors: Sequence["torch.Tensor"],
        *,
        kind: PICHandleKind,
        token_start: int,
        token_end: int,
        allocated: list[int],
    ) -> int | None:
        if not tensors:
            return None
        handle = self.pool.allocate_tensor_snapshot(
            tensors,
            kind=kind,
            token_start=token_start,
            token_end=token_end,
        )
        allocated.append(handle.handle_id)
        return handle.handle_id

    def _restore_one(
        self,
        handle_ids: tuple[int, ...],
        tensors: Sequence["torch.Tensor"],
        name: str,
    ) -> None:
        if not handle_ids and not tensors:
            return
        if len(handle_ids) != 1 or not tensors:
            raise ValueError(
                f"PIC {name} restore requires one handle and non-empty tensors"
            )
        self.pool.restore_tensor_snapshot(handle_ids[0], tensors)

    @staticmethod
    def _ordered_state_tensors(
        layout: PICStateLayout,
        state_tensors: Mapping[PICStateKey, "torch.Tensor"],
    ) -> tuple[tuple["torch.Tensor", ...], tuple["torch.Tensor", ...]]:
        specs = tuple(spec for group in layout.groups for spec in group)
        expected_keys = {(spec.group_id, spec.state_index) for spec in specs}
        if set(state_tensors) != expected_keys:
            raise ValueError(
                "PIC hybrid state keys do not match PICStateLayout: "
                f"expected={sorted(expected_keys)}, "
                f"actual={sorted(state_tensors)}"
            )

        recurrent: list["torch.Tensor"] = []
        conv_tail: list["torch.Tensor"] = []
        for spec in specs:
            tensor = state_tensors[(spec.group_id, spec.state_index)]
            if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
                raise ValueError(
                    "PIC hybrid state tensor does not match PICStateLayout: "
                    f"group={spec.group_id}, index={spec.state_index}, "
                    f"expected_shape={spec.shape}, actual_shape={tuple(tensor.shape)}, "
                    f"expected_dtype={spec.dtype}, actual_dtype={tensor.dtype}"
                )
            if spec.kind == PICHandleKind.RECURRENT_STATE:
                recurrent.append(tensor)
            elif spec.kind == PICHandleKind.CONV_TAIL:
                conv_tail.append(tensor)
            else:
                raise ValueError(
                    "PICStateLayout contains unsupported snapshot kind "
                    f"{spec.kind!r}"
                )
        return tuple(recurrent), tuple(conv_tail)
