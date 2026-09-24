# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-request PIC runtime planning.

The runtime plan is the boundary between PIC metadata and the vLLM worker.
It contains native KV alias/private-materialization plans and the miss ranges
that must be sent through the model. It never changes ``num_computed_tokens``;
the scheduler still owns that scalar and the worker treats this plan as a
request-local execution view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from vllm.v1.pic.native_kv import (
    PICNativeKVKind,
    PICNativeKVReference,
    PICNativeKVSlotPlan,
    build_native_kv_slot_plan,
)
from vllm.v1.pic.single_request import PICSingleRequestPlan
from vllm.v1.pic.worker_plan import PICWorkerUnsupported

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PICRuntimeRange:
    """One model invocation or one already-materialized native range."""

    segment_index: int
    start: int
    end: int
    action: str
    context_end: int

    @property
    def token_count(self) -> int:
        return self.end - self.start

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end))


@dataclass(frozen=True)
class PICSingleRequestRuntimePlan:
    """Worker-consumable native-slot and miss-forward plan."""

    ranges: tuple[PICRuntimeRange, ...]
    native_slot_plans: tuple[PICNativeKVSlotPlan, ...]

    @property
    def recompute_ranges(self) -> tuple[PICRuntimeRange, ...]:
        return tuple(item for item in self.ranges if item.action == "recompute")

    @property
    def reused_ranges(self) -> tuple[PICRuntimeRange, ...]:
        return tuple(item for item in self.ranges if item.action == "reuse")

    def validate(self, prompt_len: int) -> None:
        if not self.ranges:
            raise ValueError("PIC runtime plan cannot be empty")
        cursor = 0
        for item in self.ranges:
            if item.start != cursor or item.end <= item.start:
                raise ValueError("PIC runtime ranges must be contiguous")
            if item.end > prompt_len or item.context_end < item.end:
                raise ValueError("PIC runtime range is outside the prompt")
            cursor = item.end
        if cursor != prompt_len:
            raise ValueError("PIC runtime ranges do not cover the prompt")
        if self.ranges[-1].action != "recompute":
            raise ValueError("the final PIC range must produce the final logits")


def build_single_request_runtime_plan(
    plan: PICSingleRequestPlan | None,
    *,
    native_kv_refs_by_group: Sequence[PICNativeKVReference],
    device: "torch.device | str",
    prompt_len: int,
    allow_fallback: bool,
    kernel_block_sizes: Sequence[int] | None = None,
    seam_tokens_by_segment: Mapping[int, int] | None = None,
) -> PICSingleRequestRuntimePlan | None:
    """Compile a safe native-slot runtime view.

    Reuse is limited to complete native blocks.  A hybrid state transition may
    cover a seam-trimmed body; the worker applies its recurrent and rolling
    conv-state components at the reused-body end.  Entries without the required
    transition components remain ordinary-forward fallbacks.
    """
    if plan is None:
        return None
    try:
        if not plan.reused_ranges:
            raise PICWorkerUnsupported("PIC runtime plan has no reused range")
        if not plan.recompute_ranges:
            raise PICWorkerUnsupported("PIC runtime plan has no miss range")
        runtime_ranges: list[PICRuntimeRange] = []
        slot_plans: list[PICNativeKVSlotPlan] = []
        available_refs = {
            (
                reference.kv_cache_group_id,
                reference.block_ids,
                reference.token_count,
                reference.canonical_start,
                reference.source_token_start,
                reference.source_block_start,
            )
            for reference in native_kv_refs_by_group
        }
        cursor = 0
        for item in plan.ranges:
            if item.start > cursor:
                runtime_ranges.append(
                    PICRuntimeRange(
                        segment_index=-1,
                        start=cursor,
                        end=item.start,
                        action="recompute",
                        context_end=item.start,
                    )
                )
            if item.action != "reuse" or item.entry is None:
                runtime_ranges.append(
                    PICRuntimeRange(
                        segment_index=item.segment_index,
                        start=item.start,
                        end=item.end,
                        action=item.action,
                        context_end=item.end,
                    )
                )
                cursor = item.end
                continue
            if not item.entry.native_kv_refs:
                raise PICWorkerUnsupported(
                    "reused PIC range has no native KV reference"
                )
            spans = {
                (reference.canonical_start, reference.token_count)
                for reference in item.entry.native_kv_refs
            }
            if len(spans) != 1:
                raise PICWorkerUnsupported(
                    "native KV references do not share one segment-local span"
                )
            segment_token_count = item.end - item.start
            seam_tokens = 0
            if seam_tokens_by_segment is not None:
                seam_tokens = max(
                    0,
                    min(
                        int(seam_tokens_by_segment.get(item.segment_index, 0)),
                        segment_token_count,
                    ),
                )
            local_start, reusable_token_count = next(iter(spans))
            local_end = local_start + reusable_token_count
            if (
                local_start < 0
                or reusable_token_count <= 0
                or local_end > segment_token_count
            ):
                raise PICWorkerUnsupported(
                    "native KV reference span is outside the PIC segment"
                )
            candidate_start = max(local_start, seam_tokens)
            candidate_end = local_end
            reusable_starts: list[int] = []
            reusable_ends: list[int] = []
            for reference in item.entry.native_kv_refs:
                if kernel_block_sizes is None:
                    alignment = reference.block_size
                else:
                    try:
                        alignment = int(
                            kernel_block_sizes[reference.kv_cache_group_id]
                        )
                    except IndexError as exc:
                        raise PICWorkerUnsupported(
                            "worker kernel block-size list has no entry for "
                            f"KV group {reference.kv_cache_group_id}"
                        ) from exc
                if alignment <= 0 or reference.block_size % alignment != 0:
                    raise PICWorkerUnsupported(
                        "PIC kernel block size must divide the allocator block size"
                    )
                origin = reference.canonical_start
                target_phase = (-item.start) % alignment
                source_phase = (
                    -int(reference.source_token_start) + origin
                ) % alignment
                # Canonical public KV is copied into a request-private row,
                # so its token-level gather/rerotation path does not require
                # the source and target request phases to coincide.  A
                # position-invariant shared reference may still use direct
                # block-table aliasing and therefore keeps the old check.
                if (
                    target_phase != source_phase
                    and reference.kind != PICNativeKVKind.CANONICAL_PUBLIC
                ):
                    raise PICWorkerUnsupported(
                        "PIC source and target token phases are not kernel aligned"
                    )
                if reference.kind == PICNativeKVKind.CANONICAL_PUBLIC:
                    aligned_start = candidate_start
                    aligned_end = candidate_end
                else:
                    aligned_start = candidate_start + (
                        target_phase - candidate_start
                    ) % alignment
                    aligned_end = candidate_end - (
                        candidate_end - target_phase
                    ) % alignment
                reusable_starts.append(aligned_start)
                reusable_ends.append(aligned_end)
            reusable_local_start = max(reusable_starts)
            reusable_local_end = min(reusable_ends)
            if reusable_local_end <= reusable_local_start:
                raise PICWorkerUnsupported(
                    "PIC seam and block alignment leave no reusable body"
                )
            reusable_start = item.start + reusable_local_start
            reusable_end = item.start + reusable_local_end
            reusable_token_count = reusable_local_end - reusable_local_start
            if (
                item.entry.conv_tail_handle is not None
                and reusable_end < item.end
                and item.entry.transition_state_handle is None
            ):
                raise PICWorkerUnsupported(
                    "partial PIC reuse has no conv-tail transition"
                )
            if (
                reusable_token_count != segment_token_count
                and item.entry.recurrent_state_handle is not None
                and item.entry.transition_state_handle is None
            ):
                raise PICWorkerUnsupported(
                    "partial PIC KV reuse has no matching hybrid-state checkpoint"
                )

            if reusable_start > item.start:
                runtime_ranges.append(
                    PICRuntimeRange(
                        segment_index=item.segment_index,
                        start=item.start,
                        end=reusable_start,
                        action="recompute",
                        context_end=reusable_start,
                    )
                )
            runtime_ranges.append(
                PICRuntimeRange(
                    segment_index=item.segment_index,
                    start=reusable_start,
                    end=reusable_end,
                    action="reuse",
                    context_end=reusable_end,
                )
            )
            if reusable_end < item.end:
                runtime_ranges.append(
                    PICRuntimeRange(
                        segment_index=item.segment_index,
                        start=reusable_end,
                        end=item.end,
                        action="recompute",
                        context_end=item.end,
                    )
                )

            for reference in item.entry.native_kv_refs:
                if (
                    reference.kv_cache_group_id,
                    reference.block_ids,
                    reference.token_count,
                    reference.canonical_start,
                    reference.source_token_start,
                    reference.source_block_start,
                ) not in available_refs:
                    raise PICWorkerUnsupported(
                        "worker native KV reference differs from scheduler reference"
                    )
                if kernel_block_sizes is not None:
                    kernel_block_size = int(
                        kernel_block_sizes[reference.kv_cache_group_id]
                    )
                else:
                    kernel_block_size = None
                slot_plan = build_native_kv_slot_plan(
                    reference,
                    target_start=reusable_start,
                    device=device,
                    kernel_block_size=kernel_block_size,
                    local_start=reusable_local_start,
                    token_count=reusable_token_count,
                )
                if (
                    not slot_plan.block_table_compatible
                    and not slot_plan.requires_private_materialization
                ):
                    raise PICWorkerUnsupported(
                        slot_plan.fallback_reason or "unaligned PIC native slot"
                    )
                slot_plans.append(slot_plan)
            cursor = item.end
        if cursor < prompt_len:
            runtime_ranges.append(
                PICRuntimeRange(
                    segment_index=-1,
                    start=cursor,
                    end=prompt_len,
                    action="recompute",
                    context_end=prompt_len,
                )
            )
        result = PICSingleRequestRuntimePlan(
            ranges=tuple(runtime_ranges),
            native_slot_plans=tuple(slot_plans),
        )
        result.validate(prompt_len)
        return result
    except (ValueError, PICWorkerUnsupported):
        if allow_fallback:
            return None
        raise
