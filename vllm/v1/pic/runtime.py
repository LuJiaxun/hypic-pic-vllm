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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.pic.env import pic_debug_enabled
from vllm.v1.pic.native_kv import (
    PICNativeKVKind,
    is_pic_exact_position_aliasable,
    PICNativeKVReference,
    PICNativeKVSlotPlan,
    build_native_kv_slot_plan,
)
from vllm.v1.pic.single_request import PICSingleRequestPlan
from vllm.v1.pic.worker_plan import PICWorkerUnsupported

logger = init_logger(__name__)


def _pic_trace_enabled() -> bool:
    """True when the diagnostics switch asks for PIC tracing."""
    return pic_debug_enabled()

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


def _plan_still_needs_compute(plan: "PICSingleRequestPlan", prompt_len: int) -> bool:
    """Whether the plan has work outside its reused ranges.

    A gap before a range and the tail after the last one are recompute positions that never
    become ranges of their own, so a plan whose only segment is reused still has a miss:
    rejecting it dropped the whole request onto the ordinary path.
    """
    cursor = 0
    for item in plan.ranges:
        if item.start > cursor:
            return True
        cursor = max(cursor, item.end)
    return cursor < prompt_len


def build_single_request_runtime_plan(
    plan: PICSingleRequestPlan | None,
    *,
    native_kv_refs_by_group: Sequence[PICNativeKVReference],
    native_kv_refs_by_segment: Mapping[
        int, Sequence[PICNativeKVReference]
    ] | None = None,
    device: torch.device | str,
    prompt_len: int,
    allow_fallback: bool,
    kernel_block_sizes: Sequence[int] | None = None,
    seam_tokens_by_segment: Mapping[int, int] | None = None,
) -> PICSingleRequestRuntimePlan | None:
    """Compile a safe native-slot runtime view.

    The reuse window runs from the seam to one token before the segment ends: the
    last token is recomputed because the request samples from it.  A window may
    therefore stop inside a page, so the published span and the window are the
    same range rather than the segment rounded to a block boundary.  A hybrid
    state transition covers that window, and the worker applies its recurrent and
    rolling conv-state components at its end.  Entries without the required
    transition components remain ordinary-forward fallbacks.
    """
    if plan is None:
        return None
    try:
        if not plan.reused_ranges:
            raise PICWorkerUnsupported("PIC runtime plan has no reused range")
        if not plan.recompute_ranges and not _plan_still_needs_compute(plan, prompt_len):
            # The whole prompt is reused, so no token is left whose hidden state could
            # become the logits. This is a deliberate fallback, not a failure: the request
            # recomputes normally. Skipping everything would leave nothing to sample from,
            # and pulling the reusable end back would move it off the transition boundary
            # the entry published for its aligned span.
            logger.debug(
                "PIC falls back to the ordinary forward for a fully reused prompt "
                "(prompt_len=%d); full reuse is not a PIC target",
                prompt_len,
            )
            return None
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
            item_native_kv_refs = tuple(
                native_kv_refs_by_segment.get(
                    item.segment_index, item.entry.native_kv_refs
                )
                if native_kv_refs_by_segment is not None
                else item.entry.native_kv_refs
            )
            if not item_native_kv_refs:
                raise PICWorkerUnsupported(
                    "reused PIC range has no native KV reference"
                )
            spans = {
                (reference.canonical_start, reference.token_count)
                for reference in item_native_kv_refs
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
            for reference in item_native_kv_refs:
                if kernel_block_sizes is None:
                    alignment = reference.block_size
                else:
                    try:
                        alignment = int(kernel_block_sizes[reference.kv_cache_group_id])
                    except IndexError as exc:
                        raise PICWorkerUnsupported(
                            "worker kernel block-size list has no entry for "
                            f"KV group {reference.kv_cache_group_id}"
                        ) from exc
                if alignment <= 0 or reference.block_size % alignment != 0:
                    raise PICWorkerUnsupported(
                        "PIC kernel block size must divide the allocator block size"
                    )
                assert reference.source_token_start is not None
                origin = reference.canonical_start
                target_phase = (-item.start) % alignment
                source_phase = (-reference.source_token_start + origin) % alignment
                # A reference at the canonical origin aliases the request row directly and must
                # keep the source and target phases aligned. Anything else is imported by
                # copying, which re-rotates token by token, so only its end has to land on a
                # block boundary - an interior segment must not be rejected by the phase rule.
                copied_path = (
                    reference.kind == PICNativeKVKind.CANONICAL_PUBLIC
                    or not is_pic_exact_position_aliasable(item.start)
                )
                if target_phase != source_phase and not copied_path:
                    if _pic_trace_enabled():
                        logger.warning(
                            "PIC phase mismatch segment=%d kind=%s start=%d "
                            "alignment=%d target_phase=%d source_phase=%d "
                            "canonical_start=%d source_token_start=%d",
                            item.segment_index,
                            getattr(reference.kind, "name", reference.kind),
                            item.start,
                            alignment,
                            target_phase,
                            source_phase,
                            origin,
                            reference.source_token_start,
                        )
                    raise PICWorkerUnsupported(
                        "PIC source and target token phases are not kernel aligned"
                    )
                if copied_path:
                    # The copied path does not need the start to be aligned. Its end still stops one
                    # token short of the segment, because the request's logits come from that token
                    # and a window that covered it could not produce them.
                    aligned_start = candidate_start
                    aligned_end = max(candidate_start + 1, candidate_end - 1)
                else:
                    aligned_start = (
                        candidate_start + (target_phase - candidate_start) % alignment
                    )
                    aligned_end = (
                        candidate_end - (candidate_end - target_phase) % alignment
                    )
                reusable_starts.append(aligned_start)
                reusable_ends.append(aligned_end)
            reusable_local_start = max(reusable_starts)
            reusable_local_end = min(reusable_ends)
            if reusable_local_end <= reusable_local_start:
                # Naming the inputs is the difference between a supported refusal and another
                # round of guessing: the seam, the published span, the alignment and the phases
                # together decide whether a body survives.
                if _pic_trace_enabled():
                    logger.warning(
                        "PIC empty reuse body segment=%d start=%d end=%d seam=%d "
                        "local=[%d,%d) candidate=[%d,%d) aligned=[%d,%d) starts=%s ends=%s "
                        "alignment=%s phase=(%d,%d) kind=%s",
                        item.segment_index,
                        item.start,
                        item.end,
                        seam_tokens,
                        local_start,
                        local_end,
                        candidate_start,
                        candidate_end,
                        reusable_local_start,
                        reusable_local_end,
                        list(reusable_starts),
                        list(reusable_ends),
                        kernel_block_sizes,
                        target_phase,
                        source_phase,
                        getattr(reference.kind, "name", reference.kind),
                    )
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

            for reference in item_native_kv_refs:
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
        # Adjacent recompute ranges describe one continuous span of the prompt,
        # so a single forward covers them.  Keeping them apart costs one launch
        # per boundary, and for a separator-sized gap it produced a one-token
        # forward that measured two seconds.
        merged_ranges: list[PICRuntimeRange] = []
        for item in runtime_ranges:
            if (
                merged_ranges
                and item.action == "recompute"
                and merged_ranges[-1].action == "recompute"
                and merged_ranges[-1].end == item.start
            ):
                merged_ranges[-1] = replace(
                    merged_ranges[-1],
                    end=item.end,
                    context_end=item.context_end,
                )
                continue
            merged_ranges.append(item)
        runtime_ranges = merged_ranges
        result = PICSingleRequestRuntimePlan(
            ranges=tuple(runtime_ranges),
            native_slot_plans=tuple(slot_plans),
        )
        result.validate(prompt_len)
        return result
    except (ValueError, PICWorkerUnsupported) as exc:
        # Both the alignment checks and the final-logits rule land here, and returning None
        # silently made a refused plan look like one that was never built. Naming the reason
        # is what turns "no reuse" into a diagnosis.
        message = (
            "PIC single-request runtime refused (prompt_len=%d): %s", prompt_len, exc
        )
        # Always a warning. A refusal costs a full prefill, which is far more expensive than
        # one line of log, and keeping it behind a switch made a refused plan
        # indistinguishable from one that was never built - the same emptiness had to be
        # re-diagnosed repeatedly while the reason sat one level below the default.
        logger.warning(*message)
        if allow_fallback:
            return None
        raise
