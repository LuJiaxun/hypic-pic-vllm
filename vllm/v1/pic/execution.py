# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Range-based PIC execution plan.

The ordinary vLLM scheduler represents progress with one contiguous prefix
length.  PIC needs a separate description because a request may reuse segment
0, recompute segment 1, and reuse segment 2.  This module compiles the Stage 0
metadata result into that description without changing vLLM's existing
``num_computed_tokens`` semantics.
"""

from dataclasses import dataclass
from typing import Literal, Sequence

from vllm.v1.pic.cache import PICCachePlan, PICSegmentEntry
from vllm.v1.pic.native_kv import PICKVPositionMode
from vllm.v1.pic.segmenter import PICSegment

PICAction = Literal["reuse", "recompute"]


@dataclass(frozen=True)
class PICExecutionRange:
    segment_index: int
    start: int
    end: int
    action: PICAction
    entry: PICSegmentEntry | None = None
    canonical_start: int = 0
    position_mode: PICKVPositionMode = PICKVPositionMode.CANONICAL_LOCAL

    @property
    def target_start(self) -> int:
        """Absolute position of this range in the current request."""
        return self.start

    @property
    def target_end(self) -> int:
        return self.end


@dataclass(frozen=True)
class PICTransition:
    """A boundary where a reused segment needs transition/seam handling."""

    segment_index: int
    seam_start: int
    seam_end: int
    transition_state_handle: int | None = None


@dataclass(frozen=True)
class PICExecutionPlan:
    ranges: tuple[PICExecutionRange, ...]
    transitions: tuple[PICTransition, ...]

    @property
    def reused_ranges(self) -> tuple[PICExecutionRange, ...]:
        return tuple(item for item in self.ranges if item.action == "reuse")

    @property
    def recompute_ranges(self) -> tuple[PICExecutionRange, ...]:
        return tuple(item for item in self.ranges if item.action == "recompute")

    @property
    def requires_physical_handles(self) -> bool:
        return any(
            item.entry is None or not item.entry.has_physical_handles
            for item in self.reused_ranges
        )


def compile_execution_plan(
    segments: Sequence[PICSegment],
    cache_plan: PICCachePlan | None,
) -> PICExecutionPlan:
    """Compile a cache match into ordered ranges and transition boundaries."""
    if not segments:
        return PICExecutionPlan((), ())

    matches = dict(cache_plan.matches) if cache_plan is not None else {}
    seam_by_segment = dict(cache_plan.seam_tokens) if cache_plan is not None else {}
    reused = set(cache_plan.reused_segments) if cache_plan is not None else set()

    ranges = tuple(
        PICExecutionRange(
            segment_index=index,
            start=segment.start,
            end=segment.end,
            action="reuse" if index in reused else "recompute",
            entry=matches.get(index),
            canonical_start=0,
            position_mode=PICKVPositionMode.CANONICAL_LOCAL,
        )
        for index, segment in enumerate(segments)
    )

    transitions: list[PICTransition] = []
    for index, seam_tokens in seam_by_segment.items():
        if index not in matches:
            continue
        segment = segments[index]
        seam_tokens = min(max(int(seam_tokens), 0), len(segment.token_ids))
        if seam_tokens == 0:
            continue
        entry = matches[index]
        transitions.append(
            PICTransition(
                segment_index=index,
                seam_start=segment.start,
                seam_end=segment.start + seam_tokens,
                transition_state_handle=entry.transition_state_handle,
            )
        )

    return PICExecutionPlan(tuple(ranges), tuple(transitions))
