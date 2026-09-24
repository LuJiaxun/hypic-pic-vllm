# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-side PIC execution gate and skip/recompute descriptors."""

from dataclasses import dataclass
from typing import Iterable

from vllm.v1.pic.execution import PICExecutionPlan, PICExecutionRange


class PICWorkerUnsupported(RuntimeError):
    """Raised when a PIC plan cannot be executed by the active worker path."""


@dataclass(frozen=True)
class PICWorkerCapabilities:
    """Capabilities advertised by a concrete PIC-aware worker backend.

    The default is deliberately disabled.  A future backend must opt in only
    after it can consume non-contiguous ranges and materialized handles.
    """

    range_execution_supported: bool = False
    zero_copy_attention_supported: bool = False
    native_kv_bridge_supported: bool = False
    single_request_execution_supported: bool = False
    batch_execution_supported: bool = False


@dataclass(frozen=True)
class PICWorkerPlan:
    """Token-level execution descriptor for a PIC-aware worker."""

    ranges: tuple[PICExecutionRange, ...]
    skip_positions: tuple[int, ...]
    recompute_positions: tuple[int, ...]

    @property
    def has_reuse(self) -> bool:
        return any(item.action == "reuse" for item in self.ranges)

    def validate(self, prompt_len: int) -> None:
        if not self.ranges:
            if prompt_len:
                raise ValueError("non-empty PIC prompt has no execution ranges")
            return

        expected_start = 0
        for item in self.ranges:
            if (
                item.start < expected_start
                or item.end < item.start
                or item.end > prompt_len
            ):
                raise ValueError("PIC execution ranges must be ordered and bounded")
            expected_start = item.end

        if set(self.skip_positions) & set(self.recompute_positions):
            raise ValueError("PIC skip and recompute positions overlap")
        if len(self.skip_positions) + len(self.recompute_positions) != prompt_len:
            raise ValueError("PIC skip/recompute positions do not cover the prompt")
        if set(self.skip_positions) | set(self.recompute_positions) != set(
            range(prompt_len)
        ):
            raise ValueError("PIC skip/recompute positions are out of bounds")


def build_worker_plan(
    execution_plan: PICExecutionPlan | None,
    *,
    prompt_len: int,
    range_execution_supported: bool,
    allow_fallback: bool,
) -> PICWorkerPlan | None:
    """Build a worker descriptor or safely select the ordinary path.

    ``range_execution_supported`` is intentionally explicit.  Until the
    attention backend accepts non-contiguous positions, a PIC hit must never
    silently turn into a partial ordinary forward.  With fallback enabled we
    return ``None`` and the caller runs the unchanged vLLM path.
    """
    if execution_plan is None or not execution_plan.ranges:
        return None

    if execution_plan.requires_physical_handles:
        if allow_fallback:
            return None
        raise PICWorkerUnsupported(
            "PIC plan contains reused ranges without materialized physical "
            "handles"
        )

    if execution_plan.reused_ranges and not range_execution_supported:
        if allow_fallback:
            return None
        raise PICWorkerUnsupported(
            "PIC plan contains reused ranges but the active worker does not "
            "support range execution"
        )

    skip_positions: list[int] = []
    recompute_positions: list[int] = []
    cursor = 0
    for item in execution_plan.ranges:
        # The segmenter removes separator tokens.  Those positions are not in
        # a Stage 3 segment range and must remain ordinary recompute tokens.
        if item.start > cursor:
            recompute_positions.extend(range(cursor, item.start))
        target = skip_positions if item.action == "reuse" else recompute_positions
        target.extend(range(item.start, item.end))
        cursor = item.end
    if cursor < prompt_len:
        recompute_positions.extend(range(cursor, prompt_len))

    result = PICWorkerPlan(
        ranges=execution_plan.ranges,
        skip_positions=tuple(skip_positions),
        recompute_positions=tuple(recompute_positions),
    )
    result.validate(prompt_len)
    return result


def merge_positions(plans: Iterable[PICWorkerPlan]) -> tuple[int, ...]:
    """Return sorted unique positions for batch-side diagnostics."""
    return tuple(sorted({position for plan in plans for position in plan.skip_positions}))
