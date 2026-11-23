# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage 9-A single-request skip/recompute safety gate.

This module compiles the already validated PIC range plan into an execution
descriptor for the narrow first runtime prototype.  It does not silently
change vLLM's ordinary forward.  A caller must explicitly prove that the
batch is one request, eager mode is active, speculative decoding is disabled,
and the worker can attach the SGLang-style native KV slot bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.pic.execution import PICExecutionPlan, PICExecutionRange
from vllm.logger import init_logger
from vllm.v1.pic.profile_level import pic_profile_enabled as _pic_trace_enabled
from vllm.v1.pic.worker_plan import (
    PICWorkerPlan,
    PICWorkerUnsupported,
    build_worker_plan,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class PICSingleRequestPlan:
    """Validated single-request execution descriptor."""

    ranges: tuple[PICExecutionRange, ...]
    skip_positions: tuple[int, ...]
    recompute_positions: tuple[int, ...]

    @property
    def reused_ranges(self) -> tuple[PICExecutionRange, ...]:
        return tuple(item for item in self.ranges if item.action == "reuse")

    @property
    def recompute_ranges(self) -> tuple[PICExecutionRange, ...]:
        return tuple(item for item in self.ranges if item.action == "recompute")

    def validate(self, prompt_len: int) -> None:
        if set(self.skip_positions) & set(self.recompute_positions):
            raise ValueError("PIC single-request skip/recompute positions overlap")
        all_positions = set(self.skip_positions) | set(self.recompute_positions)
        if all_positions != set(range(prompt_len)):
            raise ValueError("PIC single-request positions do not cover the prompt")


def build_single_request_plan(
    execution_plan: PICExecutionPlan | None,
    *,
    prompt_len: int,
    num_requests: int = 1,
    eager_mode: bool = True,
    range_execution_supported: bool = True,
    speculative_decoding: bool,
    zero_copy_attention_supported: bool,
    allow_fallback: bool,
    native_kv_bridge_supported: bool | None = None,
    worker_plan: PICWorkerPlan | None = None,
) -> PICSingleRequestPlan | None:
    """Build a single-request plan or return the safe ordinary-path fallback.

    Range control alone is insufficient: a reused range also needs the native
    KV bridge. ``None`` preserves the Stage 9-A prototype call contract; new
    callers must pass the explicit native-bridge capability.

    ``worker_plan`` lets a caller hand in a plan it already built. The runner does
    exactly that: it resolves the worker descriptor when the request first enters the
    batch and would otherwise build the same per-token position lists a second time.
    """
    if execution_plan is None or not execution_plan.ranges:
        return None

    if native_kv_bridge_supported is None:
        native_kv_bridge_supported = zero_copy_attention_supported

    reason: str | None = None
    if num_requests != 1:
        reason = "Stage 9-A requires exactly one request"
    elif not eager_mode:
        reason = "Stage 9-A requires eager execution"
    elif speculative_decoding:
        reason = "Stage 9-A does not support speculative decoding"
    elif execution_plan.reused_ranges and not range_execution_supported:
        reason = "worker range execution capability is disabled"
    elif execution_plan.reused_ranges and not native_kv_bridge_supported:
        reason = "native PIC KV slot bridge capability is disabled"

    if reason is not None:
        # A fallback is correct here, but a silent one is not: every one of these refusals looks
        # identical from the outside (the request simply recomputes), and telling them apart has
        # cost several rounds. The reason is named at trace level and still falls back safely.
        if _pic_trace_enabled():
            logger.warning("PIC single-request plan refused: %s", reason)
        if allow_fallback:
            return None
        raise PICWorkerUnsupported(reason)

    if worker_plan is None:
        worker_plan = build_worker_plan(
            execution_plan,
            prompt_len=prompt_len,
            range_execution_supported=range_execution_supported,
            allow_fallback=allow_fallback,
        )
    if worker_plan is None:
        if _pic_trace_enabled():
            logger.warning(
                "PIC worker plan refused: execution_ranges=%d reused=%d "
                "needs_physical_handles=%s range_execution=%s",
                len(execution_plan.ranges),
                len(execution_plan.reused_ranges),
                execution_plan.requires_physical_handles,
                range_execution_supported,
            )
        return None
    result = PICSingleRequestPlan(
        ranges=worker_plan.ranges,
        skip_positions=worker_plan.skip_positions,
        recompute_positions=worker_plan.recompute_positions,
    )
    result.validate(prompt_len)
    return result
