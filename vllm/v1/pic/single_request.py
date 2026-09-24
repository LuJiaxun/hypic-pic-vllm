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
from vllm.v1.pic.worker_plan import (
    PICWorkerPlan,
    PICWorkerUnsupported,
    build_worker_plan,
)


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
            raise ValueError(
                "PIC single-request positions do not cover the prompt"
            )


def build_single_request_plan(
    execution_plan: PICExecutionPlan | None,
    *,
    prompt_len: int,
    num_requests: int,
    eager_mode: bool,
    speculative_decoding: bool,
    range_execution_supported: bool,
    zero_copy_attention_supported: bool,
    allow_fallback: bool,
    native_kv_bridge_supported: bool | None = None,
) -> PICSingleRequestPlan | None:
    """Build a single-request plan or return the safe ordinary-path fallback.

    Range control alone is insufficient: a reused range also needs the native
    KV bridge. ``None`` preserves the Stage 9-A prototype call contract; new
    callers must pass the explicit native-bridge capability.
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
        if allow_fallback:
            return None
        raise PICWorkerUnsupported(reason)

    worker_plan = build_worker_plan(
        execution_plan,
        prompt_len=prompt_len,
        range_execution_supported=range_execution_supported,
        allow_fallback=allow_fallback,
    )
    if worker_plan is None:
        return None
    result = PICSingleRequestPlan(
        ranges=worker_plan.ranges,
        skip_positions=worker_plan.skip_positions,
        recompute_positions=worker_plan.recompute_positions,
    )
    result.validate(prompt_len)
    return result
