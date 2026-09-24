# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage 9-B request-level batch planning and isolation checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from vllm.v1.pic.runtime import PICSingleRequestRuntimePlan


@dataclass(frozen=True)
class PICBatchRuntimePlan:
    """Immutable mapping of request IDs to independent runtime plans.

    The scheduler's batch is deliberately not represented as one flattened
    PIC range list. Each request owns its cursor, native slots, and state
    transition sequence; this prevents one request's reuse range from changing
    another request's logical position.
    """

    request_ids: tuple[str, ...]
    runtime_plans: tuple[tuple[str, PICSingleRequestRuntimePlan], ...]

    def plan_for(self, request_id: str) -> PICSingleRequestRuntimePlan | None:
        return dict(self.runtime_plans).get(request_id)

    def validate(self) -> None:
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("PIC batch request IDs must be unique")
        request_set = set(self.request_ids)
        plan_ids = {request_id for request_id, _ in self.runtime_plans}
        if not plan_ids <= request_set:
            raise ValueError("PIC batch plan contains an unknown request")
        if len(plan_ids) != len(self.runtime_plans):
            raise ValueError("PIC batch plan contains duplicate request plans")
        for _, plan in self.runtime_plans:
            if not plan.ranges:
                raise ValueError("PIC batch plan contains an empty runtime plan")


@dataclass(frozen=True)
class PICPackedBatchPlan(PICBatchRuntimePlan):
    """Request-local plans validated for packed range-round execution.

    A packed round advances every request through one recompute range. Reuse
    ranges are applied between rounds, so a request cannot be flattened into
    another request's token sequence or transition cursor.
    """

    def validate_packed(self) -> None:
        self.validate()
        recompute_counts = {
            sum(item.action == "recompute" for item in plan.ranges)
            for _, plan in self.runtime_plans
        }
        if len(recompute_counts) > 1:
            raise ValueError(
                "PIC packed batch requires the same number of recompute ranges "
                "for every request"
            )
        if not recompute_counts or next(iter(recompute_counts)) == 0:
            raise ValueError("PIC packed batch requires a recompute range")


def build_pic_batch_runtime_plan(
    request_ids: tuple[str, ...],
    runtime_plans: Mapping[str, PICSingleRequestRuntimePlan],
) -> PICBatchRuntimePlan:
    """Build a request-isolated batch view without flattening ranges."""

    result = PICBatchRuntimePlan(
        request_ids=request_ids,
        runtime_plans=tuple(
            (request_id, runtime_plans[request_id])
            for request_id in request_ids
            if request_id in runtime_plans
        ),
    )
    result.validate()
    return result


def build_pic_packed_batch_plan(
    request_ids: tuple[str, ...],
    runtime_plans: Mapping[str, PICSingleRequestRuntimePlan],
) -> PICPackedBatchPlan:
    """Build the Stage 10-A packed-round plan.

    Ordinary requests are added by the worker as synthetic recompute plans;
    this helper only validates the request-local PIC plans.
    """

    result = PICPackedBatchPlan(
        request_ids=request_ids,
        runtime_plans=tuple(
            (request_id, runtime_plans[request_id])
            for request_id in request_ids
            if request_id in runtime_plans
        ),
    )
    result.validate_packed()
    return result
