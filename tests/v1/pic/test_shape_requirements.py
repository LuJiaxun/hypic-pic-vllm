# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the shape the single-request runtime can accept.

A runtime plan has to end with a recompute range, because that range produces the logits.
When the reusable document sits behind the query, the last range is the reuse range and the
plan is refused - correctly - but the refusal used to be indistinguishable from a plan that
was never built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.v1.pic import runtime
from vllm.v1.pic.execution import PICExecutionRange


def _plan(*ranges):
    """Build a plan whose ranges are the type the runtime actually walks.

    A bare SimpleNamespace is not enough: the runtime reads ``segment_index`` and ``entry``
    while building the runtime ranges, so a stub missing them fails deep inside the call
    instead of where the test is wrong.
    """
    items = tuple(
        PICExecutionRange(
            segment_index=index,
            start=start,
            end=end,
            action=action,
            entry=None,
            canonical_start=0,
        )
        for index, (start, end, action) in enumerate(ranges)
    )
    return SimpleNamespace(
        ranges=items,
        reused_ranges=tuple(item for item in items if item.action == "reuse"),
        recompute_ranges=tuple(item for item in items if item.action != "reuse"),
    )


def test_a_trailing_reuse_is_refused() -> None:
    # query then doc: nothing recomputes at the end, so no range produces the logits.
    plan = _plan((0, 272, "recompute"), (272, 5113, "reuse"))

    result = runtime.build_single_request_runtime_plan(
        plan,
        native_kv_refs_by_group=(),
        device="cpu",
        prompt_len=5113,
        allow_fallback=True,
    )

    assert result is None


def test_the_refusal_is_named_in_the_log() -> None:
    source = __import__("pathlib").Path("vllm/v1/pic/runtime.py").read_text(
        encoding="utf-8"
    )

    assert "PIC single-request runtime refused" in source
    assert "the final PIC range must produce the final logits" in source
