# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the deliberate fallback for a fully reused prompt.

Full reuse is not a PIC target: every token would be skipped, including the one whose
hidden state becomes the logits, and pulling the reusable end back would move it off the
transition boundary the entry published. The plan therefore declines, and the request
recomputes normally - a fallback that must stay quiet and correct, not an error.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("torch")

from types import SimpleNamespace

from vllm.v1.pic import runtime


def _plan(*ranges):
    """A plan stub carrying the two properties the runtime reads.

    ``build_single_request_runtime_plan`` asks the plan for its reused and recompute
    ranges before anything else, so a bare SimpleNamespace is rejected deeper in the call.
    """
    items = tuple(
        SimpleNamespace(start=start, end=end, action=action)
        for start, end, action in ranges
    )
    return SimpleNamespace(
        ranges=items,
        reused_ranges=tuple(item for item in items if item.action == "reuse"),
        recompute_ranges=tuple(item for item in items if item.action != "reuse"),
    )


def test_a_fully_reused_plan_declines_without_raising() -> None:
    plan = _plan((0, 5120, "reuse"))

    result = runtime.build_single_request_runtime_plan(
        plan,
        native_kv_refs_by_group=(),
        device="cpu",
        prompt_len=5120,
        allow_fallback=True,
    )

    assert result is None, "a full reuse must fall back, not build a plan"


def test_the_fallback_is_explained_in_the_source() -> None:
    source = pathlib.Path("vllm/v1/pic/runtime.py").read_text(encoding="utf-8")

    # Checked line by line: the comment is wrapped, and an exact substring spanning two
    # lines would carry the "#" markers between them.
    assert "full reuse is not a PIC target" in source
    assert "transition boundary" in source
    assert "the entry published for its aligned span" in source
