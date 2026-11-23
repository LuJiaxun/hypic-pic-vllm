# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the miss-range check in the single-request runtime plan.

A prompt that is one reused segment followed by a tail has no recompute range: the tail is
expressed as recompute positions, never as a range of its own. The runtime plan used to
require a recompute range, so exactly that shape fell back to the ordinary path and
recomputed the whole prompt with every reuse counter at zero.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.v1.pic.runtime import _plan_still_needs_compute


def _plan(*ranges):
    return SimpleNamespace(
        ranges=tuple(
            SimpleNamespace(start=start, end=end, action=action)
            for start, end, action in ranges
        )
    )


def test_a_tail_after_the_reused_segment_counts_as_work() -> None:
    plan = _plan((0, 4352, "reuse"))

    assert _plan_still_needs_compute(plan, 4911)


def test_a_gap_before_the_reused_segment_counts_as_work() -> None:
    plan = _plan((128, 4352, "reuse"))

    assert _plan_still_needs_compute(plan, 4352)


def test_full_prompt_reuse_leaves_nothing_to_compute() -> None:
    plan = _plan((0, 4911, "reuse"))

    assert not _plan_still_needs_compute(plan, 4911)


def test_a_trailing_recompute_range_leaves_nothing_to_compute() -> None:
    plan = _plan((0, 4352, "reuse"), (4352, 4911, "recompute"))

    assert not _plan_still_needs_compute(plan, 4911)
