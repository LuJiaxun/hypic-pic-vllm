# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the deferred block zeroing.

A step that schedules no tokens is the engine polling for a parked import. Zeroing
there issues a kernel while the import thread owns the CUDA context, which measured as
one millisecond of kernel followed by about a hundred and eighty milliseconds of host
wait - on every empty step, for every first-seen segment. The worker therefore collects
the ids and zeroes them on the next step that actually schedules tokens.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _ZeroRecorder:
    """Stands in for the runner: records what was zeroed, when."""

    def __init__(self) -> None:
        self.zeroed: list[list[int]] = []

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        self.zeroed.append(list(block_ids))


def _step(*, zero=(), scheduled=0):
    return SimpleNamespace(
        new_block_ids_to_zero=list(zero),
        total_num_scheduled_tokens=scheduled,
        num_scheduled_tokens={} if scheduled == 0 else {"req-1": scheduled},
    )


def run_step(runner: _ZeroRecorder, step) -> None:
    """The zeroing half of ``_update_states``, without the rest of the batch."""
    if step.new_block_ids_to_zero:
        pending = getattr(runner, "_pic_deferred_zero_ids", None)
        if pending is None:
            pending = []
            runner._pic_deferred_zero_ids = pending
        pending.extend(step.new_block_ids_to_zero)
    deferred = getattr(runner, "_pic_deferred_zero_ids", None)
    if deferred and int(step.total_num_scheduled_tokens) > 0:
        zero_ids = list(dict.fromkeys(deferred))
        deferred.clear()
        runner._zero_block_ids(zero_ids)


def test_an_empty_step_does_not_zero() -> None:
    runner = _ZeroRecorder()
    run_step(runner, _step(zero=[1, 2, 3], scheduled=0))

    assert runner.zeroed == [], "an empty step must not issue a kernel"
    assert runner._pic_deferred_zero_ids == [1, 2, 3]


def test_the_next_scheduling_step_zeroes_the_accumulated_ids() -> None:
    runner = _ZeroRecorder()
    run_step(runner, _step(zero=[1, 2], scheduled=0))
    run_step(runner, _step(zero=[3], scheduled=0))
    run_step(runner, _step(scheduled=128))

    assert runner.zeroed == [[1, 2, 3]]
    assert runner._pic_deferred_zero_ids == []


def test_duplicates_collapse_and_nothing_is_lost() -> None:
    runner = _ZeroRecorder()
    run_step(runner, _step(zero=[7, 8], scheduled=0))
    run_step(runner, _step(zero=[8, 9], scheduled=0))
    run_step(runner, _step(scheduled=1))

    assert runner.zeroed == [[7, 8, 9]]


def test_zero_requests_still_run_on_a_scheduling_step() -> None:
    runner = _ZeroRecorder()
    run_step(runner, _step(zero=[5], scheduled=64))

    assert runner.zeroed == [[5]]
    assert getattr(runner, "_pic_deferred_zero_ids", []) == []
