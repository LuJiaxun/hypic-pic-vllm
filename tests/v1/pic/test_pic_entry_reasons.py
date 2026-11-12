# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the PIC entry-condition report.

The fast path has five entry conditions and used to fall through silently, so a batch that
never entered it looked identical to a plan without reuse: no ranges ran, and every reuse
counter stayed at zero. At higher concurrency the prompt stops fitting in one batch budget,
which is exactly the case this report has to name.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _Output:
    def __init__(self, scheduled: dict[str, int]) -> None:
        self.num_scheduled_tokens = scheduled


def _runner(*, copyback: bool = False, states: dict | None = None):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(pic_config=SimpleNamespace(copyback_kv=copyback)),
        requests=states or {},
    )


def _state(*, plan=None, computed: int = 0, prompt: int = 4911):
    return SimpleNamespace(
        pic_runtime_plan=plan,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
    )


def _reasons(runner, output, req_ids, num_reqs):
    return GPUModelRunner._pic_entry_skip_reasons(
        runner, output, req_ids, num_reqs
    )


def test_a_multi_request_batch_is_named() -> None:
    reasons = _reasons(_runner(), _Output({}), ["a", "b"], 2)

    assert "batch=2" in reasons


def test_a_partial_schedule_is_named() -> None:
    runner = _runner(states={"a": _state(plan=object())})

    reasons = _reasons(runner, _Output({"a": 2048}), ["a"], 1)

    assert reasons == "scheduled=2048/4911", reasons


def test_a_missing_plan_is_named() -> None:
    runner = _runner(states={"a": _state(plan=None)})

    reasons = _reasons(runner, _Output({"a": 4911}), ["a"], 1)

    assert "no_runtime_plan" in reasons


def test_copyback_blocks_the_fast_path() -> None:
    runner = _runner(copyback=True, states={"a": _state(plan=object())})

    reasons = _reasons(runner, _Output({"a": 4911}), ["a"], 1)

    assert "copyback_kv" in reasons


def test_resumed_tokens_are_named() -> None:
    runner = _runner(states={"a": _state(plan=object(), computed=512)})

    reasons = _reasons(runner, _Output({"a": 4911}), ["a"], 1)

    assert "computed=512" in reasons
