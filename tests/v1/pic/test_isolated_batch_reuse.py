# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the isolated batch reuse.

A trace showed the one-row PIC batch costing about 2.4 ms of allocation and 1.3 ms of block
table construction per step, even though its shape never changes: one request, one model
length. The instance is kept and refilled - remove the previous row, condense, add the new
request - and any surprise while doing that rebuilds from scratch, because the reuse is only
an optimisation and must never be able to break a step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.v1.worker import gpu_model_runner


class _FakeBatch:
    """Records the calls the reuse path makes."""

    instances = 0

    def __init__(self, **_kwargs) -> None:
        _FakeBatch.instances += 1
        self.req_ids: list[str | None] = []
        self.removed: list[str] = []
        self.condensed = 0
        self.added: list[str] = []
        self.fail_condense = False

    def add_request(self, request_state) -> int:
        self.added.append(request_state.req_id)
        self.req_ids = [request_state.req_id]
        return 0

    def remove_request(self, req_id: str) -> int | None:
        self.removed.append(req_id)
        self.req_ids = [None]
        return 0

    def condense(self) -> None:
        if self.fail_condense:
            raise RuntimeError("condense failed")
        self.condensed += 1
        self.req_ids = []


def _runner():
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            vocab_size=100,
            logitsprocs=None,
            logitsprocs_need_output_token_ids=False,
        ),
        max_model_len=512,
        max_encoder_len=0,
        max_num_tokens=512,
        device="cpu",
        pin_memory=False,
        _init_block_sizes=[16],
        _init_kernel_block_sizes=[16],
        is_pooling_model=False,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        vllm_config=SimpleNamespace(reasoning_config=None),
        pic_metrics=SimpleNamespace(record_materialization=lambda **_: None),
    )


def _state(name: str):
    return SimpleNamespace(req_id=name)


@pytest.fixture(autouse=True)
def _patch_batch(monkeypatch):
    monkeypatch.setattr(gpu_model_runner, "InputBatch", _FakeBatch)
    _FakeBatch.instances = 0


def test_the_second_call_refills_the_same_instance() -> None:
    runner = _runner()

    first = gpu_model_runner.GPUModelRunner._make_pic_isolated_input_batch(
        runner, _state("a")
    )
    second = gpu_model_runner.GPUModelRunner._make_pic_isolated_input_batch(
        runner, _state("b")
    )

    assert first is second, "the batch must be refilled, not rebuilt"
    assert _FakeBatch.instances == 1
    assert first.removed == ["a"], "the previous row is removed first"
    assert first.condensed == 1
    assert first.added == ["a", "b"]


def test_a_failure_during_reuse_rebuilds_instead_of_breaking() -> None:
    runner = _runner()
    first = gpu_model_runner.GPUModelRunner._make_pic_isolated_input_batch(
        runner, _state("a")
    )
    first.fail_condense = True

    second = gpu_model_runner.GPUModelRunner._make_pic_isolated_input_batch(
        runner, _state("b")
    )

    assert second is not first, "a failure must fall back to a fresh batch"
    assert _FakeBatch.instances == 2
    assert second.added == ["b"]
