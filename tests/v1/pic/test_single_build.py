# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard that a worker plan handed in is used, not rebuilt.

The runner resolves the worker descriptor when a request enters the batch. Building the
single-request plan then resolved it again, regenerating the per-token position lists for
the whole prompt - a second copy of work that had already been done.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.v1.pic import single_request
from vllm.v1.pic.single_request import build_single_request_plan


def _execution_plan():
    return SimpleNamespace(
        ranges=(
            SimpleNamespace(start=0, end=2, action="reuse", segment_index=0),
            SimpleNamespace(start=2, end=4, action="recompute", segment_index=1),
        ),
        reused_ranges=(SimpleNamespace(start=0, end=2, action="reuse"),),
    )


def _builder(**kwargs):
    return build_single_request_plan(
        _execution_plan(),
        prompt_len=4,
        num_requests=1,
        eager_mode=True,
        speculative_decoding=False,
        range_execution_supported=True,
        zero_copy_attention_supported=True,
        allow_fallback=True,
        native_kv_bridge_supported=True,
        **kwargs,
    )


def test_a_handed_in_worker_plan_is_not_rebuilt(monkeypatch) -> None:
    calls = []

    def _spy(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            ranges=(), skip_positions=(0, 1), recompute_positions=(2, 3)
        )

    monkeypatch.setattr(single_request, "build_worker_plan", _spy)
    handed_in = SimpleNamespace(
        ranges=(), skip_positions=(0, 1), recompute_positions=(2, 3)
    )

    _builder(worker_plan=handed_in)

    assert calls == [], "a plan already built must not be rebuilt"


def test_without_a_worker_plan_it_is_built(monkeypatch) -> None:
    calls = []

    def _spy(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            ranges=(), skip_positions=(0, 1), recompute_positions=(2, 3)
        )

    monkeypatch.setattr(single_request, "build_worker_plan", _spy)

    _builder()

    assert len(calls) == 1
