# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the batched transition apply.

A segment carries one linear transition per GDN layer, and applying them one at a time
costs a matmul launch and a Python round trip per layer - eighteen of each for this model,
which a trace showed at roughly ten milliseconds inside the non-forward part of a step. When
every layer has the same square matrix and state shape they are applied as one batched
matmul; anything else keeps the per-layer loop, so shapes that would break the batch cannot
be silently mis-applied.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from vllm.v1.pic.state import PICLinearTransition, PICTransitionOperator


def _operator(layers: int, *, size: int = 8, varied: bool = False) -> PICTransitionOperator:
    # apply() requires the state shape to equal zero_state exactly, and the state is the
    # [value, key] matrix that multiplies the square transition, so zero_state carries the
    # same full shape rather than a single broadcastable row.
    transitions = tuple(
        PICLinearTransition(
            matrix=torch.eye(size if not varied or index == 0 else size + 1),
            zero_state=torch.zeros(
                size if not varied or index == 0 else size + 1,
                size if not varied or index == 0 else size + 1,
            ),
        )
        for index in range(layers)
    )
    return PICTransitionOperator(transitions=transitions, token_start=0, token_end=64)


def test_identical_layers_are_batched() -> None:
    operator = _operator(4)

    assert operator._batched is not None


def test_varied_shapes_keep_the_loop() -> None:
    operator = _operator(3, varied=True)

    assert operator._batched is None


def test_the_batched_result_matches_the_per_layer_loop() -> None:
    torch.manual_seed(0)
    operator = _operator(4)
    states = [torch.randn(8, 8) for _ in range(4)]

    batched = operator.apply(states)
    looped = tuple(
        transition.apply(state)
        for transition, state in zip(operator.transitions, states)
    )

    # A batched matmul and eighteen small ones may pick different kernels, so compare
    # within tolerance rather than requiring bit equality.
    for got, want in zip(batched, looped):
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-6), (got, want)


def test_a_varied_batch_still_applies_every_state() -> None:
    operator = _operator(3, varied=True)
    states = [torch.zeros(8, 8), torch.zeros(9, 9), torch.zeros(9, 9)]

    applied = operator.apply(states)

    assert len(applied) == 3
    assert all(torch.count_nonzero(item) == 0 for item in applied)


def test_a_state_count_mismatch_still_raises() -> None:
    operator = _operator(3)

    with pytest.raises(ValueError, match="state count mismatch"):
        operator.apply([torch.zeros(8, 8)])
