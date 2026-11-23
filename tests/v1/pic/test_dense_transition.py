# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard recovering the dense transition from two kernel runs.

The module is not wired into a call site yet, so what can be guarded here is the algebra: seeding
with the identity recovers the linear part, seeding with zeros recovers the write term, a
non-square state is refused instead of silently truncated, and the gate rejects a mismatched
operator rather than passing it through.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm.v1.pic import dense_transition
from vllm.v1.pic.worker_plan import PICWorkerUnsupported

HEADS = 2
DIM = 4


def _affine(state):
    """A stand-in for the kernel: ``state @ T + U`` with fixed T and U."""
    torch.manual_seed(0)
    matrix = torch.randn(DIM, DIM) * 0.1 + torch.eye(DIM)
    zero_state = torch.randn(HEADS, DIM, DIM) * 0.1
    return state.matmul(matrix) + zero_state


def _build(**kwargs):
    return dense_transition.build_dense_transition(
        _affine,
        heads=HEADS,
        key_dim=DIM,
        value_dim=kwargs.get("value_dim", DIM),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_seeding_with_the_identity_recovers_the_linear_part() -> None:
    matrix, zero_state = _build()

    torch.manual_seed(0)
    expected_matrix = torch.randn(DIM, DIM) * 0.1 + torch.eye(DIM)
    expected_zero = torch.randn(HEADS, DIM, DIM) * 0.1

    assert torch.allclose(matrix, expected_matrix.expand(HEADS, DIM, DIM), atol=1e-5)
    assert torch.allclose(zero_state, expected_zero, atol=1e-5)


def test_the_recovered_operator_reproduces_the_kernel() -> None:
    matrix, zero_state = _build()
    torch.manual_seed(7)
    probe = torch.randn(HEADS, DIM, DIM)

    ours = dense_transition.apply_dense_transition(matrix, zero_state, probe)

    assert torch.allclose(ours, _affine(probe), atol=1e-5)


def test_it_costs_exactly_two_applications() -> None:
    calls: list[tuple[int, ...]] = []

    def counting(state):
        calls.append(tuple(state.shape))
        return _affine(state)

    dense_transition.build_dense_transition(
        counting,
        heads=HEADS,
        key_dim=DIM,
        value_dim=DIM,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert len(calls) == 2, "one seed with the identity, one with zeros"
    assert calls[0] == (HEADS, DIM, DIM), "the identity is seeded in the state layout"
    assert calls[1] == (HEADS, DIM, DIM)


def test_a_non_square_state_is_refused() -> None:
    with pytest.raises(PICWorkerUnsupported, match="square"):
        dense_transition.build_dense_transition(
            _affine,
            heads=HEADS,
            key_dim=DIM,
            value_dim=DIM + 1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_the_gate_accepts_the_recovered_operator() -> None:
    ok, worst = dense_transition.verify_against_kernel(
        _affine,
        heads=HEADS,
        key_dim=DIM,
        value_dim=DIM,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert ok is True
    assert worst <= 1e-3


def test_the_gate_rejects_a_mismatched_operator() -> None:
    def almost(state):
        # Same linear part, different write term.
        return _affine(state) + 0.5

    ok, worst = dense_transition.verify_against_kernel(
        almost,
        heads=HEADS,
        key_dim=DIM,
        value_dim=DIM,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert ok is False
    assert worst > 1e-3
