# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense per-segment transition, recovered by running the kernel twice.

The paper's primitive is a matrix: ``S_out = S_in @ T_C + S_C0`` with ``T_C = prod_t T_t``.
The segment-cumulative matrix does not need a closed form and does not need a per-token loop:
the map the kernel applies is affine, so feeding it two known starting states recovers both
halves.

    apply(I) = I @ T + U = T + U
    apply(0) = 0 @ T + U = U
    T       = apply(I) - apply(0)

``apply(0)`` is exactly the zero-start end state the capture already computes, so the dense
operator costs **one extra kernel run per segment** - the same run the sglang reference
implementation performs (``gdn_backend.py``, ``initial_state`` filled with the identity and
``INPLACE_UPDATE``). The state layout is ``[heads, v, k]`` and the transition multiplies on the
right, so the identity is ``[heads, k, k]``; when ``v != k`` the identity is not a state and the
caller must fall back to the closure representation.

This module is deliberately **not wired into any call site**: step 3 of
PLAN_TRANSITION_MATRIX.md switches the callers only after ``verify_against_kernel`` agrees on
real segments.
"""

from __future__ import annotations

from typing import Callable

import torch

from vllm.v1.pic.worker_plan import PICWorkerUnsupported


def build_dense_transition(
    apply: Callable[[torch.Tensor], torch.Tensor],
    *,
    heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(T_C, S_C0)`` from two applications of the kernel.

    ``apply`` must be the segment's own transition application: ``apply(S) = S @ T + U``. It is
    called twice, with the identity and with zeros, which is what makes the result exact rather
    than derived.
    """
    if value_dim != key_dim:
        raise PICWorkerUnsupported(
            "PIC dense transition needs a square state to seed with the identity: "
            f"value_dim={value_dim} key_dim={key_dim}"
        )

    identity = torch.eye(key_dim, dtype=dtype, device=device)
    identity = identity.expand(heads, key_dim, key_dim).contiguous()
    zeros = torch.zeros((heads, value_dim, key_dim), dtype=dtype, device=device)

    with_seed_identity = apply(identity)
    zero_start = apply(zeros)
    matrix = with_seed_identity - zero_start
    return matrix.contiguous(), zero_start.contiguous()


def apply_dense_transition(
    matrix: torch.Tensor, zero_state: torch.Tensor, state: torch.Tensor
) -> torch.Tensor:
    """``S_out = S_in @ T_C + S_C0``, the layout the recurrent state uses."""
    return state.to(matrix.dtype).matmul(matrix) + zero_state


def verify_against_kernel(
    apply: Callable[[torch.Tensor], torch.Tensor],
    *,
    heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    tolerance: float = 1e-3,
) -> tuple[bool, float]:
    """Check the recovered operator against the kernel on a fresh random state.

    This is the gate that keeps a wrong or stale assumption from shipping: the same state goes
    through both paths and the worst relative difference comes back. Keeping it means a model
    whose decay bookkeeping changes degrades to the closure instead of silently reusing a wrong
    state.
    """
    matrix, zero_state = build_dense_transition(
        apply,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        dtype=dtype,
        device=device,
    )
    probe = torch.randn((heads, value_dim, key_dim), dtype=dtype, device=device)
    ours = apply_dense_transition(matrix, zero_state, probe)
    theirs = apply(probe).to(matrix.dtype)
    error = (ours - theirs).abs()
    scale = theirs.abs().clamp_min(1e-6)
    worst = float((error / scale).max().item()) if error.numel() else 0.0
    return worst <= tolerance, worst
