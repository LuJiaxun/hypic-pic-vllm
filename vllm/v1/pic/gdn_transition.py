# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Torch reference transition extraction for Gated DeltaNet state."""

from __future__ import annotations

import torch

from vllm.v1.pic.state import PICGDNTransition, PICTransitionOperator
from vllm.v1.pic.worker_plan import PICWorkerUnsupported


def apply_native_gdn_transition(
    *,
    backend: str,
    key: "torch.Tensor",
    value: "torch.Tensor",
    log_decay: "torch.Tensor",
    beta: "torch.Tensor",
    chunk_w: "torch.Tensor | None",
    chunk_u: "torch.Tensor | None",
    chunk_g: "torch.Tensor | None",
    chunk_size: int,
    state: "torch.Tensor",
) -> "torch.Tensor":
    """Apply a cached transition using the model's active GDN backend."""
    if not state.is_cuda:
        raise PICWorkerUnsupported("native GDN transition requires CUDA tensors")

    if backend == "triton":
        if chunk_w is None or chunk_u is None or chunk_g is None:
            raise PICWorkerUnsupported(
                "Triton GDN transition requires prepared chunk w/u/g tensors"
            )
        from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h,
        )

        # This is the same state-only Triton kernel used by the normal
        # Triton/FLA GDN prefill path.  PIC does not run a separate Python
        # recurrence on the production CUDA path.
        with torch.no_grad():
            _, _, final_state = chunk_gated_delta_rule_fwd_h(
                k=key.unsqueeze(0).contiguous(),
                w=chunk_w.unsqueeze(0).contiguous(),
                u=chunk_u.unsqueeze(0).contiguous(),
                g=chunk_g.unsqueeze(0).contiguous(),
                initial_state=state.unsqueeze(0).contiguous(),
                output_final_state=True,
                chunk_size=chunk_size,
            )
        if final_state is None:
            raise PICWorkerUnsupported(
                "Triton GDN transition did not return final state"
            )
        if not bool(torch.isfinite(final_state).all().item()):
            raise PICWorkerUnsupported(
                "Triton GDN transition produced a non-finite final state"
            )
        return final_state[0].to(dtype=state.dtype)

    if backend == "flashinfer":
        from flashinfer.gdn_prefill import (
            chunk_gated_delta_rule as flashinfer_gdn,
        )

        # The final recurrent state is independent of q.  Reusing k as q
        # avoids introducing a second cached model input while keeping the
        # same FlashInfer state-update implementation as normal forward.
        with torch.no_grad():
            _, final_state = flashinfer_gdn(
                q=key.contiguous(),
                k=key.contiguous(),
                v=value.contiguous(),
                g=torch.exp(log_decay.float()),
                beta=beta.float(),
                initial_state=state.unsqueeze(0).float().contiguous(),
                output_final_state=True,
            )
        if final_state is None:
            raise PICWorkerUnsupported(
                "FlashInfer GDN transition did not return final state"
            )
        if not bool(torch.isfinite(final_state).all().item()):
            raise PICWorkerUnsupported(
                "FlashInfer GDN transition produced a non-finite final state"
            )
        return final_state[0].to(dtype=state.dtype)

    raise PICWorkerUnsupported(
        "PIC transition does not support the active GDN backend: "
        f"{backend!r}"
    )


def _build_fla_chunk_reference(
    key: "torch.Tensor",
    value: "torch.Tensor",
    log_decay: "torch.Tensor",
    beta: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", int] | None:
    """Build the same WY tensors consumed by vLLM's native GDN path.

    This deliberately reuses existing vLLM FLA/Triton helpers.  No kernel is
    added or changed by PIC.  CPU/unit-test inputs use the raw recurrence;
    live CUDA capture uses this representation so ``PICGDNTransition.apply``
    follows the numerical ordering of ``chunk.py`` and ``chunk_delta_h.py``.
    """
    if not key.is_cuda:
        return None
    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
    from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
    from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd

    chunk_size = int(FLA_CHUNK_SIZE)
    k = key.unsqueeze(0).contiguous()
    v = value.unsqueeze(0).contiguous()
    g = log_decay.unsqueeze(0).contiguous()
    b = beta.unsqueeze(0).contiguous()
    g_cumsum = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=None,
        output_dtype=torch.float32,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=b,
        g=g_cumsum,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
        output_dtype=key.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=b,
        g_cumsum=g_cumsum,
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
    )
    return w.squeeze(0), u.squeeze(0), g_cumsum.squeeze(0), chunk_size


def build_gdn_transition_operator(
    key: "torch.Tensor",
    value: "torch.Tensor",
    log_decay: "torch.Tensor",
    beta: "torch.Tensor",
    *,
    token_start: int,
    token_end: int,
    state_dtype: "torch.dtype | None" = None,
    use_fla_reference: bool | None = None,
    backend: str | None = None,
) -> PICTransitionOperator:
    """Build a segment transition from Qwen GDN post-conv inputs.

    For one value head, the GDN recurrent update is:

    ``S' = exp(g) * S + beta * (v - S @ k) outer k``

    The returned operator stores the post-convolution inputs themselves. This
    is both the exact reference recurrence and the compact representation that
    a later worker execution stage can apply to a different incoming state;
    it does not materialize a dense matrix for every token.

    Inputs use the post-``fused_post_conv_prep`` layout:

    * ``key``: ``[tokens, key_heads, key_dim]``;
    * ``value``: ``[tokens, value_heads, value_dim]``;
    * ``log_decay`` and ``beta``: ``[tokens, value_heads]``.
    """
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError("PIC GDN transition expects rank-3 key/value tensors")
    if log_decay.ndim != 2 or beta.ndim != 2:
        raise ValueError("PIC GDN transition expects rank-2 gate tensors")
    token_count = value.shape[0]
    if token_count <= 0 or token_end - token_start != token_count:
        raise ValueError("PIC GDN transition token range does not match inputs")
    if key.shape[0] != token_count or log_decay.shape != beta.shape:
        raise ValueError("PIC GDN transition input lengths are inconsistent")
    if value.shape[1] != log_decay.shape[1]:
        raise ValueError("PIC GDN key/value head counts are inconsistent")

    value_heads = value.shape[1]
    key_heads = key.shape[1]
    if value_heads % key_heads != 0:
        raise ValueError("PIC GDN value heads must be divisible by key heads")

    # Only Triton/FLA consumes the WY representation.  FlashInfer retains
    # the post-conv inputs and dispatches to its own state operator instead.
    chunk_reference = (
        _build_fla_chunk_reference(key, value, log_decay, beta)
        if (backend in (None, "triton") and use_fla_reference is not False)
        else None
    )

    # Keep key/value in the model's storage dtype.  PICGDNTransition applies
    # them in FP32 and expands GQA heads transiently, so this avoids both the
    # permanent FP32 copy and the permanent key-head replication.
    target_dtype = value.dtype if state_dtype is None else state_dtype
    zero_seed = torch.zeros(
        (value.shape[1], value.shape[2], key.shape[2]),
        dtype=target_dtype,
        device=value.device,
    )
    transition_with_seed = PICGDNTransition(
        key=key,
        value=value,
        log_decay=log_decay,
        beta=beta,
        zero_state=zero_seed,
        chunk_w=None if chunk_reference is None else chunk_reference[0],
        chunk_u=None if chunk_reference is None else chunk_reference[1],
        chunk_g=None if chunk_reference is None else chunk_reference[2],
        chunk_size=64 if chunk_reference is None else chunk_reference[3],
        use_fla_reference=False,
        backend=backend,
    )
    # ``zero_state`` is the operator's zero-start end state, not merely a
    # shape placeholder.  Keeping it materialized makes composition and
    # validation unambiguous while ``key/value/gates`` remain the compact
    # representation used to apply the operator to arbitrary live state.
    zero_end_state = transition_with_seed.apply(zero_seed)
    transition = PICGDNTransition(
        key=key,
        value=value,
        log_decay=log_decay,
        beta=beta,
        zero_state=zero_end_state,
        chunk_w=None if chunk_reference is None else chunk_reference[0],
        chunk_u=None if chunk_reference is None else chunk_reference[1],
        chunk_g=None if chunk_reference is None else chunk_reference[2],
        chunk_size=64 if chunk_reference is None else chunk_reference[3],
        use_fla_reference=chunk_reference is not None,
        backend=backend,
    )
    return PICTransitionOperator(
        transitions=(transition,),
        token_start=token_start,
        token_end=token_end,
    )
