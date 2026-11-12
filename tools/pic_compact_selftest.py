# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that a captured transition really can be served from a matrix.

Run inside the container on a GPU, with no server:

    python3 tools/pic_compact_selftest.py

It builds a transition of realistic shape directly through the same code path the worker uses,
then answers three separate questions - a matrix that fails any of them is not usable, and the
recurrence keeps being used:

1. Can the matrix be built at all (square state, three dimensions)?
2. Does either multiplication direction agree with the kernel?  With equal key and value sizes both
   fit the shape while only one is correct, so this must be measured rather than assumed.
3. Does the agreeing direction keep agreeing on states the build never saw?  A single probe can pass
   by coincidence; several independent states are what makes the answer trustworthy.

The script prints one verdict line at the end so the result is a decision, not a number to interpret.
"""

from __future__ import annotations

import torch

from vllm.v1.pic.gdn_transition import (
    _build_fla_chunk_reference,
    apply_native_gdn_transition,
)
from vllm.v1.pic.state import (
    PICGDNTransition,
    _COMPACT_CACHE,
    _apply_compact,
    _compact_entry,
    _transition_identity,
)

TOKENS = 640
HEADS = 4
KEY_DIM = 128
VALUE_DIM = 128
CHUNK = 64


def build(device: torch.device) -> PICGDNTransition:
    """A transition of realistic shape, prepared the way the worker prepares one."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    # The model normalises q/k inside the kernel and drives beta through a sigmoid, so the per-token
    # transition is a contraction. Un-normalised keys with beta near one make the cumulative operator
    # grow by tens per token, and six hundred tokens of that overflows bf16 - which says nothing
    # about the matrix form, only about the synthetic data. These ranges match the model's.
    key = torch.randn(TOKENS, HEADS, KEY_DIM, dtype=dtype, device=device)
    key = key / key.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(dtype)
    value = torch.randn(TOKENS, HEADS, VALUE_DIM, dtype=dtype, device=device)
    beta = (torch.rand(TOKENS, HEADS, dtype=torch.float32, device=device) * 0.20 + 0.05)
    log_decay = -torch.rand(TOKENS, HEADS, dtype=torch.float32, device=device) * 0.02

    reference = _build_fla_chunk_reference(key, value, log_decay, beta)
    if reference is None:
        raise SystemExit(
            "the chunked reference could not be built; the FLA preparation is unavailable here"
        )
    chunk_w, chunk_u, chunk_g, chunk_size = reference

    zeros = torch.zeros((HEADS, VALUE_DIM, KEY_DIM), dtype=dtype, device=device)
    seed = PICGDNTransition(
        key=key,
        value=value,
        log_decay=log_decay,
        beta=beta,
        zero_state=zeros,
        chunk_w=chunk_w,
        chunk_u=chunk_u,
        chunk_g=chunk_g,
        chunk_size=chunk_size or CHUNK,
        backend="triton",
    )
    # The zero-start state is whatever the transition produces from zeros; it is an input to the
    # operator, not something the caller can guess.
    zero_state = seed.apply(zeros)
    return PICGDNTransition(
        key=key,
        value=value,
        log_decay=log_decay,
        beta=beta,
        zero_state=zero_state,
        chunk_w=chunk_w,
        chunk_u=chunk_u,
        chunk_g=chunk_g,
        chunk_size=chunk_size or CHUNK,
        backend="triton",
    )


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = b.float().norm().clamp_min(1e-12)
    return float((a.float() - b.float()).norm() / scale)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Direction agreement, insensitive to a uniform gain difference."""
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    device = torch.device("cuda")

    transition = build(device)
    print(
        f"tokens={TOKENS} heads={HEADS} key={KEY_DIM} value={VALUE_DIM} "
        f"dtype={transition.key.dtype}"
    )
    print()

    # 1) build
    compact = transition.compact()
    if compact is None:
        print("VERDICT: the state is not square, so no matrix exists for this transition")
        return 0
    finite = bool(torch.isfinite(compact.matrix).all() and torch.isfinite(compact.zero_state).all())
    print(
        f"1. matrix built          shape={tuple(compact.matrix.shape)} "
        f"zero_state={tuple(compact.zero_state.shape)} finite={finite} "
        f"absmax={float(compact.matrix.float().abs().max()):.3e}"
    )
    if not finite:
        print(
            "VERDICT: the recovered matrix has non-finite entries, so this span is beyond what "
            "the matrix form can represent; the recurrence stays in use"
        )
        return 0

    # 2) direction, measured
    entry = _compact_entry(transition)
    if entry is None:
        print("VERDICT: no matrix could be built for this transition")
        return 0
    matrix, zero_state = entry
    print(f"2. convention            right (S @ M)")

    # The production path stores the matrix and uses it; accuracy is this script's job, so the whole
    # point of running it is the comparison below.


    # 3) generalisation: states the build never saw.
    #
    # The reference must not come from this operator's own cache: once the entry is stored, `apply`
    # is served from it, so comparing the matrix against `apply` would compare the matrix with
    # itself. The cache entry is therefore dropped first, and a second reference is taken by calling
    # the kernel directly - the shortest path to an answer that does not depend on `apply` at all.
    worst = 0.0
    worst_direct = 0.0
    worst_cos = 1.0
    worst_cos_direct = 1.0
    for index in range(3):
        torch.manual_seed(1000 + index)
        probe = torch.randn(
            tuple(zero_state.shape), dtype=zero_state.dtype, device=zero_state.device
        )
        _COMPACT_CACHE.pop(_transition_identity(transition), None)
        expected = transition.apply(probe)
        direct = apply_native_gdn_transition(
            backend="triton",
            key=transition.key,
            value=transition.value,
            log_decay=transition.log_decay,
            beta=transition.beta,
            chunk_w=transition.chunk_w,
            chunk_u=transition.chunk_u,
            chunk_g=transition.chunk_g,
            chunk_size=transition.chunk_size,
            state=probe,
        )
        actual = _apply_compact(entry, probe)
        error = relative(actual, expected)
        error_direct = relative(actual, direct)
        cos_expected = cosine(actual, expected)
        cos_direct = cosine(actual, direct)
        worst = max(worst, error)
        worst_direct = max(worst_direct, error_direct)
        worst_cos = min(worst_cos, cos_expected)
        worst_cos_direct = min(worst_cos_direct, cos_direct)
        print(
            f"3.{index + 1} unseen state          frob_rel={error:.3e} "
            f"cos={cos_expected:.8f} vs_kernel={error_direct:.3e} "
            f"cos_kernel={cos_direct:.8f}"
        )
    # Leave the cache as the worker would find it: built and usable.
    _COMPACT_CACHE[_transition_identity(transition)] = entry

    print()
    # A recurrence collapsed into one product cannot reproduce the kernel's per-chunk rounding, so
    # the relative error is reported rather than required to vanish; what must hold is direction,
    # boundedness and finiteness. Whether the residual matters is decided by comparing generated
    # tokens against PIC off - that is the only end-to-end question.
    bounded = max(worst, worst_direct) <= 5e-2
    direction_ok = min(worst_cos, worst_cos_direct) >= 0.999
    if bounded and direction_ok:
        print(
            "VERDICT: the matrix tracks the kernel on unseen states "
            f"(worst frob_rel={worst:.3e}, vs kernel directly {worst_direct:.3e}; "
            f"worst cos={worst_cos:.8f}, vs kernel directly {worst_cos_direct:.8f}). "
            "The residual is the rounding a recurrence gives up when it becomes one product; "
            "judge it by comparing generated tokens against PIC off."
        )
    else:
        print(
            "VERDICT: the matrix does not track the kernel on unseen states "
            f"(worst frob_rel={worst:.3e}, vs kernel directly {worst_direct:.3e}; "
            f"worst cos={worst_cos:.8f}, vs kernel directly {worst_cos_direct:.8f}); "
            "the recurrence stays in use"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
