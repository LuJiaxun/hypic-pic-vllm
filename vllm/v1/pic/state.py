# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PIC state layout for hybrid Mamba/GDN attention.

vLLM already owns the authoritative state shapes and dtypes in ``MambaSpec``.
This module only translates that information into a PIC-specific descriptor;
it does not allocate or copy model state yet.  Keeping this translation
centralized is important because Mamba, short-conv, and GDN use different
numbers and meanings of state tensors.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from vllm.v1.pic.handles import PICHandleKind

if TYPE_CHECKING:
    import torch

    from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec


@dataclass(frozen=True)
class PICStateSpec:
    """Description of one recurrent or convolutional state tensor."""

    group_id: int
    state_index: int
    kind: PICHandleKind
    shape: tuple[int, ...]
    dtype: "torch.dtype"


@dataclass
class PICRequestStateBinding:
    """Request-owned cursor for hybrid state and native KV reuse.

    The binding is deliberately independent of batch row indices. A vLLM
    request may move between rows after preemption or batch reordering, while
    its recurrent/conv block IDs, logical cursor and transition cursor must
    remain attached to the request itself.
    """

    request_id: str
    logical_token_position: int = 0
    segment_index: int = -1
    range_cursor: int = 0
    transition_cursor: int = 0
    mapping_generation: int = 0
    decode_steps: int = 0
    state_block_ids: tuple[tuple[int, ...], ...] = ()
    fallback_reason: str | None = None

    def bind_mapping(self, mapping_present: bool) -> None:
        """Advance the mapping generation when a request plan is refreshed."""
        self.mapping_generation += 1
        if not mapping_present:
            self.fallback_reason = "native_mapping_unavailable"
        else:
            self.fallback_reason = None

    def sync(
        self,
        *,
        num_computed_tokens: int,
        block_ids: Sequence[Sequence[int]],
        segment_index: int = -1,
        range_cursor: int | None = None,
        transition_cursor: int | None = None,
    ) -> None:
        """Update request-owned state after scheduler row/block changes."""
        if num_computed_tokens < 0:
            raise ValueError("PIC logical token position must be non-negative")
        self.logical_token_position = int(num_computed_tokens)
        self.state_block_ids = tuple(tuple(int(block) for block in group) for group in block_ids)
        self.segment_index = int(segment_index)
        if range_cursor is not None:
            self.range_cursor = int(range_cursor)
        if transition_cursor is not None:
            self.transition_cursor = int(transition_cursor)

    def mark_decode_step(
        self,
        *,
        num_computed_tokens: int,
        scheduled_tokens: int,
        mapping_present: bool,
    ) -> bool:
        """Validate and record one decode step using the persistent mapping."""
        if num_computed_tokens <= 0 or scheduled_tokens <= 0:
            self.fallback_reason = "invalid_decode_cursor"
            return False
        if self.mapping_generation <= 0 or not mapping_present:
            self.fallback_reason = "native_mapping_unavailable"
            return False
        if not self.state_block_ids:
            self.fallback_reason = "state_blocks_unavailable"
            return False
        if self.logical_token_position != num_computed_tokens:
            self.fallback_reason = "logical_cursor_mismatch"
            return False
        self.decode_steps += 1
        self.logical_token_position = num_computed_tokens + scheduled_tokens
        self.fallback_reason = None
        return True

    def invalidate(self, reason: str) -> None:
        self.fallback_reason = reason


@dataclass(frozen=True)
class PICAffineTransition:
    """One compact affine recurrent-state transition.

    The transition is represented as ``out = decay * state + zero_state``.
    ``decay`` may be broadcast over the state tensor. ``zero_state`` is the
    output produced by running the segment from zero; it is not an end-state
    snapshot from an unrelated request.
    """

    decay: "torch.Tensor"
    zero_state: "torch.Tensor"

    def __post_init__(self) -> None:
        if tuple(self.zero_state.shape) == ():
            raise ValueError("PIC transition zero_state must be a tensor")
        if self.decay.device != self.zero_state.device:
            raise ValueError("PIC transition tensors must use the same device")
        if self.zero_state.dtype != self.decay.dtype:
            raise ValueError("PIC transition tensors must use the same dtype")

    def apply(self, state: "torch.Tensor") -> "torch.Tensor":
        """Apply this transition to an incoming recurrent state."""
        if tuple(state.shape) != tuple(self.zero_state.shape):
            raise ValueError(
                "PIC transition state shape mismatch: "
                f"expected={tuple(self.zero_state.shape)}, "
                f"actual={tuple(state.shape)}"
            )
        if state.dtype != self.zero_state.dtype or state.device != self.zero_state.device:
            raise ValueError("PIC transition state dtype/device mismatch")
        try:
            return self.decay * state + self.zero_state
        except RuntimeError as exc:
            raise ValueError("PIC transition decay is not broadcastable") from exc

    def compose(self, following: "PICAffineTransition") -> "PICAffineTransition":
        """Return ``following(self(state))`` without materializing a matrix."""
        if self.zero_state.shape != following.zero_state.shape:
            raise ValueError("PIC transitions have incompatible state shapes")
        if self.zero_state.dtype != following.zero_state.dtype:
            raise ValueError("PIC transitions have incompatible state dtypes")
        if self.zero_state.device != following.zero_state.device:
            raise ValueError("PIC transitions have incompatible state devices")
        return PICAffineTransition(
            decay=following.decay * self.decay,
            zero_state=following.decay * self.zero_state + following.zero_state,
        )

    def detached_clone(self) -> "PICAffineTransition":
        return PICAffineTransition(
            decay=self.decay.detach().clone(),
            zero_state=self.zero_state.detach().clone(),
        )


@dataclass(frozen=True)
class PICLinearTransition:
    """A recurrent transition represented as ``state @ matrix + zero_state``.

    Gated delta-rule state updates are not element-wise affine: each token
    applies a low-rank right-side matrix to the ``[value_dim, key_dim]`` state.
    This representation keeps that structure in Torch without materializing a
    dense operator over the flattened recurrent state.
    """

    matrix: "torch.Tensor"
    zero_state: "torch.Tensor"

    def __post_init__(self) -> None:
        if self.matrix.ndim < 2 or self.matrix.shape[-1] != self.matrix.shape[-2]:
            raise ValueError("PIC linear transition matrix must be square")
        if self.zero_state.ndim < 2:
            raise ValueError("PIC linear transition zero_state must be non-scalar")
        if self.matrix.device != self.zero_state.device:
            raise ValueError("PIC transition tensors must use the same device")
        if self.matrix.dtype != self.zero_state.dtype:
            raise ValueError("PIC transition tensors must use the same dtype")
        if self.zero_state.shape[-1] != self.matrix.shape[-1]:
            raise ValueError("PIC linear transition state/matrix shapes mismatch")

    def apply(self, state: "torch.Tensor") -> "torch.Tensor":
        if tuple(state.shape) != tuple(self.zero_state.shape):
            raise ValueError(
                "PIC linear transition state shape mismatch: "
                f"expected={tuple(self.zero_state.shape)}, "
                f"actual={tuple(state.shape)}"
            )
        if state.dtype != self.zero_state.dtype or state.device != self.zero_state.device:
            raise ValueError("PIC linear transition state dtype/device mismatch")
        try:
            return state.matmul(self.matrix) + self.zero_state
        except RuntimeError as exc:
            raise ValueError("PIC linear transition cannot be applied") from exc

    def compose(self, following: "PICLinearTransition") -> "PICLinearTransition":
        """Return ``following(self(state))``."""
        if self.zero_state.shape != following.zero_state.shape:
            raise ValueError("PIC linear transitions have incompatible state shapes")
        if self.zero_state.dtype != following.zero_state.dtype:
            raise ValueError("PIC linear transitions have incompatible state dtypes")
        if self.zero_state.device != following.zero_state.device:
            raise ValueError("PIC linear transitions have incompatible state devices")
        return PICLinearTransition(
            matrix=self.matrix.matmul(following.matrix),
            zero_state=self.zero_state.matmul(following.matrix) + following.zero_state,
        )

    def detached_clone(self) -> "PICLinearTransition":
        return PICLinearTransition(
            matrix=self.matrix.detach().clone(),
            zero_state=self.zero_state.detach().clone(),
        )


@dataclass(frozen=True)
class PICGDNTransition:
    """Structured GDN transition without materializing a dense matrix.

    ``zero_state`` stores the end state produced by applying the transition
    to an all-zero incoming state.  It is also used for shape validation; the
    recurrence itself is represented by the captured token tensors.
    """

    key: "torch.Tensor"
    value: "torch.Tensor"
    log_decay: "torch.Tensor"
    beta: "torch.Tensor"
    zero_state: "torch.Tensor"
    # The native FLA/Triton prefill path does not update the state directly
    # from (k, v, beta, g).  It first builds the chunk-local WY representation
    # (w, u, local-cumsum(g)) and then runs chunk_delta_h.  These optional
    # tensors keep the PIC operator numerically aligned with that path while
    # retaining the raw tensors for CPU/reference fallback and serialization.
    chunk_w: "torch.Tensor | None" = None
    chunk_u: "torch.Tensor | None" = None
    chunk_g: "torch.Tensor | None" = None
    chunk_size: int = 64
    use_fla_reference: bool = False
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.key.ndim != 3 or self.value.ndim != 3:
            raise ValueError("PIC GDN transition expects rank-3 key/value tensors")
        if self.log_decay.ndim != 2 or self.beta.ndim != 2:
            raise ValueError("PIC GDN transition expects rank-2 gate tensors")
        if self.key.shape[0] != self.value.shape[0]:
            raise ValueError("PIC GDN transition token counts do not match")
        if self.log_decay.shape != self.beta.shape:
            raise ValueError("PIC GDN transition gate shapes do not match")
        if self.value.shape[:2] != self.log_decay.shape:
            raise ValueError("PIC GDN transition value/gate heads do not match")
        if self.value.shape[1] % self.key.shape[1] != 0:
            raise ValueError(
                "PIC GDN value heads must be divisible by key heads"
            )
        if not (
            self.key.device
            == self.value.device
            == self.log_decay.device
            == self.beta.device
            == self.zero_state.device
        ):
            raise ValueError("PIC GDN transition tensors must use one device")
        expected_state_shape = (
            self.value.shape[1],
            self.value.shape[2],
            self.key.shape[2],
        )
        if tuple(self.zero_state.shape) != expected_state_shape:
            raise ValueError(
                "PIC GDN transition zero state shape mismatch: "
                f"expected={expected_state_shape}, actual={tuple(self.zero_state.shape)}"
            )
        chunked = (self.chunk_w, self.chunk_u, self.chunk_g)
        if any(item is not None for item in chunked):
            if not all(item is not None for item in chunked):
                raise ValueError(
                    "PIC GDN chunked reference must provide w, u and g"
                )
            assert self.chunk_w is not None
            assert self.chunk_u is not None
            assert self.chunk_g is not None
            if self.chunk_size <= 0:
                raise ValueError("PIC GDN chunk size must be positive")
            if self.chunk_w.ndim != 3 or self.chunk_u.ndim != 3:
                raise ValueError("PIC GDN chunked w/u tensors must be rank 3")
            if self.chunk_g.ndim != 2:
                raise ValueError("PIC GDN chunked g tensor must be rank 2")
            if (
                self.chunk_w.shape[0] != self.key.shape[0]
                or self.chunk_u.shape[0] != self.key.shape[0]
                or self.chunk_g.shape[0] != self.key.shape[0]
            ):
                raise ValueError("PIC GDN chunked reference token counts mismatch")
            if self.chunk_w.shape[1:] != (
                self.value.shape[1],
                self.key.shape[2],
            ):
                raise ValueError("PIC GDN chunked w shape mismatch")
            if self.chunk_u.shape[1:] != tuple(self.value.shape[1:]):
                raise ValueError("PIC GDN chunked u shape mismatch")
            if self.chunk_g.shape[1] != self.value.shape[1]:
                raise ValueError("PIC GDN chunked g head count mismatch")
            if not (
                self.chunk_w.device
                == self.chunk_u.device
                == self.chunk_g.device
                == self.zero_state.device
            ):
                raise ValueError("PIC GDN chunked tensors must use one device")

    def _apply_chunked_reference(
        self,
        state: "torch.Tensor",
        *,
        chunk_w: "torch.Tensor | None" = None,
        chunk_u: "torch.Tensor | None" = None,
        chunk_g: "torch.Tensor | None" = None,
        chunk_size: int | None = None,
    ) -> "torch.Tensor":
        """Apply the FLA/Triton chunk_delta_h recurrence in Torch.

        ``chunk_w``, ``chunk_u`` and ``chunk_g`` are produced by the same
        chunk-local WY preparation used by ``chunk_gated_delta_rule_fwd``.
        The ordering below mirrors ``chunk_delta_h.py``: materialize the
        chunk-start state, subtract ``w @ h``, apply the within-chunk gate,
        downcast the update to the key dtype, then accumulate ``k.T @ delta``
        in FP32.  This is intentionally a correctness reference, not a new
        execution kernel.
        """
        chunk_w = self.chunk_w if chunk_w is None else chunk_w
        chunk_u = self.chunk_u if chunk_u is None else chunk_u
        chunk_g = self.chunk_g if chunk_g is None else chunk_g
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        assert chunk_w is not None
        assert chunk_u is not None
        assert chunk_g is not None
        work = state.float()
        key = self.key
        repeat = work.shape[0] // key.shape[1]
        for chunk_start in range(0, key.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, key.shape[0])
            for value_head in range(work.shape[0]):
                key_head = value_head // repeat
                w = chunk_w[chunk_start:chunk_end, value_head]
                u = chunk_u[chunk_start:chunk_end, value_head]
                g = chunk_g[chunk_start:chunk_end, value_head]
                # Triton casts the resident FP32 state to the w dtype for the
                # dot product and keeps the dot accumulator in FP32.
                memory = torch.matmul(
                    w.float(), work[value_head].float().transpose(0, 1)
                )
                delta = u.float() - memory
                last_gate = g[-1]
                delta = delta * torch.exp(last_gate - g).unsqueeze(-1)
                work[value_head] = work[value_head] * torch.exp(last_gate)
                # chunk_delta_h explicitly quantizes the update to k.dtype
                # before the state update; omitting this is the main source of
                # the previous Python-vs-Triton drift.
                delta = delta.to(key.dtype)
                update = torch.matmul(
                    delta.float().transpose(0, 1),
                    key[chunk_start:chunk_end, key_head].float(),
                )
                work[value_head] = work[value_head] + update
        return work.to(state.dtype)

    def apply(self, state: "torch.Tensor") -> "torch.Tensor":
        if tuple(state.shape) != tuple(self.zero_state.shape):
            raise ValueError(
                "PIC GDN transition state shape mismatch: "
                f"expected={tuple(self.zero_state.shape)}, "
                f"actual={tuple(state.shape)}"
            )
        if state.device != self.zero_state.device:
            raise ValueError("PIC GDN transition state device mismatch")

        # Live CUDA capture records the active model backend.  Do not silently
        # replace that backend with the Torch reference on the production path.
        if self.backend is not None:
            from vllm.v1.pic.gdn_transition import apply_native_gdn_transition

            return apply_native_gdn_transition(
                backend=self.backend,
                key=self.key,
                value=self.value,
                log_decay=self.log_decay,
                beta=self.beta,
                chunk_w=self.chunk_w,
                chunk_u=self.chunk_u,
                chunk_g=self.chunk_g,
                chunk_size=self.chunk_size,
                state=state,
            )

        # An explicitly supplied chunk representation is already the fully
        # prepared reference and must take precedence over the mode flag.
        # This also keeps direct construction in tests and future callers
        # from silently falling back to the raw recurrence.
        if self.use_fla_reference or self.chunk_w is not None:
            chunk_w = self.chunk_w
            chunk_u = self.chunk_u
            chunk_g = self.chunk_g
            chunk_size = self.chunk_size
            if chunk_w is None or chunk_u is None or chunk_g is None:
                # Avoid retaining a second copy of the WY representation in
                # every cached operator.  The helper is the same existing
                # FLA/Triton preparation used by the model forward.
                from vllm.v1.pic.gdn_transition import (
                    _build_fla_chunk_reference,
                )

                reference = _build_fla_chunk_reference(
                    self.key, self.value, self.log_decay, self.beta
                )
                if reference is not None:
                    chunk_w, chunk_u, chunk_g, chunk_size = reference
            if chunk_w is not None and chunk_u is not None and chunk_g is not None:
                return self._apply_chunked_reference(
                    state,
                    chunk_w=chunk_w,
                    chunk_u=chunk_u,
                    chunk_g=chunk_g,
                    chunk_size=chunk_size,
                )

        # Keep the captured model dtype in the persistent operator (normally
        # BF16), but perform the reference recurrence in FP32.  The GQA
        # key-head expansion is also transient rather than stored per token.
        work = state.float()
        key = self.key.float()
        value = self.value.float()
        log_decay = self.log_decay.float()
        beta = self.beta.float()
        head_repeat = work.shape[0] // key.shape[1]
        for token_index in range(key.shape[0]):
            key_token = key[token_index]
            if key_token.shape[0] != work.shape[0]:
                key_token = key_token.repeat_interleave(head_repeat, dim=0)
            value_token = value[token_index]
            decay = log_decay[token_index].exp().unsqueeze(-1).unsqueeze(-1)
            beta_token = beta[token_index].unsqueeze(-1).unsqueeze(-1)
            memory = (work * key_token.unsqueeze(-2)).sum(dim=-1)
            work = work * decay + (
                beta_token
                * (value_token - memory).unsqueeze(-1)
                * key_token.unsqueeze(-2)
            )
        return work.to(state.dtype)

    def compose(self, following: "PICGDNTransition") -> "PICGDNTransition":
        if self.key.device != following.key.device:
            raise ValueError("PIC GDN transitions have incompatible devices")
        if self.zero_state.shape != following.zero_state.shape:
            raise ValueError("PIC GDN transitions have incompatible state shapes")
        combined_zero = following.apply(self.zero_state)
        return PICGDNTransition(
            key=torch.cat((self.key, following.key), dim=0),
            value=torch.cat((self.value, following.value), dim=0),
            log_decay=torch.cat((self.log_decay, following.log_decay), dim=0),
            beta=torch.cat((self.beta, following.beta), dim=0),
            zero_state=combined_zero,
            chunk_w=(
                None
                if self.chunk_w is None or following.chunk_w is None
                else torch.cat((self.chunk_w, following.chunk_w), dim=0)
            ),
            chunk_u=(
                None
                if self.chunk_u is None or following.chunk_u is None
                else torch.cat((self.chunk_u, following.chunk_u), dim=0)
            ),
            chunk_g=(
                None
                if self.chunk_g is None or following.chunk_g is None
                else torch.cat((self.chunk_g, following.chunk_g), dim=0)
            ),
            chunk_size=self.chunk_size,
            backend=self.backend or following.backend,
            use_fla_reference=(
                self.use_fla_reference or following.use_fla_reference
            ),
        )

    def slice(self, local_start: int, local_end: int) -> "PICGDNTransition":
        """Return the transition for a local token sub-range."""
        if not (0 <= local_start < local_end <= self.key.shape[0]):
            raise ValueError("PIC GDN transition slice is outside the token range")
        key = self.key[local_start:local_end].contiguous()
        value = self.value[local_start:local_end].contiguous()
        log_decay = self.log_decay[local_start:local_end].contiguous()
        beta = self.beta[local_start:local_end].contiguous()

        # chunk_w/u/g are local to the original FLA chunk boundaries.  They
        # cannot be sliced at an arbitrary token offset and then passed to
        # chunk_delta_h: the new range would have the wrong chunk origin and
        # wrong local cumulative gates.  Rebuild the WY representation from
        # the raw post-conv inputs for the active Triton backend.
        chunk_w = None
        chunk_u = None
        chunk_g = None
        chunk_size = self.chunk_size
        if self.backend == "triton":
            from vllm.v1.pic.gdn_transition import _build_fla_chunk_reference

            chunk_reference = _build_fla_chunk_reference(
                key, value, log_decay, beta
            )
            if chunk_reference is None:
                from vllm.v1.pic.worker_plan import PICWorkerUnsupported

                raise PICWorkerUnsupported(
                    "Triton GDN transition slice requires CUDA tensors"
                )
            chunk_w, chunk_u, chunk_g, chunk_size = chunk_reference
        elif self.backend == "flashinfer":
            # FlashInfer consumes the raw post-conv inputs directly.
            pass
        else:
            # CPU/reference operators retain their existing test semantics.
            chunk_w = (
                None
                if self.chunk_w is None
                else self.chunk_w[local_start:local_end].contiguous()
            )
            chunk_u = (
                None
                if self.chunk_u is None
                else self.chunk_u[local_start:local_end].contiguous()
            )
            chunk_g = (
                None
                if self.chunk_g is None
                else self.chunk_g[local_start:local_end].contiguous()
            )

        use_fla_reference = chunk_w is not None
        result = PICGDNTransition(
            key=key,
            value=value,
            log_decay=log_decay,
            beta=beta,
            zero_state=torch.zeros_like(self.zero_state),
            chunk_w=chunk_w,
            chunk_u=chunk_u,
            chunk_g=chunk_g,
            chunk_size=chunk_size,
            backend=self.backend,
            use_fla_reference=use_fla_reference,
        )
        return PICGDNTransition(
            key=result.key,
            value=result.value,
            log_decay=result.log_decay,
            beta=result.beta,
            zero_state=result.apply(torch.zeros_like(self.zero_state)),
            chunk_w=result.chunk_w,
            chunk_u=result.chunk_u,
            chunk_g=result.chunk_g,
            chunk_size=result.chunk_size,
            backend=result.backend,
            use_fla_reference=use_fla_reference,
        )

    def detached_clone(self) -> "PICGDNTransition":
        return PICGDNTransition(
            key=self.key.detach().clone(),
            value=self.value.detach().clone(),
            log_decay=self.log_decay.detach().clone(),
            beta=self.beta.detach().clone(),
            zero_state=self.zero_state.detach().clone(),
            backend=self.backend,
            chunk_w=(
                None if self.chunk_w is None else self.chunk_w.detach().clone()
            ),
            chunk_u=(
                None if self.chunk_u is None else self.chunk_u.detach().clone()
            ),
            chunk_g=(
                None if self.chunk_g is None else self.chunk_g.detach().clone()
            ),
            chunk_size=self.chunk_size,
            use_fla_reference=self.use_fla_reference,
        )


@dataclass(frozen=True)
class PICConvTransition:
    """Rolling convolution-state transition for one PIC segment.

    vLLM's causal convolution state is the last ``state_length`` raw mixed
    inputs.  Unlike the GDN recurrent state, it does not need the convolution
    weights to advance: applying a segment is a rolling-window operation.
    Keeping the raw inputs lets a partial reuse range produce the conv state at
    the reuse-body end instead of restoring the segment's final state.
    """

    inputs: "torch.Tensor"
    state_length: int

    def __post_init__(self) -> None:
        if self.inputs.ndim != 2:
            raise ValueError("PIC conv transition expects [tokens, channels]")
        if self.state_length < 0:
            raise ValueError("PIC conv transition state length must be non-negative")

    def apply(self, state: "torch.Tensor") -> "torch.Tensor":
        if state.device != self.inputs.device:
            raise ValueError("PIC conv transition state device mismatch")
        if state.ndim != 2:
            raise ValueError("PIC conv transition state must be rank-2")
        inputs = (
            self.inputs
            if state.dtype == self.inputs.dtype
            else self.inputs.to(dtype=state.dtype)
        )
        channels = inputs.shape[1]
        if self.state_length == 0:
            return state[..., :0].clone()
        if tuple(state.shape) == (self.state_length, channels):
            history = torch.cat((state, inputs), dim=0)
            return history[-self.state_length :].contiguous()
        if tuple(state.shape) == (channels, self.state_length):
            history = torch.cat((state.transpose(0, 1), inputs), dim=0)
            return history[-self.state_length :].transpose(0, 1).contiguous()
        raise ValueError(
            "PIC conv transition state shape mismatch: "
            f"expected={(self.state_length, channels)} or "
            f"{(channels, self.state_length)}, actual={tuple(state.shape)}"
        )

    def compose(self, following: "PICConvTransition") -> "PICConvTransition":
        if self.state_length != following.state_length:
            raise ValueError("PIC conv transitions have different state lengths")
        if self.inputs.shape[1] != following.inputs.shape[1]:
            raise ValueError("PIC conv transitions have different channel counts")
        return PICConvTransition(
            inputs=torch.cat((self.inputs, following.inputs), dim=0),
            state_length=self.state_length,
        )

    def slice(self, local_start: int, local_end: int) -> "PICConvTransition":
        if not (0 <= local_start < local_end <= self.inputs.shape[0]):
            raise ValueError("PIC conv transition slice is outside the range")
        return PICConvTransition(
            inputs=self.inputs[local_start:local_end],
            state_length=self.state_length,
        )

    def storage_tensors(self) -> tuple["torch.Tensor", ...]:
        return (self.inputs,)

    def detached_clone(self) -> "PICConvTransition":
        return PICConvTransition(
            inputs=self.inputs.detach().clone(),
            state_length=self.state_length,
        )


PICStateTransition = (
    PICAffineTransition | PICLinearTransition | PICGDNTransition
)


@dataclass(frozen=True)
class PICTransitionOperator:
    """Composable transition for recurrent tensors in one PIC segment."""

    transitions: tuple[PICStateTransition, ...]
    conv_transitions: tuple[PICConvTransition, ...] = ()
    token_start: int = 0
    token_end: int = 0

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("PIC transition operator needs at least one state")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("PIC transition operator range is invalid")

    @property
    def zero_start_end_state(self) -> tuple["torch.Tensor", ...]:
        return tuple(item.zero_state for item in self.transitions)

    def apply(self, states: Sequence["torch.Tensor"]) -> tuple["torch.Tensor", ...]:
        if len(states) != len(self.transitions):
            raise ValueError(
                "PIC transition state count mismatch: "
                f"expected={len(self.transitions)}, actual={len(states)}"
            )
        return tuple(
            transition.apply(state)
            for transition, state in zip(self.transitions, states)
        )

    def apply_conv(
        self, states: Sequence["torch.Tensor"]
    ) -> tuple["torch.Tensor", ...]:
        if len(states) != len(self.conv_transitions):
            raise ValueError(
                "PIC conv transition state count mismatch: "
                f"expected={len(self.conv_transitions)}, actual={len(states)}"
            )
        return tuple(
            transition.apply(state)
            for transition, state in zip(self.conv_transitions, states)
        )

    def compose(self, following: "PICTransitionOperator") -> "PICTransitionOperator":
        """Compose this segment followed by ``following``."""
        if len(self.transitions) != len(following.transitions):
            raise ValueError("PIC transition operators have different state counts")
        if len(self.conv_transitions) != len(following.conv_transitions):
            raise ValueError("PIC conv transition operators have different state counts")
        composed: list[PICStateTransition] = []
        for current, next_transition in zip(
            self.transitions, following.transitions
        ):
            if type(current) is not type(next_transition):
                raise ValueError("PIC transition operators have different state kinds")
            composed.append(current.compose(next_transition))  # type: ignore[arg-type]
        return PICTransitionOperator(
            transitions=tuple(composed),
            conv_transitions=tuple(
                current.compose(next_transition)
                for current, next_transition in zip(
                    self.conv_transitions, following.conv_transitions
                )
            ),
            token_start=self.token_start,
            token_end=following.token_end,
        )

    def slice(self, local_start: int, local_end: int) -> "PICTransitionOperator":
        """Return the operator for a local sub-range of this segment."""
        length = self.token_end - self.token_start
        if not (0 <= local_start < local_end <= length):
            raise ValueError("PIC transition operator slice is outside the range")
        sliced: list[PICStateTransition] = []
        for transition in self.transitions:
            if isinstance(transition, PICGDNTransition):
                sliced.append(transition.slice(local_start, local_end))
            elif local_start == 0 and local_end == length:
                sliced.append(transition)
            else:
                raise ValueError(
                    "PIC transition slicing is unsupported for this state kind"
                )
        sliced_conv = tuple(
            transition.slice(local_start, local_end)
            for transition in self.conv_transitions
        )
        return PICTransitionOperator(
            transitions=tuple(sliced),
            conv_transitions=sliced_conv,
            token_start=self.token_start + local_start,
            token_end=self.token_start + local_end,
        )

    def storage_tensors(self) -> tuple["torch.Tensor", ...]:
        """Flatten operator tensors for a physical snapshot allocation."""
        tensors: list["torch.Tensor"] = []
        for transition in self.transitions:
            first = (
                transition.decay
                if isinstance(transition, PICAffineTransition)
                else (
                    transition.matrix
                    if isinstance(transition, PICLinearTransition)
                    else transition.key
                )
            )
            if isinstance(transition, PICGDNTransition):
                tensors.extend(
                    (
                        first,
                        transition.value,
                        transition.log_decay,
                        transition.beta,
                        transition.zero_state,
                    )
                )
            else:
                tensors.extend((first, transition.zero_state))
        for transition in self.conv_transitions:
            tensors.extend(transition.storage_tensors())
        return tuple(tensors)

    def detached_clone(self) -> "PICTransitionOperator":
        return PICTransitionOperator(
            transitions=tuple(item.detached_clone() for item in self.transitions),
            conv_transitions=tuple(
                item.detached_clone() for item in self.conv_transitions
            ),
            token_start=self.token_start,
            token_end=self.token_end,
        )

    @classmethod
    def from_step_transitions(
        cls,
        steps: Sequence[Sequence[PICStateTransition]],
        *,
        token_start: int,
        token_end: int,
    ) -> "PICTransitionOperator":
        """Compose per-token affine updates into one segment operator."""
        if not steps:
            raise ValueError("PIC transition capture needs at least one step")
        state_count = len(steps[0])
        if state_count == 0 or any(len(step) != state_count for step in steps):
            raise ValueError("PIC transition steps have inconsistent state counts")
        operator = cls(
            transitions=tuple(item.detached_clone() for item in steps[0]),
            token_start=token_start,
            token_end=token_end,
        )
        for step in steps[1:]:
            operator = operator.compose(
                cls(
                    transitions=tuple(item.detached_clone() for item in step),
                    token_start=token_start,
                    token_end=token_end,
                )
            )
        return operator


@dataclass(frozen=True)
class PICStateLayout:
    """Per-KV-group state layout used by the PIC backend."""

    groups: tuple[tuple[PICStateSpec, ...], ...]

    @property
    def has_state(self) -> bool:
        return any(self.groups)

    @property
    def recurrent_states(self) -> tuple[PICStateSpec, ...]:
        return tuple(
            state
            for group in self.groups
            for state in group
            if state.kind == PICHandleKind.RECURRENT_STATE
        )

    @property
    def conv_tails(self) -> tuple[PICStateSpec, ...]:
        return tuple(
            state
            for group in self.groups
            for state in group
            if state.kind == PICHandleKind.CONV_TAIL
        )

    @classmethod
    def from_kv_cache_config(cls, kv_cache_config: "KVCacheConfig") -> "PICStateLayout":
        """Build state descriptors from vLLM's hybrid KV cache config.

        The mapping follows vLLM's Mamba state ordering:

        * ``LINEAR`` has one temporal/recurrent tensor;
        * ``MAMBA1`` and ``MAMBA2`` have ``(conv, temporal)``;
        * ``SHORT_CONV`` has one convolutional tensor;
        * ``GDN_ATTN`` has ``(conv, temporal)``.

        Unknown/custom backends are rejected instead of guessing, because an
        incorrect state order would silently corrupt a reused request.
        """
        # Local import avoids making vllm.v1.pic import the full cache
        # interface in processes that only need token segmentation.
        from vllm.v1.kv_cache_interface import MambaSpec

        groups: list[tuple[PICStateSpec, ...]] = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if not isinstance(spec, MambaSpec):
                groups.append(())
                continue

            backend_name = spec.mamba_type.name
            if backend_name == "LINEAR":
                state_kinds = (PICHandleKind.RECURRENT_STATE,)
            elif backend_name in {"MAMBA1", "MAMBA2", "GDN_ATTN"}:
                state_kinds = (
                    PICHandleKind.CONV_TAIL,
                    PICHandleKind.RECURRENT_STATE,
                )
            elif backend_name == "SHORT_CONV":
                state_kinds = (PICHandleKind.CONV_TAIL,)
            else:
                raise ValueError(
                    "PIC does not know the state ordering for Mamba backend "
                    f"{backend_name!r}; add an explicit mapping before enabling it."
                )

            if len(spec.shapes) != len(state_kinds) or len(spec.dtypes) != len(
                state_kinds
            ):
                raise ValueError(
                    "MambaSpec state metadata does not match the registered PIC "
                    f"layout for backend {backend_name}: "
                    f"shapes={len(spec.shapes)}, dtypes={len(spec.dtypes)}, "
                    f"kinds={len(state_kinds)}"
                )

            groups.append(
                tuple(
                    PICStateSpec(
                        group_id=group_id,
                        state_index=state_index,
                        kind=kind,
                        shape=tuple(int(dim) for dim in shape),
                        dtype=dtype,
                    )
                    for state_index, (kind, shape, dtype) in enumerate(
                        zip(state_kinds, spec.shapes, spec.dtypes)
                    )
                )
            )

        return cls(groups=tuple(groups))
