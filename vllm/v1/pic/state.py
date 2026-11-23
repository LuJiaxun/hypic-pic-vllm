# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PIC state layout for hybrid Mamba/GDN attention.

vLLM already owns the authoritative state shapes and dtypes in ``MambaSpec``.
This module only translates that information into a PIC-specific descriptor;
it does not allocate or copy model state yet.  Keeping this translation
centralized is important because Mamba, short-conv, and GDN use different
numbers and meanings of state tensors.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING


import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm.v1.pic.env import pic_debug_enabled
from vllm.v1.pic.handles import PICHandleKind

if TYPE_CHECKING:
    import torch

    from vllm.v1.kv_cache_interface import KVCacheConfig


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
        self.state_block_ids = tuple(
            tuple(int(block) for block in group) for group in block_ids
        )
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
        if (
            state.dtype != self.zero_state.dtype
            or state.device != self.zero_state.device
        ):
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


# Matrices recovered from captured transitions, keyed by object identity. A transition is a
# per-segment object held for the lifetime of the process, so identity is a stable key here; the
# bound keeps a long run from accumulating operators that are no longer referenced.
_COMPACT_CACHE: dict[int, "PICLinearTransition"] = {}
# The build runs through `apply`, which is the very method that asks for a matrix, so a
# transition being built must be recognisable: while it is, the request for a matrix answers
# None and the kernel recurrence runs underneath instead. This is control flow, not a
# judgement about the result.
_COMPACT_BUILDING: set[tuple[int, int, int]] = set()
_COMPACT_CACHE_LIMIT = 256
# Below this many tokens the one-off build is not repaid: the recurrence for a short span is cheap,
# while the build costs a full pass over the same span.
_COMPACT_MIN_TOKENS = 512


def _transition_identity(transition: "PICGDNTransition") -> tuple[int, int, int]:
    """A key that survives re-importing an operator.

    ``id()`` is stable only while the object lives, and a reader builds its operator from what the
    store returned - a new object every time - so an id-based cache would miss on every request. The
    tensors are what the matrix is made of, so they identify it: same storage, same span.
    """
    key = transition.key
    return (int(key.data_ptr()), int(transition.value.data_ptr()), int(key.shape[0]))


def _compact_entry(transition: "PICGDNTransition") -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the matrix form of ``transition``, building it at most once.

    This is where the recurrence becomes a single product. Nothing here compares the result against
    the recurrence: PIC reuse is approximate by construction, every request would pay for the
    comparison, and it would answer a question the production path cannot act on. Accuracy is
    measured by the test scripts, which can afford to run both paths.
    """
    key = _transition_identity(transition)
    cached = _COMPACT_CACHE.get(key)
    if cached is not None:
        return cached
    # `compact()` derives the matrix by applying this operator to an identity state, which is the
    # same call that just asked for a matrix. Answering None here is what stops that from recurring
    # into itself and lets the kernel do the one pass the build needs.
    if key in _COMPACT_BUILDING:
        return None
    _COMPACT_BUILDING.add(key)
    try:
        compact = transition.compact()
    finally:
        _COMPACT_BUILDING.discard(key)
    if compact is None:
        return None
    # A matrix that cannot be multiplied - non-finite entries, or a state that is not square - would
    # fail deep inside a request; keep that one guard and nothing else.
    if not bool(torch.isfinite(compact.matrix).all()):
        if pic_debug_enabled():
            logger.warning(
                "PIC compact skipped: the matrix has non-finite entries tokens=%d",
                int(transition.key.shape[0]),
            )
        return None
    entry = (compact.matrix, compact.zero_state)
    if len(_COMPACT_CACHE) >= _COMPACT_CACHE_LIMIT:
        _COMPACT_CACHE.clear()
    _COMPACT_CACHE[key] = entry
    if pic_debug_enabled():
        logger.warning(
            "PIC compact built tokens=%d shape=%s",
            int(transition.key.shape[0]),
            tuple(compact.matrix.shape),
        )
    return entry


def _apply_compact(entry: tuple[torch.Tensor, torch.Tensor], state: torch.Tensor):
    """Apply a stored matrix to an incoming state.

    The state is multiplied on the left by the matrix, which is the convention the kernel's state
    update follows: ``h_new = h_old @ T + C`` with ``h`` shaped ``[heads, value, key]``. The
    arithmetic stays in the dtype the kernel produced, because that is the reference this path is
    approximating and a wider product would move away from it rather than towards it.
    """
    matrix, zero_state = entry
    work = state.to(matrix.dtype)
    return work.matmul(matrix) + zero_state


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
        if (
            state.dtype != self.zero_state.dtype
            or state.device != self.zero_state.device
        ):
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
            raise ValueError("PIC GDN value heads must be divisible by key heads")
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
                f"expected={expected_state_shape}, "
                f"actual={tuple(self.zero_state.shape)}"
            )
        chunked = (self.chunk_w, self.chunk_u, self.chunk_g)
        if any(item is not None for item in chunked):
            if not all(item is not None for item in chunked):
                raise ValueError("PIC GDN chunked reference must provide w, u and g")
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

    def retimed(self) -> "PICGDNTransition":
        """Return this transition with chunk metadata rebuilt for its own window.

        The chunk tensors that arrive with a capture were computed on a grid anchored at the
        segment start. Trimming tokensequence off the head therefore leaves a boundary chunk whose
        stored metadata belongs to a different window, and the cumulative operator derived from it
        would describe the wrong span. Rebuilding the grid on the trimmed sequence keeps the decay
        relative to the window, which is what the transition of that window means.
        """
        from vllm.v1.pic.gdn_transition import _build_fla_chunk_reference

        reference = _build_fla_chunk_reference(
            self.key, self.value, self.log_decay, self.beta
        )
        if reference is None:
            return self
        chunk_w, chunk_u, chunk_g, chunk_size = reference
        return PICGDNTransition(
            key=self.key,
            value=self.value,
            log_decay=self.log_decay,
            beta=self.beta,
            zero_state=self.zero_state,
            chunk_w=chunk_w,
            chunk_u=chunk_u,
            chunk_g=chunk_g,
            chunk_size=chunk_size or self.chunk_size,
            backend=self.backend,
        )

    def compact(self) -> "PICLinearTransition | None":
        """Materialize the segment-cumulative operator for wire storage.

        The GDN state update is linear in the incoming state.  When the
        value dimension equals the key dimension, applying the captured
        operator to an identity state recovers its right-side transition
        matrix; subtracting ``zero_state`` removes the additive term.  The
        resulting operator is independent of segment length and can be
        stored as ``PICLinearTransition``.  Other shapes keep the legacy
        token-level representation and are handled by the old wire format.
        """
        if self.zero_state.ndim != 3:
            return None
        key_dim = int(self.key.shape[-1])
        if int(self.zero_state.shape[1]) != key_dim:
            return None
        basis = torch.eye(
            key_dim,
            dtype=self.zero_state.dtype,
            device=self.zero_state.device,
        ).unsqueeze(0).expand(self.zero_state.shape[0], -1, -1).contiguous()
        if tuple(basis.shape) != tuple(self.zero_state.shape):
            return None
        with torch.no_grad():
            transformed = self.apply(basis)
        matrix = (transformed - self.zero_state).contiguous()
        return PICLinearTransition(matrix=matrix, zero_state=self.zero_state)

    def apply(
        self, state: "torch.Tensor", *, validate: bool = False
    ) -> "torch.Tensor":
        if tuple(state.shape) != tuple(self.zero_state.shape):
            raise ValueError(
                "PIC GDN transition state shape mismatch: "
                f"expected={tuple(self.zero_state.shape)}, "
                f"actual={tuple(state.shape)}"
            )
        if state.device != self.zero_state.device:
            raise ValueError("PIC GDN transition state device mismatch")

        # A long span is worth serving from a matrix; a short one is not, and the build itself
        # runs through this method, which the building set keeps out of this branch.
        if int(self.key.shape[0]) >= _COMPACT_MIN_TOKENS:
            entry = _compact_entry(self)
            if entry is not None:
                return _apply_compact(entry, state)

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
                validate=validate,
            )

        # A recorded CUDA backend is applied above, with the model's own GDN kernel. What stood here
        # was a Torch recurrence, and it did not agree with that kernel: the self-test measured a
        # relative error near eleven with a cosine near zero, which is a different operator rather
        # than rounding. It is deleted instead of kept as a quiet alternative. Reaching this line
        # means the operator has no backend to apply with - a programming error, not a request the
        # runtime should degrade around - so it fails here, where the stack says why.
        raise ValueError(
            "PIC GDN transition requires a recorded GDN prefill backend "
            f"(got {self.backend!r}); the Torch recurrence was removed"
        )

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

            chunk_reference = _build_fla_chunk_reference(key, value, log_decay, beta)
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
        )

    def detached_clone(self) -> "PICGDNTransition":
        return PICGDNTransition(
            key=self.key.detach().clone(),
            value=self.value.detach().clone(),
            log_decay=self.log_decay.detach().clone(),
            beta=self.beta.detach().clone(),
            zero_state=self.zero_state.detach().clone(),
            backend=self.backend,
            chunk_w=(None if self.chunk_w is None else self.chunk_w.detach().clone()),
            chunk_u=(None if self.chunk_u is None else self.chunk_u.detach().clone()),
            chunk_g=(None if self.chunk_g is None else self.chunk_g.detach().clone()),
            chunk_size=self.chunk_size,
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
    source_token_count: int | None = None

    def __post_init__(self) -> None:
        if self.inputs.ndim != 2:
            raise ValueError("PIC conv transition expects [tokens, channels]")
        if self.state_length < 0:
            raise ValueError("PIC conv transition state length must be non-negative")
        if self.source_token_count is not None and self.source_token_count <= 0:
            raise ValueError("PIC conv source token count must be positive")
        if (
            self.source_token_count is not None
            and self.inputs.shape[0] > self.source_token_count
        ):
            raise ValueError("PIC compact conv tail exceeds source token count")

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

    def compact(self, source_token_count: int) -> "PICConvTransition":
        """Keep only the tail needed for complete-segment reuse."""
        if source_token_count <= 0:
            raise ValueError("PIC conv source token count must be positive")
        inputs = (
            self.inputs[-self.state_length :]
            if self.state_length > 0
            else self.inputs[:0]
        )
        return PICConvTransition(
            inputs=inputs.contiguous(),
            state_length=self.state_length,
            source_token_count=source_token_count,
        )

    def compose(self, following: "PICConvTransition") -> "PICConvTransition":
        if self.source_token_count is not None or following.source_token_count is not None:
            raise ValueError("PIC compact conv tails cannot be composed")
        if self.state_length != following.state_length:
            raise ValueError("PIC conv transitions have different state lengths")
        if self.inputs.shape[1] != following.inputs.shape[1]:
            raise ValueError("PIC conv transitions have different channel counts")
        return PICConvTransition(
            inputs=torch.cat((self.inputs, following.inputs), dim=0),
            state_length=self.state_length,
        )

    def slice(self, local_start: int, local_end: int) -> "PICConvTransition":
        source_length = (
            self.inputs.shape[0]
            if self.source_token_count is None
            else self.source_token_count
        )
        if not (0 <= local_start < local_end <= source_length):
            raise ValueError("PIC conv transition slice is outside the range")
        if self.source_token_count is not None:
            if local_start != 0 or local_end != self.source_token_count:
                raise ValueError("PIC compact conv tail cannot be partially sliced")
            return self
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
            source_token_count=self.source_token_count,
        )


PICStateTransition = PICAffineTransition | PICLinearTransition | PICGDNTransition


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
        # A segment carries one linear transition per GDN layer, and applying them one at
        # a time costs a matmul launch and a Python round trip per layer - eighteen of
        # each for this model, which a trace showed at roughly ten milliseconds. When
        # every layer has the same square matrix and state shape they can be applied as
        # one batched matmul, so the stack is prepared once here.
        object.__setattr__(self, "_batched", self._build_batched())

    def _build_batched(self) -> tuple[list[int], torch.Tensor, torch.Tensor] | None:
        """Collect the layers that share a matrix shape into one stack, with their positions.

        A segment normally carries one matrix per layer, so one batched product replaces eighteen
        launches. Layers that stayed token-level, or that have a different shape, are simply left out
        of the stack and applied on their own: one unusual layer must not cost the other seventeen
        their batch, which is exactly what an all-or-nothing rule did.
        """
        if not self.transitions:
            return None
        positions: list[int] = []
        matrices: list[torch.Tensor] = []
        zeros: list[torch.Tensor] = []
        shape: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        dtype = None
        device = None
        for index, item in enumerate(self.transitions):
            if type(item) is not PICLinearTransition:
                continue
            if item.matrix.ndim != 2 or item.zero_state.ndim != 2:
                continue
            if shape is None:
                shape = (tuple(item.matrix.shape), tuple(item.zero_state.shape))
                dtype = item.matrix.dtype
                device = item.matrix.device
            elif (
                tuple(item.matrix.shape) != shape[0]
                or tuple(item.zero_state.shape) != shape[1]
                or item.matrix.dtype != dtype
                or item.matrix.device != device
            ):
                continue
            positions.append(index)
            matrices.append(item.matrix)
            zeros.append(item.zero_state)
        if not positions:
            return None
        try:
            return positions, torch.stack(matrices), torch.stack(zeros)
        except RuntimeError:
            return None

    @property
    def zero_start_end_state(self) -> tuple["torch.Tensor", ...]:
        return tuple(item.zero_state for item in self.transitions)

    def apply(self, states: Sequence["torch.Tensor"]) -> tuple["torch.Tensor", ...]:
        if len(states) != len(self.transitions):
            raise ValueError(
                "PIC transition state count mismatch: "
                f"expected={len(self.transitions)}, actual={len(states)}"
            )
        batched = getattr(self, "_batched", None)
        results: list[torch.Tensor | None] = [None] * len(self.transitions)
        if batched is not None and states:
            positions, matrices, zeros = batched
            subset = [states[index] for index in positions]
            first = subset[0]
            shape = tuple(first.shape)
            if first.ndim == 2 and all(
                tuple(state.shape) == shape
                and state.dtype == first.dtype
                and state.device == first.device
                for state in subset
            ):
                stacked = torch.stack(subset, dim=0)
                applied = torch.bmm(stacked, matrices) + zeros
                for slot, index in enumerate(positions):
                    results[index] = applied[slot]
        for index, transition in enumerate(self.transitions):
            if results[index] is None:
                # A token-level layer replays its own recurrence; this is the path every layer used
                # before matrices existed, so a layer that could not be compacted costs no more than
                # it always did.
                results[index] = transition.apply(states[index])
        return tuple(results)

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
            raise ValueError(
                "PIC conv transition operators have different state counts"
            )
        composed: list[PICStateTransition] = []
        for current, next_transition in zip(self.transitions, following.transitions):
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
                # A compact transition only describes the whole range it was built
                # for.  Report it as a controlled worker limitation so the request
                # falls back to recomputation instead of taking the engine down.
                from vllm.v1.pic.worker_plan import PICWorkerUnsupported

                raise PICWorkerUnsupported(
                    "PIC transition slicing is unsupported for this state kind: "
                    f"requested=[{local_start},{local_end}) length={length} "
                    f"kind={type(transition).__name__}"
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

    def compact_for_full_reuse(self) -> "PICTransitionOperator | None":
        """Build the length-independent operator for a complete segment.

        Partial reuse must continue to use ``self`` because a compact linear
        transition cannot be sliced at an arbitrary token boundary.
        """
        token_count = self.token_end - self.token_start
        if token_count <= 0:
            return None

        compacted: list[PICStateTransition] = []
        needs_compaction = False
        for transition in self.transitions:
            if isinstance(transition, PICGDNTransition):
                compact = transition.compact()
                if compact is None:
                    return None
                compacted.append(compact)
                needs_compaction = True
            elif isinstance(transition, (PICAffineTransition, PICLinearTransition)):
                compacted.append(transition)
            else:
                return None

        compact_conv = tuple(
            transition.compact(token_count) for transition in self.conv_transitions
        )
        needs_compaction = needs_compaction or any(
            transition.source_token_count is None
            for transition in self.conv_transitions
        )
        if not needs_compaction:
            return self
        return PICTransitionOperator(
            transitions=tuple(compacted),
            conv_transitions=compact_conv,
            token_start=self.token_start,
            token_end=self.token_end,
        )

    def storage_tensors(self) -> tuple["torch.Tensor", ...]:
        """Flatten operator tensors for a physical snapshot allocation."""
        tensors: list[torch.Tensor] = []
        for transition in self.transitions:
            if isinstance(transition, PICAffineTransition):
                first = transition.decay
            elif isinstance(transition, PICLinearTransition):
                first = transition.matrix
            else:
                first = transition.key
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
        for conv_transition in self.conv_transitions:
            tensors.extend(conv_transition.storage_tensors())
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
            state_kinds: tuple[PICHandleKind, ...]
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
