# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opaque PIC handle lifecycle primitives.

Stage 1 deliberately keeps the handle registry independent from vLLM's normal
prefix-cache block pool.  A handle is a stable control-plane identifier for a
future physical PIC allocation; it is not itself a CUDA pointer or a vLLM
block ID.  Later stages may attach a backend payload (for example, a set of
non-contiguous GPU slots) without changing request or scheduler interfaces.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class PICHandleKind(str, Enum):
    """Kinds of state that can be reused by a PIC segment."""

    FULL_KV = "full_kv"
    RECURRENT_STATE = "recurrent_state"
    TRANSITION_STATE = "transition_state"
    CONV_TAIL = "conv_tail"


@dataclass
class PICHandle:
    """An opaque reference to one materialized PIC state object."""

    handle_id: int
    kind: PICHandleKind
    group_id: int = -1
    token_start: int = 0
    token_end: int = 0
    slot_ids: tuple[int, ...] = ()
    ref_count: int = 1
    payload: Any = None


class PICHandlePool:
    """Reference-counted registry for PIC state handles.

    The pool owns metadata only in Stage 1.  ``payload`` is intentionally
    opaque so the normal vLLM path never needs to know whether a later backend
    stores a handle in a block pool, a tensor pool, or an external device
    allocator.
    """

    def __init__(self, max_handles: int | None = None) -> None:
        self.max_handles = max_handles
        self._next_handle_id = 1
        self._handles: dict[int, PICHandle] = {}

    def __len__(self) -> int:
        return len(self._handles)

    def allocate(
        self,
        kind: PICHandleKind,
        *,
        group_id: int = -1,
        token_start: int = 0,
        token_end: int = 0,
        slot_ids: tuple[int, ...] = (),
        payload: Any = None,
    ) -> PICHandle:
        if self.max_handles is not None and len(self._handles) >= self.max_handles:
            raise MemoryError("PIC handle pool is full")
        handle = PICHandle(
            handle_id=self._next_handle_id,
            kind=kind,
            group_id=group_id,
            token_start=token_start,
            token_end=token_end,
            slot_ids=tuple(int(slot) for slot in slot_ids),
            payload=payload,
        )
        self._next_handle_id += 1
        self._handles[handle.handle_id] = handle
        return handle

    def get(self, handle_id: int) -> PICHandle | None:
        return self._handles.get(int(handle_id))

    def retain(self, handle_id: int) -> PICHandle:
        handle = self._require(handle_id)
        handle.ref_count += 1
        return handle

    def release(self, handle_id: int) -> PICHandle | None:
        handle = self._require(handle_id)
        handle.ref_count -= 1
        if handle.ref_count < 0:
            raise RuntimeError(f"PIC handle {handle_id} released too many times")
        if handle.ref_count == 0:
            return self._handles.pop(handle.handle_id)
        return None

    def release_many(self, handle_ids: Iterable[int]) -> list[PICHandle]:
        released: list[PICHandle] = []
        for handle_id in handle_ids:
            handle = self.release(handle_id)
            if handle is not None:
                released.append(handle)
        return released

    def clear(self) -> list[PICHandle]:
        released = list(self._handles.values())
        self._handles.clear()
        return released

    def _require(self, handle_id: int) -> PICHandle:
        handle = self.get(handle_id)
        if handle is None:
            raise KeyError(f"unknown PIC handle {handle_id}")
        return handle
