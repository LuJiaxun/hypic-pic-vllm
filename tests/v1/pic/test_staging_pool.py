# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the pinned staging pool used for transition transfers.

Pinning and registering a host arena cost more than the copy that uses it, and an
arena sized to one object cannot serve the next object of a different size, so a run
over many distinct objects used to pay both costs once per object. The pool keeps a
few arenas at the largest size seen and hands them out one at a time.

These tests pin the invariants that make that safe: an arena is never handed to two
owners at once, a bigger object grows the pool instead of adding a size, and a
release always returns the arena.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("torch")

import torch

from vllm.v1.pic.mooncake_kv import _PinnedStagingPool


def _pool(slots: int = 2, granularity: int = 1 << 20) -> _PinnedStagingPool:
    registered: list[tuple[int, int]] = []
    pool = _PinnedStagingPool(
        slots=slots,
        granularity=granularity,
        register=lambda address, nbytes, role: registered.append((address, nbytes)),
        unregister=lambda address: None,
    )
    pool.registered = registered  # type: ignore[attr-defined]
    return pool


def test_one_arena_serves_every_object_of_the_same_size() -> None:
    pool = _pool()
    first = pool.acquire(4 << 20, pin=False)
    pool.release(first)
    second = pool.acquire(4 << 20, pin=False)

    assert second.numel() == first.numel()
    assert pool.allocations == 1, "a repeat of the same size must not allocate again"


def test_a_larger_object_grows_the_arena_instead_of_adding_a_size() -> None:
    pool = _pool()
    small = pool.acquire(2 << 20, pin=False)
    pool.release(small)
    big = pool.acquire(6 << 20, pin=False)

    assert int(big.numel()) >= 6 << 20
    pool.release(big)

    # The grown capacity now serves the small object too: one size for the process.
    again = pool.acquire(2 << 20, pin=False)
    assert again.numel() == big.numel()
    assert pool.allocations == 2


def test_capacity_is_rounded_up_so_sizes_do_not_multiply() -> None:
    pool = _pool(granularity=1 << 20)
    arena = pool.acquire((3 << 20) + 12345, pin=False)
    assert int(arena.numel()) == 4 << 20
    pool.release(arena)


def test_two_owners_never_share_one_arena() -> None:
    pool = _pool(slots=2)
    first = pool.acquire(1 << 20, pin=False)
    second = pool.acquire(1 << 20, pin=False)

    assert first.data_ptr() != second.data_ptr()
    pool.release(first)
    pool.release(second)


def test_a_third_owner_waits_for_a_release() -> None:
    pool = _pool(slots=1)
    held = pool.acquire(1 << 20, pin=False)
    acquired: list[torch.Tensor] = []

    def worker() -> None:
        acquired.append(pool.acquire(1 << 20, pin=False))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.05)
    assert acquired == [], "the pool must not exceed its slot count"

    pool.release(held)
    thread.join(timeout=2)
    assert acquired, "the waiter must get the released arena"
    pool.release(acquired[0])


def test_registration_happens_once_per_arena() -> None:
    pool = _pool()
    for _ in range(3):
        pool.release(pool.acquire(2 << 20, pin=False))

    registered = pool.registered  # type: ignore[attr-defined]
    assert len(registered) == 1, "one arena must be registered once, not per use"
    assert registered[0][1] == 2 << 20
