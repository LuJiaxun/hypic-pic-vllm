# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the retained KV arena pool.

Dropping the warm blocks only returned them to the pinned allocator's cache, which then
may decline to serve a slightly smaller request - measured as the same forty to two
hundred millisecond pin landing on a request anyway. The pool keeps them, so a request
gets a plain memcpy and nothing is allocated on its path.

The upload reads the arena asynchronously, so a released arena goes to a pending list and
only returns to the free list once its copy has finished.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from vllm.v1.pic.mooncake_kv import MooncakeNativeKVObjectStore


class _Done:
    def query(self) -> bool:
        return True


class _Pending:
    def query(self) -> bool:
        return False


def _store() -> MooncakeNativeKVObjectStore:
    from threading import Lock

    store = object.__new__(MooncakeNativeKVObjectStore)
    store._native_arenas = {}
    store._native_arena_cache_bytes = 512 * 1024 * 1024
    store._native_arena_evictions = 0
    store._free_native_arenas = []
    store._pending_native_arenas = []
    store._native_pool_lock = Lock()
    return store


def test_a_warm_arena_is_handed_out_without_allocating() -> None:
    store = _store()
    warm = torch.empty(64, dtype=torch.uint8)
    store._free_native_arenas.append(warm)

    arena, release = store._rent_native_arena("key-1", 32)

    assert arena is warm
    release()


def test_a_released_arena_returns_only_after_its_copy_finishes() -> None:
    store = _store()
    warm = torch.empty(64, dtype=torch.uint8)
    store._free_native_arenas.append(warm)
    arena, release = store._rent_native_arena("key-1", 32)

    release(_Pending())
    assert store._free_native_arenas == [], "an in-flight copy still reads the buffer"
    assert store._take_warm_arena(32) is None

    # The next acquire reaps it once the event reports completion.
    store._pending_native_arenas = [(arena, _Done())]
    assert store._take_warm_arena(32) is arena


def test_a_finished_copy_is_recycled_on_the_next_acquire() -> None:
    store = _store()
    warm = torch.empty(64, dtype=torch.uint8)
    store._free_native_arenas.append(warm)
    arena, release = store._rent_native_arena("key-1", 32)

    release(_Done())
    assert store._take_warm_arena(32) is arena


def test_an_arena_too_small_is_not_handed_out() -> None:
    store = _store()
    store._free_native_arenas.append(torch.empty(8, dtype=torch.uint8))

    assert store._take_warm_arena(64) is None
    assert len(store._free_native_arenas) == 1
