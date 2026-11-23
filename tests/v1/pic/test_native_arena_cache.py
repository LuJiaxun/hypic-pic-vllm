# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the pinned arena cache: parsing, eviction, and the event that blocked it.

Two defects here cost a full test round each. The byte budget was parsed with ``int()``,
so a value like "512M" raised inside the import path and surfaced three imports later as
"import disabled" - which reads as a code regression. And eviction skipped any entry whose
upload event was not ``None``, but that event only got cleared when the same key was rented
again, so every entry that had uploaded once became permanently unevictable and the cache
grew past its budget.
"""

from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch

from vllm.v1.pic.mooncake_kv import MooncakeNativeKVObjectStore, _parse_byte_size


class _Done:
    """A CUDA event stand-in that has completed."""

    def query(self) -> bool:
        return True


class _Pending:
    """A CUDA event stand-in that is still in flight."""

    def query(self) -> bool:
        return False


def _store(*, budget: int) -> MooncakeNativeKVObjectStore:
    store = object.__new__(MooncakeNativeKVObjectStore)
    store._native_arenas = {}
    store._native_arena_cache_bytes = budget
    store._native_arena_evictions = 0
    return store


def _entry(nbytes: int, *, pending=None, busy: bool = False):
    return {
        "arena": torch.empty(nbytes, dtype=torch.uint8),
        "busy": busy,
        "pending": pending,
        "lock": Lock(),
    }


def test_size_suffixes_are_accepted() -> None:
    assert _parse_byte_size("512M", name="X") == 512 * 1024 * 1024
    assert _parse_byte_size("2G", name="X") == 2 * 1024 * 1024 * 1024
    assert _parse_byte_size("1024K", name="X") == 1024 * 1024
    assert _parse_byte_size("123", name="X") == 123
    assert _parse_byte_size("1.5G", name="X") == int(1.5 * 1024 * 1024 * 1024)


def test_a_bad_value_fails_loudly() -> None:
    with pytest.raises(ValueError, match="HYPIC_TEST_BYTES"):
        _parse_byte_size("512MB!", name="HYPIC_TEST_BYTES")
    with pytest.raises(ValueError, match="positive"):
        _parse_byte_size("0", name="HYPIC_TEST_BYTES")


def test_a_completed_upload_does_not_protect_an_arena() -> None:
    store = _store(budget=100)
    store._native_arenas["a"] = _entry(60, pending=_Done())

    store._evict_native_arenas(60)

    assert "a" not in store._native_arenas, "a finished upload must be evictable"
    assert store._native_arena_evictions == 1


def test_an_inflight_upload_still_protects_the_arena() -> None:
    store = _store(budget=100)
    store._native_arenas["a"] = _entry(60, pending=_Pending())

    store._evict_native_arenas(60)

    assert "a" in store._native_arenas, "the copy may still be reading the buffer"


def test_a_busy_arena_is_never_evicted() -> None:
    store = _store(budget=100)
    store._native_arenas["a"] = _entry(60, busy=True)

    store._evict_native_arenas(60)

    assert "a" in store._native_arenas


def test_eviction_stops_once_the_budget_fits() -> None:
    store = _store(budget=200)
    store._native_arenas["a"] = _entry(80, pending=_Done())
    store._native_arenas["b"] = _entry(80, pending=_Done())

    store._evict_native_arenas(60)

    # 160 + 60 > 200, so one eviction suffices; the second arena stays.
    assert len(store._native_arenas) == 1
    assert store._native_arena_evictions == 1
