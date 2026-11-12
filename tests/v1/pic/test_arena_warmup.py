# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the pinned arena warmup.

The first pin of a given size costs far more than later ones because the pinned
allocator has nothing to hand back yet; measured anywhere from forty to two hundred
milliseconds, and it landed on the first request of a cold process. Allocating and
dropping blocks at construction leaves them in the allocator's cache, which is what the
request path then reuses.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from vllm.v1.pic.mooncake_kv import MooncakeNativeKVObjectStore


def _store() -> MooncakeNativeKVObjectStore:
    store = object.__new__(MooncakeNativeKVObjectStore)
    store._native_arenas = {}
    store._native_arena_cache_bytes = 512 * 1024 * 1024
    store._native_arena_evictions = 0
    return store


def test_warmup_does_not_keep_the_blocks(monkeypatch) -> None:
    monkeypatch.setenv("HYPIC_PIC_KV_ARENA_BYTES", "1M")
    store = _store()

    sizes: list[int] = []
    real_empty = torch.empty

    def fake_empty(*args, **kwargs):
        if kwargs.get("pin_memory") and kwargs.get("dtype") is torch.uint8:
            nbytes = int(args[0]) if args else int(kwargs["size"])
            sizes.append(nbytes)
            return real_empty(8, dtype=torch.uint8)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fake_empty)
    store._warm_native_arenas()

    assert sizes == [1 << 20] * 4, "one block per staging slot, at the asked size"
    # Nothing is retained: the store must not hold warmup references.
    assert store._native_arenas == {}
    assert not hasattr(store, "_warm_blocks")


def test_a_bad_size_is_skipped_not_fatal(monkeypatch) -> None:
    monkeypatch.setenv("HYPIC_PIC_KV_ARENA_BYTES", "not-a-size")
    store = _store()

    store._warm_native_arenas()  # must not raise

    assert store._native_arenas == {}
