# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the per-layer publish check.

The manifest read-back catches a missing manifest, but an object is mostly its layer
blobs, and a missing one only surfaces later as a failed import. Each layer is confirmed
with ``is_exist`` while its bytes are still in hand, so a lost blob is rewritten instead
of discovered on the consumer.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from vllm.v1.pic import mooncake_kv
from vllm.v1.pic.mooncake_kv import MooncakeNativeKVObjectStore


class _Store:
    """Store that only reports a key as existing after N writes."""

    def __init__(self, *, present_after: int = 1) -> None:
        self.values: dict[str, bytes] = {}
        self.writes: dict[str, int] = {}
        self.present_after = present_after
        self.puts = 0

    def put(self, key: str, value) -> int:
        self.puts += 1
        self.writes[key] = self.writes.get(key, 0) + 1
        self.values[key] = bytes(value)
        return 0

    def is_exist(self, key: str) -> bool:
        return self.writes.get(key, 0) >= self.present_after


class _NoProbeStore(_Store):
    is_exist = None  # type: ignore[assignment]


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args) -> None:
        self.warnings.append(message % args if args else message)

    def text(self) -> str:
        return "\n".join(self.warnings)


def _objects(store) -> MooncakeNativeKVObjectStore:
    objects = object.__new__(MooncakeNativeKVObjectStore)
    objects.store = store
    return objects


def test_a_present_layer_is_written_once(monkeypatch) -> None:
    store = _Store()

    attempts = _objects(store)._put_layer_until_present("k/layers/a", b"payload")

    assert attempts == 1
    assert store.puts == 1


def test_a_lost_layer_is_rewritten(monkeypatch) -> None:
    store = _Store(present_after=2)

    attempts = _objects(store)._put_layer_until_present("k/layers/a", b"payload")

    assert attempts == 2, "the second write is what made the key exist"
    assert store.puts == 2


def test_a_layer_that_never_appears_is_reported(monkeypatch) -> None:
    store = _Store(present_after=99)
    logger = _Logger()
    monkeypatch.setattr(mooncake_kv, "logger", logger)

    attempts = _objects(store)._put_layer_until_present("k/layers/a", b"payload")

    assert attempts == 2
    assert "not present after 2 attempt(s)" in logger.text()


def test_a_store_without_a_probe_keeps_one_write(monkeypatch) -> None:
    store = _NoProbeStore(present_after=99)

    attempts = _objects(store)._put_layer_until_present("k/layers/a", b"payload")

    assert attempts == 1
    assert store.puts == 1
