# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the publish guarantee.

A successful put is not a promise of visibility: a burst of twenty publishes was accepted
with status zero and only one could be read back, so the consumer saw a missing manifest
and silently recomputed the whole prompt. The publish now reads the manifest back and
repeats the put a bounded number of times, so a publish that returns has landed; strict
mode turns an unverifiable publish into an error instead of a warning.

The manifest is the pointer to the layer blobs, so verifying it catches the failure that
matters without transferring the object twice.

The warning assertions replace the module logger rather than reading ``caplog``: vLLM's
logger does not propagate to the root logger, so ``caplog.text`` stays empty even when the
line is emitted.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from vllm.v1.pic import mooncake_kv
from vllm.v1.pic.mooncake_kv import MooncakeNativeKVObjectStore


class _Store:
    """Store whose visibility can be delayed by a number of reads."""

    def __init__(self, *, hidden_reads: int = 0) -> None:
        self.values: dict[str, bytes] = {}
        self.hidden_reads = hidden_reads
        self.gets = 0
        self.puts = 0

    def put(self, key: str, value) -> int:
        self.puts += 1
        self.values[key] = bytes(value)
        return 0

    def get(self, key: str):
        self.gets += 1
        if self.hidden_reads > 0:
            self.hidden_reads -= 1
            return b""
        return self.values.get(key, b"")

    def remove(self, key: str, force: bool = False) -> int:
        self.values.pop(key, None)
        return 0


class _Logger:
    """Collect warnings without touching the logging configuration."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args) -> None:
        self.warnings.append(message % args if args else message)

    def debug(self, *_args, **_kwargs) -> None:
        pass

    def info(self, *_args, **_kwargs) -> None:
        pass

    def text(self) -> str:
        return "\n".join(self.warnings)


def _objects(store) -> MooncakeNativeKVObjectStore:
    objects = object.__new__(MooncakeNativeKVObjectStore)
    objects._owned_native_keys = set()
    objects.store = store
    return objects


def test_a_visible_publish_returns_after_one_attempt(monkeypatch) -> None:
    store = _Store()

    _objects(store)._publish_manifest_until_visible("k/manifest", b"payload")

    assert store.puts == 1
    assert store.gets == 1


def test_a_lagging_publish_is_repeated_until_visible(monkeypatch) -> None:
    store = _Store(hidden_reads=1)

    _objects(store)._publish_manifest_until_visible("k/manifest", b"payload")

    assert store.puts == 2, "the second put is what made it visible"
    assert store.values["k/manifest"] == b"payload"


def test_an_unverifiable_publish_warns_by_default(monkeypatch) -> None:
    store = _Store(hidden_reads=99)
    logger = _Logger()
    monkeypatch.setattr(mooncake_kv, "logger", logger)

    _objects(store)._publish_manifest_until_visible("k/manifest", b"payload")

    assert "not readable after 2 attempt(s)" in logger.text()
    assert store.puts == 2
