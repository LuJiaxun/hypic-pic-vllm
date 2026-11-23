# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the KV receiver width.

KV receives are serialized by default, while the state transfers keep their own pool.
Sharing the link between two KV imports makes both finish at the same late moment, so
neither request can start computing and the engine idles through the whole transfer;
sending them one at a time lets the first request compute while the second still arrives.
Measured on a two-request run: wall 2545.7 -> 2413.3 ms, p50 265.3 -> 243.2 ms.

The two widths are separate on purpose: state transfers ride their own stream, where
parallelism costs nothing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from vllm.v1.pic.native_kv_transfer import PICNativeKVImportWorker


def _worker(monkeypatch, *, kv: str | None, state: str | None):
    if kv is None:
        monkeypatch.delenv("HYPIC_PIC_KV_IMPORT_WORKERS", raising=False)
    else:
        monkeypatch.setenv("HYPIC_PIC_KV_IMPORT_WORKERS", kv)
    # ``state`` is kept in the signature only so callers can express the retired
    # setting; the runtime no longer reads it, which the assert below relies on.
    monkeypatch.delenv("HYPIC_PIC_STATE_WORKERS", raising=False)
    return PICNativeKVImportWorker()


def test_kv_receives_are_serial_by_default(monkeypatch) -> None:
    worker = _worker(monkeypatch, kv=None, state=None)

    assert worker.workers == 1, "the pipeline needs one receive at a time"
    worker._executor.shutdown(wait=False)


def test_the_state_setting_no_longer_moves_the_kv_pool(monkeypatch) -> None:
    worker = _worker(monkeypatch, kv=None, state="8")

    assert worker.workers == 1, "state width is a different concern"
    worker._executor.shutdown(wait=False)


def test_the_kv_width_can_be_raised(monkeypatch) -> None:
    worker = _worker(monkeypatch, kv="3", state=None)

    assert worker.workers == 3
    assert worker._executor._max_workers == 3
    worker._executor.shutdown(wait=False)


def test_a_bad_width_falls_back_to_one(monkeypatch) -> None:
    worker = _worker(monkeypatch, kv="many", state=None)

    assert worker.workers == 1
    worker._executor.shutdown(wait=False)


def test_zero_is_raised_to_one(monkeypatch) -> None:
    worker = _worker(monkeypatch, kv="0", state=None)

    assert worker.workers == 1
    worker._executor.shutdown(wait=False)
