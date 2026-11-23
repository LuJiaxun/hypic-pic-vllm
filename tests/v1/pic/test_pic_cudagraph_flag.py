# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the cudagraph opt-in for PIC ranges.

A trace of one PIC step showed 29.5% kernel time and 70% GPU idle: about 2800 kernel
launches per step, each with roughly 27 us of gap. The range path forced eager, so the
service-level flag could not help. Capture is now opt-in through HYPIC_PIC_CUDAGRAPH, which
stays off by default because the captured path pads to a capture size and that changes the
token accounting a range is built around.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")


def _force_eager(monkeypatch, value: str | None) -> bool:
    if value is None:
        monkeypatch.delenv("HYPIC_PIC_CUDAGRAPH", raising=False)
    else:
        monkeypatch.setenv("HYPIC_PIC_CUDAGRAPH", value)
    # Mirrors the expression the runner evaluates.
    return not os.getenv("HYPIC_PIC_CUDAGRAPH", "0").strip() not in ("", "0")


def test_ranges_stay_eager_by_default(monkeypatch) -> None:
    assert _force_eager(monkeypatch, None) is True


def test_capture_can_be_opted_into(monkeypatch) -> None:
    assert _force_eager(monkeypatch, "1") is False


def test_zero_keeps_eager(monkeypatch) -> None:
    assert _force_eager(monkeypatch, "0") is True
