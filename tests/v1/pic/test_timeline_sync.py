# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the phase-timeline synchronisation.

Phase records synchronise the device at every boundary when the diagnostics switch is on: that
makes the timeline truthful, at the cost of charging each phase the tail of the previous step's
work. The timeline exists for a profile run, so it follows the profile rather than carrying a
switch of its own - a production run never pays for it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from vllm.v1.worker import gpu_model_runner


def test_a_profile_run_synchronises_the_timeline(monkeypatch) -> None:
    monkeypatch.setenv("HYPIC_PIC_DEBUG", "1")

    assert gpu_model_runner._pic_timeline_sync_enabled() is True


def test_production_does_not_synchronise(monkeypatch) -> None:
    monkeypatch.delenv("HYPIC_PIC_DEBUG", raising=False)

    assert gpu_model_runner._pic_timeline_sync_enabled() is False
