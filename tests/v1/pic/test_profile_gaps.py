# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the profiling lines added to attribute per-segment work.

Two kinds of records are new: the gap between instrumented phases, and the timing
of the per-new-segment steps (remote lookup, destination resolution, block
zeroing). They exist so a distinct-versus-repeated comparison can attribute time
instead of inferring it, and they must stay silent when profiling is off.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker import gpu_model_runner as runner


class _Recorder:
    """Collect formatted log lines without depending on logging configuration."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def warning(self, fmt: str, *args: object) -> None:
        self.lines.append(fmt % args if args else fmt)

    def debug(self, fmt: str, *args: object) -> None:
        self.lines.append(fmt % args if args else fmt)


@pytest.fixture(autouse=True)
def clear_phase_state():
    runner._PIC_PHASE_LAST_END.clear()
    yield
    runner._PIC_PHASE_LAST_END.clear()


def _phase(runner_module, stage: str, *, request: str = "r1", mode: str = "pic") -> None:
    start = runner_module._pic_profile_phase_start(True, None)
    runner_module._pic_profile_phase_end(
        True, start, request_ids=request, mode=mode, stage=stage
    )


def test_gap_is_emitted_between_phases(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(runner, "logger", recorder)

    _phase(runner, "first")
    time.sleep(0.002)
    _phase(runner, "second")

    text = "\n".join(recorder.lines)
    assert "stage=first " in text
    assert "stage=second.gap" in text
    gap_line = next(line for line in recorder.lines if "second.gap" in line)
    gap_ms = float(gap_line.split("duration_ms=")[1])
    assert gap_ms >= 1.0


def test_no_gap_for_the_first_phase_of_a_request(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(runner, "logger", recorder)

    _phase(runner, "first", request="only-one")

    assert not any(".gap" in line for line in recorder.lines)


def test_gap_is_per_request_and_mode(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(runner, "logger", recorder)

    _phase(runner, "a", request="r1")
    time.sleep(0.001)
    # A different request must not inherit r1's previous phase end.
    _phase(runner, "b", request="r2")

    assert not any(".gap" in line for line in recorder.lines)


def test_phase_records_are_silent_when_disabled(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(runner, "logger", recorder)

    runner._pic_profile_phase_end(
        False, 1, request_ids="r1", mode="pic", stage="first"
    )

    assert recorder.lines == []


class _Index:
    group_block_sizes = (2,)

    def __init__(self, found) -> None:
        self.found = found

    def lookup(self, segment, groups):
        return self.found


def _discover(monkeypatch, *, found, level: str):
    recorder = _Recorder()
    monkeypatch.setattr(scheduler_module, "logger", recorder)
    if level and level != "0":
        monkeypatch.setenv("HYPIC_PIC_DEBUG", "1")
    else:
        monkeypatch.delenv("HYPIC_PIC_DEBUG", raising=False)
    stub = SimpleNamespace(
        pic_mooncake_reference_index=_Index(found),
        vllm_config=SimpleNamespace(pic_config=SimpleNamespace(debug=False)),
    )
    discover = Scheduler._discover_pic_mooncake_segment.__get__(stub, Scheduler)
    return discover, recorder


def test_lookup_logs_its_duration(monkeypatch) -> None:
    """The lookup line follows the profiling switch, not ``--pic-debug``.

    A run that profiled without debug logging went blind here once already.
    """
    discover, recorder = _discover(monkeypatch, found="entry", level="1")

    assert discover(SimpleNamespace(seg_hash=b"0123456789abcdef")) == "entry"

    line = next(line for line in recorder.lines if "pic-lookup" in line)
    assert "found=True" in line
    assert "ms=" in line


def test_lookup_is_quiet_without_profiling(monkeypatch) -> None:
    discover, recorder = _discover(monkeypatch, found=None, level="0")

    assert discover(SimpleNamespace(seg_hash=b"0123456789abcdef")) is None
    assert recorder.lines == []


def test_trace_switch_reads_the_environment(monkeypatch) -> None:
    """One switch drives both the trace and the profile."""
    trace = Scheduler._pic_trace_enabled
    monkeypatch.setenv("HYPIC_PIC_DEBUG", "1")
    assert trace() is True
    monkeypatch.setenv("HYPIC_PIC_DEBUG", "0")
    assert trace() is False
    monkeypatch.setenv("HYPIC_PIC_DEBUG", "")
    assert trace() is False
