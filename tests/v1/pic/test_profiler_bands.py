# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the profiler bands.

A trace only shows what has a band. The PIC phases ran as bare Python calls, so a torch
profile showed the model kernels and nothing about the preparation that dominates a small
reuse: plan application, slot attachment, the isolated batch. Each of those now opens a
``record_function`` scope, which vLLM gates behind ``VLLM_CUSTOM_SCOPES_FOR_PROFILING`` so
the normal path keeps paying nothing.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("torch")

RUNNER = pathlib.Path("vllm/v1/worker/gpu_model_runner.py")

EXPECTED = (
    "pic:plan_apply",
    "pic:slot_attach",
    "pic:isolated_batch",
    "pic:range_attach",
    "pic:ranges",
)


def test_every_documented_band_exists() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    for name in EXPECTED:
        assert f'record_function_or_nullcontext("{name}")' in source, (
            f"missing profiler band {name}: the trace will not show that phase"
        )


def test_plan_apply_is_banded_at_both_call_sites() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert source.count('record_function_or_nullcontext("pic:plan_apply")') == 2
