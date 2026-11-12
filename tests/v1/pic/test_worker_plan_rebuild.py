# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the worker-plan rebuild after adopt.

The worker plan is built when a request first enters the batch, which happens before the
pending import installs its native references. An entry carrying external references has no
physical handles at that moment, so the plan is refused and the request silently recomputes
its whole prompt. The plan is rebuilt after the adopt, which costs about a millisecond per
refused request instead of a full prefill.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("torch")

SOURCE = pathlib.Path("vllm/v1/worker/gpu_model_runner.py")


def test_the_rebuild_runs_after_the_adopt() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    adopt = source.index("self._pic_adopt_park_import(request_state)")
    rebuild = source.index("worker plan rebuilt after adopt")
    assert rebuild > adopt, "the rebuild must follow the adopt, or refs are still missing"


def test_the_rebuild_recompiles_the_execution_plan() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "recompiled = compile_execution_plan(" in source
    assert "request_state.pic_execution_plan = recompiled" in source


def test_the_rebuild_only_touches_refused_plans() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "request_state.pic_worker_plan is None" in source


def test_the_plan_builder_is_reachable_without_request_data() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "def _build_pic_worker_plan_for(" in source
    assert "self._build_pic_worker_plan_for(" in source
