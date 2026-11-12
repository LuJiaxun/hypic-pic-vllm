# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the coordinate space of a captured transition operator.

The manifest records the operator's span in segment-local terms: the reader compares
``operator.token_start`` with ``manifest.span_start``, and that span start is the seam
measured from the segment head. Building the operator with the segment's absolute start
therefore worked only while the reusable segment sat at position zero - the prefix shape -
and raised "captured transition does not cover the reusable span" as soon as a query
segment came first, which is the shape A ordering.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("torch")

PIC_CAPTURE = pathlib.Path("vllm/v1/pic/live_capture.py")
RUNNER = pathlib.Path("vllm/v1/worker/gpu_model_runner.py")


def test_capture_builds_operators_in_segment_local_coordinates() -> None:
    source = PIC_CAPTURE.read_text(encoding="utf-8")

    assert "token_start=segment.start" not in source, (
        "a captured operator must stay segment-local; the manifest span_start is the seam"
    )
    assert "token_start=0," in source


def test_the_runner_publish_path_is_segment_local_too() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "token_start=segment.start" not in source


def test_the_operator_end_is_local_as_well() -> None:
    # An untrimmed operator must report its length, not the segment's absolute end: the
    # publish path measures it before slicing.
    for path in (PIC_CAPTURE, RUNNER):
        source = path.read_text(encoding="utf-8")
        assert "token_end=segment.end" not in source, path.name


def test_the_reader_still_compares_against_the_local_span() -> None:
    source = pathlib.Path("vllm/v1/pic/mooncake_kv.py").read_text(encoding="utf-8")

    assert "operator.token_start != manifest.span_start" in source
