# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The committed acceptance fixtures describe the shapes they claim to describe.

These are structural checks only - no engine, no GPU, no tokenizer. They exist because a fixture that is
subtly wrong does not fail; it produces a run that looks like a PIC fault. Two ways that happened:

  * a segment shorter than ``--pic-min-segment-tokens`` is dropped by the scheduler, so the request
    reuses nothing and the log shows a lookup miss;
  * a document written as one sentence repeated makes the model continue the repetition instead of
    answering, so a correct reuse reads as a wrong answer.

The fixtures are committed under ``tests/v1/pic/fixtures/`` so a newcomer does not have to generate
them, and one test here insists they still match the definitions that produced them.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TOOL = _ROOT / "tools" / "pic_fixture_shapes.py"
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load():
    spec = importlib.util.spec_from_file_location("pic_fixture_shapes", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


shapes = _load()


def _files(shape: str) -> dict[str, str]:
    directory = FIXTURES / shape
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("*.txt"))
    }


@pytest.mark.parametrize("shape", shapes.SHAPES)
def test_fixtures_are_committed(shape: str) -> None:
    """Every shape ships its prompts, so a run needs no generation step."""
    files = _files(shape)
    assert files, f"{shape}: no fixtures committed"
    assert "reuse" in files, f"{shape}: no reuse prompt"
    assert "baseline" in files, f"{shape}: no baseline prompt"
    assert any(name.startswith("publish") for name in files), f"{shape}: nothing to publish"


@pytest.mark.parametrize("shape", shapes.SHAPES)
def test_committed_files_match_the_definitions(shape: str) -> None:
    """A fixture that drifted from its definition is a run nobody maintains."""
    expected = shapes.build(shape)
    files = _files(shape)
    assert set(files) == set(expected), (
        f"{shape}: files on disk {sorted(files)} differ from {sorted(expected)}; "
        "regenerate with tools/pic_make_fixtures.py"
    )
    for name, text in expected.items():
        assert files[name] == text, f"{shape}/{name}.txt does not match pic_fixture_shapes.py"


@pytest.mark.parametrize("shape", shapes.SHAPES)
def test_baseline_is_one_plain_prompt(shape: str) -> None:
    """The baseline is the same content without segments, so it has nothing to reuse."""
    assert shapes.SEP not in _files(shape)["baseline"]


@pytest.mark.parametrize("shape", shapes.SHAPES)
def test_baseline_carries_every_document(shape: str) -> None:
    for name, text in _files(shape).items():
        if not name.startswith(("publish", "reuse")):
            continue
        for part in shapes.segments(text):
            needle = part.strip().replace(" ", "")[:200]
            assert needle in _files(shape)["baseline"].replace(" ", ""), (
                f"{shape}/{name}: a document is missing from the baseline"
            )


@pytest.mark.parametrize("shape", shapes.SHAPES)
def test_cached_segments_clear_the_minimum(shape: str) -> None:
    """Only the enclosed parts are cached, and the scheduler drops any below its minimum.

    Measured in words here so no tokenizer is needed: this prose runs about 1.3 words per token, so
    400 words is comfortably past 512 tokens.
    """
    for name, text in _files(shape).items():
        if not name.startswith("publish") or name.endswith("_0"):
            continue
        for part in shapes.reusable_parts(text):
            words = len(part.split())
            assert words >= 400, f"{shape}/{name}: a cached segment is only {words} words"


@pytest.mark.parametrize("shape", shapes.SHAPES)
def test_reuse_carries_every_published_document(shape: str) -> None:
    files = _files(shape)
    published = [
        part.strip()
        for name, text in files.items()
        if name.startswith("publish")
        for part in shapes.segments(text)
    ]
    assert published, f"{shape}: nothing is published"
    for part in published:
        assert part[:200] in files["reuse"], (
            f"{shape}: a published document is missing from the reuse prompt"
        )


def test_documents_are_prose_not_one_repeated_sentence() -> None:
    part = shapes.reusable_parts(_files("c")["publish_a"])[0]
    sentences = [line for line in part.split(".") if line.strip()]
    assert len(set(sentences)) >= 3, "the document repeats a single sentence"


def test_the_question_matches_the_document_that_is_reused() -> None:
    """The answer has to be readable from the reused document, or the run proves nothing."""
    assert shapes.TARGET in _files("c")["publish_b"].lower()
    assert "segment b" in _files("a")["reuse"].lower()
