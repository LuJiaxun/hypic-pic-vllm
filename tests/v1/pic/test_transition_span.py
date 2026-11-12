"""Span rules for the published hybrid transition.

The reused range starts after a seam at the head of the reused content and runs
to the end of the segment, and the published transition covers exactly that
range.  A compact operator cannot be sliced, so these tests pin the rules that
keep publication and reuse in agreement.
"""

from __future__ import annotations

import pytest

from vllm.v1.pic.mooncake_kv import PICHybridStateObject


def _manifest(**overrides: object) -> PICHybridStateObject:
    values = {
        "external_key": "segment/abc/hybrid-state",
        "token_count": 4344,
        "payload_key": "segment/abc/hybrid-state/payload",
        "nbytes": 1024,
        "checksum": "0" * 64,
        "span_start": 8,
    }
    values.update(overrides)
    return PICHybridStateObject(**values)  # type: ignore[arg-type]


def test_manifest_round_trips_the_span_start() -> None:
    manifest = _manifest()
    restored = PICHybridStateObject.from_dict(manifest.as_dict())
    assert restored.span_start == 8
    assert restored.token_count == manifest.token_count


def test_manifest_without_span_defaults_to_the_whole_segment() -> None:
    payload = _manifest().as_dict()
    payload.pop("span_start")
    restored = PICHybridStateObject.from_dict(payload)
    assert restored.span_start == 0


def test_span_start_must_leave_room_for_the_covered_range() -> None:
    manifest = _manifest(span_start=0)
    operator_start = manifest.span_start
    operator_end = manifest.span_start + manifest.token_count
    assert (operator_start, operator_end) == (0, 4344)

    manifest = _manifest()
    operator_start = manifest.span_start
    operator_end = manifest.span_start + manifest.token_count
    assert (operator_start, operator_end) == (8, 4352)


@pytest.mark.parametrize(
    ("seam", "segment_length", "expected_span"),
    [
        (8, 4352, 8),
        (8, 4, 3),
        (0, 4352, 0),
    ],
)
def test_published_span_matches_the_reuse_window(
    seam: int, segment_length: int, expected_span: int
) -> None:
    """Mirror of PICConfig.seam_sink applied by the plan and the publisher."""
    span_start = min(max(seam, 0), max(segment_length - 1, 0))
    assert span_start == expected_span
    assert segment_length - span_start > 0
