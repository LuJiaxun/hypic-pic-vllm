"""The serialized range payload must stay integer-only.

A range used to be a ``(start, end)`` pair.  Separator pairing first tried to add
a third element, and later a ``bool`` inside an ``int`` tuple; both shapes were
rejected by the serializer, which took the engine down before any request ran.
The flag therefore travels in a parallel ``list[bool]`` while the ranges stay
pairs.

These checks read the declarations instead of an instance, so they fail fast and
without a running engine.
"""

from __future__ import annotations

import inspect

import pytest
from typing import Any, get_args, get_origin, get_type_hints

from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request

_RANGES = "pic_segment_ranges"
_FLAGS = "pic_segment_cacheable"


def _inner_list(annotation: Any) -> Any:
    optional_args = [item for item in get_args(annotation) if item is not type(None)]
    assert len(optional_args) == 1, annotation
    return optional_args[0]


def _element_annotation(annotation: Any) -> Any:
    list_arg = _inner_list(annotation)
    assert get_origin(list_arg) is list, list_arg
    (element,) = get_args(list_arg)
    return element


def _request_hints() -> dict[str, Any]:
    signature = inspect.signature(Request.__init__)
    if isinstance(signature.parameters[_RANGES].annotation, str):
        return get_type_hints(Request.__init__)
    return {
        name: parameter.annotation
        for name, parameter in signature.parameters.items()
    }


def test_engine_core_request_ranges_are_integer_pairs() -> None:
    hints = get_type_hints(EngineCoreRequest)
    element = _element_annotation(hints[_RANGES])
    assert get_origin(element) is tuple
    assert get_args(element) == (int, int)


def test_engine_core_request_carries_the_flag_separately() -> None:
    hints = get_type_hints(EngineCoreRequest)
    assert _element_annotation(hints[_FLAGS]) is bool


def test_request_ranges_are_integer_pairs() -> None:
    hints = _request_hints()
    element = _element_annotation(hints[_RANGES])
    assert get_origin(element) is tuple
    assert get_args(element) == (int, int)
    assert _element_annotation(hints[_FLAGS]) is bool


def test_flag_list_decides_cacheability() -> None:
    from vllm.v1.pic.segmenter import segments_from_ranges

    segments = segments_from_ranges(
        [1, 2, 3, 4], [(0, 2), (2, 4)], [False, True]
    )
    assert [segment.cacheable for segment in segments] == [False, True]


def test_three_element_ranges_still_build_cacheable_segments() -> None:
    from vllm.v1.pic.segmenter import segments_from_ranges

    segments = segments_from_ranges([1, 2, 3, 4], [(0, 2, True), (2, 4, False)])
    assert [segment.cacheable for segment in segments] == [True, False]

def test_engine_core_request_round_trips_the_pic_payload() -> None:
    """Drive the real serializer instead of only reading the annotation.

    Both earlier failures - a three-element range, then a bool inside an integer
    tuple - passed an annotation check and only failed when the engine decoded a
    request.  This encodes and decodes the actual struct, so the payload and its
    declaration have to agree.
    """
    msgspec = pytest.importorskip("msgspec")
    from vllm.v1.engine import EngineCoreRequest

    request = EngineCoreRequest(
        request_id="req-pic",
        prompt_token_ids=[1, 2, 3, 4],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        pic_enabled=True,
        pic_segment_ranges=[(0, 2), (2, 4)],
        pic_segment_cacheable=[True, False],
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(request), type=EngineCoreRequest
    )
    assert restored.pic_segment_ranges == [(0, 2), (2, 4)]
    assert restored.pic_segment_cacheable == [True, False]
