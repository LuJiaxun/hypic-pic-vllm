"""Separator pairing decides which segments may be reused.

Content enclosed by a separator pair is a document or a prefix block and may be
cached; the text outside a pair is the request-specific query and must be
recomputed.  A prompt without separators keeps its single block reusable.

``split_token_ids`` stays a low-level structural helper that marks every span
cacheable, so these tests go through the text path the engine actually uses.
"""

from __future__ import annotations

from vllm.v1.pic.segmenter import (
    PICSegment,
    segments_from_ranges,
    split_text_and_tokenize,
)

SEPARATOR = "<SEP>"


class _CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


def _segments(text: str) -> list[PICSegment]:
    token_ids, ranges = split_text_and_tokenize(text, _CharTokenizer(), SEPARATOR)
    return segments_from_ranges(token_ids, ranges)


def _flags(text: str) -> list[bool]:
    return [segment.cacheable for segment in _segments(text)]


def test_enclosed_blocks_are_cacheable_and_the_query_is_not() -> None:
    # <SEP>AAAA<SEP> BBBB <SEP> CCCC <SEP>
    text = f"{SEPARATOR}AAAA{SEPARATOR} BBBB {SEPARATOR} CCCC {SEPARATOR}"
    assert _flags(text) == [True, False, True]


def test_double_separator_makes_every_block_cacheable() -> None:
    # <SEP>AAAA<SEP><SEP> BBBB<SEP> <SEP> CCCC <SEP>
    text = (
        f"{SEPARATOR}AAAA{SEPARATOR}{SEPARATOR} BBBB{SEPARATOR} "
        f"{SEPARATOR} CCCC {SEPARATOR}"
    )
    assert _flags(text) == [True, True, False, True]


def test_prompt_without_the_separator_stays_one_reusable_block() -> None:
    assert _flags("just a prompt") == [True]


def test_leading_text_before_the_first_separator_is_not_cacheable() -> None:
    text = f" preamble {SEPARATOR}AAAA{SEPARATOR}"
    assert _flags(text)[-1] is True
    assert _flags(text)[0] is False


def test_ranges_carry_the_flag_and_default_to_cacheable() -> None:
    ids = [1, 2, 3, 4]
    segments = segments_from_ranges(ids, [(0, 2, False), (2, 4, True)])
    assert [segment.cacheable for segment in segments] == [False, True]
    legacy = segments_from_ranges(ids, [(0, 2)])
    assert legacy[0].cacheable is True


def test_low_level_splitter_marks_every_span_cacheable() -> None:
    from vllm.v1.pic.segmenter import split_token_ids

    segments = split_token_ids([9, 1, 1, 9, 2, 2, 9], [9])
    assert [segment.cacheable for segment in segments] == [True, True]
    assert segments[1].token_ids == (2, 2)
