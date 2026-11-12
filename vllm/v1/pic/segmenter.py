# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Token-based segmentation for position-independent prompt reuse."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PICSegment:
    """A reusable prompt segment in prompt-token coordinates.

    ``start``/``end`` are positions in the current target request.  The
    reusable KV payload itself is canonical and starts at local position zero;
    callers must not treat ``start`` as the source KV position.
    """

    start: int
    end: int
    token_ids: tuple[int, ...]
    seg_hash: bytes
    # Content enclosed by a separator pair is reusable; content outside the pair
    # is a query or instruction that differs per request and must be recomputed.
    # A prompt without separators keeps its single segment reusable.
    cacheable: bool = True

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def canonical_range(self) -> tuple[int, int]:
        return (0, self.length)


def _segment_hash(token_ids: Sequence[int]) -> bytes:
    payload = b"".join(
        int(token).to_bytes(8, "little", signed=True) for token in token_ids
    )
    return hashlib.sha256(payload).digest()[:16]


def split_text_and_tokenize(
    text: str,
    tokenizer: Any,
    separator: str,
    pretokenized_token_ids: Sequence[int] | None = None,
    separator_token_ids: Sequence[int] | None = None,
) -> tuple[list[int], list[tuple[int, int, bool]]]:
    """Split raw text before tokenization and return explicit token ranges.

    This follows the SGLang PIC input path. The separator is removed at the
    string level, each non-empty part is tokenized independently, and the
    resulting token ranges are recorded while concatenating the parts. This
    avoids relying on a separator token sequence that may change under BPE
    whitespace/context merges.
    """
    parts = text.split(separator) if separator else [text]
    # Content between a separator pair is reusable; the text outside a pair is a
    # query that differs per request.  Parts alternate, so odd positions are the
    # enclosed ones.  When the separator never occurs the whole prompt is one
    # reusable block.
    has_separator = bool(separator) and len(parts) > 1
    enclosed = [
        (not has_separator) or index % 2 == 1 for index in range(len(parts))
    ]

    # InputProcessor normally receives a complete tokenization from the
    # renderer.  If the separator is represented as an exact token sequence in
    # that input, reuse it instead of encoding every segment again.  This is a
    # safe fast path: contextual BPE boundaries that prevent an exact match
    # simply fall through to the original per-segment tokenization below.
    if pretokenized_token_ids is not None:
        expected_segments = sum(bool(part) for part in parts)
        if not separator:
            if expected_segments:
                ids = [int(token) for token in pretokenized_token_ids]
                return ids, ([(0, len(ids), True)] if ids else [])
        else:
            if separator_token_ids is None:
                try:
                    separator_ids = tokenizer.encode(
                        separator, add_special_tokens=False
                    )
                except Exception:
                    separator_ids = []
            else:
                separator_ids = separator_token_ids
            if separator_ids:
                token_ids, ranges = _split_pretokenized_token_ids(
                    pretokenized_token_ids, separator_ids, enclosed
                )
                if len(ranges) == expected_segments:
                    return token_ids, ranges

    token_ids: list[int] = []
    ranges: list[tuple[int, int, bool]] = []
    for index, part in enumerate(parts):
        segment_ids = [
            int(token) for token in tokenizer.encode(part, add_special_tokens=False)
        ]
        if not segment_ids:
            continue
        start = len(token_ids)
        token_ids.extend(segment_ids)
        ranges.append((start, len(token_ids), enclosed[index]))
    return token_ids, ranges


def segments_from_ranges(
    token_ids: Sequence[int],
    ranges: Sequence[tuple],
    cacheable: Sequence[bool] | None = None,
) -> list[PICSegment]:
    """Build PIC segments from explicit token ranges.

    ``cacheable`` is a parallel list telling whether each part sits inside a
    separator pair.  Ranges that carry the flag as a third element are still
    accepted, which keeps older callers and the internal split path working.
    """
    ids = tuple(int(token) for token in token_ids)
    segments: list[PICSegment] = []
    previous_end = 0
    flags = list(cacheable) if cacheable is not None else None
    for index, item in enumerate(ranges):
        start, end = int(item[0]), int(item[1])
        if flags is not None and index < len(flags):
            allowed = bool(flags[index])
        elif len(item) >= 3:
            allowed = bool(item[2])
        else:
            allowed = True
        if not (0 <= start < end <= len(ids)) or start < previous_end:
            raise ValueError("PIC segment ranges are invalid or overlap")
        segment_ids = ids[start:end]
        segments.append(
            PICSegment(
                start,
                end,
                segment_ids,
                _segment_hash(segment_ids),
                cacheable=allowed,
            )
        )
        previous_end = end
    return segments


def _split_pretokenized_token_ids(
    token_ids: Sequence[int],
    separator_ids: Sequence[int],
    enclosed: Sequence[bool] | None = None,
) -> tuple[list[int], list[tuple[int, int, bool]]]:
    """Split already-tokenized input without constructing PICSegment objects.

    ``split_token_ids`` is the public structural helper and intentionally
    returns hashed ``PICSegment`` objects.  The InputProcessor only needs the
    flattened token IDs and ranges, so constructing those objects and then
    flattening them again is unnecessary work on every request.

    ``enclosed`` marks, per original part index, whether that part sits inside a
    separator pair.  Empty parts keep their slot so the parity stays aligned.
    """
    ids = token_ids if isinstance(token_ids, list) else list(token_ids)
    separator = [int(token) for token in separator_ids]
    if not ids:
        return [], []
    if not separator:
        return ids, [(0, len(ids), True)]
    matched_any = any(
        ids[index : index + len(separator)] == separator
        for index in range(len(ids) - len(separator) + 1)
    )

    output: list[int] = []
    ranges: list[tuple[int, int, bool]] = []
    segment_start = 0
    output_cursor = 0
    separator_len = len(separator)
    first_separator_token = separator[0]
    index = 0
    part_index = 0
    while index <= len(ids) - separator_len:
        if (
            ids[index] == first_separator_token
            and ids[index : index + separator_len] == separator
        ):
            if segment_start < index:
                segment = ids[segment_start:index]
                output.extend(segment)
                end = output_cursor + len(segment)
                ranges.append(
                    (
                        output_cursor,
                        end,
                        True
                        if not matched_any
                        or enclosed is None
                        or part_index >= len(enclosed)
                        else bool(enclosed[part_index]),
                    )
                )
                output_cursor = end
            index += separator_len
            segment_start = index
            part_index += 1
        else:
            index += 1

    if segment_start < len(ids):
        segment = ids[segment_start:]
        output.extend(segment)
        ranges.append(
            (
                output_cursor,
                output_cursor + len(segment),
                True
                if not matched_any
                or enclosed is None
                or part_index >= len(enclosed)
                else bool(enclosed[part_index]),
            )
        )

    return output, ranges


def split_token_ids(
    token_ids: Sequence[int],
    separator_ids: Sequence[int],
) -> list[PICSegment]:
    """Split token IDs on an exact separator-token sequence.

    This is the low-level structural helper: every returned segment describes a
    non-empty token span and is cacheable.  Whether a span may be cached depends
    on the prompt format, and ``split_text_and_tokenize`` marks the parts that
    sit inside a separator pair.
    """
    if not token_ids:
        return []
    if not separator_ids:
        ids = tuple(int(token) for token in token_ids)
        return [PICSegment(0, len(ids), ids, _segment_hash(ids))]

    token_ids = tuple(int(token) for token in token_ids)
    separator_ids = tuple(int(token) for token in separator_ids)
    segments: list[PICSegment] = []
    start = 0
    i = 0
    separator_len = len(separator_ids)
    while i <= len(token_ids) - separator_len:
        if token_ids[i : i + separator_len] == separator_ids:
            if start < i:
                ids = token_ids[start:i]
                segments.append(
                    PICSegment(
                        start,
                        i,
                        ids,
                        _segment_hash(ids),
                        cacheable=True,
                    )
                )
            i += separator_len
            start = i
        else:
            i += 1

    if start < len(token_ids):
        ids = token_ids[start:]
        segments.append(
            PICSegment(
                start,
                len(token_ids),
                ids,
                _segment_hash(ids),
                cacheable=True,
            )
        )
    return segments
