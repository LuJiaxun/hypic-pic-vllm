# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Token-based segmentation for position-independent prompt reuse."""

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence


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
) -> tuple[list[int], list[tuple[int, int]]]:
    """Split raw text before tokenization and return explicit token ranges.

    This follows the SGLang PIC input path. The separator is removed at the
    string level, each non-empty part is tokenized independently, and the
    resulting token ranges are recorded while concatenating the parts. This
    avoids relying on a separator token sequence that may change under BPE
    whitespace/context merges.
    """
    parts = text.split(separator) if separator else [text]
    token_ids: list[int] = []
    ranges: list[tuple[int, int]] = []
    for part in parts:
        segment_ids = [
            int(token)
            for token in tokenizer.encode(part, add_special_tokens=False)
        ]
        if not segment_ids:
            continue
        start = len(token_ids)
        token_ids.extend(segment_ids)
        ranges.append((start, len(token_ids)))
    return token_ids, ranges


def segments_from_ranges(
    token_ids: Sequence[int], ranges: Sequence[tuple[int, int]]
) -> list[PICSegment]:
    """Build PIC segments from explicit token ranges."""
    ids = tuple(int(token) for token in token_ids)
    segments: list[PICSegment] = []
    previous_end = 0
    for start, end in ranges:
        if not (0 <= start < end <= len(ids)) or start < previous_end:
            raise ValueError("PIC segment ranges are invalid or overlap")
        segment_ids = ids[start:end]
        segments.append(PICSegment(start, end, segment_ids, _segment_hash(segment_ids)))
        previous_end = end
    return segments


def split_token_ids(
    token_ids: Sequence[int],
    separator_ids: Sequence[int],
) -> list[PICSegment]:
    """Split token IDs on an exact separator-token sequence."""
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
                segments.append(PICSegment(start, i, ids, _segment_hash(ids)))
            i += separator_len
            start = i
        else:
            i += 1

    if start < len(token_ids):
        ids = token_ids[start:]
        segments.append(PICSegment(start, len(token_ids), ids, _segment_hash(ids)))
    return segments
