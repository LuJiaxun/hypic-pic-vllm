# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boundary arithmetic for windows that do not end on a block boundary.

Every test in this directory used a window length that divides evenly into the page or kernel block -
544, 1088, 512 - and so did every measurement. While a window ends on a boundary, ``floor`` and ``ceil``
give the same answer, an exact block count and a covering one are the same number, and "the blocks
equal the quotient" holds. Ten places in the import path derived that number independently and they all
agreed by accident.

Lengthening the window to a segment's last token made it stop inside a page, and the disagreements
surfaced one at a time: a truncated token count, an allocation that refused a partial end, a provider
that called one block too few "incomplete". None of them needed a GPU to find; they needed a window
length that is not a multiple of the block size.

These cases are that length.
"""

from __future__ import annotations

import pytest

from vllm.v1.pic.native_kv import (
    PICNativeKVAllocation,
    PICWindow,
    get_covering_local_block_span,
    get_full_local_block_span,
)

KERNEL_BLOCK = 32
ALLOCATOR_BLOCK = 544
PAGE_ALIGNED = 544 * 8

#: Windows that stop inside a page, plus the two neighbours of a page boundary.
UNALIGNED_LENGTHS = [
    PAGE_ALIGNED,  # control: aligned, the two answers must coincide
    PAGE_ALIGNED + 1,
    PAGE_ALIGNED - 1,
    4839,  # the measured document: 544 * 8 + 487
    4837,
    543,
    545,
]


def _ceil_div(value: int, size: int) -> int:
    return -(-value // size)


@pytest.mark.parametrize("length", UNALIGNED_LENGTHS)
def test_covering_span_rounds_up(length: int) -> None:
    """A covering span needs the page its end falls inside."""
    covering = get_covering_local_block_span(length, KERNEL_BLOCK)
    assert covering.block_count == _ceil_div(length, KERNEL_BLOCK)
    assert covering.reusable_token_count == length


@pytest.mark.parametrize("length", UNALIGNED_LENGTHS)
def test_full_span_rounds_down(length: int) -> None:
    """The whole-block prefix keeps its own meaning: it stops before the partial page."""
    full = get_full_local_block_span(length, KERNEL_BLOCK)
    assert full.block_count == length // KERNEL_BLOCK
    assert full.reusable_token_count == (length // KERNEL_BLOCK) * KERNEL_BLOCK


@pytest.mark.parametrize("length", UNALIGNED_LENGTHS)
def test_window_accepts_a_partial_end(length: int) -> None:
    """An allocation may end inside a page; only its start has to be a block boundary."""
    blocks = tuple(range(_ceil_div(length, ALLOCATOR_BLOCK)))
    allocation = PICNativeKVAllocation(
        kv_cache_group_id=0,
        target_start=0,
        target_end=length,
        block_size=ALLOCATOR_BLOCK,
        public_block_ids=blocks,
    )
    assert allocation.target_end == length


@pytest.mark.parametrize("length", UNALIGNED_LENGTHS)
def test_window_asks_for_coverage_not_equality(length: int) -> None:
    """One block more than the quotient is the page tail, and it has to be enough."""
    window = PICWindow(0, length, KERNEL_BLOCK)
    needed = _ceil_div(length, KERNEL_BLOCK)
    assert window.blocks() == needed
    assert window.fits(needed)
    assert window.fits(needed + 1), "the page tail must not be rejected"
    assert not window.fits(needed - 1), "one block short must be rejected"


@pytest.mark.parametrize("length", UNALIGNED_LENGTHS)
def test_window_counts_in_another_unit(length: int) -> None:
    """The same range counted in pages and in kernel blocks agrees with the definitions."""
    window = PICWindow(0, length, KERNEL_BLOCK)
    assert window.blocks_at(ALLOCATOR_BLOCK) == _ceil_div(length, ALLOCATOR_BLOCK)
    assert window.covers(PICWindow(0, length - 1, KERNEL_BLOCK))
    assert not window.covers(PICWindow(0, length + 1, KERNEL_BLOCK))


def test_published_window_and_reuse_window_agree() -> None:
    """The publisher describes the window the reader will use, one token short of the segment.

    The last token of a segment is recomputed because the request samples from it, so a transition
    published for the whole segment cannot be served; this is the off-by-one that made publication and
    the reader disagree by a single token.
    """
    segment_length = 4839
    seam = 8
    reuse = PICWindow(seam, segment_length - 1, KERNEL_BLOCK)
    published = PICWindow(seam, segment_length - 1, KERNEL_BLOCK)

    assert reuse.token_count == segment_length - seam - 1
    assert published.token_count == reuse.token_count
    assert published.covers(reuse)


def test_relative_window_translates_to_the_operator_frame() -> None:
    """``slice`` takes coordinates relative to the operator, not to the segment."""
    operator_window = PICWindow(271, 5110, KERNEL_BLOCK)
    segment_window = PICWindow(279, 5109, KERNEL_BLOCK)
    relative = segment_window.relative_to(operator_window.token_start)

    assert relative.token_start == 8
    assert relative.token_end == 4838
    assert operator_window.covers(segment_window)
