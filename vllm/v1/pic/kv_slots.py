# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Where a PIC segment's tokens live in vLLM's own KV cache.

Two questions, no state: which layer targets make up a request's KV cache, and which slot in those
blocks a token maps to. Both sides of the bridge use this - the exporter writes through
`do_kv_cache_update`, the importer reads the same mapping - so the numbering is defined once.

It does not add a second attention layout, alter the block table, or make the attention kernel read the
PIC pool directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.pic.cache import PICSegmentEntry
from vllm.v1.pic.segmenter import PICSegment
from vllm.v1.pic.snapshot import PICPhysicalSnapshot, PICSnapshotStore

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PICKVLayerTarget:
    """One existing vLLM attention layer that can receive restored KV data."""

    layer_name: str
    slot_mapping: torch.Tensor
    copy_kv: Callable[[torch.Tensor, torch.Tensor], None]


def build_kv_slot_mapping(
    block_ids: Sequence[int],
    *,
    block_size: int,
    token_start: int,
    token_end: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build vLLM kernel slots for a contiguous logical token range.

    ``block_ids`` must already be in the attention kernel's block-id space.
    This helper is intentionally single-rank: context-parallel remapping is a
    later zero-copy/backend concern and must not be guessed here.
    """
    import torch

    if block_size <= 0:
        raise ValueError("KV block_size must be positive")
    if token_start < 0 or token_end < token_start:
        raise ValueError("invalid KV token range")
    if token_start == token_end:
        return torch.empty(0, dtype=torch.int64, device=device)
    if not block_ids:
        raise ValueError("KV block_ids cannot be empty for a non-empty range")

    positions = torch.arange(token_start, token_end, dtype=torch.int64, device=device)
    block_indices = torch.div(positions, block_size, rounding_mode="floor")
    if int(block_indices.max().item()) >= len(block_ids):
        raise ValueError(
            "KV token range exceeds the supplied block table: "
            f"last_block={int(block_indices.max().item())}, "
            f"num_blocks={len(block_ids)}"
        )
    blocks = torch.as_tensor(
        tuple(int(block) for block in block_ids), dtype=torch.int64, device=device
    )
    return blocks[block_indices] * block_size + torch.remainder(positions, block_size)
