# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage 8-A copy-back from PIC snapshots into vLLM's KV cache.

This module deliberately uses the attention implementation's existing
``do_kv_cache_update`` path.  It does not add a second attention layout, alter
the block table, or make the attention kernel read the PIC pool directly.
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
    slot_mapping: "torch.Tensor"
    copy_kv: Callable[["torch.Tensor", "torch.Tensor"], None]


def build_kv_slot_mapping(
    block_ids: Sequence[int],
    *,
    block_size: int,
    token_start: int,
    token_end: int,
    device: "torch.device | str",
) -> "torch.Tensor":
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
    blocks = torch.as_tensor(tuple(int(block) for block in block_ids),
                             dtype=torch.int64, device=device)
    return blocks[block_indices] * block_size + torch.remainder(positions, block_size)


class PICAttentionKVCacheCopyBack:
    """Restore a full-KV PIC snapshot through existing attention backends."""

    def __init__(self, store: PICSnapshotStore) -> None:
        self.store = store

    def copy_entry(
        self,
        segment: PICSegment,
        entry: PICSegmentEntry,
        targets: Sequence[PICKVLayerTarget],
    ) -> int:
        """Copy one snapshot into target layers and return copied layer count.

        The Stage 8-A snapshot contract is two tensors per target layer in
        target order: ``key_0, value_0, key_1, value_1, ...``.  The pool
        validates the original shape and dtype before the backend is called.
        """
        import torch

        if entry.seg_hash != segment.seg_hash or entry.token_ids != segment.token_ids:
            raise ValueError("PIC KV materialization does not match segment")
        if not entry.full_kv_handles:
            return 0
        if len(entry.full_kv_handles) != 1:
            raise ValueError("PIC KV copy-back requires exactly one full-KV handle")
        if not targets:
            raise ValueError("PIC KV copy-back requires at least one target layer")

        handle_id = entry.full_kv_handles[0]
        allocation = self.store.pool.get_allocation(handle_id)
        expected_tensor_count = 2 * len(targets)
        if len(allocation.regions) != expected_tensor_count:
            raise ValueError(
                "PIC KV snapshot tensor count does not match target layers: "
                f"expected={expected_tensor_count}, "
                f"actual={len(allocation.regions)}"
            )

        restored = tuple(
            torch.empty(
                region.shape,
                dtype=region.dtype,
                device=self.store.pool.device,
            )
            for region in allocation.regions
        )
        snapshot = PICPhysicalSnapshot(
            seg_hash=segment.seg_hash,
            token_start=segment.start,
            token_end=segment.end,
            full_kv_handles=(handle_id,),
        )
        self.store.restore(snapshot, full_kv=restored)

        token_count = segment.end - segment.start
        for index, target in enumerate(targets):
            if target.slot_mapping.numel() != token_count:
                raise ValueError(
                    "PIC KV slot mapping length does not match segment: "
                    f"layer={target.layer_name}, expected={token_count}, "
                    f"actual={target.slot_mapping.numel()}"
                )
            target.copy_kv(restored[2 * index], restored[2 * index + 1])
        return len(targets)
