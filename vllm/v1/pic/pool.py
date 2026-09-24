# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent device storage for PIC snapshots.

This is intentionally not a replacement for vLLM's ``BlockPool``.  PIC
segments can be non-prefix and therefore need an independent lifetime and
mapping layer.  The pool stores opaque tensor snapshots; attention backends
may later copy them into the model KV cache or consume their slot metadata.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from vllm.v1.pic.handles import PICHandle, PICHandleKind, PICHandlePool

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PICTensorRegion:
    offset_bytes: int
    size_bytes: int
    shape: tuple[int, ...]
    dtype: "torch.dtype"


@dataclass(frozen=True)
class PICPhysicalAllocation:
    """A physical allocation containing one or more tensor regions."""

    offset_bytes: int
    size_bytes: int
    regions: tuple[PICTensorRegion, ...]


@dataclass(frozen=True)
class PICSlotMapping:
    """Mapping from logical token positions to independent PIC slots."""

    logical_positions: tuple[int, ...]
    physical_slots: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.logical_positions) != len(self.physical_slots):
            raise ValueError(
                "logical and physical PIC slot mappings must have equal size"
            )
        if len(set(self.logical_positions)) != len(self.logical_positions):
            raise ValueError("logical PIC positions must be unique")
        if len(set(self.physical_slots)) != len(self.physical_slots):
            raise ValueError("physical PIC slots must be unique")

    @classmethod
    def contiguous(
        cls, logical_start: int, physical_slots: Sequence[int]
    ) -> "PICSlotMapping":
        return cls(
            logical_positions=tuple(
                logical_start + index for index in range(len(physical_slots))
            ),
            physical_slots=tuple(int(slot) for slot in physical_slots),
        )


class PICPhysicalPool:
    """Reference-counted byte-addressable device pool for PIC snapshots."""

    def __init__(
        self,
        capacity_bytes: int,
        *,
        device: "torch.device | str",
        alignment_bytes: int = 256,
        max_handles: int | None = None,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("PIC physical pool capacity must be positive")
        if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
            raise ValueError("PIC pool alignment must be a positive power of two")

        import torch

        self.capacity_bytes = int(capacity_bytes)
        self.alignment_bytes = int(alignment_bytes)
        self.device = torch.device(device)
        self.storage = torch.empty(
            self.capacity_bytes, dtype=torch.uint8, device=self.device
        )
        self.handles = PICHandlePool(max_handles=max_handles)
        self._free_ranges: list[tuple[int, int]] = [(0, self.capacity_bytes)]
        self._allocations: dict[int, PICPhysicalAllocation] = {}

    @property
    def free_bytes(self) -> int:
        return sum(size for _, size in self._free_ranges)

    def allocate_tensor_snapshot(
        self,
        tensors: Sequence["torch.Tensor"],
        *,
        kind: PICHandleKind,
        group_id: int = -1,
        token_start: int = 0,
        token_end: int = 0,
    ) -> PICHandle:
        """Allocate storage and copy a sequence of tensors into it."""
        import torch

        if not tensors:
            raise ValueError("PIC snapshot must contain at least one tensor")

        regions: list[PICTensorRegion] = []
        cursor = 0
        for tensor in tensors:
            if tensor.device != self.device:
                raise ValueError(
                    f"PIC tensor is on {tensor.device}, expected {self.device}"
                )
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            size_bytes = tensor.numel() * tensor.element_size()
            if size_bytes <= 0:
                raise ValueError("PIC snapshot tensors must have non-zero size")
            cursor = self._align_up(cursor)
            regions.append(
                PICTensorRegion(
                    offset_bytes=cursor,
                    size_bytes=size_bytes,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                )
            )
            cursor += size_bytes

        allocation = self._allocate_range(cursor, tuple(regions))
        try:
            handle = self.handles.allocate(
                kind,
                group_id=group_id,
                token_start=token_start,
                token_end=token_end,
                slot_ids=tuple(region.offset_bytes for region in regions),
                payload=allocation,
            )
        except Exception:
            self._free_range(allocation.offset_bytes, allocation.size_bytes)
            raise
        self._allocations[handle.handle_id] = allocation

        for tensor, region in zip(tensors, allocation.regions):
            self._copy_tensor_to_region(tensor, region)
        return handle

    def restore_tensor_snapshot(
        self, handle_id: int, tensors: Sequence["torch.Tensor"]
    ) -> None:
        """Restore a snapshot into tensors with the original shapes/dtypes."""
        import torch

        allocation = self._require_allocation(handle_id)
        if len(tensors) != len(allocation.regions):
            raise ValueError("PIC restore tensor count does not match allocation")
        for tensor, region in zip(tensors, allocation.regions):
            if tuple(tensor.shape) != region.shape or tensor.dtype != region.dtype:
                raise ValueError("PIC restore tensor shape or dtype does not match")
            source = self.storage[
                region.offset_bytes : region.offset_bytes + region.size_bytes
            ]
            if tensor.is_contiguous():
                tensor.reshape(-1).view(torch.uint8).copy_(source.reshape(-1))
            else:
                restored = source.view(region.dtype).reshape(region.shape)
                tensor.copy_(restored)

    def retain(self, handle_id: int) -> PICHandle:
        return self.handles.retain(handle_id)

    def release(self, handle_id: int) -> PICHandle | None:
        released = self.handles.release(handle_id)
        if released is not None:
            allocation = self._allocations.pop(handle_id)
            self._free_range(allocation.offset_bytes, allocation.size_bytes)
        return released

    def clear(self) -> None:
        self.handles.clear()
        self._allocations.clear()
        self._free_ranges = [(0, self.capacity_bytes)]

    def get_allocation(self, handle_id: int) -> PICPhysicalAllocation:
        return self._require_allocation(handle_id)

    def _allocate_range(
        self, size_bytes: int, regions: tuple[PICTensorRegion, ...]
    ) -> PICPhysicalAllocation:
        size_bytes = self._align_up(size_bytes)
        for index, (offset, available) in enumerate(self._free_ranges):
            aligned_offset = self._align_up(offset)
            padding = aligned_offset - offset
            if available - padding < size_bytes:
                continue

            before = padding
            after = available - padding - size_bytes
            replacement: list[tuple[int, int]] = []
            if before:
                replacement.append((offset, before))
            if after:
                replacement.append((aligned_offset + size_bytes, after))
            self._free_ranges[index : index + 1] = replacement
            absolute_regions = tuple(
                PICTensorRegion(
                    offset_bytes=aligned_offset + region.offset_bytes,
                    size_bytes=region.size_bytes,
                    shape=region.shape,
                    dtype=region.dtype,
                )
                for region in regions
            )
            return PICPhysicalAllocation(
                aligned_offset, size_bytes, absolute_regions
            )

        raise MemoryError(
            f"PIC physical pool exhausted: requested {size_bytes} bytes, "
            f"free {self.free_bytes} bytes"
        )

    def _free_range(self, offset: int, size_bytes: int) -> None:
        self._free_ranges.append((offset, size_bytes))
        self._free_ranges.sort()
        merged: list[tuple[int, int]] = []
        for current_offset, current_size in self._free_ranges:
            if merged and merged[-1][0] + merged[-1][1] == current_offset:
                prev_offset, prev_size = merged[-1]
                merged[-1] = (prev_offset, prev_size + current_size)
            else:
                merged.append((current_offset, current_size))
        self._free_ranges = merged

    def _copy_tensor_to_region(
        self, tensor: "torch.Tensor", region: PICTensorRegion
    ) -> None:
        import torch

        contiguous = tensor.contiguous()
        source = contiguous.view(torch.uint8).reshape(-1)
        target = self.storage[
            region.offset_bytes : region.offset_bytes + region.size_bytes
        ]
        target.copy_(source)

    def _require_allocation(self, handle_id: int) -> PICPhysicalAllocation:
        allocation = self._allocations.get(int(handle_id))
        if allocation is None:
            raise KeyError(f"unknown PIC physical allocation {handle_id}")
        return allocation

    def _align_up(self, value: int) -> int:
        mask = self.alignment_bytes - 1
        return (int(value) + mask) & ~mask
