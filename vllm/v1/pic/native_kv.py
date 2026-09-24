# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SGLang-style native KV slot bridge for PIC segments.

PIC full-attention KV is deliberately represented by vLLM-native block IDs.
The independent :class:`PICPhysicalPool` is reserved for recurrent, conv-tail,
and transition state; it is not an attention KV backing store.

This module builds the control-plane mapping and provides the small Torch
gather/rerotation bridge used when a hit cannot be attached as whole request
blocks.  It does not change vLLM's scheduler progress or attention kernels.
The worker either aliases complete blocks or materializes the hit into the
current request's ordinary private KV row before the existing backend runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class PICNativeKVKind(str, Enum):
    """Position/storage semantics of a native KV reference."""

    SHARED = "shared"
    CANONICAL_PUBLIC = "canonical_public"
    PRIVATE = "private"


class PICKVPositionMode(str, Enum):
    """Coordinate system used by the cached segment."""

    CANONICAL_LOCAL = "canonical_local"
    REQUEST_ABSOLUTE = "request_absolute"


@dataclass(frozen=True)
class PICLocalBlockSpan:
    """Complete native blocks covered by one PIC segment in local space.

    PIC segments are independent objects: their block origin is the beginning
    of the segment, not the beginning of the request that happens to consume
    them.  The incomplete suffix is deliberately excluded from the reusable
    span and must be recomputed by the current request.
    """

    segment_token_count: int
    block_size: int
    reusable_start: int
    reusable_end: int

    @property
    def reusable_token_count(self) -> int:
        return self.reusable_end - self.reusable_start

    @property
    def block_count(self) -> int:
        return self.reusable_token_count // self.block_size


def get_full_local_block_span(
    segment_token_count: int, block_size: int
) -> PICLocalBlockSpan:
    """Return the complete block prefix of a segment-local token sequence."""
    if segment_token_count < 0:
        raise ValueError("PIC segment token count must be non-negative")
    if block_size <= 0:
        raise ValueError("PIC native block size must be positive")
    reusable_end = (segment_token_count // block_size) * block_size
    return PICLocalBlockSpan(
        segment_token_count=segment_token_count,
        block_size=block_size,
        reusable_start=0,
        reusable_end=reusable_end,
    )


def is_pic_public_segment_cacheable(segment_index: int, segment_count: int) -> bool:
    """Return whether a segment may receive a reusable public KV allocation.

    Like SGLang PIC, a multi-segment request keeps its final segment as the
    active continuation and does not publish it.  A single-segment warmup is
    cacheable, because a later request may consume that segment as an interior
    reusable segment.
    """
    if segment_count <= 0 or segment_index < 0 or segment_index >= segment_count:
        raise ValueError("PIC segment index/count is invalid")
    return segment_count == 1 or segment_index < segment_count - 1


@dataclass
class PICNativeKVLease:
    """A ref-count lease over vLLM-native KV blocks.

    ``BlockPool.touch`` removes blocks from the eviction queue and increments
    their ref-count.  The matching ``free_blocks`` call is delayed until the
    PIC entry is released.  The callbacks keep this module usable with a
    small fake pool in unit tests without coupling the metadata object to a
    concrete scheduler process.
    """

    block_ids: tuple[int, ...]
    _release_fn: Callable[[], None]
    _released: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_block_pool(
        cls,
        block_pool: Any,
        block_ids: Sequence[int],
    ) -> "PICNativeKVLease":
        ids = tuple(int(block_id) for block_id in block_ids)
        if not ids:
            raise ValueError("PIC native KV lease requires at least one block")
        if len(set(ids)) != len(ids):
            raise ValueError("PIC native KV lease block IDs must be unique")
        try:
            blocks = tuple(block_pool.blocks[block_id] for block_id in ids)
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("invalid vLLM native KV block ID") from exc
        block_pool.touch(blocks)

        def release() -> None:
            block_pool.free_blocks(blocks)

        return cls(ids, release)

    @classmethod
    def from_owned_block_pool(
        cls,
        block_pool: Any,
        block_ids: Sequence[int],
    ) -> "PICNativeKVLease":
        """Adopt blocks allocated by the caller without touching twice.

        ``BlockPool.get_new_blocks`` already gives the allocator owner one
        reference.  Public PIC blocks use that reference as their cache lease;
        calling :meth:`from_block_pool` here would increment the count again
        and leak one reference on every publication.
        """
        ids = tuple(int(block_id) for block_id in block_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("owned PIC native KV blocks must be non-empty and unique")
        try:
            blocks = tuple(block_pool.blocks[block_id] for block_id in ids)
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("invalid vLLM native KV block ID") from exc

        def release() -> None:
            block_pool.free_blocks(blocks)

        return cls(ids, release)

    def release(self) -> bool:
        """Release the lease once; repeated release is harmless."""
        if self._released:
            return False
        self._release_fn()
        self._released = True
        return True


@dataclass(frozen=True)
class PICNativeKVReference:
    """Native KV blocks containing one segment in local/canonical order."""

    kv_cache_group_id: int
    block_ids: tuple[int, ...]
    block_size: int
    token_count: int
    kind: PICNativeKVKind = PICNativeKVKind.CANONICAL_PUBLIC
    position_mode: PICKVPositionMode = PICKVPositionMode.CANONICAL_LOCAL
    canonical_start: int = 0
    # Physical source coordinates are optional because scheduler-side entries
    # normally carry only canonical/local metadata.  Worker-created entries may
    # retain the absolute source range used to build the reference.  This lets
    # us describe an arbitrary segment without confusing source block offsets
    # with the segment-local origin.
    source_token_start: int | None = None
    source_block_start: int | None = None
    dtype: str | None = None
    layout: tuple[int, ...] = ()
    lease: PICNativeKVLease | None = field(default=None, compare=False, repr=False)
    # External references are scheduler metadata only.  They intentionally do
    # not carry local block IDs until a worker-local provider imports them.
    external_provider: str | None = None
    external_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_ids", tuple(int(block_id) for block_id in self.block_ids)
        )
        if self.kv_cache_group_id < 0:
            raise ValueError("PIC native KV group ID must be non-negative")
        if self.block_size <= 0 or self.token_count <= 0:
            raise ValueError("PIC native KV block size and token count must be positive")
        if self.canonical_start < 0:
            raise ValueError("PIC native KV canonical start must be non-negative")
        source_token_start = (
            self.canonical_start
            if self.source_token_start is None
            else self.source_token_start
        )
        source_block_start = (
            source_token_start // self.block_size
            if self.source_block_start is None
            else self.source_block_start
        )
        if source_token_start < 0 or source_block_start < 0:
            raise ValueError("PIC native KV source coordinates must be non-negative")
        if (self.external_provider is None) != (self.external_key is None):
            raise ValueError(
                "external PIC KV references require both provider and object key"
            )
        if self.is_external:
            if self.block_ids:
                raise ValueError(
                    "external PIC KV references cannot contain local block IDs"
                )
            if self.lease is not None:
                raise ValueError(
                    "external PIC KV references cannot carry a local lease"
                )
        else:
            expected_blocks = ceil(
                (source_token_start + self.token_count) / self.block_size
            ) - source_block_start
            if len(self.block_ids) != expected_blocks:
                raise ValueError(
                    "PIC native KV reference does not cover the canonical range: "
                    f"expected_blocks={expected_blocks}, actual={len(self.block_ids)}"
                )
        if any(block_id < 0 for block_id in self.block_ids):
            raise ValueError("PIC native KV block IDs must be non-negative")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("PIC native KV block IDs must be unique")
        if self.position_mode == PICKVPositionMode.REQUEST_ABSOLUTE:
            raise ValueError(
                "request-absolute PIC KV cannot be published as a reusable entry"
            )

        object.__setattr__(self, "source_token_start", source_token_start)
        object.__setattr__(self, "source_block_start", source_block_start)

    @property
    def canonical_end(self) -> int:
        return self.canonical_start + self.token_count

    @property
    def is_external(self) -> bool:
        return self.external_provider is not None

    def release(self) -> bool:
        return self.lease.release() if self.lease is not None else False

    def with_lease(self, lease: PICNativeKVLease) -> "PICNativeKVReference":
        if lease.block_ids != self.block_ids:
            raise ValueError("native KV lease does not match reference block IDs")
        return PICNativeKVReference(
            kv_cache_group_id=self.kv_cache_group_id,
            block_ids=self.block_ids,
            block_size=self.block_size,
            token_count=self.token_count,
            kind=self.kind,
            position_mode=self.position_mode,
            canonical_start=self.canonical_start,
            source_token_start=self.source_token_start,
            source_block_start=self.source_block_start,
            dtype=self.dtype,
            layout=self.layout,
            lease=lease,
            external_provider=self.external_provider,
            external_key=self.external_key,
        )
    def attach_to_block_pool(self, block_pool: Any) -> "PICNativeKVReference":
        """Pin this reference in a vLLM ``BlockPool`` for request lifetime."""
        return self.with_lease(
            PICNativeKVLease.from_block_pool(block_pool, self.block_ids)
        )

    def without_lease(self) -> "PICNativeKVReference":
        """Return the serializable scheduler-side form of this reference."""
        return PICNativeKVReference(
            kv_cache_group_id=self.kv_cache_group_id,
            block_ids=self.block_ids,
            block_size=self.block_size,
            token_count=self.token_count,
            kind=self.kind,
            position_mode=self.position_mode,
            canonical_start=self.canonical_start,
            source_token_start=self.source_token_start,
            source_block_start=self.source_block_start,
            dtype=self.dtype,
            layout=self.layout,
            external_provider=self.external_provider,
            external_key=self.external_key,
        )

    def validate_layout(
        self,
        *,
        kv_cache_group_id: int,
        block_size: int,
        dtype: str | None = None,
        layout: tuple[int, ...] = (),
    ) -> None:
        """Reject a reference that cannot be consumed by a KV cache group."""
        if self.kv_cache_group_id != kv_cache_group_id:
            raise ValueError("PIC native KV group ID does not match target group")
        if self.block_size != block_size:
            raise ValueError("PIC native KV block size does not match target group")
        if dtype is not None and self.dtype is not None and self.dtype != dtype:
            raise ValueError("PIC native KV dtype does not match target group")
        if layout and self.layout and self.layout != layout:
            raise ValueError("PIC native KV layout does not match target group")


def has_complete_native_kv_reference(
    references: Sequence[PICNativeKVReference],
    *,
    kv_cache_group_id: int,
    block_count: int,
    token_count: int,
) -> bool:
    """Return whether a cache entry has the complete native KV span.

    A scheduler hit is not sufficient for the 9-A runtime: older entries may
    contain only recurrent/transition state and therefore have no native KV
    reference.  Those entries must be public-materialized before they can be
    used by the native-slot runtime.
    """
    return any(
        reference.kv_cache_group_id == kv_cache_group_id
        and len(reference.block_ids) == block_count
        and reference.token_count == token_count
        for reference in references
    )


@dataclass(frozen=True)
class PICNativeKVSlotPlan:
    """Target-request slot mapping for one canonical native KV reference."""

    kv_cache_group_id: int
    target_start: int
    target_end: int
    source_start: int
    block_size: int
    logical_block_ids: tuple[int, ...]
    physical_block_ids: tuple[int, ...]
    slot_mapping: "torch.Tensor"
    block_table_compatible: bool
    requires_private_materialization: bool = False
    fallback_reason: str | None = None

    @property
    def token_count(self) -> int:
        return self.target_end - self.target_start

    @property
    def block_table(self) -> tuple[tuple[int, int], ...]:
        """Logical-block to native-block pairs for ``BlockTable.add_row``."""
        return tuple(zip(self.logical_block_ids, self.physical_block_ids))


@dataclass(frozen=True)
class PICNativeKVAllocation:
    """Allocation-time public/private native KV ownership record.

    ``public_block_ids`` identify immutable canonical blocks owned by the PIC
    cache.  ``private_block_ids`` identify the ordinary native blocks owned by
    the active request.  The two sets are deliberately represented separately
    so a request can be reattached after a scheduler row move without ever
    overwriting canonical public KV.

    The private tuple is the request's allocator row, not a new external
    storage pool.  This keeps attention on vLLM's native KV cache while making
    edge-prefix/edge-suffix handling explicit for unaligned ranges.
    """

    kv_cache_group_id: int
    target_start: int
    target_end: int
    block_size: int
    public_block_ids: tuple[int, ...] = ()
    private_block_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kv_cache_group_id < 0:
            raise ValueError("PIC native allocation group ID must be non-negative")
        if self.target_start < 0 or self.target_end <= self.target_start:
            raise ValueError("PIC native allocation target range is invalid")
        if self.block_size <= 0:
            raise ValueError("PIC native allocation block size must be positive")

        public = tuple(int(block_id) for block_id in self.public_block_ids)
        private = tuple(int(block_id) for block_id in self.private_block_ids)
        if any(block_id < 0 for block_id in public + private):
            raise ValueError("PIC native allocation block IDs must be non-negative")
        if len(public) != len(set(public)):
            raise ValueError("PIC public allocation block IDs must be unique")
        if len(private) != len(set(private)):
            raise ValueError("PIC private allocation block IDs must be unique")
        if public and private:
            raise ValueError(
                "PIC allocation cannot own public and private blocks at once"
            )
        if not public and not private:
            raise ValueError("PIC allocation must contain public or private blocks")
        if public:
            if self.target_start % self.block_size != 0 or self.target_end % self.block_size != 0:
                raise ValueError(
                    "PIC public allocation must cover complete native blocks"
                )
            expected_blocks = (self.target_end - self.target_start) // self.block_size
            if len(public) != expected_blocks:
                raise ValueError(
                    "PIC public allocation block count does not match target range"
                )
        object.__setattr__(self, "public_block_ids", public)
        object.__setattr__(self, "private_block_ids", private)

    @property
    def token_count(self) -> int:
        return self.target_end - self.target_start

    @property
    def is_public(self) -> bool:
        return bool(self.public_block_ids)

    @property
    def is_private(self) -> bool:
        return bool(self.private_block_ids)

    @property
    def edge_prefix_tokens(self) -> int:
        """Tokens before the first complete native block in the target range."""
        first_complete = ((self.target_start + self.block_size - 1)
                          // self.block_size) * self.block_size
        return max(0, min(self.target_end, first_complete) - self.target_start)

    @property
    def edge_suffix_tokens(self) -> int:
        """Tokens after the last complete native block in the target range."""
        last_complete = (self.target_end // self.block_size) * self.block_size
        return max(0, self.target_end - max(self.target_start, last_complete))

    @property
    def has_edge_tokens(self) -> bool:
        return self.edge_prefix_tokens > 0 or self.edge_suffix_tokens > 0

    @classmethod
    def from_slot_plan(
        cls,
        slot_plan: PICNativeKVSlotPlan,
        *,
        private_block_ids: Sequence[int] = (),
    ) -> "PICNativeKVAllocation":
        """Bind a slot plan to its allocation owner at attach time.

        Whole-block plans alias the public physical blocks.  All other plans
        bind the current request row and are copied/rerotated privately.  A
        shifted or partial segment therefore never aliases canonical public
        storage.
        """
        if slot_plan.block_table_compatible:
            if not slot_plan.physical_block_ids:
                raise ValueError("compatible PIC slot plan has no public blocks")
            return cls(
                kv_cache_group_id=slot_plan.kv_cache_group_id,
                target_start=slot_plan.target_start,
                target_end=slot_plan.target_end,
                block_size=slot_plan.block_size,
                public_block_ids=slot_plan.physical_block_ids,
            )

        private = tuple(int(block_id) for block_id in private_block_ids)
        required_blocks = (
            (slot_plan.target_end + slot_plan.block_size - 1)
            // slot_plan.block_size
        )
        if len(private) < required_blocks:
            raise ValueError(
                "PIC private allocation row does not cover target range: "
                f"required={required_blocks}, actual={len(private)}"
            )
        return cls(
            kv_cache_group_id=slot_plan.kv_cache_group_id,
            target_start=slot_plan.target_start,
            target_end=slot_plan.target_end,
            block_size=slot_plan.block_size,
            private_block_ids=private,
        )


@dataclass(frozen=True)
class PICNativeKVRequestMapping:
    """Persistent request-local logical-to-native KV mapping.

    This is the vLLM counterpart of SGLang's ``req_to_token`` row.  The
    mapping is built once from the request's runtime plan and remains attached
    to the request across packed rounds and decode steps.  The worker still
    attaches the mapping to the current ``BlockTable`` row each round because
    vLLM may reorder request rows, but it does not rebuild the PIC mapping.
    """

    slot_plans: tuple[PICNativeKVSlotPlan, ...]

    def __post_init__(self) -> None:
        plans = tuple(self.slot_plans)
        object.__setattr__(self, "slot_plans", plans)
        keys = tuple(
            (plan.kv_cache_group_id, plan.target_start, plan.target_end)
            for plan in plans
        )
        if len(keys) != len(set(keys)):
            raise ValueError(
                "PIC native request mapping contains duplicate logical ranges"
            )

    def for_group(self, group_id: int) -> tuple[PICNativeKVSlotPlan, ...]:
        """Return the persistent ranges for one KV cache group."""
        return tuple(
            plan for plan in self.slot_plans if plan.kv_cache_group_id == group_id
        )

    @property
    def is_empty(self) -> bool:
        return not self.slot_plans


def build_native_kv_slot_plan(
    reference: PICNativeKVReference,
    *,
    target_start: int,
    device: "torch.device | str",
    kernel_block_size: int | None = None,
    local_start: int | None = None,
    token_count: int | None = None,
) -> PICNativeKVSlotPlan:
    """Map canonical segment slots to a target request's native slots.

    vLLM's block table encodes ``logical_block -> physical_block``.  The KV
    manager block can be larger than the block consumed by the attention
    kernel (for example, Qwen3.5 uses a 544-token allocator block and a
    32-token attention block).  In that case both the logical and physical
    block IDs must be expanded exactly like ``BlockTable.append_row`` does.
    The returned token mapping always follows the segment-local coordinates.
    Whole-block attachment is checked at the same granularity as vLLM's
    ``BlockTable``.  When an allocator block is expanded into kernel blocks,
    the source and target only need to be aligned to the kernel block size;
    they may cross allocator-block boundaries.
    """
    import torch

    if target_start < 0:
        raise ValueError("PIC native KV target start must be non-negative")
    effective_local_start = (
        reference.canonical_start
        if local_start is None
        else int(local_start)
    )
    effective_token_count = (
        reference.token_count if token_count is None else int(token_count)
    )
    if (
        effective_local_start < reference.canonical_start
        or effective_token_count <= 0
        or effective_local_start + effective_token_count
        > reference.canonical_end
    ):
        raise ValueError("PIC native KV local subrange is outside the reference")
    allocator_block_size = reference.block_size
    kernel_block_size = (
        allocator_block_size if kernel_block_size is None else kernel_block_size
    )
    if kernel_block_size <= 0 or allocator_block_size % kernel_block_size != 0:
        raise ValueError(
            "PIC native KV kernel block size must divide allocator block size: "
            f"allocator={allocator_block_size}, kernel={kernel_block_size}"
        )
    blocks_per_allocator_block = allocator_block_size // kernel_block_size
    target_end = target_start + effective_token_count
    source_token_start = int(reference.source_token_start)
    source_block_start = int(reference.source_block_start)

    local_positions = torch.arange(
        effective_local_start,
        effective_local_start + effective_token_count,
        dtype=torch.int64,
        device=device,
    )
    source_positions = source_token_start + (
        local_positions - reference.canonical_start
    )
    allocator_block_indices = torch.div(
        source_positions, allocator_block_size, rounding_mode="floor"
    ) - source_block_start
    physical_allocator_blocks = torch.as_tensor(
        reference.block_ids, dtype=torch.int64, device=device
    )
    physical_kernel_block_indices = (
        physical_allocator_blocks[allocator_block_indices]
        * blocks_per_allocator_block
        + (source_positions % allocator_block_size) // kernel_block_size
    )
    slot_mapping = physical_kernel_block_indices * kernel_block_size + (
        source_positions % kernel_block_size
    )

    # BlockTable stores kernel-sized blocks (and expands allocator blocks into
    # those entries). For position-dependent KV (notably RoPE), kernel
    # alignment alone is not enough: an already-rotated source K is valid at
    # the source absolute position only. Direct aliasing is therefore limited
    # to the exact same source/target absolute range. All shifted hits use the
    # request-private materialization path, which can rerotate K safely.
    source_range_start = source_token_start + (
        effective_local_start - reference.canonical_start
    )
    # Canonical public K is position-dependent even when source and target
    # happen to start at the same absolute offset.  Keep the public allocation
    # immutable and always create a request-private view for it.  Direct alias
    # remains available only for explicitly position-invariant shared KV.
    position_invariant = reference.kind == PICNativeKVKind.SHARED
    aligned = (
        position_invariant
        and source_range_start % kernel_block_size == 0
        and target_start % kernel_block_size == 0
        and effective_token_count % kernel_block_size == 0
        and source_range_start == target_start
    )

    if aligned:
        first_logical_block = target_start // kernel_block_size
        logical_blocks = tuple(
            first_logical_block + index
            for index in range(
                effective_token_count // allocator_block_size
                * blocks_per_allocator_block
            )
        )
        num_kernel_blocks = effective_token_count // kernel_block_size
        source_block_positions = torch.arange(
            source_range_start,
            source_range_start + effective_token_count,
            kernel_block_size,
            dtype=torch.int64,
            device=device,
        )
        source_allocator_indices = torch.div(
            source_block_positions,
            allocator_block_size,
            rounding_mode="floor",
        ) - source_block_start
        source_kernel_blocks = (
            physical_allocator_blocks[source_allocator_indices]
            * blocks_per_allocator_block
            + (source_block_positions % allocator_block_size) // kernel_block_size
        )
        if source_kernel_blocks.numel() != num_kernel_blocks:
            raise ValueError("PIC native kernel block mapping has an unexpected size")
        physical_blocks = tuple(int(item) for item in source_kernel_blocks.tolist())
        reason = None
    else:
        logical_blocks = ()
        physical_blocks = ()
        reason = (
            "PIC native KV source and target positions differ or are not whole "
            "kernel blocks; use request-private native KV materialization"
        )

    return PICNativeKVSlotPlan(
        kv_cache_group_id=reference.kv_cache_group_id,
        target_start=target_start,
        target_end=target_end,
        source_start=source_range_start,
        block_size=kernel_block_size,
        logical_block_ids=logical_blocks,
        physical_block_ids=physical_blocks,
        slot_mapping=slot_mapping,
        block_table_compatible=aligned,
        requires_private_materialization=not aligned,
        fallback_reason=reason,
    )


def gather_native_kv_slots(
    kv_cache: "torch.Tensor",
    slot_mapping: "torch.Tensor",
    *,
    block_size: int,
    layout: str,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Gather K/V vectors from an existing vLLM native KV cache.

    The helper deliberately handles only the unquantized five-dimensional
    layouts used by the stock attention backends.  Quantized caches and
    backend-specific layouts must fall back until they provide an explicit
    dequantize/gather contract; copying their raw bytes would be incorrect.
    """
    import torch

    if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 5:
        raise ValueError("PIC private KV materialization requires a 5-D KV cache")
    if layout not in ("NHD", "HND"):
        raise ValueError(f"unsupported PIC native KV layout: {layout!r}")
    if block_size <= 0:
        raise ValueError("PIC native KV block size must be positive")
    if not kv_cache.dtype.is_floating_point or str(kv_cache.dtype).startswith(
        "torch.float8"
    ):
        raise ValueError(
            "PIC private KV materialization requires an unquantized floating cache"
        )

    if kv_cache.shape[0] == 2 and kv_cache.shape[1] != 2:
        key_cache, value_cache = kv_cache[0], kv_cache[1]
    elif kv_cache.shape[1] == 2 and kv_cache.shape[0] != 2:
        key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]
    else:
        raise ValueError("cannot identify the K/V dimension of the native KV cache")

    slots = slot_mapping.to(device=kv_cache.device, dtype=torch.int64).flatten()
    block_indices = torch.div(slots, block_size, rounding_mode="floor")
    offsets = torch.remainder(slots, block_size)
    if block_indices.numel() and int(block_indices.max()) >= key_cache.shape[0]:
        raise ValueError("PIC native KV source slot exceeds the cache capacity")

    if layout == "NHD":
        key = key_cache[block_indices, offsets]
        value = value_cache[block_indices, offsets]
    else:
        key = key_cache.permute(0, 2, 1, 3)[block_indices, offsets]
        value = value_cache.permute(0, 2, 1, 3)[block_indices, offsets]
    return key.contiguous(), value.contiguous()


def rerotate_native_key(
    key: "torch.Tensor",
    *,
    rotary_emb: Any,
    source_positions: "torch.Tensor",
    target_positions: "torch.Tensor",
) -> "torch.Tensor":
    """Move a cached RoPE key from source to target absolute positions.

    vLLM stores the already-rotated key in its native cache.  For a private
    request row, copying that tensor verbatim is wrong when the segment starts
    at a different absolute position.  The stock Qwen/standard RoPE object
    exposes the same cosine/sine cache used by the model, so we invert the
    source rotation and apply the target rotation with the exact native Torch
    primitive.  Rotary variants without this contract safely fall back.
    """
    import torch

    if key.ndim != 3:
        raise ValueError("PIC native KV key must have shape [tokens, heads, dim]")
    source_positions = source_positions.to(device=key.device, dtype=torch.long)
    target_positions = target_positions.to(device=key.device, dtype=torch.long)
    if source_positions.numel() != key.shape[0] or target_positions.numel() != key.shape[0]:
        raise ValueError("PIC RoPE position count does not match the KV key")

    # Canonical public materialization may be followed by a request-private
    # materialization at the same absolute positions (for example, a hit at
    # position zero).  Applying inverse RoPE and forward RoPE in BF16 is not
    # bit-exact and needlessly perturbs the cached K.  Preserve the native
    # values exactly when the position vectors are identical.
    if torch.equal(source_positions, target_positions):
        return key.contiguous()

    required = ("cos_sin_cache", "rotary_dim", "is_neox_style")
    if any(not hasattr(rotary_emb, name) for name in required):
        raise ValueError("PIC private KV RoPE rerotation is unsupported by this layer")

    cache = rotary_emb.cos_sin_cache.to(device=key.device, dtype=key.dtype)
    if source_positions.numel():
        max_position = cache.shape[0]
        if int(source_positions.max()) >= max_position or int(target_positions.max()) >= max_position:
            raise ValueError("PIC RoPE position exceeds the rotary cache")
    source_cos, source_sin = cache.index_select(0, source_positions).chunk(2, dim=-1)
    target_cos, target_sin = cache.index_select(0, target_positions).chunk(2, dim=-1)

    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    rotary_dim = int(rotary_emb.rotary_dim)
    rotated = key[..., :rotary_dim]
    passthrough = key[..., rotary_dim:]
    rotated = ApplyRotaryEmb.forward_static(
        rotated,
        source_cos,
        -source_sin,
        bool(rotary_emb.is_neox_style),
    )
    rotated = ApplyRotaryEmb.forward_static(
        rotated,
        target_cos,
        target_sin,
        bool(rotary_emb.is_neox_style),
    )
    return torch.cat((rotated, passthrough), dim=-1).contiguous()


@dataclass(frozen=True)
class PICRoPERerotationPlan:
    """Position plan for materializing canonical K into private native slots."""

    token_count: int
    source_start: int
    target_start: int

    @property
    def source_end(self) -> int:
        return self.source_start + self.token_count

    @property
    def target_end(self) -> int:
        return self.target_start + self.token_count

    def positions(
        self, device: "torch.device | str"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        source = torch.arange(
            self.source_start, self.source_end, dtype=torch.int64, device=device
        )
        target = torch.arange(
            self.target_start, self.target_end, dtype=torch.int64, device=device
        )
        return source, target


def materialize_private_rope_key(
    public_key: "torch.Tensor",
    plan: PICRoPERerotationPlan,
    rerotate: Callable[["torch.Tensor", "torch.Tensor", "torch.Tensor"], "torch.Tensor"],
) -> "torch.Tensor":
    """Apply a caller-provided RoPE transform without mutating public K.

    The model/backend supplies ``rerotate`` because rotary variants differ
    across models.  Cloning before the callback is intentional: a callback
    implemented with in-place Torch operations must still leave canonical
    public KV immutable.
    """
    source_positions, target_positions = plan.positions(public_key.device)
    private_key = rerotate(public_key.clone(), source_positions, target_positions)
    if private_key.data_ptr() == public_key.data_ptr():
        private_key = private_key.clone()
    return private_key
