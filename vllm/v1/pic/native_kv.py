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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import Any

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
        # A span that does not end on a page boundary still needs the page it ends inside, because the
        # tokens in that page are published with it. Rounding down charged that partial page to the
        # recompute on every reuse.
        if self.reusable_token_count <= 0:
            return 0
        return (self.reusable_token_count + self.block_size - 1) // self.block_size


@dataclass(frozen=True)
class PICWindow:
    """A token range together with the unit it is counted in.

    The same window is described to the scheduler (which allocates pages), to the kernels (which count
    kernel blocks), to the transition (which counts chunk tokens) and to the wire (which carries token
    counts). Carrying the unit with the value is what keeps those four descriptions from being derived
    independently - they only agreed before because a page-aligned window divides evenly in all of them.
    """

    token_start: int
    token_end: int
    block_size: int

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("PIC window block size must be positive")
        if self.token_start < 0 or self.token_end < self.token_start:
            raise ValueError("PIC window range is not a valid token range")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    def blocks(self) -> int:
        """Blocks needed to cover the window, rounding up: the page it ends inside counts."""
        if self.token_count <= 0:
            return 0
        return (self.token_count + self.block_size - 1) // self.block_size

    def blocks_at(self, block_size: int) -> int:
        """Same window counted in another unit."""
        return PICWindow(self.token_start, self.token_end, block_size).blocks()

    def fits(self, blocks: int) -> bool:
        """Whether `blocks` blocks reach the end of this window.

        The single form of the question every consumer asks: not "is the count equal" but "does it
        reach". Equality only held while windows ended on a page boundary.
        """
        return int(blocks) >= self.blocks()

    def covers(self, other: "PICWindow") -> bool:
        """Whether this window reaches at least as far as ``other``."""
        return self.token_start <= other.token_start and self.token_end >= other.token_end

    def relative_to(self, origin: int) -> "PICWindow":
        """The same window expressed against another origin."""
        return PICWindow(
            self.token_start - origin, self.token_end - origin, self.block_size
        )


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


def get_covering_local_block_span(
    segment_token_count: int, block_size: int
) -> PICLocalBlockSpan:
    """Return a span that reaches the segment's last token, partial page included."""
    if segment_token_count < 0:
        raise ValueError("PIC segment token count must be non-negative")
    if block_size <= 0:
        raise ValueError("PIC native block size must be positive")
    return PICLocalBlockSpan(
        segment_token_count=segment_token_count,
        block_size=block_size,
        reusable_start=0,
        reusable_end=segment_token_count,
    )


def get_allocator_block_count(token_count: int, allocator_block_size: int) -> int:
    """Return allocator-granularity blocks needed to cover a token prefix.

    PIC native-KV references may describe kernel blocks while scheduler-owned
    request block IDs remain in allocator granularity. Keep this conversion
    explicit at the boundary where request-owned IDs are selected.
    """
    if token_count < 0:
        raise ValueError("PIC token count must be non-negative")
    if allocator_block_size <= 0:
        raise ValueError("PIC allocator block size must be positive")
    return (token_count + allocator_block_size - 1) // allocator_block_size


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


def is_pic_exact_position_aliasable(
    segment_start: int, *, canonical_start: int = 0
) -> bool:
    """Return whether a segment can alias request-native KV safely.

    Canonical PIC KV is stored with a segment-local position origin.  A
    request block can therefore be used as the canonical source only when its
    request position already equals that origin.  The conservative first
    implementation handles the common prefix case (start == 0); shifted or
    interior segments must continue through the canonical public copy and
    later RoPE rerotation path.
    """
    return int(segment_start) == int(canonical_start)


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
    ) -> PICNativeKVLease:
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
    ) -> PICNativeKVLease:
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
    # The scheduler lookup already fetched the manifest.  Carry its small,
    # immutable descriptor across the scheduler/worker boundary so the worker
    # does not issue the same Mooncake metadata Get again during import.
    external_manifest: Mapping[str, Any] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_ids", tuple(int(block_id) for block_id in self.block_ids)
        )
        if self.kv_cache_group_id < 0:
            raise ValueError("PIC native KV group ID must be non-negative")
        if self.block_size <= 0 or self.token_count <= 0:
            raise ValueError(
                "PIC native KV block size and token count must be positive"
            )
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
        if self.external_manifest is not None:
            if not isinstance(self.external_manifest, Mapping):
                raise ValueError("external PIC KV manifest must be a mapping")
            object.__setattr__(self, "external_manifest", dict(self.external_manifest))
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
            expected_blocks = (
                ceil((source_token_start + self.token_count) / self.block_size)
                - source_block_start
            )
            # Coverage, not equality: one more block than the range divides into is the page its
            # end falls inside, and no reader reaches past the range.
            if len(self.block_ids) < expected_blocks:
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

    def with_lease(self, lease: PICNativeKVLease) -> PICNativeKVReference:
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
            external_manifest=self.external_manifest,
        )

    def attach_to_block_pool(self, block_pool: Any) -> PICNativeKVReference:
        """Pin this reference in a vLLM ``BlockPool`` for request lifetime."""
        return self.with_lease(
            PICNativeKVLease.from_block_pool(block_pool, self.block_ids)
        )

    def without_lease(self) -> PICNativeKVReference:
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
            external_manifest=self.external_manifest,
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
    slot_mapping: torch.Tensor
    block_table_compatible: bool
    # The position the stored key is already rotated to. A published object is stored at canonical
    # position zero, so moving it to a real position is one rotation by that position; a key captured
    # in this process is stored at the position it was produced at. It has a default, so it sits after
    # every required field.
    source_origin: int = 0
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
            if self.target_start % self.block_size != 0:
                raise ValueError(
                    "PIC public allocation must start on a block boundary"
                )
            # The end may fall inside a page: the page it lands in is allocated whole and read only
            # as far as the window reaches, which is the same state every request's last page is in.
            window = PICWindow(self.target_start, self.target_end, self.block_size)
            # Coverage, not equality: what matters is that the blocks reach the end of the range.
            # Requiring an exact count made a rounding difference - one block more or less than the
            # range divides into - look like a corrupt allocation, and the answer to that was to stop
            # reusing the segment altogether.
            if not window.fits(len(public)):
                raise ValueError(
                    "PIC public allocation does not cover the target range: "
                    f"blocks={len(public)} expected={window.blocks()}"
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
        first_complete = (
            (self.target_start + self.block_size - 1) // self.block_size
        ) * self.block_size
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
    ) -> PICNativeKVAllocation:
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
            slot_plan.target_end + slot_plan.block_size - 1
        ) // slot_plan.block_size
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
    device: torch.device | str,
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

    if target_start < 0:
        raise ValueError("PIC native KV target start must be non-negative")
    effective_local_start = (
        reference.canonical_start if local_start is None else int(local_start)
    )
    effective_token_count = (
        reference.token_count if token_count is None else int(token_count)
    )
    if (
        effective_local_start < reference.canonical_start
        or effective_token_count <= 0
        or effective_local_start + effective_token_count > reference.canonical_end
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
    assert reference.source_token_start is not None
    assert reference.source_block_start is not None
    source_token_start = reference.source_token_start
    source_block_start = reference.source_block_start

    local_positions = torch.arange(
        effective_local_start,
        effective_local_start + effective_token_count,
        dtype=torch.int64,
        device=device,
    )
    source_positions = source_token_start + (
        local_positions - reference.canonical_start
    )
    allocator_block_indices = (
        torch.div(source_positions, allocator_block_size, rounding_mode="floor")
        - source_block_start
    )
    physical_allocator_blocks = torch.as_tensor(
        reference.block_ids, dtype=torch.int64, device=device
    )
    physical_kernel_block_indices = (
        physical_allocator_blocks[allocator_block_indices] * blocks_per_allocator_block
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
                effective_token_count
                // allocator_block_size
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
        source_allocator_indices = (
            torch.div(
                source_block_positions,
                allocator_block_size,
                rounding_mode="floor",
            )
            - source_block_start
        )
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
        source_origin=(
            0
            if getattr(reference, "kind", None) is PICNativeKVKind.CANONICAL_PUBLIC
            else source_range_start
        ),
        block_size=kernel_block_size,
        logical_block_ids=logical_blocks,
        physical_block_ids=physical_blocks,
        slot_mapping=slot_mapping,
        block_table_compatible=aligned,
        requires_private_materialization=not aligned,
        fallback_reason=reason,
    )


def gather_native_kv_slots(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    block_size: int,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V vectors from an existing vLLM native KV cache.

    The helper deliberately handles only the unquantized five-dimensional
    layouts used by the stock attention backends.  Quantized caches and
    backend-specific layouts must fall back until they provide an explicit
    dequantize/gather contract; copying their raw bytes would be incorrect.
    """

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


# The rotary table and the relative-angle rows are the same for every layer of a request, so they are
# computed once and reused. Keying the table on the device and dtype keeps a model that mixes dtypes
# from sharing the wrong copy.
_PIC_ROTARY_TABLE: dict[tuple[str, torch.dtype, int], torch.Tensor] = {}
_PIC_DELTA_ROWS: dict[tuple[int, int, int, str, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]] = {}
_PIC_DELTA_ROWS_LIMIT = 64


def _pic_rotary_table(rotary_emb: Any, key: torch.Tensor) -> torch.Tensor:
    """The model's cos/sin table, on the key's device and dtype, transferred at most once."""
    cache = rotary_emb.cos_sin_cache
    entry_key = (str(key.device), key.dtype, int(cache.data_ptr()))
    table = _PIC_ROTARY_TABLE.get(entry_key)
    if table is None:
        table = cache.to(device=key.device, dtype=key.dtype)
        if len(_PIC_ROTARY_TABLE) >= 8:
            _PIC_ROTARY_TABLE.clear()
        _PIC_ROTARY_TABLE[entry_key] = table
    return table


def _pic_delta_rows(
    table: torch.Tensor,
    *,
    source_start: int,
    target_start: int,
    token_count: int,
    rotary_emb: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for the angle that moves one window onto the other, built once per request.

    A position p carries the angle ``p * inv_freq``, so the angle needed to go from a source position
    to a target position is the angle of their difference - the table already holds exactly that on
    the row named by the difference. Every token in the window shares the same difference, so this is
    one row broadcast over the window rather than a per-token table.
    """
    delta = int(target_start) - int(source_start)
    rows_key = (delta, token_count, 0, str(table.device), table.dtype, int(rotary_emb.rotary_dim))
    cached = _PIC_DELTA_ROWS.get(rows_key)
    if cached is not None:
        return cached
    if delta < 0:
        # A negative difference is the same rotation with the sine negated; keeping the table lookup
        # non-negative avoids depending on the table's indexing behaviour at negative positions.
        row = table.index_select(0, torch.tensor([-delta], device=table.device, dtype=torch.int64))
        cos, sin = row.chunk(2, dim=-1)
        sin = -sin
    else:
        row = table.index_select(0, torch.tensor([delta], device=table.device, dtype=torch.int64))
        cos, sin = row.chunk(2, dim=-1)
    if token_count > 1:
        cos = cos.expand(token_count, -1)
        sin = sin.expand(token_count, -1)
    cos = cos.contiguous()
    sin = sin.contiguous()
    if len(_PIC_DELTA_ROWS) >= _PIC_DELTA_ROWS_LIMIT:
        _PIC_DELTA_ROWS.clear()
    _PIC_DELTA_ROWS[rows_key] = (cos, sin)
    return cos, sin


# An operator whose device tensors are still being copied is not readable yet. The copy runs on the
# copy engine while the head of the prompt computes, so the wait belongs where the operator is first
# read - not where it was uploaded.
_PIC_PENDING_UPLOADS: dict[int, Any] = {}


def register_pending_upload(operator: Any, event: Any) -> None:
    """Remember that `operator`'s tensors become readable when `event` completes."""
    if event is not None:
        _PIC_PENDING_UPLOADS[id(operator)] = event


def wait_pending_upload(operator: Any, *, device: Any = None) -> None:
    """Wait on the copying stream before the operator is read, if a copy is outstanding.

    `wait_event` is a device-side wait, so the host keeps submitting work while the copy finishes.
    """
    event = _PIC_PENDING_UPLOADS.pop(id(operator), None)
    if event is None:
        return
    try:
        torch.cuda.current_stream(device).wait_event(event)
    except Exception:  # noqa: BLE001 - a non-CUDA build has no copy to wait for
        pass


_PIC_ROTARY_OPS: dict[bool, object] = {}


def _pic_rotary_op(is_neox: bool):
    """The registered rotary operator, built once per style.

    `ApplyRotaryEmb` is a custom op: calling it runs the fused kernel on CUDA and falls back to the
    Torch implementation elsewhere. The static helper is the fallback on its own - eight elementwise
    kernels and two concatenations per call, which is the slow way to do this on a GPU.
    """
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    op = _PIC_ROTARY_OPS.get(is_neox)
    if op is None:
        op = ApplyRotaryEmb(enforce_enable=True, is_neox_style=is_neox)
        _PIC_ROTARY_OPS[is_neox] = op
    return op


def _pic_rotate(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox: bool
) -> torch.Tensor:
    """Rotate `x` with the fused kernel when it is available."""
    try:
        return _pic_rotary_op(is_neox)(x, cos, sin)
    except Exception:  # noqa: BLE001 - the Torch path is correct, only slower
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        return ApplyRotaryEmb.forward_static(x, cos, sin, is_neox)


def rerotate_native_key(
    key: torch.Tensor,
    *,
    rotary_emb: Any,
    source_positions: torch.Tensor | None = None,
    target_positions: torch.Tensor | None = None,
    source_start: int | None = None,
    target_start: int | None = None,
    token_count: int | None = None,
) -> torch.Tensor:
    """Move a cached RoPE key from source to target absolute positions.

    vLLM stores the already-rotated key, so a private request row must be rotated again when the
    segment lands at a different absolute position. Rotating back by the source angle and forward by
    the target angle is the same as rotating once by their difference - ``R(t)·R(-s) = R(t-s)`` - and
    the difference is a single angle shared by the whole window, so the table is one row rather than
    one row per token. Doing it in one step also removes a bf16 round trip: the cached key is
    perturbed once, not twice.

    A window that did not move is returned untouched, which is what a segment anchored at position
    zero looks like.
    """

    if key.ndim != 3:
        raise ValueError("PIC native KV key must have shape [tokens, heads, dim]")

    # A caller that knows the window supplies it as integers, which is what these facts are: whether
    # the window moved, by how much, and whether every token moved by the same amount. Deriving them
    # from device tensors needs the answers on the host, and a segment's layers all ask the same
    # question - so the cheap form is also the correct form to prefer.
    if source_start is not None and target_start is not None:
        count = int(key.shape[0]) if token_count is None else int(token_count)
        first_source = int(source_start)
        first_target = int(target_start)
        moved = first_source != first_target
        if not moved:
            return key.contiguous()
        rows = int(rotary_emb.cos_sin_cache.shape[0])
        if abs(first_target - first_source) >= rows:
            raise ValueError("PIC RoPE position exceeds the rotary cache")
        table = _pic_rotary_table(rotary_emb, key)
        cos, sin = _pic_delta_rows(
            table,
            source_start=first_source,
            target_start=first_target,
            token_count=count,
            rotary_emb=rotary_emb,
        )
        rotary_dim = int(rotary_emb.rotary_dim)
        is_neox = bool(rotary_emb.is_neox_style)
        cos = cos.to(key.dtype)
        sin = sin.to(key.dtype)
        if rotary_dim >= key.shape[-1]:
            return _pic_rotate(key, cos, sin, is_neox).contiguous()
        target = key if key.is_contiguous() else key.contiguous()
        rotated = target[..., :rotary_dim].contiguous()
        target[..., :rotary_dim] = _pic_rotate(rotated, cos, sin, is_neox)
        return target

    if source_positions is None or target_positions is None:
        raise ValueError("PIC RoPE rerotation needs either positions or a window")
    source_positions = source_positions.to(device=key.device, dtype=torch.long)
    target_positions = target_positions.to(device=key.device, dtype=torch.long)
    if (
        source_positions.numel() != key.shape[0]
        or target_positions.numel() != key.shape[0]
    ):
        raise ValueError("PIC RoPE position count does not match the KV key")

    # Reached only by a caller that hands over position vectors instead of a window. The production
    # path names the window in integers, because the answers to the questions below are integers the
    # caller already holds; reading them back out of device vectors costs a synchronisation per layer
    # and cannot differ between the layers of a segment.
    # A hit that lands where it was captured needs no rotation at all, and skipping it keeps the
    # cached values bit-exact.
    if torch.equal(source_positions, target_positions):
        return key.contiguous()

    required = ("cos_sin_cache", "rotary_dim", "is_neox_style")
    if any(not hasattr(rotary_emb, name) for name in required):
        raise ValueError("PIC private KV RoPE rerotation is unsupported by this layer")

    table = _pic_rotary_table(rotary_emb, key)
    if source_positions.numel():
        largest_shift = int((target_positions - source_positions).abs().max())
        if largest_shift >= int(table.shape[0]):
            raise ValueError("PIC RoPE position exceeds the rotary cache")

    uniform = bool(
        source_positions.numel()
        and torch.equal(
            target_positions - source_positions,
            torch.full_like(source_positions, int(target_positions[0] - source_positions[0])),
        )
    )
    if uniform:
        first_source = int(source_positions[0])
        first_target = int(target_positions[0])
        cos, sin = _pic_delta_rows(
            table,
            source_start=first_source,
            target_start=first_target,
            token_count=key.shape[0],
            rotary_emb=rotary_emb,
        )
    else:
        delta = target_positions - source_positions
        rows = table.index_select(0, delta.abs())
        cos, sin = rows.chunk(2, dim=-1)
        sin = torch.where(
            (delta < 0).unsqueeze(-1), -sin, sin
        )

    rotary_dim = int(rotary_emb.rotary_dim)
    is_neox = bool(rotary_emb.is_neox_style)
    cos = cos.to(key.dtype)
    sin = sin.to(key.dtype)

    if rotary_dim >= key.shape[-1]:
        return _pic_rotate(key, cos, sin, is_neox).contiguous()

    # Only the rotary part moves; the remaining channels are copied through untouched. Rotating the
    # slice and writing it back touches rotary_dim channels per token, where rebuilding the whole
    # tensor to concatenate the two halves would touch all of them twice and allocate a second copy
    # of the cache row - most of the traffic for a partial-rotary model, none of it useful.
    target = key if key.is_contiguous() else key.contiguous()
    rotary = target[..., :rotary_dim].contiguous()
    target[..., :rotary_dim] = _pic_rotate(rotary, cos, sin, is_neox)
    return target


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
        self, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:

        source = torch.arange(
            self.source_start, self.source_end, dtype=torch.int64, device=device
        )
        target = torch.arange(
            self.target_start, self.target_end, dtype=torch.int64, device=device
        )
        return source, target


def materialize_private_rope_key(
    public_key: torch.Tensor,
    plan: PICRoPERerotationPlan,
    rerotate: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
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
