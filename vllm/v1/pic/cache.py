# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent metadata cache and matching plan for PIC requests."""

from dataclasses import dataclass
from time import monotonic
from typing import Sequence

from vllm.v1.pic.native_kv import PICNativeKVReference
from vllm.v1.pic.segmenter import PICSegment


@dataclass
class PICSegmentEntry:
    """A reusable segment and its state/native-KV references.

    ``native_kv_refs`` contains canonical/local references into vLLM's native
    KV pool.  Full-attention KV is no longer represented by a
    ``PICPhysicalPool`` byte allocation.
    """

    seg_hash: bytes
    token_ids: tuple[int, ...]
    full_kv_handles: tuple[int, ...] = ()
    recurrent_state_handle: int | None = None
    transition_state_handle: int | None = None
    conv_tail_handle: int | None = None
    native_kv_refs: tuple[PICNativeKVReference, ...] = ()
    hit_count: int = 0
    last_access_time: float = 0.0

    @property
    def has_physical_handles(self) -> bool:
        """Whether this entry owns materialized backend state."""
        return bool(
            self.full_kv_handles
            or self.recurrent_state_handle is not None
            or self.transition_state_handle is not None
            or self.conv_tail_handle is not None
            or bool(self.native_kv_refs)
        )

    def release_native_kv(self) -> None:
        """Release native block leases owned by this cache entry."""
        for reference in self.native_kv_refs:
            reference.release()


@dataclass(frozen=True)
class PICCachePlan:
    """Per-request segment match result for the isolated PIC path."""

    segment_count: int
    reused_segments: tuple[int, ...]
    recompute_segments: tuple[int, ...]
    seam_tokens: tuple[tuple[int, int], ...]
    # The matching entries are carried to the worker in Stage 1.  Keeping the
    # index with the entry avoids a second hash lookup after multiprocessing.
    matches: tuple[tuple[int, PICSegmentEntry], ...] = ()


class PICSegmentCache:
    """Request-side segment index, isolated from vLLM prefix caching."""

    def __init__(self, enabled: bool = False, max_cache_bytes: int | None = None):
        self.enabled = enabled
        self.max_cache_bytes = max_cache_bytes
        self._entries: dict[bytes, PICSegmentEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        for entry in self._entries.values():
            entry.release_native_kv()
        self._entries.clear()

    def lookup(self, segment: PICSegment) -> PICSegmentEntry | None:
        if not self.enabled:
            return None
        entry = self._entries.get(segment.seg_hash)
        if entry is None or entry.token_ids != segment.token_ids:
            return None
        entry.hit_count += 1
        entry.last_access_time = monotonic()
        return entry

    def insert(
        self,
        segment: PICSegment,
        *,
        full_kv_handles: Sequence[int] = (),
        recurrent_state_handle: int | None = None,
        transition_state_handle: int | None = None,
        conv_tail_handle: int | None = None,
        native_kv_refs: Sequence[PICNativeKVReference] = (),
    ) -> PICSegmentEntry | None:
        """Insert a segment after the backend materializes its state."""
        if not self.enabled:
            return None
        existing = self._entries.get(segment.seg_hash)
        if existing is not None:
            if existing.token_ids != segment.token_ids:
                return None
            if full_kv_handles:
                existing.full_kv_handles = tuple(
                    int(handle) for handle in full_kv_handles
                )
            if recurrent_state_handle is not None:
                existing.recurrent_state_handle = recurrent_state_handle
            if transition_state_handle is not None:
                existing.transition_state_handle = transition_state_handle
            if conv_tail_handle is not None:
                existing.conv_tail_handle = conv_tail_handle
            if native_kv_refs:
                existing.release_native_kv()
                existing.native_kv_refs = tuple(native_kv_refs)
            existing.last_access_time = monotonic()
            return existing
        entry = PICSegmentEntry(
            seg_hash=segment.seg_hash,
            token_ids=segment.token_ids,
            full_kv_handles=tuple(int(handle) for handle in full_kv_handles),
            recurrent_state_handle=recurrent_state_handle,
            transition_state_handle=transition_state_handle,
            conv_tail_handle=conv_tail_handle,
            native_kv_refs=tuple(native_kv_refs),
            last_access_time=monotonic(),
        )
        self._entries[segment.seg_hash] = entry
        return entry

    def clear_native_kv(self, segment: PICSegment) -> None:
        """Drop a stale native reference while retaining other state handles."""
        entry = self._entries.get(segment.seg_hash)
        if entry is None:
            return
        entry.release_native_kv()
        entry.native_kv_refs = ()

    def evict_hash(self, seg_hash: bytes) -> bool:
        """Remove one worker-invalidated segment and release its native lease."""
        entry = self._entries.pop(seg_hash, None)
        if entry is None:
            return False
        entry.release_native_kv()
        return True

    def build_plan(
        self,
        segments: Sequence[PICSegment] | None,
        *,
        mode: str = "transition_rope_recompute",
        seam_sink: int = 8,
    ) -> PICCachePlan:
        """Match reusable segments without assuming a common prefix."""
        if not self.enabled or not segments:
            count = len(segments) if segments else 0
            return PICCachePlan(count, (), tuple(range(count)), (), ())

        hits: list[int] = []
        misses: list[int] = []
        seams: list[tuple[int, int]] = []
        matches: list[tuple[int, PICSegmentEntry]] = []
        last_index = len(segments) - 1
        for index, segment in enumerate(segments):
            entry = self.lookup(segment) if index != last_index else None
            if entry is None:
                misses.append(index)
                continue
            hits.append(index)
            matches.append((index, entry))
            if mode == "transition_rope_recompute" and index > 0:
                seam = min(max(seam_sink, 0), max(len(segment.token_ids) - 1, 0))
                if seam:
                    seams.append((index, seam))

        return PICCachePlan(
            segment_count=len(segments),
            reused_segments=tuple(hits),
            recompute_segments=tuple(misses),
            seam_tokens=tuple(seams),
            matches=tuple(matches),
        )
