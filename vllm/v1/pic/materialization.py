# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-to-scheduler metadata for materialized PIC segments."""

from dataclasses import dataclass

from vllm.v1.pic.native_kv import PICNativeKVReference


@dataclass(frozen=True)
class PICMaterialization:
    """Worker metadata made available for future requests.

    Native references in this cross-process record must be lease-free. The
    worker-local ``BlockPool`` lease is acquired only after the reference is
    received by the consumer that owns the pool.
    """

    request_id: str
    segment_index: int
    seg_hash: bytes
    full_kv_handles: tuple[int, ...] = ()
    recurrent_state_handle: int | None = None
    transition_state_handle: int | None = None
    conv_tail_handle: int | None = None
    native_kv_refs: tuple[PICNativeKVReference, ...] = ()
    # Worker-to-scheduler invalidation event.  Evictions carry the segment
    # hash but no live handles; the scheduler drops its metadata and native
    # KV lease before any subsequent lookup can use the entry.
    evicted: bool = False

    def __post_init__(self) -> None:
        if any(reference.lease is not None for reference in self.native_kv_refs):
            raise ValueError("PIC materialization cannot carry a live BlockPool lease")
        if self.evicted and (
            self.segment_index != -1
            or self.full_kv_handles
            or self.recurrent_state_handle is not None
            or self.transition_state_handle is not None
            or self.conv_tail_handle is not None
            or self.native_kv_refs
        ):
            raise ValueError("PIC eviction materialization cannot carry handles")
