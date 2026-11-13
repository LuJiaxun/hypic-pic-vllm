# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mooncake storage for vLLM-native KV blocks.

The object payload is a byte-for-byte copy of selected vLLM native KV blocks.
Mooncake owns only the remote object; local vLLM block IDs are allocated by the
current request and are never serialized as portable physical IDs.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, Protocol
from weakref import finalize

from vllm.v1.pic.external_kv import PICExternalKVImportRequest
from vllm.v1.pic.native_kv import (
    PICWindow,
    PICKVPositionMode,
    PICNativeKVKind,
    PICNativeKVReference,
    get_covering_local_block_span,
    get_full_local_block_span,
)
from vllm.v1.pic.profile_level import pic_profile_enabled
from vllm.v1.pic.segmenter import PICSegment
from vllm.v1.pic.state import (
    PICAffineTransition,
    PICConvTransition,
    PICGDNTransition,
    PICLinearTransition,
    PICStateTransition,
    PICTransitionOperator,
)
from vllm.logger import init_logger
from vllm.v1.pic.worker_plan import PICWorkerUnsupported

# Mooncake disables its TCP connection pool by default
# (tcp_transport.h: "Use MC_TCP_ENABLE_CONNECTION_POOL=1 to enable"), so every
# transfer opens a fresh socket.  A PIC request touches about sixty objects and
# the resulting TIME_WAIT pile-up exhausts the ephemeral port range within two
# benchmark rounds, which surfaces as "Cannot assign requested address".  The
# transport reads this variable when it is constructed, so defaulting it here
# (before any store connects) is early enough; an explicit 0 is respected.
os.environ.setdefault("MC_TCP_ENABLE_CONNECTION_POOL", "1")

_FORMAT = "hypic.vllm.native-kv"
_VERSION = 1
_STATE_FORMAT = "hypic.vllm.hybrid-transition"
_STATE_VERSION = 1
_COMPACT_TRANSITION_MAGIC = b"HYPICPIC"
_COMPACT_TRANSITION_VERSION = 4
# Version 3 packs tensor bytes back to back.  Version 4 pads every tensor
# offset to ``_COMPACT_TRANSITION_ALIGNMENT`` so the receiver can upload the
# whole data region with a single H2D copy and then build zero-copy views.
_COMPACT_TRANSITION_READ_VERSIONS = frozenset({3, 4})
_COMPACT_TRANSITION_ALIGNMENT = 16
# Native KV host staging reuses the same alignment so a layer payload can be
# reinterpreted in place after the single device upload.
_NATIVE_STAGING_ALIGNMENT = 16
# Consecutive native batch failures before the Store import is switched off.
_NATIVE_BATCH_FAILURE_BUDGET = 3
_COMPACT_TRANSITION_HEADER = struct.Struct("<I")
_COMPACT_TRANSITION_DTYPES = frozenset(
    {
        "bool",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
    }
)

logger = init_logger(__name__)


def _pic_import_profile_enabled() -> bool:
    """Every per-request and per-transfer record shares the single switch."""
    return pic_profile_enabled()


def _pic_memory_profile_enabled() -> bool:
    """Memory counters are part of the same profile run."""
    return pic_profile_enabled()


def _pic_cuda_memory_snapshot(torch: Any, device: Any) -> dict[str, int] | None:
    """Read allocator and driver memory counters without synchronizing CUDA."""
    try:
        target_device = torch.device(device)
        if target_device.type != "cuda" or not torch.cuda.is_available():
            return None
        # Free/total needs cudaMemGetInfo, a driver call measured at tens of milliseconds; the
        # allocator counters below come from the caching allocator and cost nothing.
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(target_device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(target_device)),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(target_device)
            ),
        }
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # Profiling must not make an otherwise valid CPU or CUDA import fail.
        return None


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Do not retry an allocation failure through the Store GPU path."""
    try:
        oom_type = getattr(_torch().cuda, "OutOfMemoryError", None)
        if not isinstance(oom_type, type):
            return False
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            if isinstance(current, oom_type):
                return True
            seen.add(id(current))
            current = current.__cause__ or current.__context__
        return False
    except (AttributeError, RuntimeError):
        return False


def _log_pic_cuda_memory(
    torch: Any,
    device: Any,
    *,
    external_key: str,
    stage: str,
    target_bytes: int,
    failed_role: str = "",
    failed_bytes: int = 0,
) -> None:
    if not _pic_memory_profile_enabled():
        return
    snapshot = _pic_cuda_memory_snapshot(torch, device)
    if snapshot is None:
        logger.warning(
            "[PIC-MEMORY] key=%s stage=%s device=%s target_bytes=%d "
            "failed_role=%s failed_bytes=%d cuda_stats=unavailable",
            external_key,
            stage,
            device,
            target_bytes,
            failed_role,
            failed_bytes,
        )
        return
    logger.warning(
        "[PIC-MEMORY] key=%s stage=%s device=%s target_bytes=%d "
        "failed_role=%s failed_bytes=%d allocated_bytes=%d "
        "reserved_bytes=%d peak_allocated_bytes=%d",
        external_key,
        stage,
        device,
        target_bytes,
        failed_role,
        failed_bytes,
        snapshot["allocated_bytes"],
        snapshot["reserved_bytes"],
        snapshot["peak_allocated_bytes"],
    )


def _transition_operator_tensor_bytes(operator: PICTransitionOperator) -> int:
    total = 0
    for transition in operator.transitions:
        for name in ("matrix", "decay", "zero_state"):
            tensor = getattr(transition, name, None)
            if tensor is not None:
                total += int(tensor.numel()) * int(tensor.element_size())
    for transition in operator.conv_transitions:
        tensor = transition.inputs
        total += int(tensor.numel()) * int(tensor.element_size())
    return total


def _pic_payload_checksum_enabled() -> bool:
    """Return whether PIC should run its application-level SHA-256 checks.

    Mooncake and the transport already provide integrity protection for the
    object transfer.  PIC keeps the SHA-256 metadata for diagnostics and
    compatibility, but does not hash every payload on the normal hot path.
    Set this by editing the corruption test hook for corruption-injection tests or
    acceptance diagnostics.
    """
    return False




def connect_mooncake_native_kv_store() -> MooncakeNativeKVObjectStore:
    """Connect to the vLLM 0.22.1/Mooncake 0.3.10 store contract.

    This helper is intentionally lazy: importing vLLM must not require
    Mooncake, and a missing or unavailable metadata service must disable
    external discovery rather than affect ordinary requests.
    """
    from mooncake.store import MooncakeDistributedStore

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        worker as mooncake_store_worker,
    )
    from vllm.utils.network_utils import get_ip

    config = mooncake_store_worker.MooncakeStoreConfig.load_from_env()
    config.device_name = rdma_utils.get_configured_worker_rnic(
        protocol=config.protocol,
        configured_device=config.device_name,
    )
    store = MooncakeDistributedStore()
    status = store.setup(
        rdma_utils.get_requester_local_hostname(get_ip()),
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
    )
    if status != 0:
        raise RuntimeError(f"Mooncake store setup failed: status={status}")
    return MooncakeNativeKVObjectStore(
        store,
        key_prefix=os.getenv("HYPIC_MOONCAKE_KEY_PREFIX", "hypic/native-kv/v1"),
    )


class PICMooncakeObjectStore(Protocol):
    """Small subset implemented by MooncakeDistributedStore."""

    def put(self, key: str, value: bytes) -> int: ...

    def get(self, key: str) -> bytes: ...

    def remove(self, key: str, force: bool = False) -> int: ...

    def register_buffer(self, base_addr: int, size: int) -> int: ...

    def batch_get_into_multi_buffers(
        self,
        keys: Sequence[str],
        addrs: Sequence[Sequence[int]],
        sizes: Sequence[Sequence[int]],
    ) -> Sequence[int]: ...


@dataclass(frozen=True)
class PICNativeKVLayerObject:
    """One layer's selected native KV block payload."""

    layer_name: str
    key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    checksum: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "key": self.key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PICNativeKVLayerObject:
        return cls(
            layer_name=str(value["layer_name"]),
            key=str(value["key"]),
            shape=tuple(int(item) for item in value["shape"]),
            dtype=str(value["dtype"]),
            nbytes=int(value["nbytes"]),
            checksum=(
                None
                if value.get("checksum") is None
                else str(value["checksum"])
            ),
        )


@dataclass(frozen=True)
class PICNativeKVObject:
    """Manifest describing one segment's native KV object."""

    external_key: str
    kv_cache_group_id: int
    block_size: int
    token_count: int
    canonical_start: int
    layout: str
    dtype: str
    block_count: int
    layers: tuple[PICNativeKVLayerObject, ...]

    @property
    def manifest_key(self) -> str:
        return self.external_key + "/manifest"

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "external_key": self.external_key,
            "kv_cache_group_id": self.kv_cache_group_id,
            "block_size": self.block_size,
            "token_count": self.token_count,
            "canonical_start": self.canonical_start,
            "layout": self.layout,
            "dtype": self.dtype,
            "block_count": self.block_count,
            "layers": [layer.as_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PICNativeKVObject:
        if value.get("format") != _FORMAT or int(value.get("version", -1)) != _VERSION:
            raise ValueError("unsupported HYPIC native KV object format")
        layers = tuple(
            PICNativeKVLayerObject.from_dict(item) for item in value["layers"]
        )
        result = cls(
            external_key=str(value["external_key"]),
            kv_cache_group_id=int(value["kv_cache_group_id"]),
            block_size=int(value["block_size"]),
            token_count=int(value["token_count"]),
            canonical_start=int(value["canonical_start"]),
            layout=str(value["layout"]),
            dtype=str(value["dtype"]),
            block_count=int(value["block_count"]),
            layers=layers,
        )
        if result.block_size <= 0 or result.token_count <= 0:
            raise ValueError("native KV object has invalid range metadata")
        if result.block_count <= 0 or not result.layers:
            raise ValueError("native KV object has no block payload")
        return result


@dataclass(frozen=True)
class PICNativeKVReceived:
    """Native KV payloads received from Mooncake, already in their destination blocks."""

    obj: PICNativeKVObject


@dataclass(frozen=True)
class PICHybridStateObject:
    """Manifest for a serialized hybrid transition operator.

    The native KV manifest and this object are deliberately separate.  A
    full-attention-only consumer can still import KV, while a hybrid PIC
    consumer requires both manifests before it enables non-prefix reuse.
    """

    external_key: str
    token_count: int
    payload_key: str
    nbytes: int
    checksum: str
    # First token of the covered span, in segment-local coordinates.  The
    # transition is published for exactly the range the reusing request will
    # take, so the reader adopts this offset instead of slicing a compact
    # operator, which cannot be sliced.
    span_start: int = 0
    # Store fixed-layout descriptor. It contains object keys and tensor
    # metadata only; the receiver batches the Store pull directly into
    # registered GPU tensors and avoids parsing the compact operator payload.
    store_layout: Mapping[str, Any] | None = None

    @property
    def manifest_key(self) -> str:
        return self.external_key + "/manifest"

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": _STATE_FORMAT,
            "version": _STATE_VERSION,
            "external_key": self.external_key,
            "token_count": self.token_count,
            "payload_key": self.payload_key,
            "nbytes": self.nbytes,
            "checksum": self.checksum,
            "span_start": self.span_start,
            "store_layout": self.store_layout,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PICHybridStateObject:
        if (
            value.get("format") != _STATE_FORMAT
            or int(value.get("version", -1)) != _STATE_VERSION
        ):
            raise ValueError("unsupported HYPIC hybrid transition format")
        result = cls(
            external_key=str(value["external_key"]),
            token_count=int(value["token_count"]),
            payload_key=str(value["payload_key"]),
            nbytes=int(value["nbytes"]),
            checksum=str(value["checksum"]),
            # Entries written before the span existed cover the whole segment.
            span_start=int(value.get("span_start", 0)),
            store_layout=(
                None
                if value.get("store_layout") is None
                else dict(value["store_layout"])
            ),
        )
        if result.token_count <= 0:
            raise ValueError("hybrid transition manifest has invalid metadata")
        if (
            result.nbytes <= 0 or not result.checksum
        ) and result.store_layout is None:
            raise ValueError(
                "hybrid transition manifest has neither a payload nor a layout"
            )
        return result


def _torch():
    import torch

    return torch


def _thread_upload_stream(torch: Any, device: Any) -> Any:
    """A fresh upload stream, so an upload waits for nothing but its own copy.

    This used to be one stream per thread, which made two things look like a slow copy: the upload
    was timed from the moment it was submitted, so it inherited whatever the thread had queued
    before it - the fifty-three megabyte KV import, for one - and the wait was reported as device
    time. A pinned copy of thirty-eight megabytes then measured a hundred and eighty-eight
    milliseconds, one six-hundredth of what the link can do.

    A stream per upload costs a handle and removes both effects: the copy runs when it is submitted
    and the timer spans only itself.
    """
    return torch.cuda.Stream(device=device)


# Last upload detail, so callers (and the profile log) can report whether the
# copy was pinned and how much of the wait was device time.
_h2d_last_upload: dict[str, Any] = {}
def _pic_block_span(kv_cache: Any, layout: str | None) -> tuple[int, int] | None:
    """Physical block geometry, when the blocks sit back to back.

    Returns ``(block_axis, span_elements)`` where ``span_elements`` is one whole block - every axis except
    the block axis - and ``stride(block_axis)`` equals it. With the cache stored block-major that holds,
    and a run of consecutive blocks is one contiguous region.
    """
    kv_axis, block_axis = _native_cache_axes(kv_cache, layout=layout)
    span = 1
    for axis in range(kv_cache.ndim):
        if axis != block_axis:
            span *= int(kv_cache.shape[axis])
    if int(kv_cache.stride(block_axis)) != span:
        return None
    return block_axis, span


def _pic_flat_run(kv_cache: Any, ids: Sequence[int], layout: str | None) -> Any | None:
    """A 1-D view over one consecutive run of blocks, or None when the run is not contiguous."""
    if not ids:
        return None
    first = int(ids[0])
    for offset, block_id in enumerate(ids):
        if int(block_id) != first + offset:
            return None
    geometry = _pic_block_span(kv_cache, layout)
    if geometry is None:
        return None
    block_axis, span = geometry
    stride = int(kv_cache.stride(block_axis))
    total = len(ids) * span
    offset = first * stride
    storage = kv_cache.untyped_storage()
    if offset < 0 or offset + total > storage.nbytes() // kv_cache.element_size():
        return None
    return _torch().as_strided(
        kv_cache,
        size=(total,),
        stride=(1,),
        storage_offset=int(kv_cache.storage_offset()) + offset,
    )


def _h2d_on_private_stream(cpu_tensor: Any, device: Any) -> Any:
    """Upload to the device on a stream that is not the default one.

    An import runs on a background thread while another request may be
    prefilling on the default stream.  Copying on the default stream puts the
    transfer behind that forward, which measured as a transfer that grew from
    forty milliseconds alone to over a hundred under load.

    The stream is per thread and the wait is an event belonging to this copy, so
    one import never waits for another import's upload; sharing a single stream
    and waiting on the whole stream did exactly that, showing up as uploads of
    forty to a hundred and thirty milliseconds among five-millisecond ones.
    """
    torch = _torch()
    try:
        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            return cpu_tensor.to(device=device)
    except Exception:  # pragma: no cover - non-CUDA builds
        return cpu_tensor.to(device=device)

    stream = _thread_upload_stream(torch, device)

    pinned = False
    try:
        pinned = bool(cpu_tensor.is_pinned())
    except Exception:  # pragma: no cover - unusual tensor types
        pinned = False

    # Device elapsed tells "the copy was slow" apart from "the host waited": a
    # pageable copy is slow on the device clock, while a queueing delay is not.
    start_event = torch.cuda.Event(enable_timing=True)
    done = torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    with torch.cuda.stream(stream):
        start_event.record(stream)
        out = cpu_tensor.to(device=device, non_blocking=pinned)
        done.record(stream)
    # The copy is left running: its reader is one forward pass away, and the copy engine works while
    # the SMs compute. Waiting here spent the whole transfer in front of the work it could hide
    # behind, which is how a thirty-eight megabyte upload reached the critical path.
    wall_ms = (time.perf_counter() - host_start) * 1000.0
    # `elapsed_time` refuses to answer until both events have completed, and the copy is
    # deliberately still running - asking anyway raised "Both events must be completed", which the
    # caller read as a broken receive and answered by disabling this path for the whole process.
    # The copy's duration is only a diagnostic, so it is reported when it is free and skipped when
    # it is not.
    device_ms = 0.0
    try:
        if done.query():
            device_ms = float(start_event.elapsed_time(done))
    except Exception:  # noqa: BLE001 - timing must never decide whether a receive works
        device_ms = 0.0
    _h2d_last_upload.update(
        {"wall_ms": wall_ms, "device_ms": device_ms, "pinned": pinned, "event": done}
    )
    if _pic_import_profile_enabled():
        # Pinned is the field that decides it: a pageable copy of 38 MB runs at
        # hundreds of megabytes per second, which is exactly the slow plateau.
        reserved_mb = allocated_mb = 0.0
        try:
            reserved_mb = torch.cuda.memory_reserved(device) / (1 << 20)
            allocated_mb = torch.cuda.memory_allocated(device) / (1 << 20)
        except Exception:  # pragma: no cover - non-CUDA builds
            pass
        # Residency is what separates "the copy is slow" from "the allocator had
        # to ask the driver for another segment in the middle of the copy".
        logger.warning(
            "[PIC-PROFILE] transition-upload bytes=%d pinned=%s wall_ms=%.3f "
            "device_ms=%.3f reserved_mb=%.1f allocated_mb=%.1f",
            int(cpu_tensor.numel()),
            pinned,
            wall_ms,
            device_ms,
            reserved_mb,
            allocated_mb,
        )
    return out


def _compact_arena_tensor(arena: Any, record: Mapping[str, Any]) -> Any:
    """Build one zero-copy GPU view of an aligned tensor inside ``arena``.

    ``arena`` is one device buffer holding the whole compact payload data
    region.  Every version-4 record offset is 16-byte aligned, so a byte view
    can be reinterpreted in place instead of paying one H2D copy per tensor.
    """
    torch = _torch()
    dtype_name = str(record["dtype"])
    if dtype_name not in _COMPACT_TRANSITION_DTYPES:
        raise ValueError(f"unsupported compact transition dtype: {dtype_name!r}")
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(
            f"torch does not provide compact transition dtype: {dtype_name!r}"
        )
    shape = tuple(int(item) for item in record["shape"])
    if any(item < 0 for item in shape):
        raise ValueError("compact transition tensor shape contains a negative extent")
    offset = int(record["offset"])
    nbytes = int(record["nbytes"])
    if offset < 0 or nbytes < 0 or offset + nbytes > int(arena.numel()):
        raise ValueError("compact transition tensor exceeds payload bounds")
    expected = 1
    for extent in shape:
        expected *= extent
    element_bytes = int(torch.empty((), dtype=dtype).element_size())
    if expected * element_bytes != nbytes:
        raise ValueError("compact transition tensor byte size is incompatible")
    if nbytes == 0:
        return torch.empty(shape, dtype=dtype, device=arena.device)
    view = arena.narrow(0, offset, nbytes)
    if offset % element_bytes:
        # Version 4 always aligns offsets.  Keep a correct path for a payload
        # written by a partially upgraded publisher instead of failing open.
        aligned = torch.empty(nbytes, dtype=torch.uint8, device=arena.device)
        aligned.copy_(view)
        view = aligned
    return view.view(dtype).reshape(shape)


def _native_cache_axes(
    kv_cache: Any, *, layout: str | None = None
) -> tuple[int, int]:
    """Return the KV and block axes for a native cache tensor.

    The leading dimensions can both have size ``2`` in small test fixtures,
    so tensor shapes alone are not sufficient to identify the layout.  Use the
    manifest/cache layout whenever available and keep the shape heuristic only
    as a compatibility fallback for old callers.
    """
    if kv_cache.ndim != 5:
        raise ValueError("native KV cache must be a 5-D tensor")
    normalized_layout = "" if layout is None else str(layout).upper()
    if normalized_layout == "NHD":
        return 0, 1
    if normalized_layout == "HND":
        return 1, 0
    if normalized_layout not in ("", "AUTO"):
        raise ValueError(f"unsupported native KV layout: {layout!r}")
    if kv_cache.shape[0] == 2 and kv_cache.shape[1] != 2:
        return 0, 1
    if kv_cache.shape[1] == 2 and kv_cache.shape[0] != 2:
        return 1, 0
    raise ValueError(
        "cannot identify native KV and block dimensions; pass layout='NHD' "
        "or layout='HND'"
    )


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> Any:
    torch = _torch()
    dtype = getattr(torch, name, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported native KV dtype: {name}")
    return dtype


def _tensor_bytes(tensor: Any) -> bytes:
    torch = _torch()
    cpu = tensor.detach().contiguous().to(device="cpu")
    raw = cpu.view(torch.uint8).reshape(-1)
    return raw.numpy().tobytes()


def _cpu_tensor(tensor: Any) -> Any:
    return tensor.detach().contiguous().to(device="cpu")


def _transition_to_payload(transition: PICStateTransition) -> dict[str, Any]:
    if isinstance(transition, PICAffineTransition):
        return {
            "kind": "affine",
            "decay": _cpu_tensor(transition.decay),
            "zero_state": _cpu_tensor(transition.zero_state),
        }
    if isinstance(transition, PICLinearTransition):
        return {
            "kind": "linear",
            "matrix": _cpu_tensor(transition.matrix),
            "zero_state": _cpu_tensor(transition.zero_state),
        }
    if isinstance(transition, PICGDNTransition):
        compact = transition.compact()
        if compact is not None:
            return {
                "kind": "gdn_compact",
                "matrix": _cpu_tensor(compact.matrix),
                "zero_state": _cpu_tensor(compact.zero_state),
            }
        return {
            "kind": "gdn",
            "key": _cpu_tensor(transition.key),
            "value": _cpu_tensor(transition.value),
            "log_decay": _cpu_tensor(transition.log_decay),
            "beta": _cpu_tensor(transition.beta),
            "zero_state": _cpu_tensor(transition.zero_state),
            "chunk_w": (
                None if transition.chunk_w is None else _cpu_tensor(transition.chunk_w)
            ),
            "chunk_u": (
                None if transition.chunk_u is None else _cpu_tensor(transition.chunk_u)
            ),
            "chunk_g": (
                None if transition.chunk_g is None else _cpu_tensor(transition.chunk_g)
            ),
            "chunk_size": transition.chunk_size,
            "backend": transition.backend,
        }
    raise ValueError(f"unsupported PIC transition type: {type(transition)!r}")


def _transition_from_payload(
    payload: Mapping[str, Any], *, device: Any
) -> PICStateTransition:
    kind = str(payload["kind"])
    move = lambda value: _h2d_on_private_stream(value, device)
    if kind == "affine":
        return PICAffineTransition(
            decay=move(payload["decay"]),
            zero_state=move(payload["zero_state"]),
        )
    if kind == "linear":
        return PICLinearTransition(
            matrix=move(payload["matrix"]),
            zero_state=move(payload["zero_state"]),
        )
    if kind == "gdn":
        return PICGDNTransition(
            key=move(payload["key"]),
            value=move(payload["value"]),
            log_decay=move(payload["log_decay"]),
            beta=move(payload["beta"]),
            zero_state=move(payload["zero_state"]),
            chunk_w=(
                None if payload.get("chunk_w") is None else move(payload["chunk_w"])
            ),
            chunk_u=(
                None if payload.get("chunk_u") is None else move(payload["chunk_u"])
            ),
            chunk_g=(
                None if payload.get("chunk_g") is None else move(payload["chunk_g"])
            ),
            chunk_size=int(payload.get("chunk_size", 64)),
            backend=(None if payload.get("backend") is None else str(payload["backend"])),
        )
    if kind == "gdn_compact":
        return PICLinearTransition(
            matrix=move(payload["matrix"]),
            zero_state=move(payload["zero_state"]),
        )
    raise ValueError(f"unsupported serialized PIC transition kind: {kind!r}")


def _compact_tensor_record(
    tensor: Any, chunks: list[bytes], offset: int
) -> tuple[dict[str, Any], int]:
    cpu = tensor.detach().contiguous().to(device="cpu")
    dtype_name = str(cpu.dtype).removeprefix("torch.")
    if dtype_name not in _COMPACT_TRANSITION_DTYPES:
        raise ValueError(f"unsupported compact transition dtype: {dtype_name!r}")
    raw = _tensor_bytes(cpu)
    padding = (-offset) % _COMPACT_TRANSITION_ALIGNMENT
    if padding:
        chunks.append(b"\x00" * padding)
        offset += padding
    chunks.append(raw)
    record = {
        "dtype": dtype_name,
        "shape": [int(item) for item in cpu.shape],
        "offset": offset,
        "nbytes": len(raw),
    }
    return record, offset + len(raw)


def _serialize_compact_transition_operator(
    operator: PICTransitionOperator,
) -> bytes | None:
    """Serialize compact transitions without pickle or torch archive metadata."""

    chunks: list[bytes] = []
    offset = 0

    def add_tensor(tensor: Any) -> dict[str, Any]:
        nonlocal offset
        record, offset = _compact_tensor_record(tensor, chunks, offset)
        return record

    transitions: list[dict[str, Any]] = []
    for item in operator.transitions:
        if isinstance(item, PICAffineTransition):
            transitions.append(
                {
                    "kind": "affine",
                    "decay": add_tensor(item.decay),
                    "zero_state": add_tensor(item.zero_state),
                }
            )
        elif isinstance(item, PICLinearTransition):
            transitions.append(
                {
                    "kind": "linear",
                    "matrix": add_tensor(item.matrix),
                    "zero_state": add_tensor(item.zero_state),
                }
            )
        elif isinstance(item, PICGDNTransition):
            compact = item.compact()
            if compact is None:
                return None
            transitions.append(
                {
                    "kind": "gdn_compact",
                    "matrix": add_tensor(compact.matrix),
                    "zero_state": add_tensor(compact.zero_state),
                }
            )
        else:
            return None

    conv_transitions: list[dict[str, Any]] = []
    for item in operator.conv_transitions:
        inputs = (
            item.inputs[-item.state_length :]
            if item.state_length > 0
            else item.inputs[:0]
        )
        conv_transitions.append(
            {
                "kind": "tail",
                "inputs": add_tensor(inputs),
                "state_length": item.state_length,
                "source_token_count": operator.token_end - operator.token_start,
            }
        )

    metadata = {
        "version": _COMPACT_TRANSITION_VERSION,
        "token_count": operator.token_end - operator.token_start,
        "transitions": transitions,
        "conv_transitions": conv_transitions,
        "data_bytes": offset,
    }
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return (
        _COMPACT_TRANSITION_MAGIC
        + _COMPACT_TRANSITION_HEADER.pack(len(header))
        + header
        + b"".join(chunks)
    )


def _compact_tensor_from_record(
    raw: bytes, data_start: int, record: Mapping[str, Any], *, device: Any
) -> Any:
    torch = _torch()
    dtype_name = str(record["dtype"])
    if dtype_name not in _COMPACT_TRANSITION_DTYPES:
        raise ValueError(f"unsupported compact transition dtype: {dtype_name!r}")
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(
            f"torch does not provide compact transition dtype: {dtype_name!r}"
        )
    shape = tuple(int(item) for item in record["shape"])
    if any(item < 0 for item in shape):
        raise ValueError("compact transition tensor shape contains a negative extent")
    offset = int(record["offset"])
    nbytes = int(record["nbytes"])
    if offset < 0 or nbytes < 0 or data_start + offset + nbytes > len(raw):
        raise ValueError("compact transition tensor exceeds payload bounds")
    expected = 1
    for extent in shape:
        expected *= extent
    if expected * torch.empty((), dtype=dtype).element_size() != nbytes:
        raise ValueError("compact transition tensor byte size is incompatible")
    if nbytes == 0:
        cpu = torch.empty(shape, dtype=dtype)
    else:
        view = memoryview(raw)[data_start + offset : data_start + offset + nbytes]
        # The bytes object is kept alive by the frombuffer storage.  Avoid a
        # second CPU clone; .to(device) can consume this view directly for the
        # CUDA import path.  The source is intentionally read-only because
        # transition tensors are never mutated after deserialization.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cpu = torch.frombuffer(view, dtype=dtype, count=expected).reshape(shape)
    return _h2d_on_private_stream(cpu, device)
def _deserialize_compact_transition_operator(
    raw: bytes, *, device: Any
) -> PICTransitionOperator:
    if not raw.startswith(_COMPACT_TRANSITION_MAGIC):
        raise ValueError("not a compact PIC transition payload")
    prefix = len(_COMPACT_TRANSITION_MAGIC)
    if len(raw) < prefix + _COMPACT_TRANSITION_HEADER.size:
        raise ValueError("compact PIC transition payload header is truncated")
    header_size = _COMPACT_TRANSITION_HEADER.unpack_from(raw, prefix)[0]
    header_start = prefix + _COMPACT_TRANSITION_HEADER.size
    data_start = header_start + header_size
    if data_start > len(raw):
        raise ValueError("compact PIC transition payload metadata is truncated")
    metadata = json.loads(raw[header_start:data_start].decode("utf-8"))
    version = int(metadata.get("version", -1))
    if version not in _COMPACT_TRANSITION_READ_VERSIONS:
        raise ValueError("unsupported compact PIC transition payload version")
    if int(metadata.get("data_bytes", -1)) != len(raw) - data_start:
        raise ValueError("compact PIC transition payload size is incompatible")

    profile = _pic_import_profile_enabled()
    arena: Any | None = None
    h2d_ms = 0.0
    if version >= 4 and data_start < len(raw):
        torch = _torch()
        h2d_start = time.perf_counter() if profile else 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # One device upload for the whole data region.  Per-tensor views
            # below are allocations-free slices of this arena.
            arena = _h2d_on_private_stream(
                torch.frombuffer(memoryview(raw)[data_start:], dtype=torch.uint8),
                device,
            )
        h2d_ms = (time.perf_counter() - h2d_start) * 1000.0 if profile else 0.0

    tensor_count = 0

    def tensor(record: Mapping[str, Any]) -> Any:
        nonlocal tensor_count
        tensor_count += 1
        if arena is None:
            return _compact_tensor_from_record(
                raw, data_start, record, device=device
            )
        return _compact_arena_tensor(arena, record)

    transitions: list[PICStateTransition] = []
    for item in metadata.get("transitions", ()):
        kind = str(item["kind"])
        if kind == "affine":
            transitions.append(
                PICAffineTransition(
                    decay=tensor(item["decay"]),
                    zero_state=tensor(item["zero_state"]),
                )
            )
        elif kind in ("linear", "gdn_compact"):
            transitions.append(
                PICLinearTransition(
                    matrix=tensor(item["matrix"]),
                    zero_state=tensor(item["zero_state"]),
                )
            )
        else:
            raise ValueError(f"unsupported compact transition kind: {kind!r}")

    conv_transitions = tuple(
        PICConvTransition(
            inputs=tensor(item["inputs"]),
            state_length=int(item["state_length"]),
            source_token_count=int(item["source_token_count"]),
        )
        for item in metadata.get("conv_transitions", ())
    )
    token_count = int(metadata["token_count"])
    if profile:
        logger.warning(
            "[PIC-PROFILE] transition-deserialize wire_version=%d tensors=%d "
            "data_bytes=%d single_h2d=%s h2d_ms=%.3f",
            version,
            tensor_count,
            len(raw) - data_start,
            arena is not None,
            h2d_ms,
        )
    return PICTransitionOperator(
        transitions=tuple(transitions),
        conv_transitions=conv_transitions,
        token_start=0,
        token_end=token_count,
    )


def _serialize_transition_operator(operator: PICTransitionOperator) -> bytes:
    torch = _torch()
    compact_raw = _serialize_compact_transition_operator(operator)
    if compact_raw is not None:
        return compact_raw
    conv_transitions = []
    for item in operator.conv_transitions:
        inputs = (
            item.inputs[-item.state_length :]
            if item.state_length > 0
            else item.inputs[:0]
        )
        conv_transitions.append(
            {
                "kind": "tail",
                "inputs": _cpu_tensor(inputs),
                "state_length": item.state_length,
                "source_token_count": operator.token_end - operator.token_start,
            }
        )
    payload = {
        "wire_version": 2,
        "token_count": operator.token_end - operator.token_start,
        "transitions": [_transition_to_payload(item) for item in operator.transitions],
        "conv_transitions": conv_transitions,
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _deserialize_transition_operator(raw: bytes, *, device: Any) -> PICTransitionOperator:
    if raw.startswith(_COMPACT_TRANSITION_MAGIC):
        return _deserialize_compact_transition_operator(raw, device=device)
    torch = _torch()
    stream = io.BytesIO(raw)
    try:
        payload = torch.load(stream, map_location=device, weights_only=True)
    except TypeError:
        stream.seek(0)
        payload = torch.load(stream, map_location=device)
    transitions = tuple(
        _transition_from_payload(item, device=device) for item in payload["transitions"]
    )
    conv_transitions = tuple(
        PICConvTransition(
            inputs=_h2d_on_private_stream(item["inputs"], device),
            state_length=int(item["state_length"]),
            source_token_count=(
                None
                if item.get("source_token_count") is None
                else int(item["source_token_count"])
            ),
        )
        for item in payload.get("conv_transitions", ())
    )
    token_count = int(payload["token_count"])
    return PICTransitionOperator(
        transitions=transitions,
        conv_transitions=conv_transitions,
        token_start=0,
        token_end=token_count,
    )



# The number of key/value heads the model has, set once at start-up.  When it is smaller than the tensor
# parallel size several ranks hold the same KV shard, and the connector this follows writes one copy:
# ``put_step = tp_size // num_kv_heads``, with ``tp_rank // put_step`` naming the shard.
_TP_NUM_KV_HEADS = 0


def pic_set_tp_kv_heads(num_kv_heads: int) -> None:
    """Tell the key builder how many KV heads the model has."""
    global _TP_NUM_KV_HEADS
    _TP_NUM_KV_HEADS = max(0, int(num_kv_heads))


def _pic_tp_rank_and_size() -> tuple[int, int]:
    """This rank's tensor-parallel identity, or (0, 1) before the groups exist."""
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except Exception:  # noqa: BLE001 - a missing group means "not sharded"
        return 0, 1


def _pic_tp_put_step(tp_size: int, num_kv_heads: int) -> int:
    """Ranks that share one KV shard, mirroring MooncakeStoreConnector."""
    if num_kv_heads <= 0 or num_kv_heads >= tp_size:
        return 1
    return max(1, tp_size // num_kv_heads)


def pic_tp_shard_rank() -> int:
    """Index of this rank's KV shard.

    Mirrors ``head_or_tp_rank`` in MooncakeStoreConnector: with fewer KV heads than ranks the shard index
    is the head index, so several ranks write the same key with the same bytes and the store keeps one.
    """
    rank, size = _pic_tp_rank_and_size()
    return rank // _pic_tp_put_step(size, _TP_NUM_KV_HEADS)


def pic_tp_shard_count() -> int:
    """How many distinct KV shards exist at this tensor-parallel size."""
    _, size = _pic_tp_rank_and_size()
    return max(1, size // _pic_tp_put_step(size, _TP_NUM_KV_HEADS))


def pic_tp_key_suffix() -> str:
    """Key components naming the shard, in the order MooncakeStoreConnector uses.

    ``@tp_rank:<shard>@pp_rank:<rank>`` - always present, at every TP size, because the addressing here is
    a key in a shared store rather than a peer address, and a shard written at one size must never be read
    as a shard at another.
    """
    _, size = _pic_tp_rank_and_size()
    if size <= 1:
        return "@tp_rank:0@pp_rank:0"
    return f"@tp_rank:{pic_tp_shard_rank()}@pp_rank:0"


def pic_tp_state_suffix() -> str:
    """Suffix for the recurrent transition.

    The state is split ``num_v_heads / tp`` heads per rank, which is not the same partition as the KV
    shard index, so the size is part of the key: a transition computed by four ranks is not the transition
    a two-rank deployment needs.
    """
    _, size = _pic_tp_rank_and_size()
    return f"@tp_rank:{pic_tp_shard_rank()}@tp_size:{size}@pp_rank:0"


def pic_peer_tp_suffixes() -> tuple[str, ...]:
    """Suffixes of the other shards, empty when there is only one."""
    count = pic_tp_shard_count()
    if count <= 1:
        return ()
    own = pic_tp_shard_rank()
    return tuple(
        f"@tp_rank:{other}@pp_rank:0" for other in range(count) if other != own
    )


def pic_native_kv_key(seg_hash: Any, group_id: int) -> str:
    """Store key for one KV group of one segment, on this rank's shard."""
    return f"segment/{seg_hash.hex()}/group-{int(group_id)}{pic_tp_key_suffix()}"


def pic_hybrid_state_key(seg_hash: Any) -> str:
    """Store key for the recurrent transition of one segment, on this rank."""
    return f"segment/{seg_hash.hex()}/hybrid-state{pic_tp_state_suffix()}"


class MooncakeNativeKVObjectStore:
    """Persist exact native KV blocks through a Mooncake object store."""

    # Consecutive fixed-layout receive failures after which the process stops
    # attempting it.  A single transient failure must not disable the fast path.
    _STORE_TRANSITION_FAILURE_BUDGET = 3

    def __init__(
        self,
        store: PICMooncakeObjectStore,
        *,
        key_prefix: str,
    ) -> None:
        prefix = str(key_prefix).strip("/")
        if not prefix:
            raise ValueError("Mooncake KV key prefix cannot be empty")
        self.store = store
        self.key_prefix = prefix
        self._registered_buffer_ptrs: set[int] = set()
        # Registered (address, length) ranges: a slice inside one is inside the region, which is
        # what a batched call needs when several layers share one scratch allocation.
        self._registered_regions: list[tuple[int, int]] = []
        # One device buffer, grown on demand, that the store can write into directly.  The host
        self._gpu_direct_available = True
        # The put call takes a replicate config.  Built once, and left None when the installed
        # binding does not expose it, in which case the three-argument form is tried.
        self._replicate_config_ready = False
        self._replicate_config_value: Any = None
        # Layer name to KV tensor, handed over once by the worker.  A receive that can address the
        # destination blocks needs them; without them it falls back to the host arena.
        self._kv_cache_layers: dict[str, Any] = {}
        self._owned_native_keys: set[str] = set()
        self._owned_state_keys: set[str] = set()
        # ``register_buffer`` is a transfer optimisation, not a correctness
        # requirement: Full Attention keeps using the destination address when
        # registration fails because the Store still copies into it.  Report
        # the degradation once so logs distinguish it from a real fallback.
        self._buffer_registration_warned = False
        # A fixed-layout transition receive that the Store rejects would
        # otherwise be retried, and paid for, by every request before the
        # payload fallback starts.
        self._store_transition_failures = 0
        self._store_transition_disabled = False
        # A transport hiccup must not turn into a connection storm: after a few
        # consecutive native batch failures the store import is switched off for
        # the process and affected requests recompute instead.
        self._native_batch_failures = 0
        self._native_store_disabled = False

    def _key(self, external_key: str) -> str:
        key = str(external_key).strip("/")
        if not key:
            raise ValueError("Mooncake native KV object key cannot be empty")
        prefix = f"{self.key_prefix}/"
        if key == self.key_prefix or key.startswith(prefix):
            return key
        return f"{prefix}{key}"

    def _tp_peers_published(self, key_without_suffix: str, suffix_for: Any) -> bool:
        """Whether every other shard of this key was published.

        A segment is only reusable when all of its shards exist: applying two shards out of four would be
        wrong in a way no shape check catches.  One ``batch_is_exist`` asks the store once, which is how
        MooncakeStoreConnector does it; stores without that call fall back to one probe per shard.
        """
        key = self._key(key_without_suffix)
        peers = [f"{key}{other}/manifest" for other in suffix_for()]
        if not peers:
            return True
        batch = getattr(self.store, "batch_is_exist", None)
        if callable(batch):
            try:
                found = batch(peers)
            except Exception:  # noqa: BLE001 - a failed probe is a miss
                return False
            return all(bool(value) for value in found)
        probe = getattr(self.store, "is_exist", None)
        if callable(probe):
            try:
                return all(bool(probe(item)) for item in peers)
            except Exception:  # noqa: BLE001 - a failed probe is a miss
                return False
        for item in peers:
            try:
                raw = self.store.get(item)
            except Exception:  # noqa: BLE001 - absent is the answer
                return False
            if not isinstance(raw, (bytes, bytearray, memoryview)) or not raw:
                return False
        return True

    @staticmethod
    def _store_layout_keys(value: Any) -> tuple[str, ...]:
        """Collect tensor object keys from a fixed-layout Store manifest."""
        keys: list[str] = []
        if isinstance(value, Mapping):
            key = value.get("key")
            if key:
                keys.append(str(key))
            for child in value.values():
                keys.extend(MooncakeNativeKVObjectStore._store_layout_keys(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                keys.extend(MooncakeNativeKVObjectStore._store_layout_keys(child))
        return tuple(dict.fromkeys(keys))

    @staticmethod
    def _check_status(result: Any, operation: str) -> None:
        if result not in (None, 0):
            raise RuntimeError(f"Mooncake {operation} failed: {result}")

    def set_kv_caches(self, kv_caches: Mapping[str, Any]) -> None:
        """Remember the layer tensors so a receive can address its destination directly."""
        self._kv_cache_layers = {
            str(name): tensor for name, tensor in kv_caches.items()
        }

    def _replicate_config(self) -> Any:
        """The store's replicate config, or None when the binding does not have one."""
        if not self._replicate_config_ready:
            self._replicate_config_ready = True
            try:
                from mooncake.store import ReplicateConfig  # type: ignore

                self._replicate_config_value = ReplicateConfig()
            except Exception:  # noqa: BLE001 - the three-argument form may still work
                self._replicate_config_value = None
        return self._replicate_config_value

    def _gpu_direct_enabled(self) -> bool:
        """Whether the store can move bytes to and from a device buffer directly."""
        if not self._gpu_direct_available:
            return False
        get_into = getattr(self.store, "batch_get_into_multi_buffers", None)
        put = getattr(self.store, "batch_put_from_multi_buffers", None)
        if not callable(get_into) or not callable(put):
            return False
        return True
    def _layer_block_segments(
        self, kv_cache: Any, ids: Sequence[int], layout: str, *, for_device: Any
    ) -> list[tuple[int, int]] | None:
        """One (address, bytes) per block, in the order ``index_select`` would produce.

        A block slice is contiguous, so a segment list is all the store needs: there is nothing to gather.
        The order is not block-major in every layout - with the KV axis leading, all key blocks come
        before all value blocks - so it is derived from the axes rather than assumed.
        """
        if int(kv_cache.data_ptr()) == 0:
            return None
        if kv_cache.device != for_device:
            return None
        kv_axis, block_axis = _native_cache_axes(kv_cache, layout=layout)
        element = kv_cache.element_size()
        strides = [int(stride) * element for stride in kv_cache.stride()]
        # What a segment list needs is that one block's trailing dimensions run back to back.  The tensor
        # as a whole may be a view of a packed allocation, and requiring is_contiguous() threw those away.
        expected = element
        for axis in range(len(kv_cache.shape) - 1, block_axis, -1):
            if strides[axis] != expected:
                return None
            expected *= int(kv_cache.shape[axis])
        # Pinning this range is optional; the call below is what decides whether the store can take it.
        self._ensure_registered(kv_cache, role="kv-cache")
        tail = 1
        for dimension in kv_cache.shape[block_axis + 1 :]:
            tail *= int(dimension)
        base = int(kv_cache.data_ptr())
        segments: list[tuple[int, int]] = []
        per_block = tail * element
        # Consecutive blocks can share one segment only when they sit back to back: if the block axis is
        # padded, a merged copy would write into the padding.  With no padding the run collapses to a
        # single copy per layer instead of one per block.
        packed = strides[block_axis] == per_block
        runs: list[tuple[int, int]] = []
        for block_id in ids:
            block = int(block_id)
            if packed and runs and block == runs[-1][0] + runs[-1][1]:
                runs[-1] = (runs[-1][0], runs[-1][1] + 1)
            else:
                runs.append((block, 1))
        if kv_axis < block_axis:
            # KV axis outside the block axis: every key block run, then every value block run.
            for kv in range(int(kv_cache.shape[kv_axis])):
                kv_offset = kv * strides[kv_axis]
                for first, count in runs:
                    segments.append(
                        (
                            base + kv_offset + first * strides[block_axis],
                            count * per_block,
                        )
                    )
        else:
            # Block axis outside the KV axis: one segment per run, both halves together.
            for first, count in runs:
                segments.append(
                    (base + first * strides[block_axis], count * per_block)
                )
        return segments

    def _ensure_registered(self, tensor: Any, *, role: str) -> bool:
        """Register a tensor's range once, keyed by its base address."""
        address = int(tensor.data_ptr())
        nbytes = int(tensor.numel()) * int(tensor.element_size())
        if address in self._registered_buffer_ptrs and self._is_device_registered(
            address, nbytes
        ):
            return True
        return self._register_store_buffer_best_effort(
            address=address, nbytes=nbytes, role=role
        )

    def _put_batch_from_segments(
        self, entries: Sequence[tuple[str, Sequence[tuple[int, int]]]]
    ) -> bool:
        """Store several objects, each described by its own list of source segments."""
        if not self._gpu_direct_enabled() or not entries:
            return False
        for _, segments in entries:
            if not segments:
                return False
        put = getattr(self.store, "batch_put_from_multi_buffers", None)
        keys = [key for key, _ in entries]
        addrs = [[address for address, _ in segments] for _, segments in entries]
        sizes = [[nbytes for _, nbytes in segments] for _, segments in entries]
        config = self._replicate_config()
        try:
            if config is not None:
                status = put(keys, addrs, sizes, config)
            else:
                status = put(keys, addrs, sizes)
        except TypeError:
            # An older binding without the parameter: retry the short form before giving up.
            try:
                status = put(keys, addrs, sizes)
            except Exception as exc:  # noqa: BLE001 - fall back to the staged path
                return False
        except Exception as exc:  # noqa: BLE001 - fall back to the staged path
            return False
        return (
            status is not None
            and len(status) == len(entries)
            and all(int(value) >= 0 for value in status)
        )

    def _get_batch_into_segments(
        self, entries: Sequence[tuple[str, Sequence[tuple[int, int]]]]
    ) -> bool:
        """Read several objects straight into their destination segments.

        The store answers with the bytes it moved for each key - not a status - so a negative value is the
        failure signal and anything else is a count.
        """
        if not self._gpu_direct_enabled() or not entries:
            return False
        for _, segments in entries:
            if not segments:
                return False
        get_into = getattr(self.store, "batch_get_into_multi_buffers", None)
        try:
            status = get_into(
                [key for key, _ in entries],
                [[address for address, _ in segments] for _, segments in entries],
                [[nbytes for _, nbytes in segments] for _, segments in entries],
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the staged path
            return False
        return (
            status is not None
            and len(status) == len(entries)
            and all(int(value) >= 0 for value in status)
        )

    def _is_device_registered(self, address: int, nbytes: int) -> bool:
        """Whether a range sits inside a buffer the store was told about."""
        for base, size in self._registered_regions:
            if base <= address and address + nbytes <= base + size:
                return True
        return False
    def _register_store_buffer_best_effort(
        self, *, address: int, nbytes: int, role: str
    ) -> bool:
        """Register a destination buffer without failing the direct path.

        A non-zero status only means Mooncake cannot pin/RDMA this range; the
        Store still copies into the address through its TCP staging path.  The
        previous behaviour raised here, which silently pushed every PIC
        transition onto the 38 MB compact payload and its multi-millisecond
        deserialize on machines without RDMA.
        """
        register_buffer = getattr(self.store, "register_buffer", None)
        if not callable(register_buffer):
            return False
        if address in self._registered_buffer_ptrs:
            return True
        status: Any
        try:
            status = register_buffer(int(address), int(nbytes))
        except Exception as exc:  # noqa: BLE001 - status is reported below
            status = exc
        if status not in (None, 0):
            if not self._buffer_registration_warned:
                self._buffer_registration_warned = True
                logger.warning(
                    "Mooncake register_buffer failed for PIC buffer role=%s "
                    "address=%#x bytes=%d status=%r; continuing with the "
                    "unregistered destination address like Full Attention. "
                    "PIC transitions retry through pinned host staging.",
                    role,
                    int(address),
                    int(nbytes),
                    status,
                )
            return False
        self._registered_buffer_ptrs.add(int(address))
        self._registered_regions.append((int(address), int(nbytes)))
        return True

    def _unregister_store_buffer(self, address: int) -> None:
        """Release one registered address, keeping the bookkeeping in step."""
        self._registered_buffer_ptrs.discard(int(address))
        unregister = getattr(self.store, "unregister_buffer", None)
        if callable(unregister):
            with suppress(BaseException):
                unregister(int(address))
    def _put_layer_until_present(self, key: str, payload: bytes) -> int:
        """Write one layer blob and confirm the key exists before moving on.

        A burst of publishes was accepted with status zero while objects were not
        readable afterwards. The manifest read-back catches a missing manifest, but the
        bulk of an object is its eighteen layer blobs, and a missing one only surfaces
        later as a failed import. ``is_exist`` asks the store without transferring the
        blob, so each layer is confirmed in place - while its bytes are still in hand for
        a retry - instead of after the fact.

        Returns the attempt count. Stores without ``is_exist`` keep the old behaviour.
        """
        attempts = 2
        probe = getattr(self.store, "is_exist", None)
        for attempt in range(1, attempts + 1):
            self._check_status(self.store.put(key, payload), f"put {key}")
            if not callable(probe):
                return attempt
            try:
                present = bool(probe(key))
            except Exception:  # noqa: BLE001 - a probe failure must not fail a publish
                return attempt
            if present:
                return attempt
            if attempt < attempts:
                time.sleep(0.01 * attempt)
        logger.warning(
            "PIC native KV layer is not present after %d attempt(s): key=%s bytes=%d",
            attempts,
            key,
            len(payload),
        )
        return attempts

    def _publish_manifest_until_visible(self, key: str, encoded: bytes) -> None:
        """Write the manifest and confirm it can be read before returning.

        A successful put is not a promise of visibility. A burst of twenty publishes was
        accepted with status zero and only one could be read back afterwards, so the
        consumer saw a missing manifest, recomputed the whole prompt, and nothing in the
        log said why. Verifying costs one small read; when the object is not there yet the
        put is repeated a bounded number of times, so a publish that returns has landed.

        The manifest is the pointer to the layer blobs, so verifying it catches the failure
        that matters. Re-reading every layer would mean transferring the object twice.
        """
        attempts = self._publish_attempts()
        start = time.perf_counter()
        read_bytes = -1
        read_error = ""
        for attempt in range(1, attempts + 1):
            self._check_status(self.store.put(key, encoded), f"put {key}")
            read_bytes, read_error = self._read_back(key)
            if read_bytes == len(encoded):
                if _pic_import_profile_enabled():
                    logger.warning(
                        "[PIC-PROFILE] publish-verify key=%s written=%d read=%d "
                        "attempts=%d ms=%.3f",
                        key,
                        len(encoded),
                        read_bytes,
                        attempt,
                        (time.perf_counter() - start) * 1000.0,
                    )
                return
            if attempt < attempts:
                # Visibility lag is not constant: under a burst only a third of the puts were
                # accepted at first, and the object appeared afterwards. The backoff grows, so a
                # publish under load waits long enough instead of giving up after milliseconds.
                time.sleep(min(0.05 * (2 ** (attempt - 1)), 0.5))
        message = (
            "PIC native KV publish is not readable after %d attempt(s): "
            "key=%s written=%d read=%d error=%s"
        )
        rendered = message % (attempts, key, len(encoded), read_bytes, read_error)
        if self._publish_strict():
            raise PICWorkerUnsupported(rendered)
        logger.warning("%s", rendered)

    @staticmethod
    def _publish_attempts() -> int:
        raw = os.getenv("HYPIC_PIC_PUBLISH_ATTEMPTS", "")
        try:
            value = int(raw)
        except ValueError:
            return 12
        return max(1, min(value, 100))

    @staticmethod
    def _publish_strict() -> bool:
        return os.getenv("HYPIC_PIC_PUBLISH_STRICT", "") not in ("", "0", "false")

    def _read_back(self, key: str) -> tuple[int, str]:
        """Return the readable byte count for a key, or -1 with the error text."""
        try:
            back = self.store.get(key)
        except Exception as exc:  # noqa: BLE001 - visibility probe only
            return -1, repr(exc)
        if isinstance(back, (bytes, bytearray, memoryview)):
            return len(back), ""
        return -1, ""

    def publish_native_blocks(
        self,
        *,
        external_key: str,
        kv_cache_group_id: int,
        block_size: int,
        token_count: int,
        canonical_start: int,
        layout: str,
        kv_caches: Mapping[str, Any],
        block_ids: Sequence[int],
    ) -> PICNativeKVObject:
        """Store selected blocks without changing their native tensor layout."""
        if block_size <= 0 or token_count <= 0 or canonical_start < 0:
            raise ValueError("invalid native KV range")
        ids = tuple(int(block_id) for block_id in block_ids)
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError("native KV block IDs must be unique and non-negative")
        expected = PICWindow(
            canonical_start, canonical_start + token_count, block_size
        ).blocks()
        if len(ids) > expected:
            # The caller's span and the block list can disagree when the request
            # row covers more whole blocks than the published token range does.
            # Publishing the prefix that the span actually describes keeps the
            # object self-consistent; the tail stays a private, unreused block.
            logger.warning(
                "PIC native KV publish trims blocks to the declared span: "
                "external_key=%s declared=%d blocks got=%d",
                external_key,
                expected,
                len(ids),
            )
            ids = ids[:expected]
        if len(ids) != expected:
            raise ValueError(
                f"native KV block span mismatch: expected {expected}, got {len(ids)}"
            )
        if not kv_caches:
            raise ValueError("native KV object requires at least one layer")

        base_key = self._key(external_key)
        written: list[str] = []
        layers: list[PICNativeKVLayerObject] = []
        first_dtype: str | None = None
        try:
            ordered = sorted(kv_caches.items())
            # Pass one: validate every layer and work out how much device staging the batch needs.
            planned: list[tuple[str, Any, int, Any, tuple[int, ...], str, int]] = []
            total_bytes = 0
            for layer_name, kv_cache in ordered:
                kv_axis, block_axis = _native_cache_axes(kv_cache, layout=layout)
                if max(ids) >= kv_cache.shape[block_axis]:
                    raise ValueError("native KV block ID exceeds cache capacity")
                if kv_cache.shape[block_axis] <= 0:
                    raise ValueError("native KV cache has no blocks")
                dtype = _dtype_name(kv_cache.dtype)
                if first_dtype is None:
                    first_dtype = dtype
                elif first_dtype != dtype:
                    raise ValueError("native KV layers have different dtypes")
                # A consecutive run of blocks is already one contiguous region in a block-major cache,
                # so the payload is a view of it: one segment per layer instead of one per block.
                flat = _pic_flat_run(kv_cache, ids, layout)
                if flat is not None:
                    shape = (int(flat.numel()),)
                    nbytes = int(flat.numel()) * kv_cache.element_size()
                else:
                    # The published shape is what index_select produces: the block axis replaced by the
                    # number of blocks taken, every other axis unchanged.
                    shape = tuple(
                        len(ids) if position == block_axis else int(size)
                        for position, size in enumerate(kv_cache.shape)
                    )
                    nbytes = 1
                    for dimension in shape:
                        nbytes *= int(dimension)
                    nbytes *= kv_cache.element_size()
                planned.append(
                    (layer_name, kv_cache, block_axis, kv_axis, shape, dtype, nbytes)
                )
                total_bytes += nbytes

            # Pass two: send the blocks where they are.  There is no host path - a refusal is an
            # error, because the silent copy through the host is what this replaced.
            sent_all = False
            if planned:
                device = planned[0][1].device
                entries: list[tuple[str, list[tuple[int, int]]]] = []
                for layer_name, kv_cache, _, _, _, _, _ in planned:
                    flat = _pic_flat_run(kv_cache, ids, layout)
                    if flat is not None:
                        segments = [
                            (
                                int(flat.data_ptr()),
                                int(flat.numel()) * kv_cache.element_size(),
                            )
                        ]
                    else:
                        segments = self._layer_block_segments(
                            kv_cache, ids, layout, for_device=device
                        )
                    if segments is None:
                        raise RuntimeError(
                            "PIC native KV publish cannot address the blocks of layer "
                            f"{layer_name}"
                        )
                    entries.append((f"{base_key}/layers/{layer_name}", segments))
                sent_all = self._put_batch_from_segments(entries)
            if not sent_all:
                raise RuntimeError(
                    "PIC native KV publish needs the store's buffer API: "
                    "batch_put_from_multi_buffers refused the blocks"
                )
            for layer_name, _, _, _, shape, dtype, nbytes in planned:
                layer_key = f"{base_key}/layers/{layer_name}"
                written.append(layer_key)
                layers.append(
                    PICNativeKVLayerObject(
                        layer_name=layer_name,
                        key=layer_key,
                        shape=tuple(int(item) for item in shape),
                        dtype=dtype,
                        nbytes=int(nbytes),
                        checksum=None,
                    )
                )
            manifest = PICNativeKVObject(
                external_key=base_key,
                kv_cache_group_id=int(kv_cache_group_id),
                block_size=int(block_size),
                token_count=int(token_count),
                canonical_start=int(canonical_start),
                layout=str(layout),
                dtype=first_dtype or "",
                block_count=len(ids),
                layers=tuple(layers),
            )
            encoded = json.dumps(manifest.as_dict(), sort_keys=True).encode("utf-8")
            self._publish_manifest_until_visible(manifest.manifest_key, encoded)
            self._owned_native_keys.add(base_key)
            return manifest
        except Exception:
            for key in reversed(written):
                with suppress(Exception):
                    self.store.remove(key, force=True)
            raise

    def load(
        self,
        external_key: str,
        *,
        manifest: Mapping[str, Any] | None = None,
    ) -> PICNativeKVObject:
        key = self._key(external_key)
        profile = _pic_import_profile_enabled()
        start = time.perf_counter() if profile else 0.0
        raw: bytes | bytearray | memoryview | None = None
        if manifest is None:
            raw = self.store.get(key + "/manifest")
        if profile:
            logger.warning(
                "[PIC-PROFILE] native-manifest key=%s source=%s get_ms=%.3f bytes=%d",
                external_key,
                "store" if manifest is None else "reference",
                (time.perf_counter() - start) * 1000.0,
                (
                    len(raw)
                    if isinstance(raw, (bytes, bytearray, memoryview))
                    else -1
                )
                if manifest is None
                else len(json.dumps(manifest, sort_keys=True).encode("utf-8")),
            )
        if manifest is None:
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                raise ValueError("Mooncake native KV manifest is not bytes")
            if not raw:
                raise ValueError(
                    f"Mooncake native KV manifest is missing: {key}/manifest"
                )
            payload = json.loads(bytes(raw).decode("utf-8"))
        else:
            payload = dict(manifest)
        return PICNativeKVObject.from_dict(payload)

    def publish_transition_operator(
        self,
        *,
        external_key: str,
        operator: PICTransitionOperator,
    ) -> PICHybridStateObject:
        """Persist a position-independent hybrid transition beside native KV."""
        token_count = operator.token_end - operator.token_start
        if token_count <= 0:
            raise ValueError("hybrid transition operator has an invalid token range")
        base_key = self._key(external_key)
        payload_key = base_key + "/payload"
        store_layout: Mapping[str, Any] | None = None
        store_objects: tuple[tuple[str, bytes], ...] = ()
        try:
            store_layout, store_objects = self._build_transition_store_layout(
                operator, source_prefix=base_key
            )
        except (ValueError, RuntimeError) as exc:
            # Some legacy GDN operators cannot be represented by the fixed
            # matrix/state ABI. Keep the compact Store payload as a
            # correctness fallback rather than rejecting publication.
            logger.info(
                "PIC fixed-layout Store state unavailable for key=%s; "
                "keeping compatibility payload: %s",
                external_key,
                exc,
            )
        # The compact payload is a compatibility fallback once a Store fixed
        # layout exists.  Writing it anyway doubles the publish bytes and adds a
        # large slice to the segment, which both wastes capacity and creates the
        # worst fragmentation source.  Serialize it only when it can be read.
        need_payload = store_layout is None
        raw = _serialize_transition_operator(operator) if need_payload else b""
        if _pic_import_profile_enabled():
            wire_version = 0
            if raw:
                wire_version = (
                    _COMPACT_TRANSITION_VERSION
                    if raw.startswith(_COMPACT_TRANSITION_MAGIC)
                    else 2
                )
            logger.warning(
                "[PIC-PROFILE] transition-publish key=%s payload_bytes=%d "
                "wire_version=%d compact=%s",
                external_key,
                len(raw),
                wire_version,
                wire_version >= 2,
            )
        manifest = PICHybridStateObject(
            external_key=base_key,
            token_count=token_count,
            payload_key=payload_key,
            nbytes=len(raw),
            checksum=hashlib.sha256(raw).hexdigest() if raw else "",
            span_start=int(operator.token_start),
            store_layout=store_layout,
        )
        try:
            if need_payload:
                self._check_status(
                    self.store.put(payload_key, raw),
                    f"put {payload_key}",
                )
            for store_key, store_payload in store_objects:
                self._check_status(
                    self.store.put(store_key, store_payload),
                    f"put {store_key}",
                )
            encoded = json.dumps(manifest.as_dict(), sort_keys=True).encode("utf-8")
            self._check_status(
                self.store.put(manifest.manifest_key, encoded),
                f"put {manifest.manifest_key}",
            )
        except Exception:
            if need_payload:
                with suppress(Exception):
                    self.store.remove(payload_key, force=True)
            for store_key, _ in store_objects:
                with suppress(Exception):
                    self.store.remove(store_key, force=True)
            raise
        if store_layout is not None:
            self._owned_state_keys.add(base_key)
        return manifest

    def _build_transition_store_layout(
        self, operator: PICTransitionOperator, *, source_prefix: str
    ) -> tuple[dict[str, Any], tuple[tuple[str, bytes], ...]]:
        """Create a fixed-layout Store representation for transition tensors.

        The payload is still written as individual tensor objects for
        compatibility, but the manifest describes every tensor independently.
        A receiver can therefore issue one ``batch_get_into_multi_buffers``
        call into GPU destinations instead of reading and deserializing the
        large compact operator blob.
        """
        blob_key = f"{source_prefix}/store/state"
        # Every tensor goes into one blob at a recorded offset: the store charges per object, and fifty
        # of them cost more than the bytes they carry.
        chunks: list[bytes] = []
        cursor = 0

        def add_tensor(tensor: Any, role: str) -> dict[str, Any]:
            nonlocal cursor
            if tensor is None:
                raise ValueError("transition Store layout contains a null tensor")
            tensor = tensor.detach().contiguous()
            raw = _tensor_bytes(tensor)
            if not raw:
                raise ValueError("transition Store layout contains an empty tensor")
            offset = cursor
            chunks.append(raw)
            cursor += len(raw)
            return {
                "key": blob_key,
                "shape": [int(item) for item in tensor.shape],
                "dtype": _dtype_name(tensor.dtype),
                "nbytes": len(raw),
                "offset": offset,
            }

        transitions: list[dict[str, Any]] = []
        for index, transition in enumerate(operator.transitions):
            if isinstance(transition, PICAffineTransition):
                transitions.append(
                    {
                        "kind": "affine",
                        "tensors": {
                            "decay": add_tensor(
                                transition.decay, f"transition-{index}-decay"
                            ),
                            "zero_state": add_tensor(
                                transition.zero_state, f"transition-{index}-zero_state"
                            ),
                        },
                    }
                )
            elif isinstance(transition, PICLinearTransition):
                transitions.append(
                    {
                        "kind": "linear",
                        "tensors": {
                            "matrix": add_tensor(
                                transition.matrix, f"transition-{index}-matrix"
                            ),
                            "zero_state": add_tensor(
                                transition.zero_state, f"transition-{index}-zero_state"
                            ),
                        },
                    }
                )
            elif isinstance(transition, PICGDNTransition):
                compact = transition.compact()
                if compact is None:
                    raise ValueError("non-compact GDN transition has no Store layout")
                transitions.append(
                    {
                        "kind": "gdn_compact",
                        "tensors": {
                            "matrix": add_tensor(
                                compact.matrix, f"transition-{index}-matrix"
                            ),
                            "zero_state": add_tensor(
                                compact.zero_state, f"transition-{index}-zero_state"
                            ),
                        },
                    }
                )
            else:
                raise ValueError(
                    f"unsupported Store transition type: {type(transition)!r}"
                )

        conv_transitions: list[dict[str, Any]] = []
        for index, transition in enumerate(operator.conv_transitions):
            tail = (
                transition.inputs[-transition.state_length :]
                if transition.state_length > 0
                else transition.inputs[:0]
            )
            conv_transitions.append(
                {
                    "state_length": int(transition.state_length),
                    "source_token_count": operator.token_end - operator.token_start,
                    "tensor": add_tensor(tail, f"conv-{index}-inputs"),
                }
            )

        blob = b"".join(chunks)
        return (
            {
                "version": 1,
                "token_count": operator.token_end - operator.token_start,
                "blob": {"key": blob_key, "nbytes": len(blob)},
                "transitions": transitions,
                "conv_transitions": conv_transitions,
            },
            ((blob_key, blob),),
        )

    def load_transition_manifest(self, external_key: str) -> PICHybridStateObject:
        key = self._key(external_key)
        profile = _pic_import_profile_enabled()
        start = time.perf_counter() if profile else 0.0
        raw = self.store.get(key + "/manifest")
        if profile:
            logger.warning(
                "[PIC-PROFILE] transition-manifest key=%s get_ms=%.3f bytes=%d",
                external_key,
                (time.perf_counter() - start) * 1000.0,
                len(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else -1,
            )
        if not isinstance(raw, (bytes, bytearray, memoryview)) or not raw:
            raise ValueError(f"Mooncake hybrid state manifest is missing: {key}/manifest")
        return PICHybridStateObject.from_dict(json.loads(bytes(raw).decode("utf-8")))

    @staticmethod
    def _store_layout_tensor_records(
        layout: Mapping[str, Any],
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        records: list[tuple[str, Mapping[str, Any]]] = []
        for index, item in enumerate(layout.get("transitions", ())):
            tensors = item["tensors"]
            for name in ("decay", "matrix", "zero_state"):
                if name in tensors:
                    records.append((f"transition[{index}].{name}", tensors[name]))
        for index, item in enumerate(layout.get("conv_transitions", ())):
            records.append((f"conv[{index}].inputs", item["tensor"]))
        return tuple(records)

    @staticmethod
    def _store_batch_succeeded(result: Any, expected: int) -> bool:
        """Whether every Store batch status is non-negative."""
        try:
            statuses = list(result)
        except TypeError:
            return False
        if len(statuses) != expected:
            return False
        return all(int(status) >= 0 for status in statuses)
    def _load_transition_operator_store(
        self, manifest: PICHybridStateObject, *, device: Any
    ) -> PICTransitionOperator:
        layout = manifest.store_layout
        if not layout or int(layout.get("version", -1)) != 1:
            raise ValueError("unsupported PIC Store transition descriptor")
        if int(layout.get("token_count", -1)) != manifest.token_count:
            raise ValueError("Store transition token count mismatch")
        batch_get = getattr(self.store, "batch_get_into_multi_buffers", None)
        if not callable(batch_get):
            raise NotImplementedError("Mooncake Store batch receive is unavailable")

        records = self._store_layout_tensor_records(layout)
        blob = layout.get("blob") or {}
        blob_key = str(blob.get("key") or "")
        total = int(blob.get("nbytes") or 0)
        if not blob_key or total <= 0:
            raise ValueError("Store transition descriptor has no blob")
        on_cuda = getattr(device, "type", None) == "cuda"
        slab = None
        representation = "device-direct"
        elapsed_ms = 0.0
        if on_cuda and self._gpu_direct_enabled():
            try:
                slab = _torch().empty(total, dtype=_torch().uint8, device=device)
                start = time.perf_counter()
                result = batch_get([blob_key], [[int(slab.data_ptr())]], [[total]])
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if not self._store_batch_succeeded(result, 1):
                    slab = None
            except Exception:  # noqa: BLE001 - fall back to the host arena
                slab = None
            if slab is not None:
                pass
        if slab is None:
            representation = "store-host-staging"
            host = _torch().empty(total, dtype=_torch().uint8, pin_memory=on_cuda)
            self._register_store_buffer_best_effort(
                address=int(host.data_ptr()),
                nbytes=int(host.numel()),
                role="transition-host-arena",
            )
            start = time.perf_counter()
            result = batch_get([blob_key], [[int(host.data_ptr())]], [[total]])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not self._store_batch_succeeded(result, 1):
                raise RuntimeError("Mooncake Store transition receive failed")
            slab = _h2d_on_private_stream(host, device)

        targets: dict[str, Any] = {}
        torch = _torch()
        views_start = time.perf_counter()
        for role, record in records:
            offset = int(record["offset"])
            nbytes = int(record["nbytes"])
            dtype = _dtype_from_name(str(record["dtype"]))
            shape = tuple(int(size) for size in record["shape"])
            targets[role] = (
                slab[offset : offset + nbytes]
                .view(dtype)
                .reshape(shape)
            )
        views_ms = (time.perf_counter() - views_start) * 1000.0
        if _pic_import_profile_enabled():
            logger.warning(
                "[PIC-PROFILE] transition-phases key=%s receive_ms=%.3f "
                "upload_ms=0.000 views_ms=%.3f total_ms=%.3f representation=%s",
                manifest.external_key,
                elapsed_ms,
                views_ms,
                elapsed_ms + views_ms,
                representation,
            )

        transitions: list[PICStateTransition] = []
        for index, item in enumerate(layout.get("transitions", ())):
            kind = str(item["kind"])
            tensors = item["tensors"]
            if kind == "affine":
                transitions.append(
                    PICAffineTransition(
                        decay=targets[f"transition[{index}].decay"],
                        zero_state=targets[f"transition[{index}].zero_state"],
                    )
                )
            elif kind in ("linear", "gdn_compact"):
                transitions.append(
                    PICLinearTransition(
                        matrix=targets[f"transition[{index}].matrix"],
                        zero_state=targets[f"transition[{index}].zero_state"],
                    )
                )
            else:
                raise ValueError(f"unsupported Store transition kind: {kind}")
        conv_transitions = tuple(
            PICConvTransition(
                inputs=targets[f"conv[{index}].inputs"],
                state_length=int(item["state_length"]),
                source_token_count=int(item["source_token_count"]),
            )
            for index, item in enumerate(layout.get("conv_transitions", ()))
        )
        operator = PICTransitionOperator(
            transitions=tuple(transitions),
            conv_transitions=conv_transitions,
            # Span-local coordinates: the operator covers [span_start, span_end)
            # of its segment and callers normalise by token_start.
            token_start=manifest.span_start,
            token_end=manifest.span_start + manifest.token_count,
        )
        # The slab belongs to this operator now: the views are slices of it, so holding the operator
        # holds the bytes.
        if _pic_import_profile_enabled():
            logger.warning(
                "[PIC-PROFILE] transition-store-total key=%s "
                "representation=%s tensors=%d bytes=%d get_ms=%.3f "
                "deserialize_ms=0.000 total_ms=%.3f",
                manifest.external_key,
                representation,
                len(records),
                sum(int(record["nbytes"]) for _, record in records),
                elapsed_ms,
                elapsed_ms,
            )
        return operator

    def load_transition_operator(
        self, external_key: str, *, device: Any
    ) -> tuple[PICHybridStateObject, PICTransitionOperator]:
        """Load a remote hybrid transition on the target device."""
        profile = _pic_import_profile_enabled()
        verify_checksum = _pic_payload_checksum_enabled()
        total_start = time.perf_counter() if profile else 0.0
        manifest = self.load_transition_manifest(external_key)
        if manifest.store_layout is not None and not self._store_transition_disabled:
            try:
                operator = self._load_transition_operator_store(
                    manifest, device=device
                )
                self._store_transition_failures = 0
                if profile:
                    logger.warning(
                        "[PIC-PROFILE] transition-transfer key=%s "
                        "representation=store-fixed-layout total_ms=%.3f",
                        external_key,
                        (time.perf_counter() - total_start) * 1000.0,
                    )
                return manifest, operator
            except (NotImplementedError, ValueError, RuntimeError) as exc:
                if _is_cuda_oom_error(exc):
                    raise
                self._store_transition_failures += 1
                if self._store_transition_failures >= self._STORE_TRANSITION_FAILURE_BUDGET:
                    self._store_transition_disabled = True
                    logger.warning(
                        "PIC fixed-layout Store transition receive failed %d "
                        "times; disabling it for this process so requests do "
                        "not pay a failed direct receive before the payload "
                        "fallback",
                        self._store_transition_failures,
                    )
                logger.info(
                    "PIC fixed-layout Store transition unavailable for key=%s; "
                    "using compatibility path: %s cause=%r",
                    external_key,
                    exc,
                    exc.__cause__,
                )
        if manifest.nbytes <= 0:
            # The publisher no longer writes the compatibility payload when a
            # Store fixed layout exists.  Surface it as a load error so the
            # caller applies its normal request-level fallback.
            raise ValueError(
                "hybrid transition manifest has no compatibility payload"
            )
        payload_start = time.perf_counter() if profile else 0.0
        raw = self.store.get(manifest.payload_key)
        payload_ms = (time.perf_counter() - payload_start) * 1000.0 if profile else 0.0
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise ValueError("Mooncake hybrid transition payload is not bytes")
        raw_bytes = raw if isinstance(raw, bytes) else bytes(raw)
        if len(raw_bytes) != manifest.nbytes:
            raise ValueError("Mooncake hybrid transition payload size is incompatible")
        checksum_start = time.perf_counter() if profile and verify_checksum else 0.0
        if verify_checksum and hashlib.sha256(raw_bytes).hexdigest() != manifest.checksum:
            raise ValueError(
                "Mooncake hybrid transition payload checksum is incompatible"
            )
        checksum_ms = (
            (time.perf_counter() - checksum_start) * 1000.0
            if profile and verify_checksum
            else 0.0
        )
        torch = _torch()
        _log_pic_cuda_memory(
            torch,
            device,
            external_key=external_key,
            stage="store_deserialize_begin",
            target_bytes=0,
        )
        deserialize_start = time.perf_counter() if profile else 0.0
        operator = _deserialize_transition_operator(raw_bytes, device=device)
        deserialize_ms = (
            (time.perf_counter() - deserialize_start) * 1000.0
            if profile
            else 0.0
        )
        _log_pic_cuda_memory(
            torch,
            device,
            external_key=external_key,
            stage="store_deserialize_ready",
            target_bytes=_transition_operator_tensor_bytes(operator),
        )
        if operator.token_end - operator.token_start != manifest.token_count:
            raise ValueError("Mooncake hybrid transition token count is incompatible")
        if operator.token_start != manifest.span_start:
            # The payload format stores a span-local range starting at zero; the
            # manifest carries the offset into the segment, so restore it here
            # and callers can treat both representations identically.
            operator = replace(
                operator,
                token_start=manifest.span_start,
                token_end=manifest.span_start + manifest.token_count,
            )
        if profile:
            logger.warning(
                "[PIC-PROFILE] transition-total key=%s representation=compat-payload "
                "payload_bytes=%d payload_get_ms=%.3f checksum_enabled=%s "
                "checksum_ms=%.3f deserialize_ms=%.3f total_ms=%.3f",
                external_key,
                len(raw_bytes),
                payload_ms,
                verify_checksum,
                checksum_ms,
                deserialize_ms,
                (time.perf_counter() - total_start) * 1000.0,
            )
        return manifest, operator

    def remove_transition(self, obj: PICHybridStateObject) -> None:
        if obj.nbytes > 0:
            self._check_status(
                self.store.remove(obj.payload_key, force=True),
                "remove hybrid transition payload",
            )
        for key in self._store_layout_keys(obj.store_layout):
            self._check_status(
                self.store.remove(key, force=True),
                "remove hybrid transition tensor",
            )
        self._check_status(
            self.store.remove(obj.manifest_key, force=True),
            "remove hybrid transition manifest",
        )

    def restore_native_blocks(
        self, obj: PICNativeKVObject, *, kv_caches: Mapping[str, Any], block_ids: Sequence[int]
    ) -> int:
        """Read a published object straight into its destination blocks."""
        ids = tuple(int(block_id) for block_id in block_ids)
        if len(ids) != obj.block_count or len(set(ids)) != len(ids):
            raise ValueError("native KV destination block span does not match object")
        entries: list[tuple[str, list[tuple[int, int]]]] = []
        for member in obj.layers:
            target = kv_caches.get(member.layer_name)
            if target is None:
                raise ValueError(f"missing native KV layer: {member.layer_name}")
            flat = _pic_flat_run(target, ids, obj.layout)
            if flat is not None:
                segments = [
                    (int(flat.data_ptr()), int(flat.numel()) * target.element_size())
                ]
            else:
                segments = self._layer_block_segments(
                    target, ids, obj.layout, for_device=target.device
                )
            if segments is None:
                raise RuntimeError(
                    "PIC native KV restore cannot address the blocks of layer "
                    f"{member.layer_name}"
                )
            entries.append((member.key, segments))
        if not self._get_batch_into_segments(entries):
            raise RuntimeError(
                "PIC native KV restore needs the store's buffer API: "
                "batch_get_into_multi_buffers refused the blocks"
            )
        return sum(int(member.nbytes) for member in obj.layers)

    def _receive_native_blocks_host_batch(
        self, obj: PICNativeKVObject, *, block_ids: Sequence[int]
    ) -> PICNativeKVReceived:
        """Receive every native KV layer with one batched Store read."""
        batch_get = getattr(self.store, "batch_get_into_multi_buffers", None)
        if not callable(batch_get):
            raise NotImplementedError("Mooncake batch direct receive is unavailable")
        ids = tuple(int(block_id) for block_id in block_ids)
        if len(ids) != obj.block_count or len(set(ids)) != len(ids):
            raise ValueError("native KV destination block span does not match object")
        # Straight into the destination blocks when the worker handed over the layers: the blocks are the
        # same span the publisher wrote, so the segment list is computable on this side too.
        if self._kv_cache_layers and self._gpu_direct_enabled():
            direct: list[tuple[str, list[tuple[int, int]]]] = []
            for member in obj.layers:
                target = self._kv_cache_layers.get(member.layer_name)
                if target is None:
                    direct = []
                    break
                flat = _pic_flat_run(target, ids, obj.layout)
                if flat is not None:
                    segments = [
                        (
                            int(flat.data_ptr()),
                            int(flat.numel()) * target.element_size(),
                        )
                    ]
                else:
                    segments = self._layer_block_segments(
                        target, ids, obj.layout, for_device=target.device
                    )
                if segments is None:
                    direct = []
                    break
                direct.append((member.key, segments))
            if direct and self._get_batch_into_segments(direct):
                return PICNativeKVReceived(obj=obj)
        raise RuntimeError(
            "PIC native KV receive needs the store's buffer API and the destination layer tensors"
        )

    def receive_native_blocks(
        self,
        obj: PICNativeKVObject,
        *,
        block_ids: Sequence[int],
    ) -> PICNativeKVReceived:
        """Receive native layer payloads without GPU materialization.

        This is the P5 receiver stage.  It performs only synchronous
        Mooncake reads and payload validation; the caller owns the later
        CPU-to-GPU/index-copy stage through :meth:`materialize_native_blocks`.
        """
        ids = tuple(int(block_id) for block_id in block_ids)
        if len(ids) != obj.block_count or len(set(ids)) != len(ids):
            raise ValueError("native KV destination block span does not match object")
        # Batched Store pull is the only native KV data path: one BatchQuery plus
        # one parallel BatchGet for every layer, instead of a Query+Get per layer.
        if self._native_store_disabled:
            raise RuntimeError(
                "PIC native KV Store import is disabled after repeated failures"
            )
        try:
            return self._receive_native_blocks_host_batch(obj, block_ids=ids)
        except NotImplementedError as exc:
            logger.info(
                "PIC batched native KV receive unavailable for key=%s: %s",
                obj.external_key,
                exc,
            )
            raise
        except (ValueError, RuntimeError) as exc:
            if _is_cuda_oom_error(exc):
                raise
            # A runtime transfer failure must not fall back to the per-layer
            # reads: each request would then open one connection per layer on
            # top of the failed batch and a single transport hiccup turns
            # into an ephemeral-port storm.  Fail the request instead and
            # stop using the store after a few consecutive failures.
            self._native_batch_failures += 1
            if self._native_batch_failures >= _NATIVE_BATCH_FAILURE_BUDGET:
                self._native_store_disabled = True
                logger.error(
                    "PIC native KV Store import disabled after %d consecutive "
                    "failures (last key=%s): %s",
                    self._native_batch_failures,
                    obj.external_key,
                    exc,
                )
            raise

    def materialize_native_blocks(
        self,
        received: PICNativeKVReceived | None,
        *,
        kv_caches: Mapping[str, Any],
        block_ids: Sequence[int],
    ) -> None:
        """A receive already wrote the destination blocks, so there is nothing to materialize."""
        del kv_caches, block_ids
        if received is not None:
            self._last_materialized_key = received.obj.external_key

    def remove(self, obj: PICNativeKVObject) -> None:
        for member in obj.layers:
            self._check_status(
                self.store.remove(member.key, force=True), "remove KV layer"
            )
        self._check_status(
            self.store.remove(obj.manifest_key, force=True), "remove KV manifest"
        )

    def make_transfer_callback(
        self,
        destination_writer: Callable[
            [PICExternalKVImportRequest, PICNativeKVObject, tuple[int, ...], Any],
            None,
        ],
    ) -> Callable[[PICExternalKVImportRequest, tuple[int, ...], Any], None]:
        """Adapt object reads to the provider's fail-closed transfer callback."""

        def transfer(
            request: PICExternalKVImportRequest,
            block_ids: tuple[int, ...],
            engine: Any,
        ) -> None:
            obj = self.load(
                request.external_key,
                manifest=request.object_manifest,
            )
            request_dtype = (
                None
                if request.dtype is None
                else str(request.dtype).removeprefix("torch.")
            )
            if (
                obj.kv_cache_group_id != request.kv_cache_group_id
                or obj.block_size != request.block_size
                or obj.token_count != request.token_count
                or obj.canonical_start != request.canonical_start
                or obj.dtype != request_dtype
            ):
                raise ValueError("Mooncake native KV object metadata mismatch")
            destination_writer(request, obj, block_ids, engine)

        return transfer

    def make_receive_callback(
        self,
    ) -> Callable[[PICExternalKVImportRequest, tuple[int, ...], Any], Any]:
        """Build the receive-only callback used by the request-level P5 path."""

        def receive(
            request: PICExternalKVImportRequest,
            block_ids: tuple[int, ...],
            _engine: Any,
        ) -> PICNativeKVReceived:
            obj = self.load(
                request.external_key,
                manifest=request.object_manifest,
            )
            request_dtype = (
                None
                if request.dtype is None
                else str(request.dtype).removeprefix("torch.")
            )
            if (
                obj.kv_cache_group_id != request.kv_cache_group_id
                or obj.block_size != request.block_size
                or obj.token_count != request.token_count
                or obj.canonical_start != request.canonical_start
                or obj.dtype != request_dtype
            ):
                raise ValueError("Mooncake native KV object metadata mismatch")
            return self.receive_native_blocks(obj, block_ids=block_ids)

        return receive


def _pic_lookup_miss(reason: str, **fields: object) -> None:
    """Say why a lookup gave up.

    A miss is invisible by design - the request simply recomputes - and every one of these paths looks
    identical from the outside. Naming the reason and the numbers behind it is the difference between
    one run that identifies the fault and several that guess at it.
    """
    if not _pic_import_profile_enabled():
        return
    detail = " ".join(f"{name}={value}" for name, value in fields.items())
    logger.warning("[PIC-LOOKUP] miss reason=%s %s", reason, detail)


class PICMooncakeReferenceIndex:
    """Discover external PIC entries from Mooncake manifests.

    The index contains metadata only.  It never exposes remote block IDs to
    attention; the worker imports each object into request-owned native blocks
    through :class:`MooncakePICExternalKVProvider`.
    """

    def __init__(
        self,
        object_store: MooncakeNativeKVObjectStore,
        *,
        group_block_sizes: Mapping[int, int],
        group_dtypes: Mapping[int, str],
        require_hybrid_state: bool = False,
    ) -> None:
        self.object_store = object_store
        self.require_hybrid_state = bool(require_hybrid_state)
        self.group_block_sizes = {
            int(group_id): int(block_size)
            for group_id, block_size in group_block_sizes.items()
        }
        self.group_dtypes = {
            int(group_id): str(dtype).removeprefix("torch.")
            for group_id, dtype in group_dtypes.items()
        }

    def lookup(
        self, segment: PICSegment, group_ids: Sequence[int]
    ) -> Any | None:
        """Return an external cache entry only when every group is present."""
        from vllm.v1.pic.cache import PICSegmentEntry

        references: list[PICNativeKVReference] = []
        for group_id in group_ids:
            block_size = self.group_block_sizes.get(int(group_id))
            if block_size is None:
                _pic_lookup_miss(
                    "no_group_block_size",
                    group=int(group_id),
                    known=sorted(self.group_block_sizes),
                )
                return None
            # ``block_size`` is the scheduler/allocator granularity.  A
            # native Mooncake object may use a smaller attention-kernel block
            # size when vLLM virtually splits one allocator block (for
            # example, 544 scheduler tokens into 17 physical blocks of 32).
            # The scheduler-sized span still defines how many logical tokens
            # the segment publishes; the manifest's own block size defines
            # how many physical native blocks must be present.
            # Covering span: the publisher describes the window up to the segment's last token, so a
            # reader that rounds down to a kernel block boundary asks for a shorter window - four
            # thousand eight hundred and thirty-two against four thousand eight hundred and
            # thirty-nine - and rejects an entry that is exactly what it needs.
            span = get_covering_local_block_span(segment.length, block_size)
            if span.block_count <= 0:
                _pic_lookup_miss(
                    "empty_span",
                    group=int(group_id),
                    length=segment.length,
                    block_size=block_size,
                )
                return None
            external_key = pic_native_kv_key(segment.seg_hash, group_id)
            base_key = f"segment/{segment.seg_hash.hex()}/group-{int(group_id)}"
            if pic_peer_tp_suffixes() and not self._tp_peers_published(
                base_key, pic_peer_tp_suffixes
            ):
                _pic_lookup_miss(
                    "tp_shard_missing",
                    key=base_key,
                    group=int(group_id),
                )
                return None
            try:
                obj = self.object_store.load(external_key)
            except Exception as exc:  # noqa: BLE001 - the miss is what matters
                _pic_lookup_miss(
                    "native_object_unavailable",
                    key=external_key,
                    group=int(group_id),
                    error=type(exc).__name__,
                )
                return None
            expected_dtype = self.group_dtypes.get(int(group_id))
            if obj.block_size <= 0:
                _pic_lookup_miss(
                    "object_block_size_invalid",
                    key=external_key,
                    block_size=obj.block_size,
                )
                return None
            object_window = PICWindow(
                obj.canonical_start,
                obj.canonical_start + obj.token_count,
                obj.block_size,
            )
            if (
                obj.kv_cache_group_id != int(group_id)
                or obj.canonical_start != 0
                or obj.token_count != span.reusable_token_count
                or not object_window.fits(obj.block_count)
                or (expected_dtype is not None and obj.dtype != expected_dtype)
            ):
                if _pic_import_profile_enabled():
                    logger.warning(
                        "[PIC-DEBUG] lookup rejected key=%s group=%d "
                        "obj=(group=%s canonical=%s token=%s blocks=%s block_size=%s dtype=%s) "
                        "expected=(group=%s canonical=0 token=%s blocks=%s dtype=%s)",
                        external_key,
                        int(group_id),
                        obj.kv_cache_group_id,
                        obj.canonical_start,
                        obj.token_count,
                        obj.block_count,
                        obj.block_size,
                        obj.dtype,
                        int(group_id),
                        span.reusable_token_count,
                        object_window.blocks(),
                        expected_dtype,
                    )
                return None
            references.append(
                PICNativeKVReference(
                    kv_cache_group_id=int(group_id),
                    block_ids=(),
                    block_size=obj.block_size,
                    token_count=obj.token_count,
                    kind=PICNativeKVKind.CANONICAL_PUBLIC,
                    position_mode=PICKVPositionMode.CANONICAL_LOCAL,
                    canonical_start=0,
                    source_token_start=0,
                    source_block_start=0,
                    dtype=obj.dtype,
                    layout=(obj.block_size,),
                    external_provider="mooncake",
                    external_key=obj.external_key,
                    external_manifest=obj.as_dict(),
                )
            )
        if not references:
            _pic_lookup_miss(
                "no_references",
                groups=len(tuple(group_ids)),
                length=segment.length,
            )
            return None
        transition_key = pic_hybrid_state_key(segment.seg_hash)
        if self.require_hybrid_state:
            base_transition_key = f"segment/{segment.seg_hash.hex()}/hybrid-state"
            if pic_peer_tp_suffixes() and not self._tp_peers_published(
                base_transition_key, pic_peer_tp_suffixes
            ):
                _pic_lookup_miss(
                    "tp_shard_missing",
                    key=base_transition_key,
                )
                return None
            try:
                state = self.object_store.load_transition_manifest(transition_key)
            except Exception as exc:  # noqa: BLE001 - the miss is what matters
                _pic_lookup_miss(
                    "transition_manifest_unavailable",
                    key=transition_key,
                    error=type(exc).__name__,
                )
                return None
            # The published transition covers the span the reuse window takes,
            # which starts after a seam and stops one token before the segment
            # ends - the last token is recomputed because the request samples from
            # it.  It therefore ends inside a page rather than on a block boundary,
            # and it does not cover the whole segment: validate the bounds instead
            # of requiring equality with the segment length.
            # ``span_start`` defaults to zero so manifests written before the
            # span existed, and light stand-ins in tests, keep working.
            span_start = int(getattr(state, "span_start", 0) or 0)
            token_count = int(state.token_count)
            if (
                token_count <= 0
                or span_start < 0
                or span_start + token_count > segment.length
            ):
                _pic_lookup_miss(
                    "transition_bounds_invalid",
                    key=transition_key,
                    span_start=span_start,
                    token_count=token_count,
                    length=segment.length,
                )
                return None
        else:
            transition_key = None
        return PICSegmentEntry(
            seg_hash=segment.seg_hash,
            token_ids=segment.token_ids,
            native_kv_refs=tuple(references),
            external_transition_key=transition_key,
        )
