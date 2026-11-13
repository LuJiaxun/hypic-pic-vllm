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
from threading import Lock
from typing import Any, Protocol
from weakref import finalize

from vllm.v1.pic.external_kv import PICExternalKVImportRequest
from vllm.v1.pic.native_kv import (
    PICWindow,
    register_pending_upload,
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


def _pic_profile_sync(torch_module: Any, device: Any, enabled: bool) -> None:
    """Synchronize only when import profiling explicitly requests it."""
    if (
        not enabled
        or getattr(device, "type", None) != "cuda"
        or not torch_module.cuda.is_available()
    ):
        return
    torch_module.cuda.synchronize(device)


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
class PICNativeKVHostStaging:
    """One host arena holding every layer payload of a native KV object.

    Produced by the batched Store receive: a single
    ``batch_get_into_multi_buffers()`` writes all layers into ``arena`` at the
    offsets in ``layers``.  Materialization uploads the arena once and scatters
    each layer into its native blocks.
    """

    arena: Any
    layers: tuple[tuple[str, int, int], ...]
    _release: Any = None

    def release(self, event: Any = None) -> None:
        if self._release is not None:
            self._release(event)
            object.__setattr__(self, "_release", None)


@dataclass(frozen=True)
class PICNativeKVReceived:
    """Native KV payloads received from Mooncake but not yet materialized."""

    obj: PICNativeKVObject
    # Batched Store receives land in one host arena and are uploaded once.
    host_staging: PICNativeKVHostStaging | None = None


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



def _parse_byte_size(value: str, *, name: str) -> int:
    """Parse a byte size that may carry a K/M/G suffix.

    ``int(os.getenv(...))`` silently rejected "512M", and the rejection surfaced
    three imports later as "import disabled", which reads like a code regression.
    Parse once, at construction, and fail loudly instead.
    """
    text = str(value).strip().upper().replace("_", "")
    multipliers = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    if text and text[-1] in multipliers:
        number, factor = text[:-1], multipliers[text[-1]]
    else:
        number, factor = text, 1
    try:
        parsed = int(float(number) * factor)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a byte count such as 536870912, 512M or 2G; "
            f"got {value!r}"
        ) from None
    if parsed <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}")
    return parsed


class _PinnedStagingPool:
    """A few reusable pinned host arenas for transition transfers.

    Pinning a buffer and registering it with the transport cost more than the copy
    that uses them, and an arena sized to one object cannot serve the next object of
    a different size - so a run over many distinct objects paid both costs once per
    object.  A fixed number of arenas, all grown to the largest object seen and
    reused across objects, pays them once.

    The transfer itself is synchronous between receive and upload, so a slot is only
    held for that window; a slot per import thread keeps concurrent imports apart.
    """

    def __init__(
        self,
        *,
        slots: int,
        granularity: int,
        register: Callable[..., Any] | None = None,
        unregister: Callable[[int], None] | None = None,
    ) -> None:
        self._slots = max(1, int(slots))
        self._granularity = max(1, int(granularity))
        self._register = register
        self._unregister = unregister
        self._free: list[Any] = []
        # Arenas handed back while a copy still reads them, with the event that ends that read.
        self._pending: list[tuple[Any, Any]] = []
        self._busy = 0
        self._capacity = 0
        self._condition = threading.Condition()
        self.allocations = 0

    def _reap(self) -> None:
        """Move arenas whose copy finished back to the free list."""
        still_reading: list[tuple[Any, Any]] = []
        for arena, event in self._pending:
            try:
                finished = bool(event.query())
            except Exception:  # noqa: BLE001 - a dead event must not pin an arena forever
                finished = True
            if finished:
                self._free.append(arena)
                self._busy = max(0, self._busy - 1)
            else:
                still_reading.append((arena, event))
        self._pending[:] = still_reading

    def _round_up(self, nbytes: int) -> int:
        return (
            (int(nbytes) + self._granularity - 1) // self._granularity
        ) * self._granularity

    def acquire(self, nbytes: int, *, pin: bool) -> Any:
        torch = _torch()
        want = self._round_up(nbytes)
        profile = _pic_import_profile_enabled()
        with self._condition:
            while True:
                if want > self._capacity:
                    self._capacity = want
                for index, arena in enumerate(self._free):
                    if int(arena.numel()) >= self._capacity:
                        self._free.pop(index)
                        self._busy += 1
                        if profile:
                            # A reuse line per transfer plus one allocation line per
                            # arena is what tells "paid once" from "paid per object".
                            logger.warning(
                                "[PIC-PROFILE] staging-pool-use bytes=%d arena=%d "
                                "allocations=%d",
                                int(nbytes),
                                int(arena.numel()),
                                self.allocations,
                            )
                        return arena
                drop = None
                if self._free:
                    # A free arena is too small for the grown capacity: replace it
                    # rather than carry two sizes around.
                    drop = self._free.pop()
                elif len(self._free) + self._busy >= self._slots:
                    if self._pending:
                        # Every slot is out, and the ones still out are only waiting for a copy
                        # that already runs on the copy engine. Waiting for that copy is short and
                        # bounded, while waiting on the condition alone could block forever because
                        # nothing signals it when a copy completes.
                        arena, event = self._pending[0]
                        with suppress(BaseException):
                            if not event.query():
                                event.synchronize()
                        self._reap()
                        continue
                    self._condition.wait()
                    continue
                if drop is not None and self._unregister is not None:
                    with suppress(BaseException):
                        self._unregister(int(drop.data_ptr()))
                previous = 0 if drop is None else int(drop.numel())
                # Pinning an arena costs more than the copy that uses it - measured at a hundred and
                # ninety milliseconds for thirty-eight megabytes - and paying it per request put that
                # cost on the critical path of every import. The first request allocates the whole
                # pool instead, so every later one borrows.
                warm = drop is None and self.allocations == 0
                self._busy += 1
                alloc_start = time.perf_counter()
                arena = torch.empty(
                    self._capacity, dtype=torch.uint8, pin_memory=pin
                )
                alloc_ms = (time.perf_counter() - alloc_start) * 1000.0
                self.allocations += 1
                register_ms = 0.0
                if self._register is not None:
                    register_start = time.perf_counter()
                    self._register(
                        address=int(arena.data_ptr()),
                        nbytes=int(arena.numel()),
                        role="transition-host-arena",
                    )
                    register_ms = (time.perf_counter() - register_start) * 1000.0
                if profile:
                    logger.warning(
                        "[PIC-PROFILE] staging-pool bytes=%d previous=%d pin=%s "
                        "alloc_ms=%.3f register_ms=%.3f allocation=%d",
                        int(arena.numel()),
                        previous,
                        pin,
                        alloc_ms,
                        register_ms,
                        self.allocations,
                    )
                if warm:
                    for _ in range(max(0, self._slots - 1)):
                        extra = torch.empty(
                            self._capacity, dtype=torch.uint8, pin_memory=pin
                        )
                        self.allocations += 1
                        if self._register is not None:
                            with suppress(BaseException):
                                self._register(
                                    address=int(extra.data_ptr()),
                                    nbytes=int(extra.numel()),
                                    role="transition-host-arena",
                                )
                        self._free.append(extra)
                    if profile:
                        logger.warning(
                            "[PIC-PROFILE] staging-pool warmed slots=%d capacity=%d",
                            self._slots,
                            self._capacity,
                        )
                    self._condition.notify_all()
                return arena

    def release(self, arena: Any, event: Any = None) -> None:
        """Return an arena to the pool once its copy has finished.

        A returned arena is about to be reused by another object, so handing it back while the copy
        engine is still reading it would corrupt that object's transfer. The event defers the return
        instead of waiting for it, which is what keeps the upload off the critical path.
        """
        if event is not None:
            try:
                finished = bool(event.query())
            except Exception:  # noqa: BLE001 - a dead event must not pin an arena
                finished = True
            if not finished:
                with self._condition:
                    self._pending.append((arena, event))
                return
        self._release_now(arena)

    def _release_now(self, arena: Any) -> None:
        with self._condition:
            self._busy = max(0, self._busy - 1)
            self._free.append(arena)
            self._condition.notify()

    def stats(self) -> tuple[int, int, int]:
        with self._condition:
            return self.allocations, self._capacity, len(self._free) + self._busy


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
        # staging pool stays for the paths that still need bytes in hand (fallback and checksum).
        self._gpu_scratch: dict[str, Any] = {}
        self._gpu_scratch_device: Any = None
        self._gpu_direct_available = True
        # The put call takes a replicate config.  Built once, and left None when the installed
        # binding does not expose it, in which case the three-argument form is tried.
        self._replicate_config_ready = False
        self._replicate_config_value: Any = None
        # Layer name to KV tensor, handed over once by the worker.  A receive that can address the
        # destination blocks needs them; without them it falls back to the host arena.
        self._kv_cache_layers: dict[str, Any] = {}
        # One arena per import thread, all grown to the largest object seen: the
        # pin and the registration are paid once instead of once per object.
        staging_slots = 4
        # Escape hatch for an A/B run: the pool is only useful if the per-object
        # arena was expensive, and that has to be measurable, not assumed.
        self._shared_staging = True
        self._staging_pool = _PinnedStagingPool(
            slots=staging_slots,
            granularity=1 << 20,
            register=self._register_store_buffer_best_effort,
            unregister=self._unregister_store_buffer,
        )
        # Parsed once so a malformed value fails at startup instead of silently
        # disabling imports a few requests later.
        self._native_arena_cache_bytes = _parse_byte_size(
            os.getenv(
                "HYPIC_PIC_NATIVE_ARENA_CACHE_BYTES", str(512 * 1024 * 1024)
            ),
            name="native arena cache",
        )
        self._native_arena_evictions = 0
        # Keep the warm blocks. Dropping them only hands them back to the pinned
        # allocator's cache, which then may decline to serve a slightly smaller
        # request - measured as the same forty to two hundred millisecond pin landing
        # on a request anyway. Holding them makes the request path a pure memcpy.
        self._free_native_arenas: list[Any] = []
        self._pending_native_arenas: list[tuple[Any, Any]] = []
        self._native_pool_lock = Lock()
        self._warm_native_arenas()
        self._owned_native_keys: set[str] = set()
        self._owned_state_keys: set[str] = set()
        # Store direct state receives use request-independent, registered GPU
        # tensors keyed by external object.  The operator keeps the tensors
        # alive for the request; the finalizer releases the cache entry when
        # the request drops its operator.  This avoids registering a fresh
        # destination range on every reuse while preserving the Store ABI.
        self._store_state_buffers: dict[str, dict[str, Any]] = {}
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
        # Batched native KV receives land in one host arena per external object.
        # The arena is rented for the receive -> materialize window so a
        # concurrent receive of the same object cannot overwrite it mid-upload.
        self._native_arenas: dict[str, dict[str, Any]] = {}

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

    def _scratch_for(self, nbytes: int, role: str) -> Any | None:
        """A registered device buffer of at least ``nbytes``, kept per role.

        One tensor per role, grown on demand and registered once: registering a fresh temporary on every
        layer would cost more than the copy it saves.  Publish and import run in different threads and can
        overlap, so they do not share a buffer.
        """
        if nbytes <= 0:
            return None
        pooled = self._gpu_scratch.get(role)
        if pooled is not None:
            tensor, capacity = pooled
            if capacity >= nbytes and tensor.device == self._gpu_scratch_device:
                return tensor[:nbytes]
        tensor = _torch().empty(nbytes, dtype=_torch().uint8, device=self._gpu_scratch_device)
        # Registering is an optimisation, not a permission: without a pin the store still copies into
        # this address through its own staging.  Only a failed call decides the path.
        self._register_store_buffer_best_effort(
            address=tensor.data_ptr(), nbytes=int(tensor.numel()), role=f"gpu-scratch-{role}"
        )
        self._gpu_scratch[role] = (tensor, int(tensor.numel()))
        return tensor

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

    def _put_batch_from_device(
        self, entries: Sequence[tuple[str, int, int]]
    ) -> bool:
        """Store several objects' bytes straight from device memory, in one call."""
        if not self._gpu_direct_enabled() or not entries:
            return False
        put = getattr(self.store, "batch_put_from_multi_buffers", None)
        try:
            status = put(
                [key for key, _, _ in entries],
                [[address] for _, address, _ in entries],
                [[nbytes] for _, _, nbytes in entries],
            )
        except Exception:  # noqa: BLE001 - fall back to the host path
            return False
        return (
            status is not None
            and len(status) == len(entries)
            and all(int(value) >= 0 for value in status)
        )

    def _get_batch_into_device(
        self, entries: Sequence[tuple[str, int, int]]
    ) -> bool:
        """Read several objects' bytes straight into device memory, in one call."""
        if not self._gpu_direct_enabled() or not entries:
            return False
        get_into = getattr(self.store, "batch_get_into_multi_buffers", None)
        try:
            status = get_into(
                [key for key, _, _ in entries],
                [[address] for _, address, _ in entries],
                [[nbytes] for _, _, nbytes in entries],
            )
        except Exception:  # noqa: BLE001 - fall back to the host path
            return False
        return (
            status is not None
            and len(status) == len(entries)
            and all(int(value) >= 0 for value in status)
        )

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

    def _release_store_state_buffers(self, entry: dict[str, Any]) -> None:
        """Unregister only the state buffers whose registration succeeded.

        The host staging arena is owned by the pool now, so it is not released
        here: the pool keeps exactly a few arenas alive for the whole process and
        unregisters one only when it replaces it with a larger arena.
        """
        registered = entry.get("registered", {})
        for role, tensor in entry.get("buffers", {}).items():
            if not registered.get(role):
                continue
            self._unregister_store_buffer(int(tensor.data_ptr()))

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

            # Pass two: send the blocks where they are, or fall back to staging them.
            staging = None
            sent_all = False
            offsets: list[int] = []
            if self._gpu_scratch_device is None and planned:
                self._gpu_scratch_device = planned[0][1].device
            # The device path cannot hash bytes it never holds, so taking it is a choice about the
            # checksum, not about profiling.
            if not _pic_payload_checksum_enabled() and planned:
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
                        entries = []
                        break
                    entries.append((f"{base_key}/layers/{layer_name}", segments))
                if entries:
                    sent_all = self._put_batch_from_segments(entries)
            if not sent_all and not _pic_payload_checksum_enabled():
                staging = self._scratch_for(int(total_bytes), "publish")
            if staging is not None and not sent_all:
                offset = 0
                for _, kv_cache, block_axis, _, shape, _, nbytes in planned:
                    flat = _pic_flat_run(kv_cache, ids, layout)
                    if flat is not None:
                        selected = flat
                    else:
                        index = _torch().tensor(
                            ids, dtype=_torch().long, device=kv_cache.device
                        )
                        selected = kv_cache.index_select(block_axis, index).contiguous()
                    staging[offset : offset + nbytes].view(kv_cache.dtype).reshape(
                        shape
                    ).copy_(selected)
                    offsets.append(offset)
                    offset += nbytes
                entries_staged = [
                    (
                        f"{base_key}/layers/{layer_name}",
                        int(staging.data_ptr()) + offset,
                        nbytes,
                    )
                    for offset, (layer_name, _, _, _, _, _, nbytes) in zip(
                        offsets, planned
                    )
                ]
                sent_all = self._put_batch_from_device(entries_staged)

            for position, (
                layer_name,
                kv_cache,
                block_axis,
                kv_axis,
                shape,
                dtype,
                nbytes,
            ) in enumerate(planned):
                layer_key = f"{base_key}/layers/{layer_name}"
                payload: bytes | None = None
                if not sent_all:
                    if staging is not None:
                        # The batch was refused: the bytes are already on device, so the fallback only
                        # pays the copy back to the host instead of gathering again.
                        offset = offsets[position]
                        payload = (
                            staging[offset : offset + nbytes]
                            .view(kv_cache.dtype)
                            .reshape(shape)
                            .cpu()
                            .numpy()
                            .tobytes()
                        )
                    else:
                        flat = _pic_flat_run(kv_cache, ids, layout)
                        if flat is not None:
                            payload = _tensor_bytes(flat.reshape(shape))
                        else:
                            index = _torch().tensor(
                                ids, dtype=_torch().long, device=kv_cache.device
                            )
                            selected = kv_cache.index_select(
                                block_axis, index
                            ).contiguous()
                            payload = _tensor_bytes(selected)
                    self._put_layer_until_present(layer_key, payload)
                    written.append(layer_key)
                else:
                    written.append(layer_key)
                layers.append(
                    PICNativeKVLayerObject(
                        layer_name=layer_name,
                        key=layer_key,
                        shape=tuple(int(item) for item in shape),
                        dtype=dtype,
                        nbytes=int(nbytes),
                        checksum=(
                            None
                            if payload is None
                            else hashlib.sha256(payload).hexdigest()
                        ),
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

    def _store_state_cache_key(self, external_key: str, device: Any) -> str:
        return f"{external_key}|device={device}"
    def _discard_store_state_entry(self, cache_key: str) -> None:
        entry = self._store_state_buffers.get(cache_key)
        if entry is None or int(entry.get("users", 0)) != 0:
            return
        self._store_state_buffers.pop(cache_key, None)
        self._release_store_state_buffers(entry)

    def _store_state_tensor(
        self,
        *,
        cache_key: str,
        entry: dict[str, Any],
        role: str,
        record: Mapping[str, Any],
        device: Any,
    ) -> Any:
        torch = _torch()
        shape = tuple(int(item) for item in record["shape"])
        dtype = _dtype_from_name(str(record["dtype"]))
        nbytes = int(record["nbytes"])
        if any(item < 0 for item in shape) or nbytes <= 0:
            raise ValueError(f"invalid Store transition tensor metadata: {role}")
        expected = 1
        for extent in shape:
            expected *= extent
        if expected * int(torch.empty((), dtype=dtype).element_size()) != nbytes:
            raise ValueError(f"Store transition tensor size mismatch: {role}")
        buffers = entry.setdefault("buffers", {})
        registered = entry.setdefault("registered", {})
        tensor = buffers.get(role)
        if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            registered[role] = self._register_store_buffer_best_effort(
                address=int(tensor.data_ptr()),
                nbytes=nbytes,
                role=role,
            )
            buffers[role] = tensor
        return tensor

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
                self._discard_store_state_entry(
                    self._store_state_cache_key(manifest.external_key, device)
                )
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
        self,
        obj: PICNativeKVObject,
        *,
        kv_caches: Mapping[str, Any],
        block_ids: Sequence[int],
    ) -> None:
        """Restore object blocks into caller-owned native KV blocks."""
        ids = tuple(int(block_id) for block_id in block_ids)
        if len(ids) != obj.block_count or len(set(ids)) != len(ids):
            raise ValueError("native KV destination block span does not match object")
        torch = _torch()
        profile = _pic_import_profile_enabled()
        verify_checksum = _pic_payload_checksum_enabled()
        total_start = time.perf_counter() if profile else 0.0
        total_bytes = 0
        total_get_ms = 0.0
        total_checksum_ms = 0.0
        total_cpu_ms = 0.0
        total_h2d_ms = 0.0
        # Straight into the request's blocks when the store can address them: the destination is a list
        # of per-block segments, so there is nothing to stage.  Validation runs first - a mismatch must
        # fail before a byte is written.
        if not profile and not verify_checksum:
            direct_entries: list[tuple[str, list[tuple[int, int]]]] = []
            for member in obj.layers:
                target = kv_caches.get(member.layer_name)
                if target is None:
                    direct_entries = []
                    break
                _, target_block_axis = _native_cache_axes(target, layout=obj.layout)
                if len(target.shape) != len(member.shape) or any(
                    left != right
                    for position, (left, right) in enumerate(
                        zip(target.shape, member.shape)
                    )
                    if position != target_block_axis
                ):
                    direct_entries = []
                    break
                if _dtype_name(target.dtype) != member.dtype:
                    direct_entries = []
                    break
                segments = self._layer_block_segments(
                    target, ids, obj.layout, for_device=target.device
                )
                if segments is None:
                    direct_entries = []
                    break
                direct_entries.append((member.key, segments))
            if direct_entries and self._get_batch_into_segments(direct_entries):
                if profile:
                    logger.warning(
                        "[PIC-PROFILE] native-direct key=%s layers=%d bytes=%d",
                        obj.external_key,
                        len(direct_entries),
                        sum(int(member.nbytes) for member in obj.layers),
                    )
                return

        # One call for the whole group when the store can write into device memory; the offsets are
        # reused per layer below.
        device_staging = None
        device_batch_offsets: dict[str, int] | None = None
        if not profile:
            total_device_bytes = sum(int(member.nbytes) for member in obj.layers)
            if self._gpu_scratch_device is None and obj.layers:
                first_layer = kv_caches.get(obj.layers[0].layer_name)
                self._gpu_scratch_device = (
                    first_layer.device if first_layer is not None else None
                )
            device_staging = self._scratch_for(int(total_device_bytes), "import")
            if device_staging is not None:
                device_batch_offsets = {}
                entries: list[tuple[str, int, int]] = []
                offset = 0
                for member in obj.layers:
                    device_batch_offsets[member.key] = offset
                    entries.append(
                        (
                            member.key,
                            int(device_staging.data_ptr()) + offset,
                            int(member.nbytes),
                        )
                    )
                    offset += int(member.nbytes)
                if not self._get_batch_into_device(entries):
                    device_staging = None
                    device_batch_offsets = None

        for member in obj.layers:
            kv_cache = kv_caches.get(member.layer_name)
            if kv_cache is None:
                raise ValueError(f"missing native KV layer: {member.layer_name}")
            _, block_axis = _native_cache_axes(kv_cache, layout=obj.layout)
            if len(kv_cache.shape) != len(member.shape) or any(
                left != right
                for index, (left, right) in enumerate(zip(kv_cache.shape, member.shape))
                if index != block_axis
            ):
                raise ValueError("native KV layer shape is incompatible")
            if _dtype_name(kv_cache.dtype) != member.dtype:
                raise ValueError("native KV layer dtype is incompatible")
            get_start = time.perf_counter() if profile else 0.0
            device_values = None
            if device_batch_offsets is not None:
                offset = device_batch_offsets.get(member.key)
                if offset is not None:
                    device_values = device_staging[
                        offset : offset + int(member.nbytes)
                    ].view(_dtype_from_name(member.dtype)).reshape(member.shape)
            if device_values is not None:
                flat = (
                    _pic_flat_run(kv_cache, ids, obj.layout)
                    if device_values.numel() == member.nbytes // kv_cache.element_size()
                    and device_values.ndim == 1
                    else None
                )
                if flat is not None:
                    flat.copy_(device_values.reshape(-1))
                else:
                    index = torch.tensor(ids, dtype=torch.long, device=kv_cache.device)
                    kv_cache.index_copy_(block_axis, index, device_values)
                total_bytes += int(member.nbytes)
                continue
            raw = self.store.get(member.key)
            get_ms = (time.perf_counter() - get_start) * 1000.0 if profile else 0.0
            if len(raw) != member.nbytes:
                raise ValueError("native KV layer payload size is incompatible")
            checksum_start = time.perf_counter() if profile and verify_checksum else 0.0
            if (
                verify_checksum
                and member.checksum is not None
                and hashlib.sha256(bytes(raw)).hexdigest() != member.checksum
            ):
                raise ValueError("native KV layer payload checksum is incompatible")
            checksum_ms = (
                (time.perf_counter() - checksum_start) * 1000.0
                if profile and verify_checksum
                else 0.0
            )
            cpu_start = time.perf_counter() if profile else 0.0
            values = torch.frombuffer(
                memoryview(raw), dtype=_dtype_from_name(member.dtype)
            )
            values = values.clone().reshape(member.shape)
            cpu_ms = (time.perf_counter() - cpu_start) * 1000.0 if profile else 0.0
            _pic_profile_sync(torch, kv_cache.device, profile)
            h2d_start = time.perf_counter() if profile else 0.0
            values = values.to(device=kv_cache.device)
            flat = (
                _pic_flat_run(kv_cache, ids, obj.layout)
                if values.ndim == 1
                else None
            )
            if flat is not None:
                flat.copy_(values.reshape(-1))
            else:
                index = torch.tensor(ids, dtype=torch.long, device=kv_cache.device)
                kv_cache.index_copy_(block_axis, index, values)
            _pic_profile_sync(torch, kv_cache.device, profile)
            h2d_ms = (time.perf_counter() - h2d_start) * 1000.0 if profile else 0.0
            total_bytes += len(raw)
            total_get_ms += get_ms
            total_checksum_ms += checksum_ms
            total_cpu_ms += cpu_ms
            total_h2d_ms += h2d_ms
            if profile:
                logger.warning(
                    "[PIC-PROFILE] native-layer key=%s layer=%s bytes=%d "
                    "get_ms=%.3f checksum_enabled=%s checksum_ms=%.3f "
                    "cpu_ms=%.3f h2d_index_ms=%.3f total_ms=%.3f",
                    obj.external_key,
                    member.layer_name,
                    len(raw),
                    get_ms,
                    verify_checksum,
                    checksum_ms,
                    cpu_ms,
                    h2d_ms,
                    (time.perf_counter() - get_start) * 1000.0,
                )
        if profile:
            logger.warning(
                "[PIC-PROFILE] native-total key=%s layers=%d bytes=%d "
                "get_ms=%.3f checksum_ms=%.3f cpu_ms=%.3f h2d_index_ms=%.3f "
                "total_ms=%.3f",
                obj.external_key,
                len(obj.layers),
                total_bytes,
                total_get_ms,
                total_checksum_ms,
                total_cpu_ms,
                total_h2d_ms,
                (time.perf_counter() - total_start) * 1000.0,
            )

    @staticmethod
    def _native_host_plan(
        obj: PICNativeKVObject,
    ) -> tuple[dict[str, tuple[int, int]], int]:
        """Byte layout of every layer inside one host arena."""
        plan: dict[str, tuple[int, int]] = {}
        total = 0
        for member in obj.layers:
            nbytes = int(member.nbytes)
            if nbytes <= 0:
                raise ValueError("native KV layer payload is empty")
            offset = (
                (total + _NATIVE_STAGING_ALIGNMENT - 1)
                // _NATIVE_STAGING_ALIGNMENT
            ) * _NATIVE_STAGING_ALIGNMENT
            plan[member.layer_name] = (offset, nbytes)
            total = offset + nbytes
        return plan, total

    def _warm_native_arenas(self) -> None:
        """Pin the first arena at construction instead of during a request.

        ``torch.empty(pin_memory=True)`` is cheap once the pinned allocator has a
        block of that size and expensive when it must ask the driver for one, and the
        expensive case landed on the first request of every cold process. Allocating
        and dropping a few blocks here leaves them in the allocator's cache, which is
        exactly what the request path then reuses.
        """
        torch = _torch()
        try:
            if not torch.cuda.is_available():
                return
        except Exception:  # pragma: no cover - non-CUDA builds
            return
        try:
            nbytes = _parse_byte_size(
                os.getenv("HYPIC_PIC_KV_ARENA_BYTES", str(64 * 1024 * 1024)),
                name="HYPIC_PIC_KV_ARENA_BYTES",
            )
            count = 4
            blocks = [
                torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(count)
            ]
            with self._native_pool_lock:
                self._free_native_arenas.extend(blocks)
            if _pic_import_profile_enabled():
                logger.warning(
                    "[PIC-PROFILE] kv-arena-warmup bytes=%d count=%d", nbytes, count
                )
        except Exception as exc:  # noqa: BLE001 - warmup must never block startup
            logger.warning("PIC KV arena warmup skipped: %s", exc)

    def _evict_native_arenas(self, incoming: int) -> None:
        """Keep the pinned arena cache inside its byte budget.

        Pinned host memory is a finite resource, so idle arenas are dropped when
        a new receive would push the cache over budget.  Busy arenas are never
        evicted: they are serving an in-flight receive/materialize window.
        """
        budget = getattr(
            self, "_native_arena_cache_bytes", 512 * 1024 * 1024
        )
        cached = sum(int(entry["arena"].numel()) for entry in self._native_arenas.values())
        if cached + incoming <= budget:
            return
        for key, entry in tuple(self._native_arenas.items()):
            if cached + incoming <= budget:
                break
            with entry["lock"]:
                if entry["busy"]:
                    continue
                pending = entry.get("pending")
                if pending is not None:
                    # A completed upload event no longer protects the buffer. Only
                    # an in-flight copy does. Leaving it set made every entry that
                    # had uploaded once permanently unevictable, so the cache grew
                    # without bound and later pinnings got slower and slower.
                    if not pending.query():
                        continue
                    entry["pending"] = None
            self._native_arenas.pop(key, None)
            self._native_arena_evictions = (
                getattr(self, "_native_arena_evictions", 0) + 1
            )
            cached -= int(entry["arena"].numel())

    def _take_warm_arena(self, total: int) -> Any | None:
        """Return a retained arena that fits, recycling finished uploads first."""
        with self._native_pool_lock:
            pending = self._pending_native_arenas
            for item in tuple(pending):
                arena, event = item
                if event is None or event.query():
                    pending.remove(item)
                    self._free_native_arenas.append(arena)
            for index, arena in enumerate(self._free_native_arenas):
                if int(arena.numel()) >= total:
                    self._free_native_arenas.pop(index)
                    return arena
        return None

    def _rent_native_arena(
        self, external_key: str, total: int
    ) -> tuple[Any, Callable[..., None]]:
        """Rent the cached arena for a receive -> materialize window.

        The device upload reads this buffer asynchronously, so the cached arena
        may only be reused once the previous upload finished.  ``release``
        therefore carries the upload event, and a rent whose predecessor is
        still in flight gets a throwaway buffer instead of blocking.  Without
        that check the profile-off path has no ``cuda.synchronize`` to drain the
        copy and the next ``batch_get`` can overwrite bytes the copy still
        reads.
        """
        torch = _torch()
        pin = bool(torch.cuda.is_available())
        pooled = self._take_warm_arena(total)
        if pooled is not None:
            def release_pooled(event: Any = None) -> None:
                # The upload reads the arena asynchronously, so it only returns to
                # the free list once that copy has finished.
                with self._native_pool_lock:
                    self._pending_native_arenas.append((pooled, event))

            return pooled, release_pooled
        entry = self._native_arenas.get(external_key)
        if entry is None or int(entry["arena"].numel()) < total:
            profile = _pic_import_profile_enabled()
            # A new object needs a new pinned arena, and pinning 53 MB is neither
            # free nor reusable across differently sized objects. This is the cost
            # that a repeated object never pays and a first-seen one always does.
            evicted = len(self._native_arenas)
            self._evict_native_arenas(total)
            evicted -= len(self._native_arenas)
            alloc_start = time.perf_counter() if profile else 0.0
            entry = {
                "arena": torch.empty(total, dtype=torch.uint8, pin_memory=pin),
                "busy": False,
                "pending": None,
                "lock": Lock(),
            }
            alloc_ms = (time.perf_counter() - alloc_start) * 1000.0 if profile else 0.0
            self._native_arenas[external_key] = entry
            if profile:
                logger.warning(
                    "[PIC-PROFILE] kv-arena key=%s bytes=%d alloc_ms=%.3f "
                    "pinned=%s evicted=%d cached_bytes=%d budget_bytes=%d",
                    external_key,
                    int(total),
                    alloc_ms,
                    pin,
                    evicted,
                    sum(
                        int(item["arena"].numel())
                        for item in self._native_arenas.values()
                    ),
                    getattr(self, "_native_arena_cache_bytes", 0),
                )
        elif _pic_import_profile_enabled():
            logger.warning(
                "[PIC-PROFILE] kv-arena key=%s bytes=%d hit=True cached=%d",
                external_key,
                int(total),
                len(self._native_arenas),
            )
        with entry["lock"]:
            if not entry["busy"]:
                pending = entry.get("pending")
                if pending is None or pending.query():
                    entry["pending"] = None
                    entry["busy"] = True
                    arena = entry["arena"]
                    released = False

                    def release(event: Any = None) -> None:
                        nonlocal released
                        with entry["lock"]:
                            if released:
                                return
                            released = True
                            if event is not None:
                                entry["pending"] = event
                            entry["busy"] = False

                    return arena, release
        # Either the same object is already in flight or its previous upload is
        # still reading the cached arena: use a throwaway buffer, never block.
        return torch.empty(total, dtype=torch.uint8, pin_memory=pin), (
            lambda event=None: None
        )

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
        plan, total = self._native_host_plan(obj)
        arena, release = self._rent_native_arena(obj.external_key, total)
        base = int(arena.data_ptr())
        keys: list[str] = []
        addrs: list[list[int]] = []
        sizes: list[list[int]] = []
        layout: list[tuple[str, int, int]] = []
        for member in obj.layers:
            offset, nbytes = plan[member.layer_name]
            keys.append(member.key)
            addrs.append([base + offset])
            sizes.append([nbytes])
            layout.append((member.layer_name, offset, nbytes))
        start = time.perf_counter()
        try:
            result = batch_get(keys, addrs, sizes)
        except BaseException:
            release()
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not self._store_batch_succeeded(result, len(keys)):
            release()
            raise RuntimeError("Mooncake native KV batch receive failed")
        if _pic_import_profile_enabled():
            logger.warning(
                "[PIC-PROFILE] native-receive-batch key=%s layers=%d bytes=%d "
                "get_ms=%.3f",
                obj.external_key,
                len(keys),
                total,
                elapsed_ms,
            )
            logger.warning(
                "[PIC-PROFILE] native-receive-total key=%s mode=batch layers=%d "
                "bytes=%d total_ms=%.3f n_gets=1 per_get_median=%.3f "
                "per_get_max=%.3f",
                obj.external_key,
                len(keys),
                total,
                elapsed_ms,
                elapsed_ms,
                elapsed_ms,
            )
        staging = PICNativeKVHostStaging(
            arena=arena,
            layers=tuple(layout),
            _release=release,
        )
        return PICNativeKVReceived(obj=obj, host_staging=staging)

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
        """Materialize a received native object into vLLM KV blocks."""
        if received is None:
            return
        if received.host_staging is not None:
            # One device upload for the whole arena, then one scatter per layer.
            torch = _torch()
            obj = received.obj
            ids = tuple(int(block_id) for block_id in block_ids)
            if len(ids) != obj.block_count or len(set(ids)) != len(ids):
                raise ValueError(
                    "native KV destination block span does not match object"
                )
            staging = received.host_staging
            members = {member.layer_name: member for member in obj.layers}
            first = next(iter(kv_caches.values()), None)
            if first is None:
                staging.release()
                raise ValueError("native KV staging has no destination cache")
            profile = _pic_import_profile_enabled()
            total_start = time.perf_counter() if profile else 0.0
            upload_start = time.perf_counter() if profile else 0.0
            device_arena = staging.arena.to(device=first.device)
            # The upload reads staging.arena asynchronously.  Record when it
            # finished so a later rent cannot overwrite bytes the copy still
            # reads, which matters on the profile-off path because there is no
            # synchronize there to drain it.
            upload_event = None
            if first.device.type == "cuda":
                with suppress(Exception):
                    upload_event = torch.cuda.Event()
                    upload_event.record()
            # Without this the upload time is only the enqueue cost and the real
            # copy shows up later in the forward bucket.
            _pic_profile_sync(torch, first.device, profile)
            upload_ms = (
                (time.perf_counter() - upload_start) * 1000.0 if profile else 0.0
            )
            scatter_start = time.perf_counter() if profile else 0.0
            try:
                for layer_name, offset, nbytes in staging.layers:
                    member = members.get(layer_name)
                    kv_cache = kv_caches.get(layer_name)
                    if member is None or kv_cache is None:
                        raise ValueError(f"missing native KV layer: {layer_name}")
                    if _dtype_name(kv_cache.dtype) != member.dtype:
                        raise ValueError("native KV layer dtype is incompatible")
                    block_axis = _native_cache_axes(kv_cache, layout=obj.layout)[1]
                    values = (
                        device_arena.narrow(0, offset, nbytes)
                        .view(_dtype_from_name(member.dtype))
                        .reshape(member.shape)
                    )
                    index = torch.tensor(
                        ids, dtype=torch.long, device=kv_cache.device
                    )
                    kv_cache.index_copy_(block_axis, index, values)
                _pic_profile_sync(torch, first.device, profile)
            finally:
                staging.release(upload_event)
            if profile:
                scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
                staged_bytes = sum(item[2] for item in staging.layers)
                logger.warning(
                    "[PIC-PROFILE] native-materialize-batch key=%s layers=%d "
                    "bytes=%d h2d_ms=%.3f scatter_ms=%.3f",
                    obj.external_key,
                    len(staging.layers),
                    staged_bytes,
                    upload_ms,
                    scatter_ms,
                )
                logger.warning(
                    "[PIC-PROFILE] native-materialize-total key=%s mode=batch "
                    "layers=%d bytes=%d cpu_ms=0.000 h2d_index_ms=%.3f "
                    "total_ms=%.3f",
                    obj.external_key,
                    len(staging.layers),
                    staged_bytes,
                    upload_ms + scatter_ms,
                    (time.perf_counter() - total_start) * 1000.0,
                )
            return

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
