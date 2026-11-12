# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import replace

import pytest
import torch

from vllm.v1.pic.gdn_transition import build_gdn_transition_operator
from vllm.v1.pic.mooncake_kv import (
    MooncakeNativeKVObjectStore,
    PICHybridStateObject,
    PICNativeKVLayerObject,
    PICNativeKVObject,
)
from vllm.v1.pic.state import PICLinearTransition


class _DirectStore:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int]] = []
        self.batch_calls: list[tuple[list[str], list[list[int]], list[list[int]]]] = []

    def register_buffer(self, base_addr: int, size: int) -> int:
        self.registered.append((base_addr, size))
        return 0

    def batch_get_into_multi_buffers(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
    ) -> list[int]:
        self.batch_calls.append((keys, addrs, sizes))
        return [0 for _ in keys]

    def put(self, key: str, value: bytes) -> int:
        raise AssertionError("direct receive test should not publish")

    def get(self, key: str) -> bytes:
        raise AssertionError("direct receive test should not use legacy get")

    def remove(self, key: str, force: bool = False) -> int:
        return 0


def _object_for_cache(cache: torch.Tensor) -> PICNativeKVObject:
    selected = cache.index_select(1 if cache.shape[0] == 2 else 0, torch.tensor([1, 3]))
    raw_bytes = selected.numel() * selected.element_size()
    member = PICNativeKVLayerObject(
        layer_name="layer",
        key="hypic/native-kv/v1/segment/direct/layers/layer",
        shape=tuple(int(item) for item in selected.shape),
        dtype="float16",
        nbytes=raw_bytes,
    )
    return PICNativeKVObject(
        external_key="hypic/native-kv/v1/segment/direct",
        kv_cache_group_id=0,
        block_size=4,
        token_count=8,
        canonical_start=0,
        layout="auto",
        dtype="float16",
        block_count=2,
        layers=(member,),
    )


def _padded_nhd_cache() -> torch.Tensor:
    # Each NHD block has four float32 payload elements but occupies an
    # eight-element page, leaving four elements of padding after every block.
    storage = torch.empty(64, dtype=torch.float32)
    return torch.as_strided(
        storage,
        size=(2, 4, 2, 1, 2),
        stride=(32, 8, 4, 4, 2),
    )


def test_default_store_pull_contract_never_reads_the_compat_payload(monkeypatch) -> None:
    """Default env: native KV is direct and transitions use host staging.

    This is the contract Stage 11E-C establishes.  No legacy switch is set, so
    both receives must use the Store batch API, the compat payload must never be
    read, and the transition operator must come from one shared device arena.
    """

    monkeypatch.setenv("HYPIC_PIC_DEBUG", "1")

    class Store:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}
            self.batch_calls: list[
                tuple[list[str], list[list[int]], list[list[int]]]
            ] = []
            self.payload_reads: list[str] = []

        def register_buffer(self, base_addr: int, size: int) -> int:
            return 0

        def batch_get_into_multi_buffers(
            self,
            keys: list[str],
            addrs: list[list[int]],
            sizes: list[list[int]],
        ) -> list[int]:
            self.batch_calls.append(
                (list(keys), [list(item) for item in addrs], [list(item) for item in sizes])
            )
            return [0] * len(keys)

        def put(self, key: str, value: bytes) -> int:
            self.objects[key] = bytes(value)
            return 0

        def get(self, key: str) -> bytes:
            if key.endswith("/payload"):
                self.payload_reads.append(key)
                raise AssertionError(
                    "default Store pull must not read the compat payload"
                )
            return self.objects[key]

        def remove(self, key: str, force: bool = False) -> int:
            self.objects.pop(key, None)
            return 0

    store = Store()
    objects = MooncakeNativeKVObjectStore(store, key_prefix="hypic")

    cache = torch.empty((2, 4, 2, 2, 2), dtype=torch.float16)
    received = objects.receive_native_blocks(_object_for_cache(cache), block_ids=(0, 2))
    assert received.host_staging is not None
    native_batch_calls = len(store.batch_calls)

    key = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[0.5, 1.0]]], dtype=torch.float32
    )
    value = torch.tensor(
        [[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]], dtype=torch.float32
    )
    operator = build_gdn_transition_operator(
        key,
        value,
        torch.zeros((4, 1), dtype=torch.float32),
        torch.ones((4, 1), dtype=torch.float32),
        token_start=0,
        token_end=4,
        state_dtype=torch.float32,
    )
    objects.publish_transition_operator(
        external_key="segment/default/hybrid-state", operator=operator
    )

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Capture()
    module_logger = logging.getLogger("vllm.v1.pic.mooncake_kv")
    module_logger.addHandler(handler)
    try:
        _, loaded = objects.load_transition_operator(
            "segment/default/hybrid-state", device="cpu"
        )
    finally:
        module_logger.removeHandler(handler)

    assert isinstance(loaded.transitions[0], PICLinearTransition)
    assert len(store.batch_calls) == native_batch_calls + 1
    assert store.payload_reads == []

    tensors = [loaded.transitions[0].matrix, loaded.transitions[0].zero_state]
    tensors.extend(item.inputs for item in loaded.conv_transitions)
    tensors = [tensor for tensor in tensors if tensor.numel() > 0]
    assert len({tensor.untyped_storage().data_ptr() for tensor in tensors}) == 1

    joined = "\n".join(records)
    assert "representation=store-host-staging" in joined
    assert "deserialize_ms=0.000" in joined


def test_native_kv_batch_receive_uses_one_call_and_skips_per_layer_get(
    monkeypatch,
) -> None:
    """Batched native receive: one Store call, one arena, no per-layer get."""

    import ctypes

    torch = pytest.importorskip("torch")


    class Store:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.get_calls = 0
            self.fill = 0x3C

        def register_buffer(self, base_addr: int, size: int) -> int:
            return 0

        def batch_get_into_multi_buffers(self, keys, addrs, sizes):
            self.batch_calls += 1
            for key_addrs, key_sizes in zip(addrs, sizes):
                for address, size in zip(key_addrs, key_sizes):
                    payload = bytes([self.fill]) * size
                    ctypes.memmove(address, payload, size)
            return [0] * len(keys)

        def get(self, key: str) -> bytes:
            self.get_calls += 1
            raise AssertionError("per-layer store.get must not be used")

        def put(self, key: str, value: bytes) -> int:
            raise AssertionError("receive path must not publish")

        def remove(self, key: str, force: bool = False) -> int:
            return 0

    store = Store()
    objects = MooncakeNativeKVObjectStore(store, key_prefix="hypic")
    cache = torch.zeros((2, 4, 2, 2, 2), dtype=torch.float16)
    obj = _object_for_cache(cache)

    received = objects.receive_native_blocks(obj, block_ids=(0, 2))

    assert received.host_staging is not None
    assert store.batch_calls == 1
    assert store.get_calls == 0

    objects.materialize_native_blocks(
        received, kv_caches={"layer": cache}, block_ids=(0, 2)
    )
    expected_value = torch.frombuffer(
        bytearray([store.fill, store.fill]), dtype=torch.float16
    )[0].item()
    expected = torch.full(obj.layers[0].shape, expected_value, dtype=torch.float16)
    assert torch.equal(cache.index_select(1, torch.tensor([0, 2])), expected)
    # Blocks that were not selected stay untouched.
    assert torch.count_nonzero(cache.index_select(1, torch.tensor([1, 3]))) == 0

    # The arena must be handed back so the next request can rent it. With the
    # retained pool it returns through the pending list, because the upload may
    # still be reading it; the next rent reaps it once its event completes.
    available = objects._free_native_arenas + [
        arena for arena, _ in objects._pending_native_arenas
    ]
    assert available, "a released arena must be available for the next request"
    assert (
        objects._take_warm_arena(int(received.host_staging.arena.numel()))
        is not None
    )


def test_native_kv_batch_failure_opens_no_per_layer_connections(monkeypatch) -> None:
    """A rejected batch must fail the request, not open per-layer connections."""

    torch = pytest.importorskip("torch")


    class Store:
        def __init__(self, layer_bytes: int) -> None:
            self.layer_bytes = layer_bytes
            self.batch_calls = 0
            self.get_calls = 0

        def register_buffer(self, base_addr: int, size: int) -> int:
            return 0

        def batch_get_into_multi_buffers(self, keys, addrs, sizes):
            self.batch_calls += 1
            return [-1] * len(keys)

        def get(self, key: str) -> bytes:
            self.get_calls += 1
            return bytes([0x11]) * self.layer_bytes

        def put(self, key: str, value: bytes) -> int:
            raise AssertionError("receive path must not publish")

        def remove(self, key: str, force: bool = False) -> int:
            return 0

    cache = torch.zeros((2, 4, 2, 2, 2), dtype=torch.float16)
    obj = _object_for_cache(cache)
    store = Store(obj.layers[0].nbytes)
    objects = MooncakeNativeKVObjectStore(store, key_prefix="hypic")

    # A rejected batch must not open a second round of per-layer connections.
    with pytest.raises(RuntimeError):
        objects.receive_native_blocks(obj, block_ids=(0, 2))

    assert store.batch_calls == 1
    assert store.get_calls == 0

def test_native_kv_batch_failure_disables_store_after_budget(monkeypatch) -> None:
    """Repeated transport failures must stop the store import, not retry forever."""

    torch = pytest.importorskip("torch")


    class Store:
        def __init__(self) -> None:
            self.batch_calls = 0

        def register_buffer(self, base_addr: int, size: int) -> int:
            return 0

        def batch_get_into_multi_buffers(self, keys, addrs, sizes):
            self.batch_calls += 1
            return [-800] * len(keys)

        def get(self, key: str) -> bytes:
            raise AssertionError("no per-layer retry after a transport failure")

        def put(self, key: str, value: bytes) -> int:
            raise AssertionError("receive path must not publish")

        def remove(self, key: str, force: bool = False) -> int:
            return 0

    cache = torch.zeros((2, 4, 2, 2, 2), dtype=torch.float16)
    obj = _object_for_cache(cache)
    store = Store()
    objects = MooncakeNativeKVObjectStore(store, key_prefix="hypic")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            objects.receive_native_blocks(obj, block_ids=(0, 2))

    # The fourth attempt must not even try the store again.
    with pytest.raises(RuntimeError):
        objects.receive_native_blocks(obj, block_ids=(0, 2))

    assert store.batch_calls == 3
    assert objects._native_store_disabled is True


def test_native_kv_missing_batch_api_raises(monkeypatch) -> None:
    """A store without the batch API fails the import instead of degrading.

    The per-layer ``store.get`` fallback is gone, so a missing
    ``batch_get_into_multi_buffers`` must surface as ``NotImplementedError`` and
    let the caller recompute rather than open one connection per layer.
    """

    torch = pytest.importorskip("torch")


    class Store:
        def __init__(self, layer_bytes: int) -> None:
            self.layer_bytes = layer_bytes
            self.get_calls = 0

        def register_buffer(self, base_addr: int, size: int) -> int:
            return 0

        def get(self, key: str) -> bytes:
            self.get_calls += 1
            return bytes([0x11]) * self.layer_bytes

        def put(self, key: str, value: bytes) -> int:
            raise AssertionError("receive path must not publish")

        def remove(self, key: str, force: bool = False) -> int:
            return 0

    cache = torch.zeros((2, 4, 2, 2, 2), dtype=torch.float16)
    obj = _object_for_cache(cache)
    store = Store(obj.layers[0].nbytes)
    objects = MooncakeNativeKVObjectStore(store, key_prefix="hypic")

    with pytest.raises(NotImplementedError):
        objects.receive_native_blocks(obj, block_ids=(0, 2))

    assert store.get_calls == 0


def test_transition_publish_never_writes_the_compat_payload(monkeypatch) -> None:
    """The Store layout is self-sufficient; no payload object may be written."""

    torch = pytest.importorskip("torch")

    class Store:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def put(self, key: str, value: bytes) -> int:
            self.objects[key] = bytes(value)
            return 0

        def get(self, key: str) -> bytes:
            return self.objects[key]

        def remove(self, key: str, force: bool = False) -> int:
            self.objects.pop(key, None)
            return 0

    key = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[0.5, 1.0]]], dtype=torch.float32
    )
    value = torch.tensor(
        [[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]], dtype=torch.float32
    )
    operator = build_gdn_transition_operator(
        key,
        value,
        torch.zeros((4, 1), dtype=torch.float32),
        torch.ones((4, 1), dtype=torch.float32),
        token_start=0,
        token_end=4,
        state_dtype=torch.float32,
    )
    store = Store()
    objects = MooncakeNativeKVObjectStore(store, key_prefix="hypic")

    manifest = objects.publish_transition_operator(
        external_key="segment/no-payload/hybrid-state", operator=operator
    )

    assert manifest.store_layout is not None
    assert manifest.nbytes == 0
    assert manifest.checksum == ""
    assert not any(name.endswith("/payload") for name in store.objects)
