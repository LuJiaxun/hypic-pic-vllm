# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vllm.config import PICConfig
from vllm.v1.pic.attention import (
    PICAttentionBackendUnsupported,
    build_pic_packed_attention_round,
)
from vllm.v1.pic.cache import (
    PICCachePlan,
    PICSegmentCache,
    PICSegmentEntry,
    pic_external_segment_indices,
)
from vllm.v1.pic.execution import compile_execution_plan
from vllm.v1.pic.external_kv import (
    PICExternalKVImportError,
    PICExternalKVImportRequest,
    PICExternalKVImportResult,
    PICExternalKVRegistry,
)
from vllm.v1.pic.gdn_transition import build_gdn_transition_operator
from vllm.v1.pic.handles import PICHandleKind, PICHandlePool
from vllm.v1.pic.kv_slots import (
    PICKVLayerTarget,
    build_kv_slot_mapping,
)
from vllm.v1.pic.lifecycle import PICLeaseRegistry
from vllm.v1.pic.live_capture import (
    PICGDNTransitionCapture,
    PICLiveCaptureManager,
    _collect_native_kv_references,
    ordered_gdn_layer_names,
)
from vllm.v1.pic.metrics import PICMetrics
from vllm.v1.pic.mooncake_kv import (
    pic_hybrid_state_key,
    pic_native_kv_key,
    MooncakeNativeKVObjectStore,
    PICHybridStateObject,
    PICMooncakeReferenceIndex,
    PICNativeKVLayerObject,
    PICNativeKVObject,
)
from vllm.v1.pic.mooncake_provider import MooncakePICExternalKVProvider
from vllm.v1.pic.native_kv import (
    PICKVPositionMode,
    PICNativeKVAllocation,
    PICNativeKVKind,
    PICNativeKVLease,
    PICNativeKVReference,
    PICNativeKVRequestMapping,
    PICNativeKVSlotPlan,
    PICRoPERerotationPlan,
    build_native_kv_slot_plan,
    gather_native_kv_slots,
    get_allocator_block_count,
    get_full_local_block_span,
    has_complete_native_kv_reference,
    is_pic_exact_position_aliasable,
    is_pic_public_segment_cacheable,
    materialize_private_rope_key,
    rerotate_native_key,
)
from vllm.v1.pic.pool import PICPhysicalPool, PICSlotMapping
from vllm.v1.pic.runtime import (
    PICRuntimeRange,
    PICSingleRequestRuntimePlan,
    build_single_request_runtime_plan,
)
from vllm.v1.pic.segmenter import split_text_and_tokenize, split_token_ids
from vllm.v1.pic.single_request import build_single_request_plan
from vllm.v1.pic.snapshot import PICSnapshotStore
from vllm.v1.pic.state import (
    PICAffineTransition,
    PICConvTransition,
    PICGDNTransition,
    PICLinearTransition,
    PICRequestStateBinding,
    PICStateLayout,
    PICStateSpec,
    PICTransitionOperator,
)
from vllm.v1.pic.worker_plan import build_worker_plan
from vllm.v1.worker.block_table import BlockTable


def test_mooncake_destination_uses_allocator_ids_then_kernel_span() -> None:
    # Qwen3.5 hybrid cache: one scheduler allocator block covers 544 tokens,
    # while the attention kernel addresses 32-token blocks (17 per allocator
    # block). The 4352-token prefix therefore needs 8 allocator IDs / 136
    # kernel IDs, not 136 allocator IDs.
    allocator_block_count = get_allocator_block_count(4352, 544)
    allocator_ids = np.arange(40, 40 + allocator_block_count, dtype=np.int64)
    expanded = BlockTable.map_to_kernel_blocks(
        allocator_ids,
        17,
        np.arange(17, dtype=np.int64).reshape(1, -1),
    )
    kernel_span = get_full_local_block_span(4352, 32)
    destination_ids = expanded[: kernel_span.block_count]

    assert allocator_block_count == 8
    assert len(expanded) == 136
    assert kernel_span.block_count == 136
    assert len(destination_ids) == 136
    assert destination_ids[0] == 680
    assert destination_ids[-1] == 815


def test_external_alias_reference_keeps_allocator_block_granularity() -> None:
    import torch

    from vllm.v1.kv_cache_interface import AttentionSpec

    spec = AttentionSpec(
        block_size=544,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
    )
    state = SimpleNamespace(
        pic_public_block_ids=((0, 0, tuple(range(100, 108))),),
        pic_mooncake_external_keys={(0, 0): "segment/native"},
        pic_mooncake_external_block_sizes={(0, 0): 32},
    )
    config = SimpleNamespace(
        kv_cache_groups=(SimpleNamespace(kv_cache_spec=spec),)
    )

    references = _collect_native_kv_references(
        request_id="alias-request",
        segment_index=0,
        segment=SimpleNamespace(start=0, end=4352),
        request_state=state,
        kv_cache_config=config,
    )

    assert len(references) == 1
    assert references[0].block_ids == ()
    assert references[0].block_size == 32
    assert references[0].token_count == 4352


def test_pic_metrics_tracks_plan_and_request_local_fallback() -> None:
    metrics = PICMetrics()
    plan = PICSingleRequestRuntimePlan(
        ranges=(
            PICRuntimeRange(0, 0, 4, "reuse", 4),
            PICRuntimeRange(-1, 4, 7, "recompute", 7),
        ),
        native_slot_plans=(),
    )

    metrics.record_lookup(hit=True, matched_segments=1)
    metrics.record_plan(plan)
    metrics.record_seam(2)
    metrics.record_fallback("request-local-mapping-mismatch")
    metrics.record_decode_mapping(reused=True)
    snapshot = metrics.snapshot(
        lease_snapshot={"tokens": 1},
        pool_capacity_bytes=1024,
        pool_free_bytes=768,
    )

    assert snapshot["lookup_hits"] == 1
    assert snapshot["matched_segments"] == 1
    assert snapshot["reused_tokens"] == 4
    assert snapshot["recompute_tokens"] == 3
    assert snapshot["seam_tokens"] == 2
    assert snapshot["fallbacks"] == 1
    assert snapshot["fallback_reasons"] == {"request-local-mapping-mismatch": 1}
    assert snapshot["decode_mapping_reused"] == 1
    assert snapshot["leases"] == {"tokens": 1}
    assert snapshot["pool"] == {"capacity_bytes": 1024, "free_bytes": 768}


def test_exact_position_alias_is_restricted_to_canonical_origin() -> None:
    assert is_pic_exact_position_aliasable(0)
    assert is_pic_exact_position_aliasable(12, canonical_start=12)
    assert not is_pic_exact_position_aliasable(12)


def test_pic_metrics_tracks_materialization_capture_and_forward() -> None:
    metrics = PICMetrics()
    metrics.record_materialization(kind="public")
    metrics.record_materialization(kind="private")
    metrics.record_materialization(kind="private", reused=True)
    metrics.record_capture()
    metrics.record_capture(deduplicated=True)
    metrics.record_eviction()
    metrics.record_forward(kind="single", token_count=5)
    metrics.record_forward(kind="packed", token_count=8, request_count=2)
    metrics.record_forward(kind="ordinary", token_count=3, request_count=2)

    snapshot = metrics.snapshot()
    assert snapshot["public_kv_materialized"] == 1
    assert snapshot["private_kv_materialized"] == 1
    assert snapshot["private_kv_materialization_reused"] == 1
    assert snapshot["capture_stored"] == 1
    assert snapshot["capture_deduplicated"] == 1
    assert snapshot["evictions"] == 1
    assert snapshot["model_forward_calls"] == 3
    assert snapshot["packed_rounds"] == 1
    assert snapshot["single_request_rounds"] == 1
    assert snapshot["forward_tokens"] == 16
    assert snapshot["forward_requests"] == 5


def test_pic_metrics_fallback_reason_cardinality_is_bounded() -> None:
    metrics = PICMetrics()
    for index in range(40):
        metrics.record_fallback(f"reason-{index}")
    snapshot = metrics.snapshot()
    assert snapshot["fallbacks"] == 40
    assert len(snapshot["fallback_reasons"]) == 33
    assert snapshot["fallback_reasons"]["other"] == 8


def test_external_native_kv_reference_fails_closed_without_provider() -> None:
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(),
        block_size=4,
        token_count=8,
        external_provider="mooncake",
        external_key="segment-key",
    )

    with pytest.raises(PICExternalKVImportError, match="provider is unavailable"):
        PICExternalKVRegistry().import_reference(reference)


def test_external_native_kv_import_binds_native_lease() -> None:
    released: list[str] = []

    class Provider:
        name = "mooncake"

        def import_native_kv(self, request):
            assert request.external_key == "segment-key"
            lease = PICNativeKVLease(
                (10, 11), lambda: released.append(request.external_key)
            )
            return PICExternalKVImportResult((10, 11), 4, 8, lease)

    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(),
        block_size=4,
        token_count=8,
        external_provider="mooncake",
        external_key="segment-key",
    )
    registry = PICExternalKVRegistry()
    registry.register(Provider())
    imported = registry.import_reference(reference)

    assert imported.block_ids == (10, 11)
    assert not imported.is_external
    assert imported.lease is not None
    assert imported.release()
    assert released == ["segment-key"]


def test_external_native_kv_import_releases_mismatched_result() -> None:
    released: list[bool] = []

    class Provider:
        name = "mooncake"

        def import_native_kv(self, request):
            lease = PICNativeKVLease((10,), lambda: released.append(True))
            return PICExternalKVImportResult((10,), 4, 8, lease)

    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(),
        block_size=4,
        token_count=8,
        external_provider="mooncake",
        external_key="segment-key",
    )
    registry = PICExternalKVRegistry()
    registry.register(Provider())

    with pytest.raises(PICExternalKVImportError, match="incomplete block span"):
        registry.import_reference(reference)
    assert released == [True]


def test_external_native_kv_registry_rejects_duplicate_provider() -> None:
    class Provider:
        name = "mooncake"

        def import_native_kv(self, request):
            raise AssertionError("not called")

    registry = PICExternalKVRegistry()
    registry.register(Provider())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Provider())
    assert registry.provider_names() == ("mooncake",)


def _mooncake_import_request() -> PICExternalKVImportRequest:
    return PICExternalKVImportRequest(
        provider="mooncake",
        external_key="segment-key",
        kv_cache_group_id=0,
        block_size=4,
        token_count=8,
        canonical_start=0,
        dtype="bfloat16",
        layout=(2, 8, 4, 1, 2),
    )


def test_mooncake_provider_import_and_release() -> None:
    transferred: list[tuple[int, ...]] = []
    released: list[tuple[int, ...]] = []
    lease_registry = PICLeaseRegistry()
    provider = MooncakePICExternalKVProvider(
        lease_registry=lease_registry,
        block_resolver=lambda request: (10, 11),
        engine="fake-engine",
        transfer=lambda request, block_ids, engine: transferred.append(block_ids),
        release=lambda request, block_ids, engine: released.append(block_ids),
    )

    result = provider.import_native_kv(_mooncake_import_request())
    assert result.block_ids == (10, 11)
    assert transferred == [(10, 11)]
    assert lease_registry.snapshot()["tokens"] == 1
    assert result.lease.release()
    assert released == [(10, 11)]
    assert lease_registry.snapshot()["tokens"] == 0


def test_mooncake_provider_reuses_active_import_span() -> None:
    transferred: list[tuple[int, ...]] = []
    released: list[tuple[int, ...]] = []
    provider = MooncakePICExternalKVProvider(
        lease_registry=PICLeaseRegistry(),
        block_resolver=lambda request: (10, 11),
        transfer=lambda request, block_ids, engine: transferred.append(block_ids),
        release=lambda request, block_ids, engine: released.append(block_ids),
    )

    first = provider.import_native_kv(_mooncake_import_request())
    second = provider.import_native_kv(_mooncake_import_request())
    assert first.block_ids == second.block_ids == (10, 11)
    assert transferred == [(10, 11)]
    assert first.lease.release()
    assert released == []
    assert second.lease.release()
    assert released == [(10, 11)]


def test_mooncake_provider_requires_native_destination() -> None:
    provider = MooncakePICExternalKVProvider(
        lease_registry=PICLeaseRegistry(),
        block_resolver=lambda request: (10, 11),
    )

    with pytest.raises(
        PICExternalKVImportError, match="destination writer is unavailable"
    ):
        provider.import_native_kv(_mooncake_import_request())


def test_release_does_not_retire_a_resource_still_in_use() -> None:
    """The first request to finish must not shut the shared key down.

    Several requests import the same stored segment, so their leases share one
    key.  Releasing one lease used to mark the key retired immediately, and the
    next import raised RuntimeError: PIC resource is retired.
    """
    registry = PICLeaseRegistry()
    key = ("mooncake", "segment/deadbeef/group-3")
    released: list[str] = []

    first = registry.acquire(key, owner="a", kind="external_native_kv",
                             release_fn=lambda: released.append("freed"))
    second = registry.acquire(key, owner="b", kind="external_native_kv")

    # Request a finishes first: its release must not retire the shared key.
    registry.retire(key)
    registry.release(first)
    assert released == [], "a shared key must stay alive while one lease is held"
    assert registry.ref_count(key) == 1

    # The other request can still import against the same key and finish.
    third = registry.acquire(key, owner="c", kind="external_native_kv")
    assert registry.ref_count(key) == 2
    registry.release(second)
    assert released == []
    registry.retire(key)
    registry.release(third)
    assert released == ["freed"]


def test_acquire_after_retire_starts_a_fresh_resource() -> None:
    registry = PICLeaseRegistry()
    key = ("mooncake", "segment/fresh/group-0")
    released: list[str] = []

    token = registry.acquire(key, owner="a", kind="external_native_kv",
                             release_fn=lambda: released.append("freed"))
    registry.release(token)
    registry.retire(key)
    assert released == ["freed"]

    # Fully retired keys are gone, so the next import re-imports into a new record.
    again = registry.acquire(key, owner="b", kind="external_native_kv")
    assert registry.ref_count(key) == 1
    assert again.token_id != token.token_id


def test_mooncake_provider_rejects_invalid_block_span() -> None:
    provider = MooncakePICExternalKVProvider(
        lease_registry=PICLeaseRegistry(),
        block_resolver=lambda request: (10,),
        transfer=lambda request, block_ids, engine: None,
    )

    with pytest.raises(PICExternalKVImportError, match="invalid native block span"):
        provider.import_native_kv(_mooncake_import_request())


def test_mooncake_provider_releases_after_transfer_failure() -> None:
    released: list[tuple[int, ...]] = []
    lease_registry = PICLeaseRegistry()
    provider = MooncakePICExternalKVProvider(
        lease_registry=lease_registry,
        block_resolver=lambda request: (10, 11),
        transfer=lambda request, block_ids, engine: (_ for _ in ()).throw(
            RuntimeError("transfer failed")
        ),
        release=lambda request, block_ids, engine: released.append(block_ids),
    )

    with pytest.raises(PICExternalKVImportError, match="transfer failed"):
        provider.import_native_kv(_mooncake_import_request())
    assert released == [(10, 11)]
    assert lease_registry.snapshot()["tokens"] == 0


def test_mooncake_native_kv_object_round_trip_preserves_native_blocks() -> None:
    import torch

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

    source = torch.arange(2 * 4 * 2 * 1 * 2, dtype=torch.float32).reshape(2, 4, 2, 1, 2)
    store = MooncakeNativeKVObjectStore(Store(), key_prefix="hypic")
    obj = store.publish_native_blocks(
        external_key="segment-1",
        kv_cache_group_id=3,
        block_size=2,
        token_count=4,
        canonical_start=0,
        layout="NHD",
        kv_caches={"layer.0": source},
        block_ids=(1, 3),
    )

    target = torch.zeros_like(source)
    loaded = store.load("segment-1")
    assert store.load(obj.external_key) == obj
    store.restore_native_blocks(
        loaded,
        kv_caches={"layer.0": target},
        block_ids=(0, 2),
    )
    assert loaded == obj
    assert torch.equal(target[:, 0], source[:, 1])
    assert torch.equal(target[:, 2], source[:, 3])
    assert torch.count_nonzero(target[:, 1]).item() == 0
    assert torch.count_nonzero(target[:, 3]).item() == 0


def test_mooncake_native_kv_object_rejects_wrong_destination_shape() -> None:
    import torch

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

    # Keep the KV axis and block axis distinguishable.  The destination may
    # have more blocks than the object, but its per-block shape must match.
    source = torch.zeros(2, 3, 2, 1, 2, dtype=torch.float32)
    store = MooncakeNativeKVObjectStore(Store(), key_prefix="hypic")
    obj = store.publish_native_blocks(
        external_key="segment-2",
        kv_cache_group_id=0,
        block_size=2,
        token_count=2,
        canonical_start=0,
        layout="NHD",
        kv_caches={"layer.0": source},
        block_ids=(0,),
    )

    with pytest.raises(ValueError, match="shape"):
        store.restore_native_blocks(
            obj,
            kv_caches={"layer.0": torch.zeros(2, 4, 3, 1, 2)},
            block_ids=(0,),
        )


def test_mooncake_reference_index_builds_external_nonfinal_hit() -> None:
    class Store:
        def load(self, key: str) -> PICNativeKVObject:
            assert key == pic_native_kv_key(segment.seg_hash, 0)
            return PICNativeKVObject(
                external_key="hypic/native-kv/v1/" + key,
                kv_cache_group_id=0,
                block_size=2,
                token_count=4,
                canonical_start=0,
                layout="NHD",
                dtype="float32",
                block_count=2,
                layers=(
                    PICNativeKVLayerObject(
                        layer_name="layer.0",
                        key="layer.0",
                        shape=(2, 2, 1, 2, 1),
                        dtype="float32",
                        nbytes=32,
                    ),
                ),
            )

    segment = split_token_ids((1, 2, 3, 4), (99,))[0]
    continuation = split_token_ids((5, 6), (99,))[0]
    # The scheduler uses a 4-token allocator block while the native object
    # is stored as two 2-token kernel blocks.  Discovery must validate both
    # granularities instead of requiring them to be equal.
    index = PICMooncakeReferenceIndex(
        Store(), group_block_sizes={0: 4}, group_dtypes={0: "float32"}
    )
    cache = PICSegmentCache(enabled=True)
    plan = cache.build_plan(
        (segment, continuation),
        external_lookup=lambda item: index.lookup(item, (0,)),
    )

    assert plan.reused_segments == (0,)
    assert plan.recompute_segments == (1,)
    assert len(plan.matches) == 1
    reference = plan.matches[0][1].native_kv_refs[0]
    assert reference.is_external
    assert reference.block_ids == ()
    assert reference.external_provider == "mooncake"
    assert reference.external_manifest is not None


def test_mooncake_import_reuses_lookup_manifest_without_second_get() -> None:
    obj = PICNativeKVObject(
        external_key="hypic/native-kv/v1/segment/manifest/group-0",
        kv_cache_group_id=0,
        block_size=2,
        token_count=4,
        canonical_start=0,
        layout="NHD",
        dtype="float32",
        block_count=2,
        layers=(
            PICNativeKVLayerObject(
                layer_name="layer.0",
                key="hypic/native-kv/v1/segment/manifest/group-0/layers/layer.0",
                shape=(2, 2, 1, 2, 1),
                dtype="float32",
                nbytes=32,
            ),
        ),
    )

    class Store:
        def __init__(self) -> None:
            self.get_calls = 0

        def get(self, key: str) -> bytes:
            self.get_calls += 1
            assert key.endswith("/manifest")
            return json.dumps(obj.as_dict()).encode("utf-8")

        def put(self, key: str, value: bytes) -> int:
            return 0

        def remove(self, key: str, force: bool = False) -> int:
            return 0

    backend = Store()
    object_store = MooncakeNativeKVObjectStore(backend, key_prefix="hypic")
    looked_up = object_store.load("segment/manifest/group-0")
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(),
        block_size=2,
        token_count=4,
        dtype="float32",
        layout=(2,),
        external_provider="mooncake",
        external_key=looked_up.external_key,
        external_manifest=looked_up.as_dict(),
    )
    request = PICExternalKVImportRequest(
        provider="mooncake",
        external_key=reference.external_key or "",
        kv_cache_group_id=reference.kv_cache_group_id,
        block_size=reference.block_size,
        token_count=reference.token_count,
        canonical_start=reference.canonical_start,
        dtype=reference.dtype,
        layout=reference.layout,
        object_manifest=reference.external_manifest,
    )
    received: list[PICNativeKVObject] = []
    callback = object_store.make_transfer_callback(
        lambda _request, manifest, _block_ids, _engine: received.append(manifest)
    )

    callback(request, (3, 4), None)

    assert backend.get_calls == 1
    assert received == [looked_up]


def test_mooncake_import_only_keeps_external_entry_out_of_reuse_plan() -> None:
    segment = split_token_ids((1, 2, 3, 4), (99,))[0]
    continuation = split_token_ids((5, 6), (99,))[0]
    entry = PICSegmentEntry(
        seg_hash=segment.seg_hash,
        token_ids=segment.token_ids,
        native_kv_refs=(
            PICNativeKVReference(
                kv_cache_group_id=0,
                block_ids=(),
                block_size=2,
                token_count=4,
                kind=PICNativeKVKind.CANONICAL_PUBLIC,
                position_mode=PICKVPositionMode.CANONICAL_LOCAL,
                canonical_start=0,
                source_token_start=0,
                source_block_start=0,
                dtype="float32",
                layout=(2,),
                external_provider="mooncake",
                external_key="hypic/segment-0",
            ),
        ),
    )
    cache = PICSegmentCache(enabled=True)
    cache.insert(segment, native_kv_refs=entry.native_kv_refs)
    plan = cache.build_plan(
        (segment, continuation),
        external_lookup=lambda item: entry if item == segment else None,
        external_import_only=True,
    )

    assert plan.reused_segments == ()
    assert plan.recompute_segments == (0, 1)
    assert plan.matches == ()
    assert plan.external_matches == ((0, entry),)


def test_external_cache_entries_are_not_republished_by_worker_capture() -> None:
    external_entry = PICSegmentEntry(
        seg_hash=b"external-segment",
        token_ids=(1, 2),
        external_transition_key="hypic/segment/external/hybrid-state",
    )
    local_entry = PICSegmentEntry(
        seg_hash=b"local-segment",
        token_ids=(3, 4),
    )
    external_kv_entry = PICSegmentEntry(
        seg_hash=b"external-kv-segment",
        token_ids=(5, 6),
        native_kv_refs=(
            PICNativeKVReference(
                kv_cache_group_id=0,
                block_ids=(),
                block_size=2,
                token_count=2,
                external_provider="mooncake",
                external_key="hypic/segment/external-kv/group-0",
            ),
        ),
    )
    plan = PICCachePlan(
        segment_count=3,
        reused_segments=(0, 1, 2),
        recompute_segments=(),
        seam_tokens=(),
        matches=(
            (0, external_entry),
            (1, external_kv_entry),
            (2, local_entry),
        ),
    )

    assert pic_external_segment_indices(plan) == frozenset({0, 1})


def test_pic_config_exposes_hybrid_mooncake_import_only() -> None:
    config = PICConfig(mooncake=True, mooncake_import_only=True)
    assert config.mooncake
    assert config.mooncake_import_only


def test_mooncake_reference_index_rejects_incomplete_group_set() -> None:
    class Store:
        def load(self, key: str) -> PICNativeKVObject:
            if key.endswith("group-1"):
                raise ValueError("manifest missing")
            return PICNativeKVObject(
                external_key="hypic/native-kv/v1/" + key,
                kv_cache_group_id=0,
                block_size=2,
                token_count=4,
                canonical_start=0,
                layout="NHD",
                dtype="float32",
                block_count=2,
                layers=(
                    PICNativeKVLayerObject(
                        layer_name="layer.0",
                        key="layer.0",
                        shape=(2, 2, 1, 2, 1),
                        dtype="float32",
                        nbytes=32,
                    ),
                ),
            )

    segment = split_token_ids((1, 2, 3, 4), (99,))[0]
    index = PICMooncakeReferenceIndex(
        Store(),
        group_block_sizes={0: 2, 1: 2},
        group_dtypes={0: "float32", 1: "float32"},
    )
    assert index.lookup(segment, (0, 1)) is None
def test_mooncake_reference_index_requires_hybrid_state_when_requested() -> None:
    segment = split_token_ids((1, 2, 3, 4), (99,))[0]

    class Store:
        def __init__(self, has_state: bool) -> None:
            self.has_state = has_state

        def load(self, key: str) -> PICNativeKVObject:
            return PICNativeKVObject(
                external_key="hypic/native-kv/v1/" + key,
                kv_cache_group_id=0,
                block_size=2,
                token_count=4,
                canonical_start=0,
                layout="NHD",
                dtype="float32",
                block_count=2,
                layers=(
                    PICNativeKVLayerObject(
                        layer_name="layer.0",
                        key="layer.0",
                        shape=(2, 2, 1, 2, 1),
                        dtype="float32",
                        nbytes=32,
                    ),
                ),
            )

        def load_transition_manifest(self, key: str) -> Any:
            if not self.has_state:
                raise ValueError("state manifest missing")
            return SimpleNamespace(token_count=4, span_start=0)

    missing = PICMooncakeReferenceIndex(
        Store(False),
        group_block_sizes={0: 4},
        group_dtypes={0: "float32"},
        require_hybrid_state=True,
    )
    assert missing.lookup(segment, (0,)) is None

    present = PICMooncakeReferenceIndex(
        Store(True),
        group_block_sizes={0: 4},
        group_dtypes={0: "float32"},
        require_hybrid_state=True,
    )
    entry = present.lookup(segment, (0,))
    assert entry is not None
    assert entry.external_transition_key == pic_hybrid_state_key(
        segment.seg_hash
    )


def test_pic_lease_registry_defers_release_until_last_reference() -> None:
    released: list[str] = []
    registry = PICLeaseRegistry()
    resource = ("private_kv", "request-1", 0)
    entry = registry.acquire(
        resource,
        owner="entry:segment",
        kind="cache_entry",
        release_fn=lambda: released.append("released"),
    )
    active = registry.acquire(resource, owner="request:1", kind="active_request")
    inflight = registry.acquire(resource, owner="inflight:1", kind="inflight")

    assert registry.ref_count(resource) == 3
    assert registry.retire(resource)
    assert released == []
    assert registry.release(inflight)
    assert registry.release(active)
    assert released == []
    assert registry.release(entry)
    assert released == ["released"]
    assert registry.ref_count(resource) == 0


def test_pic_lease_registry_owner_cleanup_is_idempotent() -> None:
    registry = PICLeaseRegistry()
    first = registry.acquire(("state", 1), owner="request:1", kind="active_request")
    second = registry.acquire(("state", 2), owner="request:1", kind="active_request")
    registry.acquire(("state", 3), owner="request:2", kind="active_request")

    assert registry.release_owner("request:1") == 2
    assert registry.release_owner("request:1") == 0
    assert not registry.release(first)
    assert not registry.release(second)
    assert registry.snapshot()["tokens"] == 1


def test_pic_request_state_binding_tracks_request_owned_cursor() -> None:
    binding = PICRequestStateBinding("request-1")
    binding.bind_mapping(True)
    binding.sync(
        num_computed_tokens=32,
        block_ids=((7, 8), (20,)),
        segment_index=1,
        range_cursor=2,
        transition_cursor=1,
    )

    assert binding.logical_token_position == 32
    assert binding.state_block_ids == ((7, 8), (20,))
    assert binding.segment_index == 1
    assert binding.range_cursor == 2
    assert binding.transition_cursor == 1


def test_pic_request_state_binding_reuses_mapping_for_decode() -> None:
    binding = PICRequestStateBinding("request-1")
    binding.bind_mapping(True)
    binding.sync(num_computed_tokens=16, block_ids=((3,),))

    assert binding.mark_decode_step(
        num_computed_tokens=16,
        scheduled_tokens=1,
        mapping_present=True,
    )
    assert binding.decode_steps == 1
    assert binding.logical_token_position == 17
    assert binding.fallback_reason is None


def test_pic_request_state_binding_falls_back_on_cursor_mismatch() -> None:
    binding = PICRequestStateBinding("request-1")
    binding.bind_mapping(True)
    binding.sync(num_computed_tokens=16, block_ids=((3,),))

    assert not binding.mark_decode_step(
        num_computed_tokens=17,
        scheduled_tokens=1,
        mapping_present=True,
    )
    assert binding.fallback_reason == "logical_cursor_mismatch"


def test_split_excludes_separator_and_keeps_offsets() -> None:
    segments = split_token_ids([1, 2, 9, 8, 3, 4], [9, 8])

    assert [(segment.start, segment.end) for segment in segments] == [(0, 2), (4, 6)]
    assert [segment.token_ids for segment in segments] == [(1, 2), (3, 4)]


def test_pic_reuse_uses_complete_segment_local_blocks() -> None:
    span = get_full_local_block_span(1360, 544)

    assert span.reusable_start == 0
    assert span.reusable_end == 1088
    assert span.reusable_token_count == 2 * 544
    assert span.block_count == 2


def test_pic_reuse_skips_segment_local_partial_suffix() -> None:
    assert get_full_local_block_span(543, 544).reusable_token_count == 0
    assert get_full_local_block_span(1089, 544).reusable_token_count == 1088


def test_native_kv_hit_requires_a_complete_group_span() -> None:
    reference = PICNativeKVReference(
        kv_cache_group_id=3,
        block_ids=(7, 8),
        block_size=544,
        token_count=1088,
    )

    assert has_complete_native_kv_reference(
        (reference,),
        kv_cache_group_id=3,
        block_count=2,
        token_count=1088,
    )
    assert not has_complete_native_kv_reference(
        (),
        kv_cache_group_id=3,
        block_count=2,
        token_count=1088,
    )
    assert not has_complete_native_kv_reference(
        (reference,),
        kv_cache_group_id=3,
        block_count=2,
        token_count=544,
    )


def test_public_kv_cacheability_matches_sglang_last_segment_policy() -> None:
    assert is_pic_public_segment_cacheable(0, 1)
    assert is_pic_public_segment_cacheable(0, 3)
    assert is_pic_public_segment_cacheable(1, 3)
    assert not is_pic_public_segment_cacheable(2, 3)


def test_split_text_before_tokenization_keeps_explicit_ranges() -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return {"segment-A ": [101], " segment-B": [202]}[text]

    token_ids, ranges = split_text_and_tokenize(
        "segment-A <<PIC_SEP>> segment-B",
        FakeTokenizer(),
        "<<PIC_SEP>>",
    )

    assert token_ids == [101, 202]
    assert ranges == [(0, 1, False), (1, 2, True)]


def test_split_text_reuses_pretokenized_separator_sequence() -> None:
    calls: list[str] = []

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            calls.append(text)
            assert text == "<<PIC_SEP>>"
            return [99]

    token_ids, ranges = split_text_and_tokenize(
        "segment-A <<PIC_SEP>> segment-B",
        FakeTokenizer(),
        "<<PIC_SEP>>",
        pretokenized_token_ids=[101, 99, 202],
    )

    assert token_ids == [101, 202]
    assert ranges == [(0, 1, False), (1, 2, True)]
    assert calls == ["<<PIC_SEP>>"]


def test_split_text_uses_precomputed_separator_sequence() -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            raise AssertionError(f"unexpected tokenizer call for {text!r}")

    token_ids, ranges = split_text_and_tokenize(
        "segment-A <<PIC_SEP>> segment-B",
        FakeTokenizer(),
        "<<PIC_SEP>>",
        pretokenized_token_ids=[101, 99, 202],
        separator_token_ids=(99,),
    )

    assert token_ids == [101, 202]
    assert ranges == [(0, 1, False), (1, 2, True)]


def test_input_processor_accepts_contextual_pic_separator_and_bool_flag() -> None:
    from vllm.v1.engine.input_processor import InputProcessor

    separator = "<<PIC_SEP>>"

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return {
                "segment-A ": [101],
                " segment-B": [202],
            }[text]

    processor = object.__new__(InputProcessor)
    processor.renderer = SimpleNamespace(tokenizer=FakeTokenizer())
    processor._pic_separator_token_cache = None
    processor.vllm_config = SimpleNamespace(
        pic_config=SimpleNamespace(
            enabled=True,
            auto_enable=False,
            separator=separator,
            debug=False,
            mode="transition_rope_recompute",
            seam_sink=8,
        )
    )
    result = processor._prepare_pic_request(  # type: ignore[attr-defined]
        [17, 2, 3, 42],
        None,
        "segment-A <<PIC_SEP>> segment-B",
        SimpleNamespace(extra_args={"pic_enabled": True}),
    )

    assert result[0] is True
    assert result[1] == [101, 202]
    assert result[2] == [(0, 1, False), (1, 2, True)]


def test_cache_matches_nonprefix_segments_and_skips_last() -> None:
    segments = split_token_ids([1, 2, 9, 3, 4, 9, 5], [9])
    cache = PICSegmentCache(enabled=True)
    cache.insert(segments[0])
    cache.insert(segments[1])

    plan = cache.build_plan(segments, seam_sink=8)

    assert plan.reused_segments == (0, 1)
    assert plan.recompute_segments == (2,)
    assert [index for index, _ in plan.matches] == [0, 1]


def test_cache_entry_can_be_materialized_after_metadata_insert() -> None:
    segment = split_token_ids([1, 2], [9])[0]
    cache = PICSegmentCache(enabled=True)
    entry = cache.insert(segment)
    assert entry is not None and not entry.has_physical_handles
    updated = cache.insert(segment, full_kv_handles=(7, 11))
    assert updated is entry
    assert entry.has_physical_handles
    assert entry.full_kv_handles == (7, 11)


def test_cache_entry_stores_native_kv_reference_and_releases_lease() -> None:
    segment = split_token_ids([1, 2, 3, 4], [9])[0]

    class Lease:
        def __init__(self) -> None:
            self.released = 0

        def release(self) -> bool:
            self.released += 1
            return True

    lease = Lease()
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(7, 2),
        block_size=2,
        token_count=4,
        lease=lease,  # type: ignore[arg-type]
    )
    cache = PICSegmentCache(enabled=True)
    entry = cache.insert(segment, native_kv_refs=(reference,))

    assert entry is not None
    assert entry.native_kv_refs == (reference,)
    assert entry.has_physical_handles
    cache.clear()
    assert lease.released == 1


def test_handle_pool_is_reference_counted() -> None:
    pool = PICHandlePool(max_handles=1)
    handle = pool.allocate(PICHandleKind.FULL_KV, token_start=0, token_end=8)
    pool.retain(handle.handle_id)
    assert pool.get(handle.handle_id).ref_count == 2  # type: ignore[union-attr]
    assert pool.release(handle.handle_id) is None
    released = pool.release(handle.handle_id)
    assert released is not None
    assert len(pool) == 0


def test_state_layout_has_no_states_for_attention_only_config() -> None:
    class AttentionOnlyConfig:
        kv_cache_groups: list[Any] = []

    layout = PICStateLayout.from_kv_cache_config(AttentionOnlyConfig())
    assert not layout.has_state
    assert layout.recurrent_states == ()
    assert layout.conv_tails == ()


def test_execution_plan_preserves_nonprefix_ranges() -> None:
    segments = split_token_ids([1, 2, 9, 3, 4, 9, 5, 6], [9])
    cache = PICSegmentCache(enabled=True)
    cache.insert(segments[0], transition_state_handle=101)
    cache.insert(segments[1], transition_state_handle=102)
    cache_plan = cache.build_plan(segments, seam_sink=1)

    plan = compile_execution_plan(segments, cache_plan)
    assert [(item.start, item.end, item.action) for item in plan.ranges] == [
        (0, 2, "reuse"),
        (3, 5, "reuse"),
        (6, 8, "recompute"),
    ]
    assert [
        (item.segment_index, item.seam_start, item.seam_end)
        for item in plan.transitions
    ] == [(1, 3, 4)]
    assert plan.transitions[0].transition_state_handle == 102


def test_slot_mapping_allows_noncontiguous_physical_slots() -> None:
    mapping = PICSlotMapping(logical_positions=(4, 5, 6), physical_slots=(17, 2, 31))
    assert mapping.logical_positions == (4, 5, 6)
    assert mapping.physical_slots == (17, 2, 31)


def test_physical_pool_round_trips_tensor_snapshot() -> None:
    torch = pytest.importorskip("torch")
    from vllm.v1.pic.handles import PICHandleKind

    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    source = torch.arange(12, dtype=torch.float32)
    handle = pool.allocate_tensor_snapshot([source], kind=PICHandleKind.FULL_KV)
    second_source = torch.arange(7, dtype=torch.int32)
    second_handle = pool.allocate_tensor_snapshot(
        [second_source], kind=PICHandleKind.FULL_KV
    )
    restored = torch.zeros_like(source)
    pool.restore_tensor_snapshot(handle.handle_id, [restored])
    second_restored = torch.zeros_like(second_source)
    pool.restore_tensor_snapshot(second_handle.handle_id, [second_restored])
    assert torch.equal(source, restored)
    assert torch.equal(second_source, second_restored)
    assert pool.get_allocation(second_handle.handle_id).regions[0].offset_bytes > 0
    pool.release(handle.handle_id)
    pool.release(second_handle.handle_id)
    assert pool.free_bytes == pool.capacity_bytes


def test_snapshot_store_materializes_and_restores_segment_state() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2], [9])
    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    store = PICSnapshotStore(pool)
    source = torch.arange(4, dtype=torch.float32)
    snapshot = store.capture(segments[0], full_kv=(source,))

    cache = PICSegmentCache(enabled=True)
    entry = store.materialize_cache_entry(cache, segments[0], snapshot)
    assert entry is not None and entry.has_physical_handles

    restored = torch.zeros_like(source)
    store.restore(snapshot, full_kv=(restored,))
    assert torch.equal(source, restored)
    store.release(snapshot)
    assert pool.free_bytes == pool.capacity_bytes


def test_snapshot_store_uses_state_layout_order() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2], [9])
    layout = PICStateLayout(
        groups=(
            (
                PICStateSpec(
                    group_id=0,
                    state_index=0,
                    kind=PICHandleKind.RECURRENT_STATE,
                    shape=(2,),
                    dtype=torch.float32,
                ),
                PICStateSpec(
                    group_id=0,
                    state_index=1,
                    kind=PICHandleKind.CONV_TAIL,
                    shape=(3,),
                    dtype=torch.float32,
                ),
            ),
        )
    )
    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    store = PICSnapshotStore(pool)
    recurrent = torch.arange(2, dtype=torch.float32)
    conv_tail = torch.arange(3, dtype=torch.float32)
    tensors = {(0, 0): recurrent, (0, 1): conv_tail}
    snapshot = store.capture_hybrid_state(segments[0], layout, tensors)

    restored_recurrent = torch.zeros_like(recurrent)
    restored_conv_tail = torch.zeros_like(conv_tail)
    store.restore_hybrid_state(
        snapshot,
        layout,
        {(0, 0): restored_recurrent, (0, 1): restored_conv_tail},
    )
    assert torch.equal(recurrent, restored_recurrent)
    assert torch.equal(conv_tail, restored_conv_tail)
    store.release(snapshot)
    assert pool.free_bytes == pool.capacity_bytes


def test_worker_plan_falls_back_without_range_execution() -> None:
    segments = split_token_ids([1, 2, 9, 3], [9])
    cache = PICSegmentCache(enabled=True)
    cache.insert(segments[0], full_kv_handles=(1,))
    execution_plan = compile_execution_plan(
        segments, cache.build_plan(segments, seam_sink=1)
    )
    assert (
        build_worker_plan(
            execution_plan,
            prompt_len=4,
            range_execution_supported=False,
            allow_fallback=True,
        )
        is None
    )


def test_worker_plan_builds_skip_and_recompute_positions() -> None:
    segments = split_token_ids([1, 2, 9, 3, 4], [9])
    cache = PICSegmentCache(enabled=True)
    cache.insert(segments[0], full_kv_handles=(1,))
    execution_plan = compile_execution_plan(
        segments, cache.build_plan(segments, seam_sink=1)
    )
    worker_plan = build_worker_plan(
        execution_plan,
        prompt_len=5,
        range_execution_supported=True,
        allow_fallback=False,
    )
    assert worker_plan is not None
    assert worker_plan.skip_positions == (0, 1)
    assert worker_plan.recompute_positions == (2, 3, 4)


def test_live_capture_publishes_and_releases_state_materialization() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2], [9])
    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    manager = PICLiveCaptureManager(pool)
    recurrent = torch.arange(4, dtype=torch.float32)
    conv_tail = torch.arange(3, dtype=torch.float32)

    materialization = manager.capture_segment(
        "request-1",
        0,
        segments[0],
        recurrent_state=(recurrent,),
        conv_tail=(conv_tail,),
    )
    assert materialization is not None
    assert materialization.recurrent_state_handle is not None
    assert materialization.conv_tail_handle is not None
    assert manager.take_pending() == [materialization]

    restored = torch.zeros_like(recurrent)
    pool.restore_tensor_snapshot(materialization.recurrent_state_handle, (restored,))
    assert torch.equal(recurrent, restored)

    manager.release_request("request-1")
    assert pool.free_bytes == pool.capacity_bytes


def test_transition_operator_composes_zero_start_and_live_states() -> None:
    torch = pytest.importorskip("torch")
    state = torch.tensor([2.0, 3.0])
    first = PICAffineTransition(
        decay=torch.tensor([2.0, 0.5]),
        zero_state=torch.tensor([1.0, 4.0]),
    )
    second = PICAffineTransition(
        decay=torch.tensor([3.0, 2.0]),
        zero_state=torch.tensor([5.0, 6.0]),
    )

    operator = PICTransitionOperator.from_step_transitions(
        ((first,), (second,)),
        token_start=10,
        token_end=12,
    )

    sequential = second.apply(first.apply(state))
    composed = operator.apply((state,))[0]
    zero_sequential = second.apply(first.apply(torch.zeros_like(state)))

    assert torch.allclose(composed, sequential)
    assert torch.allclose(operator.zero_start_end_state[0], zero_sequential)
    assert operator.token_start == 10
    assert operator.token_end == 12


def test_live_capture_imports_shared_external_transition_until_last_request() -> None:
    torch = pytest.importorskip("torch")
    manager = PICLiveCaptureManager(
        PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    )
    operator = PICTransitionOperator(
        transitions=(
            PICAffineTransition(
                decay=torch.tensor([2.0]),
                zero_state=torch.tensor([1.0]),
            ),
        ),
        token_start=0,
        token_end=1,
    )

    first = manager.import_transition_operator(
        request_id="request-a", external_key="segment/state", operator=operator
    )
    second = manager.import_transition_operator(
        request_id="request-b", external_key="segment/state", operator=operator
    )
    assert first == second
    assert manager.has_transition(first)
    assert torch.equal(
        manager.apply_transition(first, (torch.tensor([3.0]),))[0],
        torch.tensor([7.0]),
    )

    manager.release_request("request-a")
    assert torch.equal(
        manager.apply_transition(second, (torch.tensor([3.0]),))[0],
        torch.tensor([7.0]),
    )
    manager.release_request("request-b")
    assert not manager.has_transition(second)
    with pytest.raises(KeyError, match="not live"):
        manager.apply_transition(second, (torch.tensor([3.0]),))


def test_gdn_transition_matches_torch_recurrent_update() -> None:
    torch = pytest.importorskip("torch")
    key = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [[[2.0, 3.0]], [[4.0, 5.0]]],
        dtype=torch.float32,
    )
    log_decay = torch.log(torch.tensor([[0.5], [0.25]], dtype=torch.float32))
    beta = torch.tensor([[0.2], [0.4]], dtype=torch.float32)
    operator = build_gdn_transition_operator(
        key,
        value,
        log_decay,
        beta,
        token_start=0,
        token_end=2,
        state_dtype=torch.float32,
    )

    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    expected = state
    for token_index in range(2):
        decay = log_decay[token_index].exp()[0]
        beta_value = beta[token_index][0]
        key_value = key[token_index][0]
        value_value = value[token_index][0]
        expected = (
            decay * expected
            + beta_value
            * (value_value - (expected * key_value).sum(dim=-1)).unsqueeze(-1)
            * key_value
        )

    actual = operator.apply((state,))[0]
    zero_state = operator.zero_start_end_state[0]
    assert torch.allclose(actual, expected)

    zero_expected = torch.zeros_like(state)
    for token_index in range(2):
        decay = log_decay[token_index].exp()[0]
        beta_value = beta[token_index][0]
        key_value = key[token_index][0]
        value_value = value[token_index][0]
        zero_expected = (
            decay * zero_expected
            + beta_value
            * (value_value - (zero_expected * key_value).sum(dim=-1)).unsqueeze(-1)
            * key_value
        )
    assert torch.allclose(zero_state, zero_expected)


def test_gdn_transition_keeps_storage_dtype_and_gqa_compact() -> None:
    torch = pytest.importorskip("torch")
    key = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]],
        dtype=torch.bfloat16,
    )
    value = torch.tensor(
        [
            [[2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0]],
        ],
        dtype=torch.bfloat16,
    )
    log_decay = torch.zeros((2, 2), dtype=torch.float32)
    beta = torch.ones((2, 2), dtype=torch.float32)
    operator = build_gdn_transition_operator(
        key,
        value,
        log_decay,
        beta,
        token_start=0,
        token_end=2,
        state_dtype=torch.float32,
    )
    transition = operator.transitions[0]
    assert isinstance(transition, PICGDNTransition)
    assert transition.key.dtype == torch.bfloat16
    assert transition.value.dtype == torch.bfloat16
    assert transition.key.shape[1] == 1
    assert transition.value.shape[1] == 2

    state = torch.zeros((2, 2, 2), dtype=torch.float32)
    expected = state
    for token_index in range(2):
        key_token = key[token_index].float().repeat_interleave(2, dim=0)
        value_token = value[token_index].float()
        memory = (expected * key_token.unsqueeze(-2)).sum(dim=-1)
        expected = expected + (
            (value_token - memory).unsqueeze(-1) * key_token.unsqueeze(-2)
        )
    actual = operator.apply((state,))[0]
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_gdn_transition_compact_operator_matches_token_operator() -> None:
    torch = pytest.importorskip("torch")
    key = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32
    )
    value = torch.tensor(
        [[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]]], dtype=torch.float32
    )
    log_decay = torch.zeros((3, 1), dtype=torch.float32)
    beta = torch.ones((3, 1), dtype=torch.float32)
    operator = build_gdn_transition_operator(
        key,
        value,
        log_decay,
        beta,
        token_start=0,
        token_end=3,
        state_dtype=torch.float32,
    )
    transition = operator.transitions[0]
    assert isinstance(transition, PICGDNTransition)
    compact = transition.compact()
    assert isinstance(compact, PICLinearTransition)

    state = torch.tensor([[[0.5, 1.0], [1.5, 2.0]]])
    expected = transition.apply(state)
    actual = compact.apply(state)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_live_capture_stores_one_matrix_for_the_segment_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2, 3], [9])
    key = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32
    )
    value = torch.tensor(
        [[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]]], dtype=torch.float32
    )
    operator = build_gdn_transition_operator(
        key,
        value,
        torch.zeros((3, 1), dtype=torch.float32),
        torch.ones((3, 1), dtype=torch.float32),
        token_start=0,
        token_end=3,
        state_dtype=torch.float32,
    )
    manager = PICLiveCaptureManager(PICPhysicalPool(16384, device="cpu", alignment_bytes=1))
    materialization = manager.capture_segment(
        "compact-transition",
        0,
        segments[0],
        recurrent_state=(torch.zeros((1, 2, 2), dtype=torch.float32),),
        transition_operator=operator,
    )

    assert materialization is not None
    handle = materialization.transition_state_handle
    assert handle is not None
    assert isinstance(
        manager._compact_transitions[handle].transitions[0], PICLinearTransition
    )

    # PIC stores one length-independent operator per segment, and that operator
    # describes exactly the reusable span of that segment.  There is no partial
    # application: a non-prefix reuse repeats the seam tokens and then applies
    # the whole span, so the stored range and the reused range are the same.
    stored = manager._compact_transitions[handle]
    assert (stored.token_start, stored.token_end) == (0, 3)
    assert manager.transition_representation(handle, 0, 3) == "compact"

    state = torch.tensor([[[0.5, 1.0], [1.5, 2.0]]])
    expected = operator.apply((state,))[0]
    actual = manager.apply_transition(handle, (state,), local_start=0, local_end=3)[0]
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    manager.release_request("compact-transition")


def test_live_capture_accepts_a_transition_covering_only_the_span() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2, 3, 4], [9])
    operator = build_gdn_transition_operator(
        torch.eye(2, dtype=torch.float32).repeat(4, 1, 1),
        torch.ones((4, 2, 2), dtype=torch.float32),
        torch.zeros((4, 2), dtype=torch.float32),
        torch.ones((4, 2), dtype=torch.float32),
        token_start=0,
        token_end=4,
        state_dtype=torch.float32,
    )
    span = operator.slice(0, 2)
    manager = PICLiveCaptureManager(PICPhysicalPool(16384, device="cpu", alignment_bytes=1))
    materialization = manager.capture_segment(
        "span-only",
        0,
        segments[0],
        recurrent_state=(torch.zeros((1, 2, 2), dtype=torch.float32),),
        transition_operator=span,
    )

    assert materialization is not None
    stored = manager._compact_transitions[materialization.transition_state_handle]
    assert (stored.token_start, stored.token_end) == (0, 2)
    manager.release_request("span-only")


def test_fixed_layout_transition_receive_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single host-staging stage avoids the payload and needs no registration."""

    torch = pytest.importorskip("torch")

    class Source:
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

    class Receiver:
        """No payload object: reading it would prove the slow fallback ran."""

        def __init__(self, source: Source) -> None:
            self.source = source
            self.batch_calls = 0
            self.register_attempts = 0
            self.register_failures = 0

        def get(self, key: str) -> bytes:
            if key.endswith("/payload"):
                raise AssertionError("compat payload must not be read")
            return self.source.objects[key]

        def register_buffer(self, base_addr: int, size: int) -> int:
            self.register_attempts += 1
            self.register_failures += 1
            return -1

        def unregister_buffer(self, base_addr: int) -> int:
            return 0

        def batch_get_into_multi_buffers(
            self,
            keys: list[str],
            addrs: list[list[int]],
            sizes: list[list[int]],
        ) -> list[int]:
            self.batch_calls += 1
            assert len(keys) == len(addrs) == len(sizes)
            return [0] * len(keys)

    key = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[0.5, 1.0]]],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]],
        dtype=torch.float32,
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
    source = Source()
    MooncakeNativeKVObjectStore(source, key_prefix="hypic").publish_transition_operator(
        external_key="segment/registered/hybrid-state", operator=operator
    )

    receiver = Receiver(source)
    objects = MooncakeNativeKVObjectStore(receiver, key_prefix="hypic")
    manifest, loaded = objects.load_transition_operator(
        "segment/registered/hybrid-state", device="cpu"
    )
    assert manifest.store_layout is not None
    assert receiver.batch_calls == 1
    assert receiver.register_attempts >= 1
    # A rejected registration must not move the receive off the fixed layout.
    assert receiver.register_failures == receiver.register_attempts
    assert isinstance(loaded.transitions[0], PICLinearTransition)

    tensors = [
        loaded.transitions[0].matrix,
        loaded.transitions[0].zero_state,
    ]
    tensors.extend(item.inputs for item in loaded.conv_transitions)
    tensors = [tensor for tensor in tensors if tensor.numel() > 0]
    storages = {tensor.untyped_storage().data_ptr() for tensor in tensors}
    assert len(storages) == 1


def test_gdn_chunked_reference_matches_chunk_delta_h_ordering() -> None:
    torch = pytest.importorskip("torch")
    key = torch.tensor(
        [
            [[1.0, 0.5]],
            [[0.25, 1.0]],
            [[1.0, -0.5]],
        ],
        dtype=torch.bfloat16,
    )
    value = torch.zeros((3, 1, 2), dtype=torch.bfloat16)
    log_decay = torch.zeros((3, 1), dtype=torch.float32)
    beta = torch.ones((3, 1), dtype=torch.float32)
    chunk_w = torch.tensor(
        [
            [[0.5, 0.0]],
            [[0.0, 0.5]],
            [[0.25, -0.25]],
        ],
        dtype=torch.bfloat16,
    )
    chunk_u = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ],
        dtype=torch.bfloat16,
    )
    chunk_g = torch.tensor([[-0.25], [-0.5], [-0.75]], dtype=torch.float32)
    zero_state = torch.zeros((1, 2, 2), dtype=torch.float32)
    transition = PICGDNTransition(
        key=key,
        value=value,
        log_decay=log_decay,
        beta=beta,
        zero_state=zero_state,
        chunk_w=chunk_w,
        chunk_u=chunk_u,
        chunk_g=chunk_g,
        chunk_size=2,
    )

    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    expected = state.float()
    for start in (0, 2):
        end = min(start + 2, key.shape[0])
        w = chunk_w[start:end, 0].float()
        u = chunk_u[start:end, 0].float()
        g = chunk_g[start:end, 0]
        delta = u - torch.matmul(w, expected[0].transpose(0, 1))
        delta = delta * torch.exp(g[-1] - g).unsqueeze(-1)
        expected[0] = expected[0] * torch.exp(g[-1])
        expected[0] += torch.matmul(
            delta.to(key.dtype).float().transpose(0, 1),
            key[start:end, 0].float(),
        )

    actual = transition.apply(state)
    assert torch.equal(actual, expected.to(state.dtype))


def test_gdn_transition_slice_matches_full_transition_suffix() -> None:
    torch = pytest.importorskip("torch")
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32)
    value = torch.tensor(
        [[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]]], dtype=torch.float32
    )
    log_decay = torch.zeros((3, 1), dtype=torch.float32)
    beta = torch.ones((3, 1), dtype=torch.float32)
    operator = build_gdn_transition_operator(
        key,
        value,
        log_decay,
        beta,
        token_start=0,
        token_end=3,
        state_dtype=torch.float32,
        use_fla_reference=False,
    )
    sliced = operator.slice(1, 3)
    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    full_after_first = operator.slice(0, 1).apply((state,))[0]
    expected = full_after_first
    for token_index in range(1, 3):
        key_token = key[token_index, 0]
        value_token = value[token_index, 0]
        expected = expected + (
            (value_token - (expected * key_token).sum(dim=-1)).unsqueeze(-1) * key_token
        )
    actual = sliced.apply((full_after_first,))[0]
    assert torch.allclose(actual, expected)


def test_conv_transition_slice_updates_dim_first_and_state_first_layouts() -> None:
    torch = pytest.importorskip("torch")
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    transition = PICConvTransition(inputs=inputs, state_length=2)
    state_sd = torch.full((2, 4), -1.0)
    state_ds = state_sd.transpose(0, 1).contiguous()

    expected = inputs[-2:]
    assert torch.equal(transition.slice(1, 3).apply(state_sd), expected)
    assert torch.equal(transition.slice(1, 3).apply(state_ds), expected.transpose(0, 1))

    compact_tail = PICConvTransition(
        inputs=inputs[-2:], state_length=2, source_token_count=3
    )
    assert torch.equal(compact_tail.apply(state_sd), expected)
    assert compact_tail.slice(0, 3) is compact_tail
    with pytest.raises(ValueError, match="partially sliced"):
        compact_tail.slice(1, 3)


def test_gdn_capture_hook_publishes_operator_and_zero_conv_tail() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2, 3], [9])
    request_state = SimpleNamespace(
        pic_transition_operators={},
        pic_transition_conv_tails={},
        pic_transition_validation={},
    )
    capture = PICGDNTransitionCapture(
        request_state=request_state,
        segments=segments,
        layer_names=("model.layers.0.linear_attn",),
    )
    key = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]],
    )
    value = torch.ones((3, 1, 2))
    log_decay = torch.zeros((3, 1))
    beta = torch.ones((3, 1))
    conv_input = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    capture.record_gdn_layer(
        "model.layers.0.linear_attn",
        key,
        value,
        log_decay,
        beta,
        state_dtype=torch.float32,
        conv_input=conv_input,
        conv_kernel_size=3,
    )

    assert tuple(request_state.pic_transition_operators) == (0,)
    assert tuple(request_state.pic_transition_conv_tails) == (0,)
    assert request_state.pic_transition_operators[0].token_end == 3
    assert torch.equal(
        request_state.pic_transition_conv_tails[0][0],
        conv_input[-2:].transpose(0, 1),
    )
    conv_state = torch.zeros((2, 4), dtype=torch.float32)
    conv_result = request_state.pic_transition_operators[0].apply_conv((conv_state,))[0]
    assert torch.equal(conv_result, conv_input[-2:])
    zero_state = torch.zeros((1, 2, 2), dtype=torch.float32)
    final_state = request_state.pic_transition_operators[0].apply((zero_state,))[0]
    assert capture.validate_gdn_layer(
        "model.layers.0.linear_attn",
        zero_state.unsqueeze(0),
        final_state.unsqueeze(0),
        token_start=0,
        token_end=3,
    )
    assert request_state.pic_transition_validation == {
        "model.layers.0.linear_attn": True
    }


def test_gdn_transition_layer_order_matches_kv_group_state_order() -> None:
    layers = {
        name: SimpleNamespace(mamba_type=SimpleNamespace(name="GDN_ATTN"))
        for name in (
            "layers.0.linear_attn",
            "layers.1.linear_attn",
            "layers.2.linear_attn",
            "layers.4.linear_attn",
            "layers.5.linear_attn",
            "layers.6.linear_attn",
        )
    }
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=(
            SimpleNamespace(
                layer_names=(
                    "layers.0.linear_attn",
                    "layers.4.linear_attn",
                )
            ),
            SimpleNamespace(
                layer_names=(
                    "layers.1.linear_attn",
                    "layers.5.linear_attn",
                )
            ),
            SimpleNamespace(
                layer_names=(
                    "layers.2.linear_attn",
                    "layers.6.linear_attn",
                )
            ),
        )
    )

    assert ordered_gdn_layer_names(kv_cache_config, layers) == (
        "layers.0.linear_attn",
        "layers.4.linear_attn",
        "layers.1.linear_attn",
        "layers.5.linear_attn",
        "layers.2.linear_attn",
        "layers.6.linear_attn",
    )


def test_live_capture_stores_and_applies_worker_local_transition() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2, 3], [9])
    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    manager = PICLiveCaptureManager(pool)
    operator = PICTransitionOperator.from_step_transitions(
        (
            (
                PICAffineTransition(
                    decay=torch.tensor([2.0, 1.0]),
                    zero_state=torch.tensor([1.0, 2.0]),
                ),
            ),
            (
                PICAffineTransition(
                    decay=torch.tensor([0.5, 3.0]),
                    zero_state=torch.tensor([4.0, 5.0]),
                ),
            ),
        ),
        token_start=segments[0].start,
        token_end=segments[0].end,
    )

    materialization = manager.capture_segment(
        "request-transition",
        0,
        segments[0],
        recurrent_state=(torch.zeros(2),),
        transition_operator=operator,
    )

    assert materialization is not None
    assert materialization.transition_state_handle is not None
    state = torch.tensor([3.0, 4.0])
    expected = operator.apply((state,))[0]
    actual = manager.apply_transition(
        materialization.transition_state_handle,
        (state,),
    )[0]
    assert torch.allclose(actual, expected)

    manager.release_request("request-transition")
    assert pool.free_bytes == pool.capacity_bytes
    with pytest.raises(KeyError):
        manager.apply_transition(materialization.transition_state_handle, (state,))


def test_snapshot_store_describes_native_kv_without_using_pic_pool() -> None:
    segments = split_token_ids([1, 2, 3, 4], [9])
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(7, 2),
        block_size=2,
        token_count=4,
    )
    pool = PICPhysicalPool(128, device="cpu", alignment_bytes=1)
    store = PICSnapshotStore(pool)
    snapshot = store.capture_native_kv(segments[0], (reference,))

    cache = PICSegmentCache(enabled=True)
    entry = store.materialize_cache_entry(cache, segments[0], snapshot)

    assert entry is not None
    assert entry.native_kv_refs == (reference,)
    assert pool.free_bytes == pool.capacity_bytes


def test_live_capture_restores_published_state_entry() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2], [9])
    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    registry = PICLeaseRegistry()
    manager = PICLiveCaptureManager(
        pool, retain_published=True, lease_registry=registry
    )
    recurrent = torch.arange(4, dtype=torch.float32)
    conv_tail = torch.arange(3, dtype=torch.float32)
    materialization = manager.capture_segment(
        "request-1",
        0,
        segments[0],
        recurrent_state=(recurrent,),
        conv_tail=(conv_tail,),
    )
    assert materialization is not None

    entry = PICSegmentEntry(
        seg_hash=materialization.seg_hash,
        token_ids=segments[0].token_ids,
        recurrent_state_handle=materialization.recurrent_state_handle,
        conv_tail_handle=materialization.conv_tail_handle,
    )
    restored_recurrent = torch.zeros_like(recurrent)
    restored_conv_tail = torch.zeros_like(conv_tail)
    manager.restore_materialized_segment(
        segments[0],
        entry,
        request_id="request-1",
        recurrent_state=(restored_recurrent,),
        conv_tail=(restored_conv_tail,),
    )
    assert torch.equal(recurrent, restored_recurrent)
    assert torch.equal(conv_tail, restored_conv_tail)
    assert registry.snapshot()["by_kind"] == {
        "active_request": 1,
        "cache_entry": 1,
    }

    manager.release_request("request-1")
    assert registry.snapshot()["by_kind"] == {"cache_entry": 1}
    assert pool.free_bytes < pool.capacity_bytes
    assert manager._evict_published_segment(segments[0].seg_hash)
    assert registry.snapshot()["tokens"] == 0
    assert pool.free_bytes == pool.capacity_bytes
    manager.clear()
    assert pool.free_bytes == pool.capacity_bytes


def test_live_capture_deduplicates_repeated_published_transition_snapshot() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2], [9])
    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    manager = PICLiveCaptureManager(pool, retain_published=True)
    operator = PICTransitionOperator.from_step_transitions(
        (
            (
                PICAffineTransition(
                    decay=torch.tensor([2.0]), zero_state=torch.tensor([1.0])
                ),
            ),
        ),
        token_start=0,
        token_end=2,
    )

    first = manager.capture_segment(
        "warmup-1",
        0,
        segments[0],
        recurrent_state=(torch.zeros(1),),
        transition_operator=operator,
    )
    assert first is not None
    free_after_first = pool.free_bytes

    second = manager.capture_segment(
        "warmup-2",
        0,
        segments[0],
        recurrent_state=(torch.ones(1),),
        transition_operator=operator,
        native_kv_refs=(),
    )
    assert second is not None
    assert second.transition_state_handle == first.transition_state_handle
    assert pool.free_bytes == free_after_first

    manager.release_request("warmup-1")
    manager.release_request("warmup-2")
    assert pool.free_bytes == free_after_first
    manager.clear()
    assert pool.free_bytes == pool.capacity_bytes


def test_live_capture_evicts_unleased_published_snapshot_on_pool_pressure() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2, 9, 3, 4, 9, 5, 6], [9])
    pool = PICPhysicalPool(32, device="cpu", alignment_bytes=1)
    manager = PICLiveCaptureManager(pool, retain_published=True)
    state = torch.arange(4, dtype=torch.float32)

    first = manager.capture_segment(
        "request-1", 0, segments[0], recurrent_state=(state,)
    )
    second = manager.capture_segment(
        "request-2", 1, segments[1], recurrent_state=(state,)
    )
    assert first is not None and second is not None
    assert manager.acquire_segment("active", segments[0].seg_hash)

    third = manager.capture_segment(
        "request-3", 2, segments[2], recurrent_state=(state,)
    )
    assert third is not None
    assert pool.free_bytes == 0

    restored = torch.zeros_like(state)
    pool.restore_tensor_snapshot(first.recurrent_state_handle, (restored,))
    assert torch.equal(restored, state)
    pending = manager.take_pending()
    evictions = [item for item in pending if item.evicted]
    assert len(evictions) == 1
    assert evictions[0].seg_hash == segments[1].seg_hash

    manager.release_request("active")
    manager.clear()
    assert pool.free_bytes == pool.capacity_bytes


def test_kv_slots_build_slots_for_noncontiguous_blocks() -> None:
    torch = pytest.importorskip("torch")
    slots = build_kv_slot_mapping(
        [7, 2, 11],
        block_size=4,
        token_start=2,
        token_end=10,
        device="cpu",
    )
    assert torch.equal(
        slots,
        torch.tensor([30, 31, 8, 9, 10, 11, 44, 45], dtype=torch.int64),
    )




def test_native_slot_plan_supports_noncontiguous_local_blocks() -> None:
    torch = pytest.importorskip("torch")
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(5, 2),
        block_size=2,
        token_count=4,
        kind=PICNativeKVKind.SHARED,
        position_mode=PICKVPositionMode.CANONICAL_LOCAL,
    )
    plan = build_native_kv_slot_plan(reference, target_start=0, device="cpu")

    assert plan.block_table_compatible
    assert plan.logical_block_ids == (0, 1)
    assert torch.equal(
        plan.slot_mapping,
        torch.tensor([10, 11, 4, 5], dtype=torch.int64),
    )


def test_native_slot_plan_expands_allocator_block_for_attention_kernel() -> None:
    torch = pytest.importorskip("torch")
    reference = PICNativeKVReference(
        kv_cache_group_id=3,
        block_ids=(5,),
        block_size=544,
        token_count=544,
        kind=PICNativeKVKind.SHARED,
    )
    plan = build_native_kv_slot_plan(
        reference,
        target_start=0,
        device="cpu",
        kernel_block_size=32,
    )

    assert plan.block_table_compatible
    assert plan.block_size == 32
    assert plan.logical_block_ids == tuple(range(17))
    assert plan.physical_block_ids == tuple(range(85, 102))
    assert torch.equal(
        plan.slot_mapping[:2],
        torch.tensor([2720, 2721], dtype=torch.int64),
    )
    assert int(plan.slot_mapping[-1]) == 3263


def test_native_slot_plan_crosses_allocator_boundary_at_kernel_granularity() -> None:
    pytest.importorskip("torch")
    reference = PICNativeKVReference(
        kv_cache_group_id=3,
        block_ids=(5, 2),
        block_size=544,
        token_count=544,
        canonical_start=0,
        source_token_start=32,
        source_block_start=0,
        kind=PICNativeKVKind.SHARED,
    )
    plan = build_native_kv_slot_plan(
        reference,
        target_start=32,
        device="cpu",
        kernel_block_size=32,
        local_start=0,
        token_count=544,
    )

    assert plan.block_table_compatible
    assert len(plan.logical_block_ids) == 17
    assert plan.physical_block_ids[0] == 86
    assert plan.physical_block_ids[-1] == 34


def test_native_slot_plan_private_copies_rope_shifted_aligned_range() -> None:
    pytest.importorskip("torch")
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(5, 2),
        block_size=32,
        token_count=64,
        source_token_start=0,
    )
    plan = build_native_kv_slot_plan(
        reference,
        target_start=32,
        device="cpu",
        kernel_block_size=32,
    )

    assert not plan.block_table_compatible
    assert plan.requires_private_materialization
    assert plan.fallback_reason is not None


def test_canonical_public_slot_plan_never_aliases_position_dependent_kv() -> None:
    pytest.importorskip("torch")
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(5, 2),
        block_size=32,
        token_count=64,
        kind=PICNativeKVKind.CANONICAL_PUBLIC,
        position_mode=PICKVPositionMode.CANONICAL_LOCAL,
    )
    plan = build_native_kv_slot_plan(
        reference, target_start=0, device="cpu", kernel_block_size=32
    )
    assert not plan.block_table_compatible
    assert plan.requires_private_materialization


def test_native_slot_plan_rejects_unaligned_target_for_block_table() -> None:
    pytest.importorskip("torch")
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(5, 2),
        block_size=2,
        token_count=3,
    )
    plan = build_native_kv_slot_plan(reference, target_start=3, device="cpu")

    assert not plan.block_table_compatible
    assert plan.requires_private_materialization
    assert plan.logical_block_ids == ()
    assert plan.fallback_reason is not None


def test_native_slot_plan_tracks_absolute_source_offsets() -> None:
    torch = pytest.importorskip("torch")
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(5, 2, 9),
        block_size=2,
        token_count=4,
        canonical_start=0,
        source_token_start=3,
        source_block_start=1,
    )
    plan = build_native_kv_slot_plan(reference, target_start=7, device="cpu")

    assert not plan.block_table_compatible
    assert torch.equal(
        plan.slot_mapping,
        torch.tensor([11, 4, 5, 18], dtype=torch.int64),
    )
    assert plan.source_start == 3


def test_native_kv_gather_supports_nhd_and_hnd_layouts() -> None:
    torch = pytest.importorskip("torch")
    # Three blocks, two tokens per block, one KV head, two head dimensions.
    nhd = torch.arange(2 * 3 * 2 * 1 * 2, dtype=torch.float32).reshape(2, 3, 2, 1, 2)
    hnd = nhd.permute(0, 1, 3, 2, 4).contiguous()
    slots = torch.tensor([0, 3, 4], dtype=torch.int64)
    expected_key = torch.stack((nhd[0, 0, 0], nhd[0, 1, 1], nhd[0, 2, 0]))
    expected_value = torch.stack((nhd[1, 0, 0], nhd[1, 1, 1], nhd[1, 2, 0]))

    nhd_key, nhd_value = gather_native_kv_slots(nhd, slots, block_size=2, layout="NHD")
    hnd_key, hnd_value = gather_native_kv_slots(hnd, slots, block_size=2, layout="HND")

    assert torch.equal(nhd_key, expected_key)
    assert torch.equal(nhd_value, expected_value)
    assert torch.equal(hnd_key, expected_key)
    assert torch.equal(hnd_value, expected_value)


def test_native_kv_private_plan_is_accepted_for_arbitrary_target_phase() -> None:
    pytest.importorskip("torch")
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(5, 2),
        block_size=4,
        token_count=8,
        source_token_start=0,
    )
    plan = build_native_kv_slot_plan(
        reference,
        target_start=3,
        device="cpu",
        kernel_block_size=2,
        token_count=6,
    )

    assert not plan.block_table_compatible
    assert plan.requires_private_materialization
    assert plan.source_start == 0
    assert plan.target_start == 3
    assert plan.token_count == 6


def test_native_kv_rerotation_is_exact_for_identical_positions() -> None:
    torch = pytest.importorskip("torch")
    key = torch.randn(4, 2, 8, dtype=torch.bfloat16)
    positions = torch.arange(4, dtype=torch.int64)

    # No rotary object is needed when source and target positions are equal:
    # the implementation must preserve the native K instead of performing an
    # inverse/forward BF16 round trip.
    result = rerotate_native_key(
        key,
        rotary_emb=object(),
        source_positions=positions,
        target_positions=positions.clone(),
    )

    assert torch.equal(result, key)


def test_native_lease_pins_and_releases_block_pool_blocks() -> None:
    class FakeBlockPool:
        def __init__(self) -> None:
            self.blocks = [object() for _ in range(4)]
            self.touched: list[tuple[object, ...]] = []
            self.freed: list[tuple[object, ...]] = []

        def touch(self, blocks: tuple[object, ...]) -> None:
            self.touched.append(blocks)

        def free_blocks(self, blocks: tuple[object, ...]) -> None:
            self.freed.append(blocks)

    pool = FakeBlockPool()
    lease = PICNativeKVLease.from_block_pool(pool, [3, 1])
    assert lease.block_ids == (3, 1)
    assert len(pool.touched) == 1
    assert lease.release()
    assert not lease.release()
    assert len(pool.freed) == 1


def test_native_lease_can_adopt_scheduler_owned_public_blocks() -> None:
    class FakeBlockPool:
        def __init__(self) -> None:
            self.blocks = [object() for _ in range(2)]
            self.touched: list[tuple[object, ...]] = []
            self.freed: list[tuple[object, ...]] = []

        def touch(self, blocks: tuple[object, ...]) -> None:
            self.touched.append(blocks)

        def free_blocks(self, blocks: tuple[object, ...]) -> None:
            self.freed.append(blocks)

    pool = FakeBlockPool()
    lease = PICNativeKVLease.from_owned_block_pool(pool, [0, 1])
    assert not pool.touched
    assert lease.release()
    assert len(pool.freed) == 1


def test_canonical_public_key_is_immutable_during_private_rerotation() -> None:
    torch = pytest.importorskip("torch")
    public_key = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    original = public_key.clone()
    plan = PICRoPERerotationPlan(token_count=3, source_start=0, target_start=8)

    def rerotate(key, source_positions, target_positions):
        key.add_((target_positions - source_positions).to(key.dtype).unsqueeze(1))
        return key

    private_key = materialize_private_rope_key(public_key, plan, rerotate)

    assert torch.equal(public_key, original)
    assert torch.equal(private_key, original + 8)


def test_single_request_plan_builds_nonprefix_skip_and_recompute() -> None:
    segments = split_token_ids([1, 2, 9, 3, 4, 9, 5], [9])
    cache = PICSegmentCache(enabled=True)
    cache.insert(segments[0], full_kv_handles=(1,))
    execution_plan = compile_execution_plan(
        segments, cache.build_plan(segments, seam_sink=0)
    )

    plan = build_single_request_plan(
        execution_plan,
        prompt_len=7,
        num_requests=1,
        eager_mode=True,
        speculative_decoding=False,
        range_execution_supported=True,
        zero_copy_attention_supported=False,
        allow_fallback=False,
        native_kv_bridge_supported=True,
    )

    assert plan is not None
    assert plan.skip_positions == (0, 1)
    assert plan.recompute_positions == (2, 3, 4, 5, 6)


def test_single_request_plan_falls_back_for_unsupported_batch_mode() -> None:
    segments = split_token_ids([1, 2, 9, 3], [9])
    cache = PICSegmentCache(enabled=True)
    cache.insert(segments[0], full_kv_handles=(1,))
    execution_plan = compile_execution_plan(
        segments, cache.build_plan(segments, seam_sink=0)
    )

    assert (
        build_single_request_plan(
            execution_plan,
            prompt_len=3,
            num_requests=2,
            eager_mode=True,
            speculative_decoding=False,
            range_execution_supported=True,
            zero_copy_attention_supported=False,
            allow_fallback=True,
            native_kv_bridge_supported=True,
        )
        is None
    )


def test_single_request_runtime_plan_maps_native_reuse_and_miss_ranges() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2, 9, 3, 4, 9, 5], [9])
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(7,),
        block_size=2,
        token_count=2,
        kind=PICNativeKVKind.SHARED,
    )
    cache = PICSegmentCache(enabled=True)
    cache.insert(segments[0], native_kv_refs=(reference,))
    execution_plan = compile_execution_plan(
        segments, cache.build_plan(segments, seam_sink=0)
    )
    worker_plan = build_single_request_plan(
        execution_plan,
        prompt_len=7,
        num_requests=1,
        eager_mode=True,
        speculative_decoding=False,
        range_execution_supported=True,
        zero_copy_attention_supported=True,
        native_kv_bridge_supported=True,
        allow_fallback=False,
    )

    runtime_plan = build_single_request_runtime_plan(
        worker_plan,
        native_kv_refs_by_group=(reference,),
        device="cpu",
        prompt_len=7,
        allow_fallback=False,
    )

    assert runtime_plan is not None
    # Contiguous recompute ranges are one span of the prompt and are issued as a
    # single forward, so the four gaps fold into one range.
    assert [(item.action, item.start, item.end) for item in runtime_plan.ranges] == [
        ("reuse", 0, 2),
        ("recompute", 2, 7),
    ]
    assert len(runtime_plan.recompute_ranges) == 1
    assert runtime_plan.native_slot_plans[0].logical_block_ids == (0,)
    assert torch.equal(
        runtime_plan.native_slot_plans[0].slot_mapping,
        torch.tensor([14, 15], dtype=torch.int64),
    )


def test_single_request_runtime_plan_splits_segment_local_partial_suffix() -> None:
    segments = split_token_ids(
        [1, 2, 3, 4, 5, 9, 6, 9, 7],
        [9],
    )
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(7, 8),
        block_size=2,
        token_count=4,
        canonical_start=0,
    )
    cache = PICSegmentCache(enabled=True)
    cache.insert(segments[0], native_kv_refs=(reference,))
    execution_plan = compile_execution_plan(
        segments, cache.build_plan(segments, seam_sink=0)
    )
    worker_plan = build_single_request_plan(
        execution_plan,
        prompt_len=9,
        num_requests=1,
        eager_mode=True,
        speculative_decoding=False,
        range_execution_supported=True,
        zero_copy_attention_supported=True,
        native_kv_bridge_supported=True,
        allow_fallback=False,
    )

    runtime_plan = build_single_request_runtime_plan(
        worker_plan,
        native_kv_refs_by_group=(reference,),
        device="cpu",
        prompt_len=9,
        allow_fallback=False,
        seam_tokens_by_segment={0: 1},
    )

    assert runtime_plan is not None
    assert (
        runtime_plan.ranges[0].action,
        runtime_plan.ranges[0].start,
        runtime_plan.ranges[0].end,
    ) == ("recompute", 0, 1)
    assert (
        runtime_plan.ranges[1].action,
        runtime_plan.ranges[1].start,
        runtime_plan.ranges[1].end,
    ) == ("reuse", 1, 4)
    # The tail after the reused body meets the following segment's tokens, so the
    # two recompute ranges are served by one forward.
    assert (
        runtime_plan.ranges[2].action,
        runtime_plan.ranges[2].start,
        runtime_plan.ranges[2].end,
    ) == ("recompute", 4, 9)
    assert runtime_plan.ranges[-1].action == "recompute"


def test_single_request_runtime_allows_partial_hybrid_range_with_transition() -> None:
    segments = split_token_ids([1, 2, 3, 4, 5, 9, 6, 9, 7], [9])
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(7, 8),
        block_size=2,
        token_count=4,
        canonical_start=0,
    )
    cache = PICSegmentCache(enabled=True)
    cache.insert(
        segments[0],
        transition_state_handle=42,
        recurrent_state_handle=43,
        native_kv_refs=(reference,),
    )
    execution_plan = compile_execution_plan(
        segments, cache.build_plan(segments, seam_sink=1)
    )
    worker_plan = build_single_request_plan(
        execution_plan,
        prompt_len=9,
        num_requests=1,
        eager_mode=True,
        speculative_decoding=False,
        range_execution_supported=True,
        zero_copy_attention_supported=True,
        native_kv_bridge_supported=True,
        allow_fallback=False,
    )

    runtime_plan = build_single_request_runtime_plan(
        worker_plan,
        native_kv_refs_by_group=(reference,),
        device="cpu",
        prompt_len=9,
        allow_fallback=False,
        seam_tokens_by_segment={0: 1},
    )

    assert runtime_plan is not None
    assert any(item.action == "reuse" for item in runtime_plan.ranges)


def test_single_request_runtime_allows_partial_conv_tail_reuse_with_transition() -> (
    None
):
    segments = split_token_ids([1, 2, 3, 4, 5, 9, 6, 9, 7], [9])
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(7, 8),
        block_size=2,
        token_count=4,
        canonical_start=0,
    )
    cache = PICSegmentCache(enabled=True)
    cache.insert(
        segments[0],
        transition_state_handle=42,
        conv_tail_handle=44,
        native_kv_refs=(reference,),
    )
    execution_plan = compile_execution_plan(
        segments, cache.build_plan(segments, seam_sink=1)
    )
    worker_plan = build_single_request_plan(
        execution_plan,
        prompt_len=9,
        num_requests=1,
        eager_mode=True,
        speculative_decoding=False,
        range_execution_supported=True,
        zero_copy_attention_supported=True,
        native_kv_bridge_supported=True,
        allow_fallback=True,
    )

    runtime_plan = build_single_request_runtime_plan(
        worker_plan,
        native_kv_refs_by_group=(reference,),
        device="cpu",
        prompt_len=9,
        allow_fallback=True,
        seam_tokens_by_segment={0: 1},
    )

    assert runtime_plan is not None
    assert any(item.action == "reuse" for item in runtime_plan.ranges)


def test_single_request_runtime_plan_uses_private_copy_for_unaligned_native_range() -> (
    None
):
    segments = split_token_ids([1, 2, 9, 3, 4, 5, 9, 6], [9])
    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(7, 8),
        block_size=2,
        token_count=3,
    )
    cache = PICSegmentCache(enabled=True)
    cache.insert(segments[1], native_kv_refs=(reference,))
    execution_plan = compile_execution_plan(
        segments, cache.build_plan(segments, seam_sink=0)
    )
    worker_plan = build_single_request_plan(
        execution_plan,
        prompt_len=8,
        num_requests=1,
        eager_mode=True,
        speculative_decoding=False,
        range_execution_supported=True,
        zero_copy_attention_supported=True,
        native_kv_bridge_supported=True,
        allow_fallback=False,
    )

    runtime_plan = build_single_request_runtime_plan(
        worker_plan,
        native_kv_refs_by_group=(reference,),
        device="cpu",
        prompt_len=8,
        allow_fallback=False,
    )
    assert runtime_plan is not None
    assert runtime_plan.native_slot_plans[0].requires_private_materialization
    assert not runtime_plan.native_slot_plans[0].block_table_compatible


def test_pic_packed_attention_round_builds_absolute_query_metadata() -> None:
    import torch

    active = {
        "pic-a": PICRuntimeRange(-1, 1088, 1120, "recompute", 1120),
        "ordinary": PICRuntimeRange(-1, 0, 5, "recompute", 5),
    }
    positions = torch.arange(37, dtype=torch.int64)
    slot_mappings = {0: torch.arange(40, dtype=torch.int64)}

    metadata = build_pic_packed_attention_round(
        ("pic-a", "ordinary"),
        active,
        positions=positions,
        slot_mappings_by_group=slot_mappings,
        num_tokens_padded=40,
    )

    assert metadata.query_lengths == (32, 5)
    assert metadata.absolute_ranges == ((1088, 1120), (0, 5))
    assert metadata.query_start_loc.tolist() == [0, 32, 37]
    assert metadata.num_tokens == 37
    assert metadata.kv_group_ids == (0,)


def test_pic_packed_attention_round_rejects_incomplete_slot_mapping() -> None:
    import torch

    active = {
        "pic-a": PICRuntimeRange(-1, 0, 2, "recompute", 2),
        "pic-b": PICRuntimeRange(-1, 4, 7, "recompute", 7),
    }

    with pytest.raises(PICAttentionBackendUnsupported, match="invalid extent"):
        build_pic_packed_attention_round(
            ("pic-a", "pic-b"),
            active,
            positions=torch.arange(5, dtype=torch.int64),
            slot_mappings_by_group={0: torch.arange(4, dtype=torch.int64)},
            num_tokens_padded=5,
        )


def test_pic_native_request_mapping_persists_logical_ranges() -> None:
    import torch

    first = PICNativeKVSlotPlan(
        kv_cache_group_id=0,
        target_start=0,
        target_end=4,
        source_start=0,
        block_size=2,
        logical_block_ids=(0, 1),
        physical_block_ids=(8, 9),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        block_table_compatible=True,
    )
    second = PICNativeKVSlotPlan(
        kv_cache_group_id=1,
        target_start=0,
        target_end=4,
        source_start=0,
        block_size=2,
        logical_block_ids=(0, 1),
        physical_block_ids=(12, 13),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        block_table_compatible=True,
    )

    mapping = PICNativeKVRequestMapping((first, second))

    assert mapping.for_group(0) == (first,)
    assert mapping.for_group(1) == (second,)
    assert not mapping.is_empty

    with pytest.raises(ValueError, match="duplicate logical ranges"):
        PICNativeKVRequestMapping((first, first))


def test_pic_native_allocation_separates_public_and_private_edges() -> None:
    import torch

    public_plan = PICNativeKVSlotPlan(
        kv_cache_group_id=0,
        target_start=0,
        target_end=4,
        source_start=0,
        block_size=2,
        logical_block_ids=(0, 1),
        physical_block_ids=(8, 9),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        block_table_compatible=True,
    )
    public = PICNativeKVAllocation.from_slot_plan(public_plan)

    assert public.is_public
    assert not public.is_private
    assert public.public_block_ids == (8, 9)
    assert public.edge_prefix_tokens == 0
    assert public.edge_suffix_tokens == 0

    private_plan = PICNativeKVSlotPlan(
        kv_cache_group_id=0,
        target_start=1,
        target_end=7,
        source_start=0,
        block_size=2,
        logical_block_ids=(),
        physical_block_ids=(),
        slot_mapping=torch.arange(6, dtype=torch.int64),
        block_table_compatible=False,
        requires_private_materialization=True,
    )
    private = PICNativeKVAllocation.from_slot_plan(
        private_plan,
        private_block_ids=(20, 21, 22, 23),
    )

    assert private.is_private
    assert not private.is_public
    assert private.has_edge_tokens
    assert private.edge_prefix_tokens == 1
    assert private.edge_suffix_tokens == 1

    with pytest.raises(ValueError, match="does not cover target range"):
        PICNativeKVAllocation.from_slot_plan(
            private_plan,
            private_block_ids=(20, 21, 22),
        )

    with pytest.raises(ValueError, match="public and private"):
        PICNativeKVAllocation(
            kv_cache_group_id=0,
            target_start=0,
            target_end=2,
            block_size=2,
            public_block_ids=(8,),
            private_block_ids=(20,),
        )
