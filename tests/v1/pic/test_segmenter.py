# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from types import SimpleNamespace

from vllm.v1.pic.cache import PICSegmentCache, PICSegmentEntry
from vllm.v1.pic.handles import PICHandleKind, PICHandlePool
from vllm.v1.pic.execution import compile_execution_plan
from vllm.v1.pic.live_capture import (
    PICGDNTransitionCapture,
    PICLiveCaptureManager,
    ordered_gdn_layer_names,
)
from vllm.v1.pic.kv_copyback import (
    PICAttentionKVCacheCopyBack,
    PICKVLayerTarget,
    build_kv_slot_mapping,
)
from vllm.v1.pic.segmenter import split_text_and_tokenize, split_token_ids
from vllm.v1.pic.pool import PICPhysicalPool, PICSlotMapping
from vllm.v1.pic.snapshot import PICSnapshotStore
from vllm.v1.pic.state import (
    PICAffineTransition,
    PICConvTransition,
    PICGDNTransition,
    PICStateLayout,
    PICStateSpec,
    PICTransitionOperator,
    PICRequestStateBinding,
)
from vllm.v1.pic.gdn_transition import build_gdn_transition_operator
from vllm.v1.pic.worker_plan import build_worker_plan
from vllm.v1.pic.native_kv import (
    PICKVPositionMode,
    PICNativeKVAllocation,
    PICNativeKVKind,
    PICNativeKVLease,
    PICNativeKVReference,
    PICNativeKVRequestMapping,
    PICRoPERerotationPlan,
    PICNativeKVSlotPlan,
    build_native_kv_slot_plan,
    gather_native_kv_slots,
    get_full_local_block_span,
    has_complete_native_kv_reference,
    is_pic_public_segment_cacheable,
    materialize_private_rope_key,
    rerotate_native_key,
)
from vllm.v1.pic.single_request import build_single_request_plan
from vllm.v1.pic.runtime import (
    PICRuntimeRange,
    PICSingleRequestRuntimePlan,
    build_single_request_runtime_plan,
)
from vllm.v1.pic.batch import (
    build_pic_batch_runtime_plan,
    build_pic_packed_batch_plan,
)
from vllm.v1.pic.attention import (
    PICAttentionBackendUnsupported,
    build_pic_packed_attention_round,
)
from vllm.v1.pic.lifecycle import PICLeaseRegistry
from vllm.v1.pic.metrics import PICMetrics
from vllm.v1.pic.external_kv import (
    PICExternalKVImportError,
    PICExternalKVImportResult,
    PICExternalKVRegistry,
)


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
    assert snapshot["fallback_reasons"] == {
        "request-local-mapping-mismatch": 1
    }
    assert snapshot["decode_mapping_reused"] == 1
    assert snapshot["leases"] == {"tokens": 1}
    assert snapshot["pool"] == {"capacity_bytes": 1024, "free_bytes": 768}


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


def test_pic_lease_registry_rejects_acquire_after_retire() -> None:
    registry = PICLeaseRegistry()
    resource = ("snapshot", b"segment")
    token = registry.acquire(resource, owner="entry:x", kind="cache_entry")
    assert registry.retire(resource)
    with pytest.raises(RuntimeError, match="retired"):
        registry.acquire(resource, owner="request:x", kind="active_request")
    assert registry.release(token)


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
    assert ranges == [(0, 1), (1, 2)]


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
    assert result[2] == [(0, 1), (1, 2)]


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
        kv_cache_groups = []

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
    mapping = PICSlotMapping(
        logical_positions=(4, 5, 6), physical_slots=(17, 2, 31)
    )
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
    assert build_worker_plan(
        execution_plan,
        prompt_len=4,
        range_execution_supported=False,
        allow_fallback=True,
    ) is None


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
    pool.restore_tensor_snapshot(
        materialization.recurrent_state_handle, (restored,)
    )
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
    log_decay = torch.log(
        torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    )
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
            * (value_value - (expected * key_value).sum(dim=-1))
            .unsqueeze(-1)
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
            * (value_value - (zero_expected * key_value).sum(dim=-1))
            .unsqueeze(-1)
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
    chunk_g = torch.tensor(
        [[-0.25], [-0.5], [-0.75]], dtype=torch.float32
    )
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
            (value_token - (expected * key_token).sum(dim=-1))
            .unsqueeze(-1)
            * key_token
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
    assert torch.equal(
        transition.slice(1, 3).apply(state_ds), expected.transpose(0, 1)
    )


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
    conv_result = request_state.pic_transition_operators[0].apply_conv(
        (conv_state,)
    )[0]
    assert torch.equal(conv_result, conv_input[-2:])
    zero_state = torch.zeros((1, 2, 2), dtype=torch.float32)
    final_state = request_state.pic_transition_operators[0].apply(
        (zero_state,)
    )[0]
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
        ((PICAffineTransition(decay=torch.tensor([2.0]), zero_state=torch.tensor([1.0])),),),
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


def test_kv_copyback_builds_slots_for_noncontiguous_blocks() -> None:
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


def test_kv_copyback_restores_snapshot_through_existing_target_callback() -> None:
    torch = pytest.importorskip("torch")
    segments = split_token_ids([1, 2], [])
    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    store = PICSnapshotStore(pool)
    source_key = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    source_value = source_key + 100
    snapshot = store.capture(
        segments[0],
        full_kv=(source_key, source_value),
    )
    entry = PICSegmentEntry(
        seg_hash=segments[0].seg_hash,
        token_ids=segments[0].token_ids,
        full_kv_handles=snapshot.full_kv_handles,
    )
    copied: list[torch.Tensor] = []
    target = PICKVLayerTarget(
        layer_name="layer.0",
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        copy_kv=lambda key, value: copied.extend((key.clone(), value.clone())),
    )

    copied_layers = PICAttentionKVCacheCopyBack(store).copy_entry(
        segments[0], entry, (target,)
    )

    assert copied_layers == 1
    assert torch.equal(copied[0], source_key)
    assert torch.equal(copied[1], source_value)
    store.release(snapshot)
    assert pool.free_bytes == pool.capacity_bytes


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
    torch = pytest.importorskip("torch")
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
    torch = pytest.importorskip("torch")
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
    torch = pytest.importorskip("torch")
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
    torch = pytest.importorskip("torch")
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
    nhd = torch.arange(2 * 3 * 2 * 1 * 2, dtype=torch.float32).reshape(
        2, 3, 2, 1, 2
    )
    hnd = nhd.permute(0, 1, 3, 2, 4).contiguous()
    slots = torch.tensor([0, 3, 4], dtype=torch.int64)
    expected_key = torch.stack((nhd[0, 0, 0], nhd[0, 1, 1], nhd[0, 2, 0]))
    expected_value = torch.stack((nhd[1, 0, 0], nhd[1, 1, 1], nhd[1, 2, 0]))

    nhd_key, nhd_value = gather_native_kv_slots(
        nhd, slots, block_size=2, layout="NHD"
    )
    hnd_key, hnd_value = gather_native_kv_slots(
        hnd, slots, block_size=2, layout="HND"
    )

    assert torch.equal(nhd_key, expected_key)
    assert torch.equal(nhd_value, expected_value)
    assert torch.equal(hnd_key, expected_key)
    assert torch.equal(hnd_value, expected_value)


def test_native_kv_private_plan_is_accepted_for_arbitrary_target_phase() -> None:
    torch = pytest.importorskip("torch")
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
    assert [(item.action, item.start, item.end) for item in runtime_plan.ranges] == [
        ("reuse", 0, 2),
        ("recompute", 2, 3),
        ("recompute", 3, 5),
        ("recompute", 5, 6),
        ("recompute", 6, 7),
    ]
    assert len(runtime_plan.recompute_ranges) == 4
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
    assert (
        runtime_plan.ranges[2].action,
        runtime_plan.ranges[2].start,
        runtime_plan.ranges[2].end,
    ) == ("recompute", 4, 5)
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


def test_single_request_runtime_allows_partial_conv_tail_reuse_with_transition() -> None:
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


def test_single_request_runtime_plan_uses_private_copy_for_unaligned_native_range() -> None:
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


def test_pic_batch_runtime_plan_keeps_request_ranges_independent() -> None:
    first = PICSingleRequestRuntimePlan(
        ranges=(
            PICRuntimeRange(0, 0, 2, "reuse", 2),
            PICRuntimeRange(-1, 2, 4, "recompute", 4),
        ),
        native_slot_plans=(),
    )
    second = PICSingleRequestRuntimePlan(
        ranges=(
            PICRuntimeRange(1, 0, 3, "reuse", 3),
            PICRuntimeRange(-1, 3, 5, "recompute", 5),
        ),
        native_slot_plans=(),
    )

    batch = build_pic_batch_runtime_plan(
        ("request-a", "request-b", "ordinary"),
        {"request-a": first, "request-b": second},
    )

    assert batch.plan_for("request-a") is first
    assert batch.plan_for("request-b") is second
    assert batch.plan_for("ordinary") is None
    assert tuple(request_id for request_id, _ in batch.runtime_plans) == (
        "request-a",
        "request-b",
    )


def test_pic_batch_runtime_plan_rejects_duplicate_request_ids() -> None:
    plan = PICSingleRequestRuntimePlan(
        ranges=(PICRuntimeRange(-1, 0, 1, "recompute", 1),),
        native_slot_plans=(),
    )

    with pytest.raises(ValueError, match="request IDs must be unique"):
        build_pic_batch_runtime_plan(("request-a", "request-a"), {"request-a": plan})


def test_pic_packed_batch_plan_requires_aligned_recompute_rounds() -> None:
    pic_plan = PICSingleRequestRuntimePlan(
        ranges=(
            PICRuntimeRange(0, 0, 2, "reuse", 2),
            PICRuntimeRange(-1, 2, 4, "recompute", 4),
        ),
        native_slot_plans=(),
    )
    ordinary_plan = PICSingleRequestRuntimePlan(
        ranges=(PICRuntimeRange(-1, 0, 4, "recompute", 4),),
        native_slot_plans=(),
    )

    packed = build_pic_packed_batch_plan(
        ("pic", "ordinary"),
        {"pic": pic_plan, "ordinary": ordinary_plan},
    )

    assert packed.plan_for("pic") is pic_plan
    assert packed.plan_for("ordinary") is ordinary_plan

    mismatched = PICSingleRequestRuntimePlan(
        ranges=(
            PICRuntimeRange(-1, 0, 1, "recompute", 1),
            PICRuntimeRange(-1, 1, 2, "recompute", 2),
        ),
        native_slot_plans=(),
    )
    with pytest.raises(ValueError, match="same number of recompute ranges"):
        build_pic_packed_batch_plan(
            ("pic", "ordinary"),
            {"pic": pic_plan, "ordinary": mismatched},
        )


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
