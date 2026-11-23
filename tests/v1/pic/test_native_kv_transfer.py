# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.v1.pic.external_kv import (
    PICExternalKVImportError,
    PICExternalKVImportRequest,
    PICExternalKVImportResult,
    PICExternalKVRegistry,
)
from vllm.v1.pic.lifecycle import PICLeaseRegistry
from vllm.v1.pic.mooncake_provider import MooncakePICExternalKVProvider
from vllm.v1.pic.native_kv import PICNativeKVLease, PICNativeKVReference
from vllm.v1.pic.native_kv_transfer import (
    PICNativeKVImportHandle,
    PICNativeKVImportWorker,
)


def _result() -> PICExternalKVImportResult:
    lease = PICNativeKVLease(block_ids=(7,), _release_fn=lambda: None)
    return PICExternalKVImportResult(
        block_ids=(7,),
        block_size=16,
        token_count=16,
        lease=lease,
    )


def test_native_kv_import_handle_waits_for_ready_result() -> None:
    handle = PICNativeKVImportHandle(request_key="segment/a")
    assert not handle.ready
    handle._set_result(_result())
    assert handle.ready
    assert not handle.failed
    assert handle.wait().block_ids == (7,)


def test_native_kv_import_handle_exposes_failure() -> None:
    handle = PICNativeKVImportHandle(request_key="segment/b")
    error = PICExternalKVImportError("transfer failed")
    handle._set_error(error)
    assert not handle.ready
    assert handle.failed
    assert handle.error is error
    with pytest.raises(PICExternalKVImportError, match="transfer failed"):
        handle.wait()


def test_native_kv_import_worker_runs_operation_in_background() -> None:
    worker = PICNativeKVImportWorker()
    try:
        handle = PICNativeKVImportHandle(request_key="segment/c")
        worker.submit(handle, _result)
        assert handle.wait().block_ids == (7,)
        assert handle.ready
    finally:
        worker.shutdown()


def test_native_kv_import_worker_completes_operation_inline() -> None:
    worker = PICNativeKVImportWorker()
    handle = PICNativeKVImportHandle(request_key="segment/inline")
    worker.complete_inline(handle, _result)
    assert handle.ready
    assert handle.wait().block_ids == (7,)
    worker.shutdown()


def test_native_kv_import_completion_finalizes_on_wait() -> None:
    from vllm.v1.pic.native_kv_transfer import PICNativeKVImportCompletion

    worker = PICNativeKVImportWorker()
    finalized: list[str] = []
    try:
        handle = PICNativeKVImportHandle(request_key="segment/split")

        def operation() -> PICNativeKVImportCompletion:
            return PICNativeKVImportCompletion(
                result=_result(),
                finalize=lambda: finalized.append("materialized"),
            )

        worker.submit(handle, operation)
        result = handle.wait()
        assert result.block_ids == (7,)
        assert finalized == ["materialized"]
        assert handle.ready
    finally:
        worker.shutdown()


def test_mooncake_p5_splits_receive_from_materialize() -> None:
    received: list[tuple[int, ...]] = []
    materialized: list[object] = []
    provider = MooncakePICExternalKVProvider(
        lease_registry=PICLeaseRegistry(),
        block_resolver=lambda request: (10, 11),
        receive=lambda request, block_ids, engine: (
            received.append(block_ids) or object()
        ),
        materialize=lambda request, payload, block_ids, engine: materialized.append(
            payload
        ),
        release=lambda request, block_ids, engine: None,
    )
    request = PICExternalKVImportRequest(
        provider="mooncake",
        external_key="segment/p5",
        kv_cache_group_id=0,
        block_size=4,
        token_count=8,
        canonical_start=0,
        dtype="bfloat16",
        layout=(2, 8, 4, 1, 2),
    )

    handle = provider.begin_import_native_kv(request)
    result = handle.wait()
    assert result.block_ids == (10, 11)
    assert received == [(10, 11)]
    assert len(materialized) == 1
    assert result.lease.release()


def test_external_kv_import_batch_waits_at_explicit_barrier() -> None:
    class Provider:
        name = "fake"

        def import_native_kv(self, request):
            raise AssertionError("the async provider entry should be used")

        def begin_import_native_kv(self, request):
            handle = PICNativeKVImportHandle(request_key=request.external_key)
            handle._set_result(_result())
            return handle

    reference = PICNativeKVReference(
        kv_cache_group_id=0,
        block_ids=(),
        block_size=16,
        token_count=16,
        external_provider="fake",
        external_key="segment/fake",
    )
    registry = PICExternalKVRegistry()
    registry.register(Provider())
    batch = registry.begin_import_references((reference,))
    imported = batch.wait()
    assert imported[0].block_ids == (7,)
    imported[0].release()
