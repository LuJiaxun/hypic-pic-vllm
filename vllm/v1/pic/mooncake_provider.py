# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail-closed Mooncake adapter for external native KV references.

The worker does not own the scheduler's BlockPool. Therefore this adapter only
accepts native block IDs supplied by an explicit resolver and only transfers data
when the caller supplies a destination-writer callback. It never invents block
IDs and never treats a remote pointer as an attention-visible native block.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from math import ceil
from typing import Any

from vllm.v1.pic.external_kv import (
    PICExternalKVImportError,
    PICExternalKVImportRequest,
    PICExternalKVImportResult,
)
from vllm.v1.pic.lifecycle import PICLeaseRegistry, PICLeaseToken
from vllm.v1.pic.native_kv import PICNativeKVLease, PICWindow
from vllm.v1.pic.native_kv_transfer import (
    PICNativeKVImportHandle,
    PICNativeKVImportCompletion,
    PICNativeKVImportWorker,
)
from vllm.v1.pic.profile_level import pic_profile_enabled
from vllm.logger import init_logger

BlockResolver = Callable[[PICExternalKVImportRequest], Sequence[int]]
TransferCallback = Callable[[PICExternalKVImportRequest, tuple[int, ...], Any], None]
ReceiveCallback = Callable[
    [PICExternalKVImportRequest, tuple[int, ...], Any], Any
]
MaterializeCallback = Callable[
    [PICExternalKVImportRequest, Any, tuple[int, ...], Any], None
]
ReleaseCallback = Callable[[PICExternalKVImportRequest, tuple[int, ...], Any], None]

logger = init_logger(__name__)


class MooncakePICExternalKVProvider:
    """Import Mooncake objects only into caller-owned native KV blocks.

    ``block_resolver`` must return already-owned native block IDs. The provider
    does not allocate scheduler blocks because the scheduler and GPU worker run
    in different processes. ``transfer`` is injected so the provider stays
    independent of a particular Mooncake object format.
    """

    name = "mooncake"

    def __init__(
        self,
        *,
        lease_registry: PICLeaseRegistry,
        block_resolver: BlockResolver | None = None,
        engine: Any = None,
        transfer: TransferCallback | None = None,
        receive: ReceiveCallback | None = None,
        materialize: MaterializeCallback | None = None,
        release: ReleaseCallback | None = None,
    ) -> None:
        self._lease_registry = lease_registry
        self._block_resolver = block_resolver
        self._engine = engine
        self._transfer = transfer
        self._receive = receive
        self._materialize = materialize
        self._release = release
        self._release_callbacks: dict[tuple[Any, ...], Callable[[], None]] = {}
        self._resource_block_ids: dict[tuple[Any, ...], tuple[int, ...]] = {}
        self._import_worker = PICNativeKVImportWorker()

    @staticmethod
    def _resource_key(request: PICExternalKVImportRequest) -> tuple[Any, ...]:
        return (
            "mooncake",
            request.external_key,
            request.kv_cache_group_id,
            request.canonical_start,
            request.block_size,
            request.token_count,
            request.dtype,
            request.layout,
        )

    @staticmethod
    def _expected_block_count(request: PICExternalKVImportRequest) -> int:
        return PICWindow(
            request.canonical_start,
            request.canonical_start + request.token_count,
            request.block_size,
        ).blocks()

    def _make_release_callback(
        self,
        key: tuple[Any, ...],
        request: PICExternalKVImportRequest,
        block_ids: tuple[int, ...],
    ) -> Callable[[], None]:
        callback = self._release_callbacks.get(key)
        if callback is not None:
            return callback

        def release_import() -> None:
            self._release_callbacks.pop(key, None)
            self._resource_block_ids.pop(key, None)
            if self._release is not None:
                self._release(request, block_ids, self._engine)

        self._release_callbacks[key] = release_import
        return release_import

    def _release_token(self, key: tuple[Any, ...], token: PICLeaseToken) -> None:
        self._lease_registry.retire(key)
        self._lease_registry.release(token)

    def begin_import_native_kv(
        self, request: PICExternalKVImportRequest
    ) -> PICNativeKVImportHandle:
        """Submit a native KV import to the provider's receiver worker."""
        handle = PICNativeKVImportHandle(request_key=request.external_key)
        operation = (
            lambda: self._import_native_kv_receive(request)
            if self._receive is not None and self._materialize is not None
            else self._import_native_kv_sync(request)
        )
        self._import_worker.submit(handle, operation)
        return handle

    def import_native_kv(
        self, request: PICExternalKVImportRequest
    ) -> PICExternalKVImportResult:
        """Keep the existing synchronous native-KV hot path.

        The request-level P5 entry point uses the split receive/materialize
        callbacks; this method intentionally preserves the original complete
        synchronous transfer for import-only and fallback callers.
        """
        handle = PICNativeKVImportHandle(request_key=request.external_key)
        self._import_worker.complete_inline(
            handle,
            lambda: self._import_native_kv_sync(request),
        )
        return handle.wait()

    def _import_native_kv_sync(
        self, request: PICExternalKVImportRequest
    ) -> PICExternalKVImportResult:
        profile = pic_profile_enabled()
        total_start = time.perf_counter() if profile else 0.0
        if (
            request.block_size <= 0
            or request.token_count <= 0
            or request.canonical_start < 0
            or request.dtype is None
            or not request.layout
        ):
            raise PICExternalKVImportError(
                "Mooncake external KV metadata is incomplete"
            )
        if self._block_resolver is None:
            raise PICExternalKVImportError(
                "Mooncake native block resolver is unavailable"
            )
        if self._transfer is None:
            raise PICExternalKVImportError(
                "Mooncake native destination writer is unavailable"
            )

        try:
            block_ids = tuple(
                int(block_id) for block_id in self._block_resolver(request)
            )
        except Exception as exc:
            raise PICExternalKVImportError(
                "Mooncake native block resolver failed"
            ) from exc
        resolve_ms = (time.perf_counter() - total_start) * 1000.0 if profile else 0.0

        window = PICWindow(
            request.canonical_start,
            request.canonical_start + request.token_count,
            request.block_size,
        )
        if (
            # Coverage, not equality. The publisher counts whole allocator pages (nine pages of
            # seventeen kernel blocks, one hundred and fifty-three) while the range divides into one
            # hundred and fifty-two at this block size; the extra block is the page tail the window
            # never reads, and requiring an exact count rejected every import.
            not window.fits(len(block_ids))
            or not block_ids
            or len(set(block_ids)) != len(block_ids)
            or any(block_id < 0 for block_id in block_ids)
        ):
            raise PICExternalKVImportError(
                "Mooncake resolver returned an invalid native block span"
            )

        key = self._resource_key(request)
        first_import = self._lease_registry.ref_count(key) == 0
        existing_block_ids = self._resource_block_ids.get(key)
        if existing_block_ids is not None:
            if existing_block_ids != block_ids:
                raise PICExternalKVImportError(
                    "Mooncake resolver changed an active native block span"
                )
            first_import = False
        release_callback = self._make_release_callback(key, request, block_ids)
        token = self._lease_registry.acquire(
            key,
            owner=f"mooncake:{request.external_key}",
            kind="external_native_kv",
            release_fn=release_callback if first_import else None,
        )
        try:
            if first_import:
                self._resource_block_ids[key] = block_ids
                transfer_start = time.perf_counter() if profile else 0.0
                self._transfer(request, block_ids, self._engine)
                transfer_ms = (
                    (time.perf_counter() - transfer_start) * 1000.0
                    if profile
                    else 0.0
                )
            else:
                transfer_ms = 0.0
        except Exception as exc:
            self._release_token(key, token)
            raise PICExternalKVImportError(
                "Mooncake native KV transfer failed"
            ) from exc

        if profile:
            logger.warning(
                "[PIC-PROFILE] native-import key=%s group=%d blocks=%d "
                "first_import=%s resolve_ms=%.3f transfer_ms=%.3f total_ms=%.3f",
                request.external_key,
                request.kv_cache_group_id,
                len(block_ids),
                first_import,
                resolve_ms,
                transfer_ms,
                (time.perf_counter() - total_start) * 1000.0,
            )

        lease = PICNativeKVLease(
            block_ids=block_ids,
            _release_fn=lambda: self._release_token(key, token),
        )
        return PICExternalKVImportResult(
            block_ids=block_ids,
            block_size=request.block_size,
            token_count=request.token_count,
            lease=lease,
        )

    def _import_native_kv_receive(
        self, request: PICExternalKVImportRequest
    ) -> PICNativeKVImportCompletion:
        """Receive native payloads without touching the destination GPU.

        The completion returned by this method is finalized by the request
        thread after transition loading and input preparation.  This keeps
        Mooncake receive overlap separate from H2D/index-copy work and avoids
        making the receiver thread a second GPU execution path.
        """
        if self._receive is None or self._materialize is None:
            raise PICExternalKVImportError(
                "Mooncake split receive/materialize callbacks are unavailable"
            )
        block_ids = self._resolve_and_validate_block_ids(request)
        key = self._resource_key(request)
        first_import = self._lease_registry.ref_count(key) == 0
        existing_block_ids = self._resource_block_ids.get(key)
        if existing_block_ids is not None:
            if existing_block_ids != block_ids:
                raise PICExternalKVImportError(
                    "Mooncake resolver changed an active native block span"
                )
            first_import = False
        release_callback = self._make_release_callback(key, request, block_ids)
        token = self._lease_registry.acquire(
            key,
            owner=f"mooncake:{request.external_key}",
            kind="external_native_kv",
            release_fn=release_callback if first_import else None,
        )
        try:
            if first_import:
                self._resource_block_ids[key] = block_ids
                received = self._receive(request, block_ids, self._engine)
            else:
                received = None
        except Exception as exc:
            self._release_token(key, token)
            raise PICExternalKVImportError(
                "Mooncake native KV receive failed"
            ) from exc

        lease = PICNativeKVLease(
            block_ids=block_ids,
            _release_fn=lambda: self._release_token(key, token),
        )
        result = PICExternalKVImportResult(
            block_ids=block_ids,
            block_size=request.block_size,
            token_count=request.token_count,
            lease=lease,
        )

        def finalize() -> None:
            if not first_import:
                return
            try:
                self._materialize(request, received, block_ids, self._engine)
            except Exception:
                result.lease.release()
                raise

        return PICNativeKVImportCompletion(result=result, finalize=finalize)

    def _resolve_and_validate_block_ids(
        self, request: PICExternalKVImportRequest
    ) -> tuple[int, ...]:
        if (
            request.block_size <= 0
            or request.token_count <= 0
            or request.canonical_start < 0
            or request.dtype is None
            or not request.layout
        ):
            raise PICExternalKVImportError(
                "Mooncake external KV metadata is incomplete"
            )
        if self._block_resolver is None:
            raise PICExternalKVImportError(
                "Mooncake native block resolver is unavailable"
            )
        try:
            block_ids = tuple(
                int(block_id) for block_id in self._block_resolver(request)
            )
        except Exception as exc:
            raise PICExternalKVImportError(
                "Mooncake resolver returned invalid native blocks"
            ) from exc
        window = PICWindow(
            request.canonical_start,
            request.canonical_start + request.token_count,
            request.block_size,
        )
        if (
            # Coverage, not equality. The publisher counts whole allocator pages (nine pages of
            # seventeen kernel blocks, one hundred and fifty-three) while the range divides into one
            # hundred and fifty-two at this block size; the extra block is the page tail the window
            # never reads, and requiring an exact count rejected every import.
            not window.fits(len(block_ids))
            or not block_ids
            or len(set(block_ids)) != len(block_ids)
            or any(block_id < 0 for block_id in block_ids)
        ):
            raise PICExternalKVImportError(
                "Mooncake resolver returned an invalid native block span"
            )
        return block_ids
