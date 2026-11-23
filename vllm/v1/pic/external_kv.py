# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional external native-KV import bridge for PIC.

The bridge deliberately contains no Mooncake import.  A connector can register
an adapter that imports an external object into vLLM-owned native KV blocks
and returns a :class:`PICNativeKVLease`.  Until such an adapter is registered,
external references fail closed and the affected request falls back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Protocol

from vllm.v1.pic.native_kv import (
    PICWindow,
    PICKVPositionMode,
    PICNativeKVKind,
    PICNativeKVLease,
    PICNativeKVReference,
)


class PICExternalKVImportError(RuntimeError):
    """Raised when an external KV object cannot be safely imported."""


@dataclass(frozen=True)
class PICExternalKVImportRequest:
    """Validated metadata passed to an external KV provider."""

    provider: str
    external_key: str
    kv_cache_group_id: int
    block_size: int
    token_count: int
    canonical_start: int
    dtype: str | None
    layout: tuple[int, ...]
    # Optional metadata fetched during scheduler lookup. Providers may use it
    # to avoid repeating a remote manifest lookup during the worker import.
    object_manifest: Mapping[str, object] | None = None


@dataclass(frozen=True)
class PICExternalKVImportResult:
    """Native blocks imported by a provider and pinned by a lease."""

    block_ids: tuple[int, ...]
    block_size: int
    token_count: int
    lease: PICNativeKVLease

    def __post_init__(self) -> None:
        block_ids = tuple(int(block_id) for block_id in self.block_ids)
        object.__setattr__(self, "block_ids", block_ids)
        if not block_ids or len(set(block_ids)) != len(block_ids):
            raise ValueError("external KV import must return unique native blocks")
        if self.block_size <= 0 or self.token_count <= 0:
            raise ValueError("external KV import has invalid layout")
        if self.lease.block_ids != block_ids:
            raise ValueError("external KV lease does not match imported blocks")


class PICExternalKVImportFuture(Protocol):
    """Completion interface exposed by an asynchronous external provider."""

    def wait(
        self, timeout: float | None = None
    ) -> PICExternalKVImportResult: ...


class _PICExternalKVImportTask:
    """One request-local external import started before its wait point."""

    def __init__(
        self,
        reference: PICNativeKVReference,
        wait_fn: Callable[[], PICExternalKVImportResult],
        resolve_fn: Callable[
            [PICNativeKVReference, PICExternalKVImportResult], PICNativeKVReference
        ],
        readiness: Callable[[], bool] | None = None,
        failure: Callable[[], bool] | None = None,
        error_fn: Callable[[], BaseException | None] | None = None,
    ) -> None:
        self._reference = reference
        self._wait_fn = wait_fn
        self._resolve_fn = resolve_fn
        self._readiness = readiness
        self._failure = failure
        self._error_fn = error_fn
        self._resolved: PICNativeKVReference | None = None

    def ready(self) -> bool:
        """Non-blocking: the transfer finished, so ``wait`` will not block long.

        A provider without an asynchronous entry point is polled as ready: its work
        happens inside ``wait_fn``, so the caller's ``wait`` performs it inline.
        Treating that case as "never ready" would park the request forever.
        """
        if self._readiness is None:
            return True
        return bool(self._readiness())

    def has_failed(self) -> bool:
        """Non-blocking: the provider reported an error."""
        if self._failure is None:
            return False
        return bool(self._failure())

    def error(self) -> BaseException | None:
        """The provider error behind ``has_failed``, for diagnostics.

        A polled import that fails is otherwise indistinguishable from one that
        timed out, and the traceback would only surface much later inside a
        fallback path.
        """
        if self._error_fn is None:
            return None
        try:
            return self._error_fn()
        except Exception:  # noqa: BLE001 - diagnostics must never raise
            return None

    def wait(self) -> PICNativeKVReference:
        if self._resolved is None:
            self._resolved = self._resolve_fn(self._reference, self._wait_fn())
        return self._resolved

    def discard(self) -> None:
        try:
            self.wait().release()
        except Exception:
            # A failed transfer has no worker-local lease to release.  The
            # provider owns cleanup for its own failed operation.
            pass


class PICExternalKVImportBatch:
    """Request-local group of imports with one explicit wait boundary."""

    def __init__(
        self,
        references: Sequence[PICNativeKVReference],
        tasks: Sequence[tuple[int, _PICExternalKVImportTask]],
    ) -> None:
        self._resolved = list(references)
        self._tasks = tuple(tasks)
        self._completed = False

    def poll(self) -> tuple[bool, bool]:
        """Return ``(all_ready, any_failed)`` without blocking.

        Used by the parked-request path: the import is started before the step's
        forward and only inspected afterwards, so a step never waits on it.
        """
        if self._completed:
            return True, False
        if any(task.has_failed() for _, task in self._tasks):
            return False, True
        if all(task.ready() for _, task in self._tasks):
            return True, False
        return False, False

    def error(self) -> BaseException | None:
        """The first provider error in this batch, if any.

        ``poll`` only reports that something failed; the exception itself lives
        here so the failure can be logged where it is observed.
        """
        for _, task in self._tasks:
            error = task.error()
            if error is not None:
                return error
        return None

    def wait(self) -> tuple[PICNativeKVReference, ...]:
        if self._completed:
            return tuple(self._resolved)
        try:
            for index, task in self._tasks:
                self._resolved[index] = task.wait()
        except Exception:
            self.discard()
            raise
        self._completed = True
        return tuple(self._resolved)

    def discard(self) -> None:
        if self._completed:
            return
        for _, task in self._tasks:
            task.discard()
        self._completed = True


class PICExternalKVProvider(Protocol):
    """Adapter contract implemented by Mooncake or another KV provider."""

    name: str

    def import_native_kv(
        self, request: PICExternalKVImportRequest
    ) -> PICExternalKVImportResult:
        """Import one external object into native vLLM KV blocks."""

    def begin_import_native_kv(
        self, request: PICExternalKVImportRequest
    ) -> PICExternalKVImportFuture:
        """Start an import whose completion is waited at an explicit barrier."""


class PICExternalKVRegistry:
    """Worker-local registry for optional external native-KV providers."""

    def __init__(self) -> None:
        self._providers: dict[str, PICExternalKVProvider] = {}

    def register(self, provider: PICExternalKVProvider) -> None:
        name = str(provider.name).strip()
        if not name:
            raise ValueError("external KV provider name cannot be empty")
        if name in self._providers:
            raise ValueError(f"external KV provider already registered: {name}")
        self._providers[name] = provider

    def unregister(self, name: str) -> bool:
        return self._providers.pop(str(name), None) is not None

    def provider_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def _provider_request(
        self, reference: PICNativeKVReference
    ) -> tuple[PICExternalKVProvider, PICExternalKVImportRequest]:
        if not reference.is_external:
            raise PICExternalKVImportError(
                "cannot build an external request for a local PIC reference"
            )
        provider_name = reference.external_provider
        external_key = reference.external_key
        if provider_name is None or external_key is None:
            raise PICExternalKVImportError(
                "external PIC reference is missing provider or object key"
            )
        provider = self._providers.get(provider_name)
        if provider is None:
            raise PICExternalKVImportError(
                f"external PIC provider is unavailable: {provider_name}"
            )
        return provider, PICExternalKVImportRequest(
            provider=provider_name,
            external_key=external_key,
            kv_cache_group_id=reference.kv_cache_group_id,
            block_size=reference.block_size,
            token_count=reference.token_count,
            canonical_start=reference.canonical_start,
            dtype=reference.dtype,
            layout=reference.layout,
            object_manifest=reference.external_manifest,
        )

    @staticmethod
    def _resolve_import_result(
        reference: PICNativeKVReference,
        result: PICExternalKVImportResult,
    ) -> PICNativeKVReference:
        window = PICWindow(
            reference.canonical_start,
            reference.canonical_start + reference.token_count,
            reference.block_size,
        )
        if result.block_size != reference.block_size:
            result.lease.release()
            raise PICExternalKVImportError(
                "external PIC provider returned a different block size"
            )
        if result.token_count != reference.token_count:
            result.lease.release()
            raise PICExternalKVImportError(
                "external PIC provider returned a different token count"
            )
        # Coverage, not equality: a provider may hand over the whole page its span ends inside,
        # which is one block more than the span divides into. The extra block is never read.
        if not window.fits(len(result.block_ids)):
            result.lease.release()
            raise PICExternalKVImportError(
                "external PIC provider returned an incomplete block span"
            )
        return PICNativeKVReference(
            kv_cache_group_id=reference.kv_cache_group_id,
            block_ids=result.block_ids,
            block_size=reference.block_size,
            token_count=reference.token_count,
            # The plan decides whether this reference is used as an alias or through the
            # copied path: that depends on where the segment sits in the request, not on the
            # provider's manifest. Keeping SHARED here preserves the zero-copy case; an
            # interior segment is handled by the plan's copied path instead.
            kind=PICNativeKVKind.SHARED,
            position_mode=PICKVPositionMode.CANONICAL_LOCAL,
            canonical_start=reference.canonical_start,
            source_token_start=reference.source_token_start,
            source_block_start=reference.source_block_start,
            dtype=reference.dtype,
            layout=reference.layout,
            lease=result.lease,
        )

    def begin_import_references(
        self, references: Sequence[PICNativeKVReference]
    ) -> PICExternalKVImportBatch:
        """Start external imports before the caller's explicit wait point."""
        references = tuple(references)
        tasks: list[tuple[int, _PICExternalKVImportTask]] = []
        try:
            for index, reference in enumerate(references):
                if not reference.is_external:
                    continue
                provider, request = self._provider_request(reference)
                begin = getattr(provider, "begin_import_native_kv", None)
                if begin is None:
                    wait_fn = lambda provider=provider, request=request: (
                        provider.import_native_kv(request)
                    )
                    readiness = None
                    failure = None
                    error_fn = None
                else:
                    future = begin(request)
                    wait_fn = future.wait
                    # ``done`` covers the split receive/materialize path, which
                    # completes without ever setting ``ready``; ``ready`` is kept
                    # as a fallback for other providers.
                    readiness = lambda future=future: getattr(
                        future, "done", future.ready
                    )
                    failure = lambda future=future: future.failed
                    error_fn = lambda future=future: getattr(future, "error", None)
                tasks.append(
                    (
                        index,
                        _PICExternalKVImportTask(
                            reference,
                            wait_fn,
                            self._resolve_import_result,
                            readiness=readiness,
                            failure=failure,
                            error_fn=error_fn,
                        ),
                    )
                )
        except Exception:
            PICExternalKVImportBatch(references, tasks).discard()
            raise
        return PICExternalKVImportBatch(references, tasks)

    def import_reference(
        self,
        reference: PICNativeKVReference,
    ) -> PICNativeKVReference:
        """Resolve one external reference into a leased native reference."""
        if not reference.is_external:
            return reference
        provider, request = self._provider_request(reference)
        try:
            result = provider.import_native_kv(request)
        except Exception as exc:
            raise PICExternalKVImportError(
                f"external PIC provider import failed: {request.provider}"
            ) from exc
        return self._resolve_import_result(reference, result)

    def import_references(
        self, references: Sequence[PICNativeKVReference]
    ) -> tuple[PICNativeKVReference, ...]:
        """Resolve a group-aligned reference tuple atomically."""
        imported: list[PICNativeKVReference] = []
        try:
            for reference in references:
                imported.append(self.import_reference(reference))
        except Exception:
            for reference in imported:
                reference.release()
            raise
        return tuple(imported)
