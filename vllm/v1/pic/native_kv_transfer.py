# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Full-Attention-style lifecycle primitives for PIC native KV imports.

The synchronous hot path remains available.  The P5 path can submit a
receive-only completion whose destination materialization is finalized by the
request thread at an explicit ready barrier.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Callable

from vllm.logger import init_logger
from vllm.v1.pic.profile_level import pic_profile_enabled

from vllm.v1.pic.external_kv import (
    PICExternalKVImportError,
    PICExternalKVImportResult,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class PICNativeKVImportCompletion:
    """A received import whose destination materialization is deferred."""

    result: PICExternalKVImportResult
    finalize: Callable[[], None]


class PICNativeKVImportHandle:
    """A request-owned completion handle for one native KV import."""

    def __init__(self, *, request_key: str) -> None:
        self.request_key = str(request_key)
        # Queue wait is invisible in every per-phase record: the transfer has not
        # started yet, so only the gap from submission to worker entry shows it.
        self.created_at = time.perf_counter()
        self._done = Event()
        self._ready = Event()
        self._lock = Lock()
        self._result: PICExternalKVImportResult | None = None
        self._completion: PICNativeKVImportCompletion | None = None
        self._error: BaseException | None = None
        self._cuda_ready_event: Any | None = None
        self._cuda_device: int | None = None

    @property
    def ready(self) -> bool:
        """Whether the destination KV blocks are ready for attention."""
        return self._ready.is_set() and self._error is None

    @property
    def done(self) -> bool:
        """Whether the transfer finished without error.

        The split receive/materialize path completes with a
        ``PICNativeKVImportCompletion`` and never sets ``_ready``: the data is in
        host staging and only ``wait()`` materializes it into the blocks.  Pollers
        must therefore treat "done without error" as the pollable state, not
        ``ready``.
        """
        return self._done.is_set() and self._error is None

    @property
    def failed(self) -> bool:
        """Whether the import completed with an error."""
        return self._done.is_set() and self._error is not None

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def _set_result(
        self,
        result: PICExternalKVImportResult | PICNativeKVImportCompletion,
        *,
        cuda_ready_event: Any | None = None,
        cuda_device: int | None = None,
    ) -> None:
        with self._lock:
            if self._done.is_set():
                raise RuntimeError("PIC native KV import handle completed twice")
            if isinstance(result, PICNativeKVImportCompletion):
                self._completion = result
            else:
                self._result = result
                self._ready.set()
            self._cuda_ready_event = cuda_ready_event
            self._cuda_device = cuda_device
            self._done.set()

    def _set_error(self, error: BaseException) -> None:
        with self._lock:
            if self._done.is_set():
                raise RuntimeError("PIC native KV import handle completed twice")
            self._error = error
            self._done.set()

    def wait(self, timeout: float | None = None) -> PICExternalKVImportResult:
        """Wait for a ready result and convert worker failures to import errors."""
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"timed out waiting for PIC native KV import: {self.request_key}"
            )
        with self._lock:
            error = self._error
            result = self._result
            completion = self._completion
            cuda_ready_event = self._cuda_ready_event
            cuda_device = self._cuda_device
        if error is not None:
            if isinstance(error, PICExternalKVImportError):
                raise error
            raise PICExternalKVImportError(
                f"PIC native KV import failed: {self.request_key}"
            ) from error
        if completion is not None:
            try:
                completion.finalize()
            except BaseException as exc:
                completion.result.lease.release()
                with self._lock:
                    self._error = exc
                    self._completion = None
                if isinstance(exc, PICExternalKVImportError):
                    raise exc
                raise PICExternalKVImportError(
                    f"PIC native KV materialization failed: {self.request_key}"
                ) from exc
            with self._lock:
                self._result = completion.result
                self._completion = None
            self._ready.set()
            result = completion.result
        if result is None:
            raise PICExternalKVImportError(
                f"PIC native KV import completed without a result: {self.request_key}"
            )
        if cuda_ready_event is not None and cuda_device is not None:
            # The receiver may enqueue H2D/index-copy work on its thread-local
            # CUDA stream.  Make the caller's current stream wait before PIC
            # attention can read the destination blocks.
            import torch

            torch.cuda.current_stream(device=cuda_device).wait_event(
                cuda_ready_event
            )
        return result


class PICNativeKVImportWorker:
    """Dedicated receiver worker used by the PIC native-KV provider.

    Mooncake's object-store ``get`` path is synchronous at the Python boundary.
    Running the receive callback here gives PIC the same lifecycle boundary as
    Full Attention's receiver thread without moving H2D/index-copy work onto a
    competing CUDA stream.
    """

    def __init__(self) -> None:
        # KV receives are serialized on purpose. Sharing the link between two imports
        # makes both finish at the same late moment, so neither request can start
        # computing and the engine idles through the whole transfer. Sending them one
        # after another lets the first request compute while the second is still
        # arriving - measured on a two-request run: wall 2545.7 -> 2413.3 ms, p50
        # 265.3 -> 243.2 ms, and the per-request curve stopped oscillating. This is
        # separate from the state transfers, which ride their own stream and lose
        # nothing from running in parallel.
        try:
            workers = max(1, int(os.getenv("HYPIC_PIC_KV_IMPORT_WORKERS", "1")))
        except ValueError:
            workers = 1
        self.workers = workers
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="pic-native-kv-recv",
        )

    def submit(
        self,
        handle: PICNativeKVImportHandle,
        operation: Callable[
            [], PICExternalKVImportResult | PICNativeKVImportCompletion
        ],
    ) -> None:
        def run() -> None:
            try:
                if pic_profile_enabled():
                    # One worker thread serves every KV import, so contention here
                    # would look like slow transfer time everywhere else.
                    logger.warning(
                        "[PIC-PROFILE] import-start key=%s queue_ms=%.3f workers=%d",
                        handle.request_key,
                        (time.perf_counter() - handle.created_at) * 1000.0,
                        int(getattr(self._executor, "_max_workers", 0)),
                    )
                result = operation()
                if isinstance(result, PICNativeKVImportCompletion):
                    # The receive-only P5 stage has no CUDA work to publish.
                    # Its finalize callback runs on the request thread at the
                    # explicit barrier, on the active attention stream.
                    handle._set_result(result)
                    return
                cuda_ready_event = None
                cuda_device = None
                try:
                    import torch

                    if torch.cuda.is_available():
                        cuda_device = torch.cuda.current_device()
                        cuda_ready_event = torch.cuda.Event()
                        cuda_ready_event.record(
                            torch.cuda.current_stream(device=cuda_device)
                        )
                except Exception:
                    # Do not expose a GPU destination without a completion
                    # event.  The caller will request-level fallback instead.
                    result.lease.release()
                    handle._set_error(
                        PICExternalKVImportError(
                            "PIC native KV import could not publish CUDA readiness"
                        )
                    )
                    return
                handle._set_result(
                    result,
                    cuda_ready_event=cuda_ready_event,
                    cuda_device=cuda_device,
                )
            except BaseException as exc:
                handle._set_error(exc)

        self._executor.submit(run)

    @staticmethod
    def complete_inline(
        handle: PICNativeKVImportHandle,
        operation: Callable[
            [], PICExternalKVImportResult | PICNativeKVImportCompletion
        ],
    ) -> None:
        """Complete a handle on the caller thread.

        The installed Mooncake ``get`` path is synchronous.  Until P5 has a
        real overlap consumer, routing this path through a worker only adds a
        thread hand-off and GIL contention.  Keeping the completion contract
        identical lets the provider expose the background entry point without
        regressing the existing synchronous hot path.
        """
        try:
            handle._set_result(operation())
        except BaseException as exc:
            handle._set_error(exc)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
