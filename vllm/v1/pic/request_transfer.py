# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-level completion boundary for PIC's asynchronous imports.

Both transfers a request needs (native KV blocks and the hybrid transition) are
started before the caller waits, and a failed operation surfaces as a
request-level fallback instead of exposing partially written KV or state.

This module owns no transport: the Store pull does the transfer itself. It used
to live beside the TransferEngine descriptor client, which PIC no longer uses.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Any

from vllm.logger import init_logger
from vllm.v1.pic.profile_level import pic_profile_enabled

logger = init_logger(__name__)


class PICTransferError(RuntimeError):
    """Raised when a PIC import operation cannot be completed safely."""


class PICAsyncOperationHandle:
    """Request-local ready/error state for a non-native transition operation."""

    def __init__(self, *, request_key: str):
        self.request_key = str(request_key)
        self.submitted_at = time.perf_counter()
        self.ready_at: float | None = None
        self._done = Event()
        self._lock = Lock()
        self._result: Any = None
        self._error: BaseException | None = None

    @property
    def ready(self) -> bool:
        """Non-blocking: finished without error."""
        return self._done.is_set() and self._error is None

    @property
    def failed(self) -> bool:
        return self._done.is_set() and self._error is not None

    def set_result(self, result: Any) -> None:
        with self._lock:
            self._result = result
            self.ready_at = time.perf_counter()
            if pic_profile_enabled():
                # The transition transfer has its own receive/upload records; this
                # line is the whole operation, so it can be compared with the KV
                # window to tell parallel from serial.
                logger.warning(
                    "[PIC-PROFILE] transfer-ready key=%s ms=%.3f",
                    self.request_key,
                    (self.ready_at - self.submitted_at) * 1000.0,
                )
            self._done.set()

    def set_error(self, error: BaseException) -> None:
        with self._lock:
            self._error = error
            self._done.set()

    def wait(self, timeout: float | None = None) -> Any:
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"timed out waiting for PIC transition: {self.request_key}"
            )
        if self._error is not None:
            raise PICTransferError(
                f"PIC transition import failed: {self.request_key}"
            ) from self._error
        return self._result


_PIC_STATE_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="pic-transition-recv",
)


class PICRequestTransfer:
    """Unified request lifecycle for native KV and transition/state imports.

    The native provider owns its lease semantics; this object owns only the
    request-level completion boundary.
    """

    def __init__(self, *, request_key: str):
        self.request_key = str(request_key)
        self.native: dict[int, Any] = {}
        self.state: dict[int, PICAsyncOperationHandle] = {}
        self._closed = False

    def add_native(self, segment_index: int, batch: Any) -> None:
        self.native[int(segment_index)] = batch

    def submit_state(self, segment_index: int, operation: Any) -> None:
        handle = PICAsyncOperationHandle(
            request_key=f"{self.request_key}:segment={segment_index}"
        )
        self.state[int(segment_index)] = handle

        def run() -> None:
            try:
                handle.set_result(operation())
            except BaseException as exc:
                handle.set_error(exc)

        if pic_profile_enabled():
            logger.warning(
                "[PIC-PROFILE] transfer-submit key=%s segment=%d",
                self.request_key,
                int(segment_index),
            )
        _PIC_STATE_EXECUTOR.submit(run)

    def wait_native(self, segment_index: int) -> Any:
        return self.native[int(segment_index)].wait()

    def wait_state(self, segment_index: int) -> Any:
        return self.state[int(segment_index)].wait()

    def poll_state(self) -> tuple[bool, bool]:
        """Return ``(all_ready, any_failed)`` for every submitted state import."""
        if not self.state:
            return True, False
        if any(handle.failed for handle in self.state.values()):
            return False, True
        if all(handle.ready for handle in self.state.values()):
            return True, False
        return False, False

    def discard(self) -> None:
        if self._closed:
            return
        self._closed = True
        for batch in self.native.values():
            discard = getattr(batch, "discard", None)
            if callable(discard):
                discard()
        # State operations may still be running.  They do not own attention
        # blocks and their result is intentionally dropped on request fallback.
        self.native.clear()
        self.state.clear()


def split_transition_result(value: Any) -> tuple[Any, Any]:
    """Split a state operation's result into ``(manifest, operator)``.

    ``MooncakeNativeKVObjectStore.load_transition_operator`` returns the pair, and
    the synchronous path unpacks it.  The asynchronous path stored the raw value, so
    callers must split it too: handing the tuple to
    ``import_transition_operator`` fails on ``operator.detached_clone``.

    A provider that returns only an operator is accepted as well.
    """
    if isinstance(value, tuple) and len(value) == 2:
        manifest, operator = value
        return manifest, operator
    return None, value
