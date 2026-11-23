# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bookkeeping for PIC imports that outlive the step that started them.

Stage 1 of the scheduler-level async work (see the stage plan, section 7.6): the
receive still happens synchronously inside ``execute_model``, so this module only
tracks *which requests have started and finished an import*.  That is enough to
park a request in ``WAITING_FOR_REMOTE_KVS`` and promote it on a later step,
because the state machine must be proven correct before any work is actually
moved off the critical path.

Stage 2 replaces the synchronous ``start``/``complete`` pair with a real
background submit: ``start`` keeps the transfer handle and ``poll`` reports
completion without blocking.  The public API below is already shaped for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class PICAsyncImportSpec:
    """Scheduler-side description of one request's pending external import.

    The worker already owns the destination state, but a parked request is not in
    the batch, so the scheduler ships the request payload it would have sent on a
    normal step (``NewRequestData``: cache plan, segments and the allocated block
    ids).  The worker builds a stand-in state from it, which is enough because
    destination resolution reads only ``pic_cache_plan``, ``pic_segments``,
    ``pic_public_block_ids`` and ``block_ids`` — never the batch row.
    """

    request_id: str
    segment_indices: tuple[int, ...]
    request_data: Any = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("PIC async import spec requires a request id")
        if not self.segment_indices:
            raise ValueError("PIC async import spec requires at least one segment")


class PICAsyncImportManager:
    """Track external imports per request until their completion is reported.

    The manager never touches the store itself: it exists so the worker can tell
    the scheduler "this request's KV is ready" on a later step, and so a failed
    or aborted request can be cleaned up exactly once.
    """

    def __init__(self) -> None:
        # request id -> opaque worker-side transfer handle
        self._pending: dict[str, Any] = {}
        self._finished: set[str] = set()
        self._failed: dict[str, str] = {}
        self._results: dict[str, Any] = {}

    # -- request lifecycle -------------------------------------------------
    def start(self, spec: PICAsyncImportSpec, transfer: Any) -> None:
        """Record a started import; ``transfer`` is released by ``take``/``abort``."""
        if spec.request_id in self._pending:
            raise ValueError(
                f"PIC async import already in flight: {spec.request_id}"
            )
        self._pending[spec.request_id] = transfer

    def complete(self, request_id: str, payload: Any = None) -> None:
        """Mark the request ready and keep the imported state for adoption."""
        if request_id not in self._pending:
            raise KeyError(f"PIC async import was not started: {request_id}")
        self._pending.pop(request_id, None)
        self._finished.add(request_id)
        if payload is not None:
            self._results[request_id] = payload

    def payload(self, request_id: str) -> Any:
        """Return the in-flight payload recorded by ``start``."""
        return self._pending[request_id]

    def pop_result(self, request_id: str) -> Any:
        """Return and forget the imported state for a finished request."""
        return self._results.pop(request_id, None)

    def peek_result(self, request_id: str) -> Any:
        """Look at the imported state without consuming it.

        Adoption validates the state before installing it; consuming it first
        would lose the import if a duplicate call hit the request.
        """
        return self._results.get(request_id)

    def fail(self, request_id: str, reason: str) -> None:
        """Record a failed import; the scheduler turns it into a fallback."""
        self._pending.pop(request_id, None)
        self._finished.discard(request_id)
        self._failed[request_id] = str(reason) or "unknown"
        # A failed import must never be adopted: drop any state recorded before
        # the failure so a later pop_result cannot hand back half-imported KV.
        self._results.pop(request_id, None)

    def abort(self, request_id: str) -> None:
        """Drop any state for a request that was preempted or aborted."""
        transfer = self._pending.pop(request_id, None)
        discard = getattr(transfer, "discard", None)
        if callable(discard):
            discard()
        self._finished.discard(request_id)
        self._failed.pop(request_id, None)
        self._results.pop(request_id, None)

    # -- scheduler-visible state -------------------------------------------
    def take_finished(self) -> set[str]:
        """Return and clear the set of requests whose import completed."""
        finished = self._finished
        self._finished = set()
        return finished

    def take_failed(self) -> dict[str, str]:
        """Return and clear the failures reported since the previous step."""
        failed = self._failed
        self._failed = {}
        return failed

    def in_flight(self) -> frozenset[str]:
        return frozenset(self._pending)

    def is_ready(self, request_id: str) -> bool:
        return request_id in self._finished

    def __len__(self) -> int:
        return len(self._pending)
