# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the transfer handle's timestamps.

The two imports a request needs (native KV and the hybrid transition) run on separate
threads. To tell a parallel transfer from a serial one, the handle records when it was
submitted and when it became ready, so the transition window can be compared with the KV
window instead of being inferred from the sum.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("torch")

from vllm.v1.pic.request_transfer import PICAsyncOperationHandle


def test_the_handle_records_submission_and_completion() -> None:
    handle = PICAsyncOperationHandle(request_key="req-1")
    submitted = handle.submitted_at

    assert handle.ready_at is None
    time.sleep(0.002)
    handle.set_result("ok")

    assert handle.ready is True
    assert handle.ready_at is not None
    assert handle.ready_at > submitted


def test_a_failed_handle_has_no_ready_time() -> None:
    handle = PICAsyncOperationHandle(request_key="req-2")
    handle.set_error(RuntimeError("boom"))

    assert handle.failed is True
    assert handle.ready_at is None


def test_two_handles_are_timed_independently() -> None:
    first = PICAsyncOperationHandle(request_key="req-a")
    time.sleep(0.001)
    second = PICAsyncOperationHandle(request_key="req-b")

    assert second.submitted_at > first.submitted_at
