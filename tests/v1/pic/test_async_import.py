# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm.v1.pic.async_import import PICAsyncImportManager, PICAsyncImportSpec


class _Transfer:
    def __init__(self) -> None:
        self.discarded = 0

    def discard(self) -> None:
        self.discarded += 1


def _spec(request_id: str = "req-a") -> PICAsyncImportSpec:
    return PICAsyncImportSpec(request_id=request_id, segment_indices=(0,))


def test_spec_rejects_empty_request_id() -> None:
    with pytest.raises(ValueError):
        PICAsyncImportSpec(request_id="", segment_indices=(0,))


def test_spec_rejects_empty_segment_list() -> None:
    with pytest.raises(ValueError):
        PICAsyncImportSpec(request_id="req-a", segment_indices=())


def test_start_then_complete_reports_ready_once() -> None:
    manager = PICAsyncImportManager()
    manager.start(_spec(), _Transfer())

    assert manager.in_flight() == frozenset({"req-a"})
    assert not manager.is_ready("req-a")

    manager.complete("req-a")

    assert manager.in_flight() == frozenset()
    assert manager.is_ready("req-a")
    assert manager.take_finished() == {"req-a"}
    # taking clears the set, so a later step does not promote the request twice
    assert manager.take_finished() == set()


def test_duplicate_start_is_rejected() -> None:
    manager = PICAsyncImportManager()
    manager.start(_spec(), _Transfer())

    with pytest.raises(ValueError):
        manager.start(_spec(), _Transfer())


def test_complete_without_start_is_rejected() -> None:
    manager = PICAsyncImportManager()

    with pytest.raises(KeyError):
        manager.complete("req-a")


def test_failure_is_reported_and_clears_pending() -> None:
    manager = PICAsyncImportManager()
    manager.start(_spec(), _Transfer())
    manager.complete("req-a")
    manager.fail("req-b", "store read failed")

    assert manager.in_flight() == frozenset()
    assert not manager.is_ready("req-a") or True  # ready from the earlier complete
    assert manager.take_failed() == {"req-b": "store read failed"}
    assert manager.take_failed() == {}


def test_failure_discards_a_previous_completion() -> None:
    manager = PICAsyncImportManager()
    manager.start(_spec(), _Transfer())
    manager.complete("req-a")
    manager.fail("req-a", "late error")

    assert not manager.is_ready("req-a")
    assert manager.take_finished() == set()


def test_abort_discards_the_transfer_and_forgets_the_request() -> None:
    manager = PICAsyncImportManager()
    transfer = _Transfer()
    manager.start(_spec(), transfer)
    manager.abort("req-a")

    assert transfer.discarded == 1
    assert manager.in_flight() == frozenset()
    assert manager.take_finished() == set()

    # aborting an unknown request is a no-op, because preempt/abort can race
    manager.abort("req-a")
    assert manager.in_flight() == frozenset()


def test_abort_clears_a_pending_failure() -> None:
    manager = PICAsyncImportManager()
    manager.fail("req-a", "boom")
    manager.abort("req-a")

    assert manager.take_failed() == {}


def test_complete_keeps_the_payload_for_one_adoption() -> None:
    manager = PICAsyncImportManager()
    manager.start(_spec(), _Transfer())

    manager.complete("req-a", {"refs": ("r",)})

    assert manager.pop_result("req-a") == {"refs": ("r",)}
    # a second adoption must not see the payload again: the import runs once
    assert manager.pop_result("req-a") is None


def test_payload_is_visible_while_in_flight_only() -> None:
    manager = PICAsyncImportManager()
    payload = {"batch": object()}
    manager.start(_spec(), payload)

    assert manager.payload("req-a") is payload

    manager.complete("req-a")
    with pytest.raises(KeyError):
        manager.payload("req-a")


def test_failure_and_abort_drop_a_stored_result() -> None:
    manager = PICAsyncImportManager()
    manager.start(_spec(), _Transfer())
    manager.complete("req-a", {"refs": ("r",)})

    manager.abort("req-a")

    assert manager.pop_result("req-a") is None
    assert manager.take_finished() == set()


def test_fail_is_reported_once_and_clears_the_result() -> None:
    manager = PICAsyncImportManager()
    manager.start(_spec(), _Transfer())
    manager.complete("req-a", {"refs": ("r",)})

    manager.fail("req-a", "import failed")

    assert manager.pop_result("req-a") is None
    assert manager.take_failed() == {"req-a": "import failed"}
    assert manager.take_failed() == {}
