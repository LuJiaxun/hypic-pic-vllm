# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard what makes a request claim a PIC step of its own.

The single-request runner rebuilds the batch around one row, so a ready PIC prefill that
shares its step with anything else loses the fast path and falls back to a full prefill.
Exclusivity used to key on ``pic_enabled``, which is set on every request once PIC is on,
so it never separated anything: concurrent ready prefills landed in one step and none of
them could reuse.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.v1.core.sched.scheduler import Scheduler


def _request(*, enabled=True, computed=0, reused=(0,), has_plan=True):
    plan = SimpleNamespace(reused_segments=reused) if has_plan else None
    return SimpleNamespace(
        pic_enabled=enabled,
        num_computed_tokens=computed,
        pic_cache_plan=plan,
    )


def _claims(request) -> bool:
    return Scheduler._pic_claims_exclusive_step(SimpleNamespace(), request)


def test_a_fresh_hit_claims_the_step() -> None:
    assert _claims(_request())


def test_a_request_without_a_plan_does_not_claim_it() -> None:
    assert not _claims(_request(has_plan=False))


def test_a_miss_does_not_claim_it() -> None:
    assert not _claims(_request(reused=()))


def test_a_continuation_does_not_claim_it() -> None:
    assert not _claims(_request(computed=3082))


def test_a_request_with_pic_off_does_not_claim_it() -> None:
    assert not _claims(_request(enabled=False))


def _awaits_park(*, enabled=True, external=(0,), has_plan=True):
    plan = SimpleNamespace(external_matches=external) if has_plan else None
    return Scheduler._pic_awaits_park(
        SimpleNamespace(pic_async_imports_enabled=enabled), SimpleNamespace(pic_cache_plan=plan)
    )


def test_a_request_with_an_external_match_passes_before_parking() -> None:
    assert _awaits_park()


def test_a_request_without_external_matches_is_still_deferred() -> None:
    assert not _awaits_park(external=())


def test_async_import_off_keeps_the_deferral() -> None:
    assert not _awaits_park(enabled=False)
