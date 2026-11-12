# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the single park path for external PIC imports.

The scheduler used to keep the last runnable request on an inline import path.
That is gone: every external import is parked, because a parked import advances
on empty steps (the import is announced on every scheduler output, the engine
calls ``execute_model`` even with zero scheduled tokens, and the worker polls
before it reads the token count).  These tests pin that decision down so a later
"optimisation" cannot quietly reinstate the timing-dependent mix.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.pic.async_import import PICAsyncImportSpec
from vllm.v1.request import RequestStatus


def _external_entry(segment_index: int):
    reference = SimpleNamespace(is_external=True, external_key="mooncake/key")
    return (segment_index, SimpleNamespace(native_kv_refs=[reference]))


class _SchedulerStub:
    """Mirror of the ``Scheduler`` state that the park path touches.

    Every attribute here exists in ``Scheduler.__init__``. This list has now been
    extended twice by a new attribute landing in the implementation, so keep the
    mirror in one place: a fresh attribute means one edit here, not a hunt through
    the test.
    """

    def __init__(self, *, debug: bool = False) -> None:
        self._pic_pending_imports: dict[str, PICAsyncImportSpec] = {}
        # A request whose import already ran is never parked again, which is what
        # stops a park/resume loop.
        self._pic_import_completed: set[str] = set()
        # Park start times, used to report how long the request waited.
        self._pic_park_started: dict[str, float] = {}
        self.vllm_config = SimpleNamespace(
            pic_config=SimpleNamespace(debug=debug)
        )
        self.debug_lines: list[str] = []

    def _pic_debug(self, message: str, *args: object) -> None:
        self.debug_lines.append(message % args if args else message)

    @staticmethod
    def _pic_trace_enabled() -> bool:
        return False


def _request(request_id: str = "req-1", *, external: bool = True):
    plan = SimpleNamespace(
        matches=[_external_entry(0)] if external else [],
        external_matches=[],
    )
    return SimpleNamespace(
        request_id=request_id,
        status=RequestStatus.WAITING,
        pic_cache_plan=plan,
    )


def _park(stub: _SchedulerStub):
    bound = Scheduler._park_pic_external_import.__get__(stub, Scheduler)
    return bound


def test_park_happens_even_when_nothing_else_is_runnable(monkeypatch) -> None:
    """A lone request still parks: an empty step advances its import."""
    monkeypatch.setattr(
        scheduler_module,
        "NewRequestData",
        SimpleNamespace(from_request=lambda request, block_ids: SimpleNamespace()),
    )
    stub = _SchedulerStub()
    request = _request()

    assert _park(stub)(request, None) is True

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert "req-1" in stub._pic_pending_imports
    spec = stub._pic_pending_imports["req-1"]
    assert spec.segment_indices == (0,)


def test_request_without_external_references_is_not_parked(monkeypatch) -> None:
    """Only requests the worker must import are parked."""
    monkeypatch.setattr(
        scheduler_module,
        "NewRequestData",
        SimpleNamespace(from_request=lambda request, block_ids: SimpleNamespace()),
    )
    stub = _SchedulerStub()
    request = _request(external=False)

    assert _park(stub)(request, None) is False

    assert request.status == RequestStatus.WAITING
    assert stub._pic_pending_imports == {}


def test_park_uses_the_plan_even_when_the_entry_is_a_local_match(
    monkeypatch,
) -> None:
    """A resolved external entry lands in ``matches``, not ``external_matches``.

    Parking reads both lists, so a lookup that resolved the object through
    ``external_lookup`` still parks instead of importing inline.
    """
    monkeypatch.setattr(
        scheduler_module,
        "NewRequestData",
        SimpleNamespace(from_request=lambda request, block_ids: SimpleNamespace()),
    )
    stub = _SchedulerStub()
    request = _request()
    request.pic_cache_plan.external_matches = []

    assert _park(stub)(request, None) is True
    assert stub._pic_pending_imports["req-1"].segment_indices == (0,)


def test_stub_covers_every_pic_attribute_the_park_path_uses() -> None:
    """A new attribute in the park path must break this test, not the suite.

    This mirror has fallen behind twice: a new ``self._pic_...`` field landed in the
    implementation and only the stub was missing it, which surfaced as an
    ``AttributeError`` in an unrelated-looking test run. Parsing the method keeps the
    two in step.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(Scheduler._park_pic_external_import))
    tree = ast.parse(source)
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr.startswith("_pic_")
    }
    used.update(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr.startswith("_pic_")
        }
    )
    assert used, "the park path is expected to touch PIC state"

    stub = _SchedulerStub()
    missing = sorted(name for name in used if not hasattr(stub, name))
    assert not missing, (
        f"_SchedulerStub is missing {missing}; add them so the mirror matches "
        f"Scheduler.__init__"
    )
