"""A captured transition operator is addressed in segment-local coordinates.

``build_gdn_transition_operator`` receives the source bounds of the tensors it
consumes, which are prompt-absolute, and must return an operator whose bounds
are the segment-local ``(0, length)``.  Keeping the source bounds made two
coordinate conventions coexist, and a reuse window measured from the segment
start was then rejected as out of range.

These checks run on CPU: the builder is asked for the Torch reference path, so
they do not need a CUDA device.
"""

from __future__ import annotations

import pytest


def _build(token_start: int, token_end: int):
    torch = pytest.importorskip("torch")
    from vllm.v1.pic.gdn_transition import build_gdn_transition_operator

    tokens = token_end - token_start
    key = torch.ones((tokens, 1, 2))
    value = torch.ones((tokens, 1, 2))
    log_decay = torch.zeros((tokens, 1))
    beta = torch.ones((tokens, 1))
    return build_gdn_transition_operator(
        key,
        value,
        log_decay,
        beta,
        token_start=token_start,
        token_end=token_end,
        state_dtype=torch.float32,
        use_fla_reference=False,
        backend=None,
    )


def _affine_operator(token_start: int, token_end: int):
    torch = pytest.importorskip("torch")
    from vllm.v1.pic.state import PICAffineTransition, PICTransitionOperator

    return PICTransitionOperator(
        transitions=(
            PICAffineTransition(
                decay=torch.tensor([0.5]),
                zero_state=torch.tensor([1.0]),
            ),
        ),
        token_start=token_start,
        token_end=token_end,
    )


@pytest.mark.parametrize(("source_start", "length"), [(0, 8), (128, 8), (4352, 64)])
def test_operator_bounds_are_segment_local(source_start: int, length: int) -> None:
    operator = _build(source_start, source_start + length)
    assert operator.token_start == 0
    assert operator.token_end == length


def test_full_range_apply_needs_no_slice() -> None:
    operator = _build(4352, 4360)
    length = operator.token_end - operator.token_start
    relative_start = 0 - operator.token_start
    relative_end = length - operator.token_start
    assert (relative_start, relative_end) == (0, length)


def test_store_span_keeps_its_offset() -> None:
    """A loaded operator covers the published span, so its start is the seam."""
    operator = _affine_operator(8, 100)
    local_start, local_end = 8, 100
    relative_start = local_start - operator.token_start
    relative_end = local_end - operator.token_start
    length = operator.token_end - operator.token_start
    assert (relative_start, relative_end) == (0, length)


def test_range_lookup_accepts_the_span_window_end_to_end() -> None:
    """A loaded operator serves its published span as one whole range.

    This drives the real lookup instead of repeating its arithmetic: an operator
    that arrives with a span offset must be returned untouched when the plan asks
    for that exact span, which is what keeps a compact transition usable.
    """
    pytest.importorskip("torch")

    from vllm.v1.pic.live_capture import PICLiveCaptureManager
    from vllm.v1.pic.pool import PICPhysicalPool

    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    manager = PICLiveCaptureManager(pool)
    handle = manager.import_transition_operator(
        request_id="req",
        external_key="segment/x/hybrid-state",
        operator=_affine_operator(8, 100),
    )

    served = manager._operator_for_range(handle, 8, 100)  # type: ignore[attr-defined]
    assert served.token_start == 8
    assert served.token_end == 100
    assert len(served.transitions) == 1


def test_range_lookup_rejects_a_window_before_the_span() -> None:
    """A request that asks for an earlier start must fail in a controlled way."""
    pytest.importorskip("torch")

    from vllm.v1.pic.live_capture import PICLiveCaptureManager
    from vllm.v1.pic.pool import PICPhysicalPool
    from vllm.v1.pic.worker_plan import PICWorkerUnsupported

    pool = PICPhysicalPool(4096, device="cpu", alignment_bytes=1)
    manager = PICLiveCaptureManager(pool)
    handle = manager.import_transition_operator(
        request_id="req",
        external_key="segment/y/hybrid-state",
        operator=_affine_operator(8, 100),
    )

    with pytest.raises(PICWorkerUnsupported):
        manager._operator_for_range(handle, 0, 100)  # type: ignore[attr-defined]
