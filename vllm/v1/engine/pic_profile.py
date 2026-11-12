# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-overhead request timeline records for HYPIC/PIC investigations.

This module intentionally contains CPU-only helpers.  GPU synchronization is
owned by the worker-side profiler; API and EngineCore timestamps are only
comparable within their own process.  Correlate records with ``request_id``.
"""

import time
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.pic.profile_level import pic_profile_enabled

logger = init_logger(__name__)


def enabled() -> bool:
    return pic_profile_enabled()


def start(is_enabled: bool | None = None) -> int:
    if is_enabled is None:
        is_enabled = enabled()
    return time.perf_counter_ns() if is_enabled else 0


def mode_for_request(request: Any) -> str:
    return "pic" if bool(getattr(request, "pic_enabled", False)) else "full"


def mode_for_requests(requests: Iterable[Any]) -> str:
    modes = {mode_for_request(request) for request in requests}
    if not modes:
        return "unknown"
    return next(iter(modes)) if len(modes) == 1 else "mixed"


def emit(
    request_id: str,
    mode: str,
    stage: str,
    start_ns: int,
    end_ns: int | None = None,
) -> None:
    if start_ns == 0:
        return
    if end_ns is None:
        end_ns = time.perf_counter_ns()
    logger.warning(
        "[PIC-PROFILE] timeline request=%s mode=%s stage=%s "
        "start_ns=%d end_ns=%d duration_ms=%.3f",
        request_id or "-",
        mode,
        stage,
        start_ns,
        end_ns,
        (end_ns - start_ns) / 1_000_000.0,
    )

