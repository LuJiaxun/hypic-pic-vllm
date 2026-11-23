# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The single PIC diagnostics switch.

PIC used to carry a dozen independent toggles; picking the right combination before a run was
harder than the measurement. There is one now:

* unset or ``0`` - production: no tracing, no profile, no verification logging;
* ``1`` - diagnostics: tracing plus the import profile.

Tuning that genuinely varies between machines stays separate: ``HYPIC_PIC_KV_ARENA_BYTES``,
``HYPIC_PIC_KV_IMPORT_WORKERS`` and ``HYPIC_MOONCAKE_KEY_PREFIX``.
"""

from __future__ import annotations

import os

DEBUG_ENV = "HYPIC_PIC_DEBUG"


def pic_debug_enabled() -> bool:
    """True when HYPIC_PIC_DEBUG asks for diagnostics."""
    raw = os.getenv(DEBUG_ENV)
    if raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "off", "no")

DEEP_PROFILE_ENV = "HYPIC_PIC_PROFILE_DEEP"


def pic_profile_deep_enabled() -> bool:
    """True when phase timing also synchronises the device.

    The import profile times phases with the wall clock. On a GPU the work is queued, so a phase can
    appear to finish before its kernel runs and the time lands on whichever phase happens to wait
    next - which is how a six-millisecond rotation was reported as two hundred and ninety. Syncing
    makes each phase honest and makes the whole request slower, so it is a separate switch: measure a
    request without it, and reach for it only when a phase needs to be attributed precisely.
    """
    raw = os.getenv(DEEP_PROFILE_ENV)
    if raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "off", "no")
