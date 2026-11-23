# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PIC profiling is the diagnostics switch, nothing else.

A profile run synchronizes on purpose so the per-stage numbers mean something, which is why it
is not a separate knob: compare production against production, and a profile against a profile.
"""

from __future__ import annotations

from vllm.v1.pic.env import pic_debug_enabled


def pic_profile_enabled() -> bool:
    """True when the diagnostics switch asks for the profile."""
    return pic_debug_enabled()
