# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the single PIC diagnostics switch.

PIC had a dozen independent toggles. There is one now: off means production, on means tracing
plus the profile. The tuning that genuinely varies between machines stays separate, so all that
is guarded here is the switch itself and that the retired names stay retired.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import re

import pytest

MODULE_PATH = pathlib.Path("vllm/v1/pic/env.py")

SURVIVORS = {
    "HYPIC_PIC_DEBUG",
    "HYPIC_PIC_KV_ARENA_BYTES",
    "HYPIC_PIC_KV_IMPORT_WORKERS",
    "HYPIC_PIC_NATIVE_ARENA_CACHE_BYTES",
    "HYPIC_MOONCAKE_KEY_PREFIX",
}


def _load():
    spec = importlib.util.spec_from_file_location("pic_env_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("HYPIC_PIC_DEBUG", raising=False)
    return monkeypatch


def test_the_switch_is_off_when_unset(clean_env) -> None:
    module = _load()

    assert module.pic_debug_enabled() is False


def test_zero_is_off(clean_env) -> None:
    module = _load()
    clean_env.setenv("HYPIC_PIC_DEBUG", "0")

    assert module.pic_debug_enabled() is False


def test_one_is_on(clean_env) -> None:
    module = _load()
    clean_env.setenv("HYPIC_PIC_DEBUG", "1")

    assert module.pic_debug_enabled() is True


def test_the_profile_follows_the_switch(monkeypatch) -> None:
    from vllm.v1.pic.profile_level import pic_profile_enabled

    monkeypatch.delenv("HYPIC_PIC_DEBUG", raising=False)
    assert pic_profile_enabled() is False

    monkeypatch.setenv("HYPIC_PIC_DEBUG", "1")
    assert pic_profile_enabled() is True


def test_only_a_few_names_survive() -> None:
    """Retired toggles must be gone from the runtime, not merely ignored."""
    names = set()
    for path in itertools.chain.from_iterable(
        root.rglob("*.py") for root in (pathlib.Path("vllm"),)
    ):
        text = path.read_text(encoding="utf-8", errors="ignore")
        names.update(re.findall(r"HYPIC_(?:PIC|MOONCAKE)_[A-Z0-9_]+", text))

    assert names <= SURVIVORS, sorted(names - SURVIVORS)
