# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm.config.pic import PICConfig


def test_async_import_defaults_to_disabled() -> None:
    config = PICConfig()

    assert config.async_import is False


def test_async_import_requires_mooncake() -> None:
    with pytest.raises(ValueError, match="requires mooncake"):
        PICConfig(async_import=True, mooncake=False)


def test_async_import_can_be_enabled_with_mooncake() -> None:
    config = PICConfig(async_import=True, mooncake=True)

    assert config.async_import is True


def test_enabling_mooncake_alone_is_unaffected() -> None:
    config = PICConfig(mooncake=True)

    assert config.mooncake is True
    assert config.async_import is False
