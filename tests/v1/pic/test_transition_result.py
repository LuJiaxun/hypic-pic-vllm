# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The parked import must hand ``import_transition_operator`` an operator.

``MooncakeNativeKVObjectStore.load_transition_operator`` returns a pair
``(manifest, operator)``.  The synchronous import path unpacks it, but the parked path
stored the raw return value, so the operator passed downstream was a tuple and
``operator.detached_clone()`` raised ``AttributeError`` — a hard EngineCore crash.
"""

import pytest

from vllm.v1.pic.request_transfer import split_transition_result


class _Operator:
    def __init__(self, name: str) -> None:
        self.name = name


class _Manifest:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count


def test_split_returns_manifest_and_operator() -> None:
    manifest = _Manifest(4352)
    operator = _Operator("transition")

    got_manifest, got_operator = split_transition_result((manifest, operator))

    assert got_manifest is manifest
    assert got_operator is operator
    # The crash was precisely this: the operator must not be a tuple.
    assert not isinstance(got_operator, tuple)


def test_split_accepts_a_bare_operator() -> None:
    operator = _Operator("bare")

    got_manifest, got_operator = split_transition_result(operator)

    assert got_manifest is None
    assert got_operator is operator


def test_split_leaves_longer_tuples_alone() -> None:
    """A three-element tuple is not the documented pair, so treat it as the payload."""
    value = (_Manifest(1), _Operator("a"), "extra")

    got_manifest, got_operator = split_transition_result(value)

    assert got_manifest is None
    assert got_operator is value
