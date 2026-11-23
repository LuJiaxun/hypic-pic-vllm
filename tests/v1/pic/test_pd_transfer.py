# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the request-scoped P/D state descriptor builder."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

# This module is intentionally torch-free.  Loading the parent vllm package
# initializes CUDA-related optional dependencies, which are not needed to test
# descriptor arithmetic and are absent from the static-check environment.
_MODULE_PATH = Path(__file__).resolve().parents[3] / "vllm" / "v1" / "pic" / "pd_transfer.py"
for _name, _path in (
    ("vllm", _MODULE_PATH.parents[2]),
    ("vllm.v1", _MODULE_PATH.parents[1]),
    ("vllm.v1.pic", _MODULE_PATH.parent),
):
    _module = sys.modules.get(_name)
    if _module is None:
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_path)]
        sys.modules[_name] = _module
_spec = importlib.util.spec_from_file_location("vllm.v1.pic.pd_transfer", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_pd_transfer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _pd_transfer
_spec.loader.exec_module(_pd_transfer)

PDMambaStateSpec = _pd_transfer.PDMambaStateSpec
build_state_descriptors = _pd_transfer.build_state_descriptors
remote_prefill_token_count = _pd_transfer.remote_prefill_token_count
state_page_index = _pd_transfer.state_page_index
validate_state_metadata = _pd_transfer.validate_state_metadata


def spec(group_id: int = 1, name: str = "gdn/state-1") -> PDMambaStateSpec:
    return PDMambaStateSpec(
        group_id=group_id,
        name=name,
        shape=(4, 8, 16),
        dtype="float16",
        block_size=32,
        strides=(512, 16, 1),
        nbytes=4 * 8 * 16 * 2,
    )


class RemotePrefillCountTest(unittest.TestCase):
    def test_plain_mooncake_uses_whole_prompt(self) -> None:
        self.assertEqual(remote_prefill_token_count({}, 100, 0), 100)

    def test_relay_excludes_prefillers_sampled_token(self) -> None:
        # D sees 101 IDs: 100 original prompt IDs plus P's first sampled token.
        # Only the original 100 have KV on P, so D forwards exactly one token.
        self.assertEqual(
            remote_prefill_token_count({"remote_prefill_tokens": 100}, 101, 0),
            100,
        )

    def test_rejects_out_of_range_count(self) -> None:
        with self.assertRaises(ValueError):
            remote_prefill_token_count({"remote_prefill_tokens": 101}, 100, 0)


class StateDescriptorTest(unittest.TestCase):
    def test_hn_page_and_natural_bytes_not_page_padding(self) -> None:
        src, dst, lengths = build_state_descriptors(
            specs=[spec()],
            local_base_addrs=[10_000],
            remote_base_addrs=[90_000],
            block_lens=[4096],
            state_nbytes=[1024],
            local_block_ids=[[], [7, 11, 13]],
            remote_block_ids=[[], [23, 29, 31]],
            prompt_tokens=65,
        )
        # h_65 lives at position 64, page 2. The source/destination use the
        # actual block ids, while the payload is the natural 1 KiB, not 4 KiB.
        self.assertEqual(state_page_index(65, 32), 2)
        self.assertEqual(src, [10_000 + 13 * 4096])
        self.assertEqual(dst, [90_000 + 31 * 4096])
        self.assertEqual(lengths, [1024])

    def test_group_ids_are_not_flattened_attention_indices(self) -> None:
        first = spec(group_id=1, name="layer-a/state-0")
        second = spec(group_id=3, name="layer-b/state-1")
        src, dst, lengths = build_state_descriptors(
            specs=[first, second],
            local_base_addrs=[1000, 2000],
            remote_base_addrs=[3000, 4000],
            block_lens=[100, 200],
            state_nbytes=[80, 160],
            local_block_ids=[[], [5], [], [7]],
            remote_block_ids=[[], [11], [], [13]],
            prompt_tokens=1,
        )
        self.assertEqual(src, [1000 + 5 * 100, 2000 + 7 * 200])
        self.assertEqual(dst, [3000 + 11 * 100, 4000 + 13 * 200])
        self.assertEqual(lengths, [80, 160])

    def test_missing_page_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "not allocated"):
            build_state_descriptors(
                specs=[spec()],
                local_base_addrs=[0],
                remote_base_addrs=[0],
                block_lens=[100],
                state_nbytes=[80],
                local_block_ids=[[], [7]],
                remote_block_ids=[[], [11]],
                prompt_tokens=64,
            )

    def test_metadata_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            validate_state_metadata([spec()], [spec(name="different")])


if __name__ == "__main__":
    unittest.main()
