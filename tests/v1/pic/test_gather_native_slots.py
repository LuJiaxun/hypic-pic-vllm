# SPDX-License-Identifier: Apache-2.0
"""Regression tests for native slot gathering in both attention layouts."""

import pytest

torch = pytest.importorskip("torch")

from vllm.v1.pic.native_kv import gather_native_kv_slots


def _cache(layout: str) -> torch.Tensor:
    # [K/V, block, token, head, dim], with a different value at every position.
    cache = torch.arange(2 * 3 * 4 * 2 * 1, dtype=torch.float32).reshape(2, 3, 4, 2, 1)
    if layout == "NHD":
        return cache
    # HND stores the K/V axis inside the block axis: [block, K/V, token, head, dim].
    return cache.permute(1, 0, 2, 3, 4).contiguous()


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_gather_preserves_heads_and_indexes_tokens(layout: str) -> None:
    cache = _cache(layout)
    # Two slots in block 0 and two in block 1; offsets are token positions, not heads.
    slots = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
    key, value = gather_native_kv_slots(cache, slots, block_size=4, layout=layout)
    assert key.shape == (4, 2, 1)
    assert value.shape == (4, 2, 1)
    expected_k = torch.tensor([2.0, 6.0, 10.0, 14.0])
    expected_v = torch.tensor([26.0, 30.0, 34.0, 38.0])
    # Head 0 across the four selected tokens.
    assert torch.equal(key[:, 0, 0], expected_k)
    assert torch.equal(value[:, 0, 0], expected_v)
    # Head 1 must remain head 1 rather than being indexed by the token offset.
    assert torch.equal(key[:, 1, 0], expected_k + 1)
    assert torch.equal(value[:, 1, 0], expected_v + 1)


def test_hnd_token_offset_can_exceed_head_count() -> None:
    # Regression for [block, :, offset]: the second advanced index addresses
    # dim2 (heads), so offset 31 fails with only two heads. Direct [block, token]
    # is the intended form and is valid.
    cache = torch.arange(2 * 3 * 32 * 2 * 1, dtype=torch.float32).reshape(2, 3, 32, 2, 1)
    cache = cache.permute(1, 0, 2, 3, 4).contiguous()
    key, value = gather_native_kv_slots(
        cache,
        torch.tensor([31], dtype=torch.int64),
        block_size=32,
        layout="HND",
    )
    assert key.shape == (1, 2, 1)
    assert value.shape == (1, 2, 1)
    assert int(key[0, 0, 0]) == 62
    assert int(key[0, 1, 0]) == 63
    assert int(value[0, 0, 0]) == 158
    assert int(value[0, 1, 0]) == 159


def test_native_axes_follow_unambiguous_shape_over_stale_layout() -> None:
    from vllm.v1.pic.mooncake_kv import _native_cache_axes

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape
            self.ndim = len(shape)

    # NHD [K/V, block, token, head, dim] but the caller still says HND.
    assert _native_cache_axes(FakeTensor((2, 15572, 32, 2, 256)), layout="HND") == (0, 1)
    # HND [block, K/V, token, head, dim] but the caller still says NHD.
    assert _native_cache_axes(FakeTensor((15572, 2, 32, 2, 256)), layout="NHD") == (1, 0)
    # Both leading dimensions are 2: genuinely ambiguous, so keep the declaration.
    assert _native_cache_axes(FakeTensor((2, 2, 32, 2, 256)), layout="HND") == (1, 0)


def test_offset_beyond_page_fails_before_cuda_indexing() -> None:
    cache = _cache("HND")
    with pytest.raises(ValueError, match="token offset exceeds the cache page"):
        gather_native_kv_slots(
            cache,
            torch.tensor([99], dtype=torch.int64),
            block_size=128,
            layout="HND",
        )
