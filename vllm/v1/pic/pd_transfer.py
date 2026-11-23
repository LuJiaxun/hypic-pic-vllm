# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped P/D state transfer helpers.

PIC's segment cache and P/D request handoff have different state semantics.  A
P/D handoff continues the same request at the same absolute position, so the
prefill side sends the *live* GDN/convolution state pages, not a transition
operator and not a canonical, segment-local KV object.

The helpers here deliberately contain no torch or Mooncake imports.  They turn
that metadata into TransferEngine descriptors and can therefore be tested
without a GPU or a running engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PDMambaStateSpec:
    """One per-layer state page transferred for each remote-prefill request."""

    group_id: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    block_size: int
    strides: tuple[int, ...]
    nbytes: int

    def __post_init__(self) -> None:
        if (
            self.group_id < 0
            or not self.name
            or self.block_size <= 0
            or self.nbytes <= 0
            or len(self.strides) != len(self.shape)
        ):
            raise ValueError("invalid P/D Mamba state metadata")


def remote_prefill_token_count(
    params: Mapping[str, Any], request_token_count: int, computed_tokens: int
) -> int:
    """Tokens D may treat as computed before its first forward.

    A hybrid P/D relay appends P's first sampled token to D's prompt.  Only the
    original prompt has KV on P, so ``remote_prefill_tokens`` must be returned
    rather than the appended prompt length.  D then forwards exactly that one
    relay token, starting from P's live state after the original prompt.
    """
    external = params.get("remote_prefill_tokens")
    if external is None:
        external = request_token_count
    external = int(external)
    if external < 0 or external > request_token_count:
        raise ValueError("remote prefill token count exceeds the request prompt")
    return max(0, external - int(computed_tokens))


def state_page_index(prompt_tokens: int, block_size: int) -> int:
    """Page holding the live state immediately after ``prompt_tokens`` tokens."""
    if prompt_tokens <= 0 or block_size <= 0:
        raise ValueError("P/D state position and block size must be positive")
    return (prompt_tokens - 1) // block_size


def validate_state_metadata(
    local: Sequence[PDMambaStateSpec], remote: Sequence[PDMambaStateSpec]
) -> None:
    """Reject a transfer whose two sides do not describe the same state pages."""
    if len(local) != len(remote):
        raise ValueError(
            "P/D Mamba state count mismatch: "
            f"local={len(local)} remote={len(remote)}"
        )
    for index, (left, right) in enumerate(zip(local, remote, strict=True)):
        if left != right:
            raise ValueError(
                "P/D Mamba state metadata mismatch at index "
                f"{index}: local={left!r} remote={right!r}"
            )


def build_state_descriptors(
    *,
    specs: Sequence[PDMambaStateSpec],
    local_base_addrs: Sequence[int],
    remote_base_addrs: Sequence[int],
    block_lens: Sequence[int],
    state_nbytes: Sequence[int],
    local_block_ids: Sequence[Sequence[int]],
    remote_block_ids: Sequence[Sequence[int]],
    prompt_tokens: int,
) -> tuple[list[int], list[int], list[int]]:
    """Build ``(src_ptrs, dst_ptrs, lengths)`` for one request's state handoff.

    Each state page is copied whole.  This includes recurrent state and the
    rolling convolution state; both already represent the same exact boundary
    on the two sides.  No segment-local re-rotation or transition application
    is involved.
    """
    if not (
        len(specs)
        == len(local_base_addrs)
        == len(remote_base_addrs)
        == len(block_lens)
        == len(state_nbytes)
    ):
        raise ValueError("P/D Mamba state address metadata length mismatch")
    src_ptrs: list[int] = []
    dst_ptrs: list[int] = []
    lengths: list[int] = []
    for index, spec in enumerate(specs):
        page_index = state_page_index(prompt_tokens, spec.block_size)
        group_id = spec.group_id
        if group_id >= len(local_block_ids) or group_id >= len(remote_block_ids):
            raise ValueError(f"P/D request has no Mamba group {group_id}")
        local_group = local_block_ids[group_id]
        remote_group = remote_block_ids[group_id]
        if page_index >= len(local_group) or page_index >= len(remote_group):
            raise ValueError(
                "P/D Mamba state page is not allocated: "
                f"group={group_id} page={page_index} "
                f"local_blocks={len(local_group)} remote_blocks={len(remote_group)}"
            )
        block_len = int(block_lens[index])
        nbytes = int(state_nbytes[index])
        if block_len <= 0 or nbytes <= 0 or nbytes > block_len:
            raise ValueError("P/D Mamba state page length is invalid")
        src_ptrs.append(
            int(local_base_addrs[index]) + int(local_group[page_index]) * block_len
        )
        dst_ptrs.append(
            int(remote_base_addrs[index]) + int(remote_group[page_index]) * block_len
        )
        lengths.append(nbytes)
    return src_ptrs, dst_ptrs, lengths
