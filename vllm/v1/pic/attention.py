# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PIC packed attention metadata contract.

Stage 10-A already routes packed ranges through vLLM's native attention
metadata and kernels.  This module makes the assumptions at that boundary
explicit before the model call: request-local query ranges are packed in a
stable order, positions cover the flattened query, and every KV cache group
has a slot mapping with the same padded token extent.

The validator is deliberately backend-neutral.  It does not replace
FlashAttention, FlashInfer, GDN, or their Triton/CUDA kernels.  A backend that
cannot satisfy this contract raises a worker-level fallback instead of
silently reading a wrong request row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from vllm.v1.pic.runtime import PICRuntimeRange


class PICAttentionBackendUnsupported(ValueError):
    """The active attention metadata cannot represent a PIC packed round."""


@dataclass(frozen=True)
class PICPackedAttentionRound:
    """Validated request-local metadata for one packed forward round.

    ``query_start_loc`` uses the same flattened token order as the native
    vLLM input batch.  ``absolute_ranges`` preserves each request's logical
    position range so future backend adapters cannot infer positions from the
    packed offset.
    """

    request_ids: tuple[str, ...]
    query_lengths: tuple[int, ...]
    query_start_loc: "torch.Tensor"
    absolute_ranges: tuple[tuple[int, int], ...]
    positions_shape: tuple[int, ...]
    num_tokens: int
    num_tokens_padded: int
    kv_group_ids: tuple[int, ...]


def build_pic_packed_attention_round(
    request_ids: Sequence[str],
    active_ranges: Mapping[str, PICRuntimeRange],
    *,
    positions: "torch.Tensor",
    slot_mappings_by_group: Mapping[int, "torch.Tensor"] | None,
    num_tokens_padded: int,
) -> PICPackedAttentionRound:
    """Validate and describe one native attention packed round.

    The function only builds metadata; it does not allocate KV blocks or copy
    KV.  ``active_ranges`` must contain one recompute range for every request.
    Reused ranges are applied by the caller before this function is called.
    """

    import torch

    ids = tuple(request_ids)
    if len(ids) < 2:
        raise PICAttentionBackendUnsupported(
            "PIC packed attention requires at least two requests"
        )
    if len(set(ids)) != len(ids):
        raise PICAttentionBackendUnsupported(
            "PIC packed attention request IDs must be unique"
        )
    if set(active_ranges) != set(ids):
        raise PICAttentionBackendUnsupported(
            "PIC packed attention ranges do not cover the request batch"
        )
    if num_tokens_padded <= 0:
        raise PICAttentionBackendUnsupported(
            "PIC packed attention requires positive padded token count"
        )

    ordered_ranges = tuple(active_ranges[request_id] for request_id in ids)
    if any(item.action != "recompute" for item in ordered_ranges):
        raise PICAttentionBackendUnsupported(
            "PIC packed attention accepts recompute ranges only"
        )
    if any(item.end <= item.start or item.start < 0 for item in ordered_ranges):
        raise PICAttentionBackendUnsupported(
            "PIC packed attention contains an invalid absolute range"
        )

    query_lengths = tuple(item.token_count for item in ordered_ranges)
    num_tokens = sum(query_lengths)
    if num_tokens <= 0 or num_tokens > num_tokens_padded:
        raise PICAttentionBackendUnsupported(
            "PIC packed attention token extent is inconsistent"
        )
    if not isinstance(positions, torch.Tensor) or positions.ndim not in (1, 2):
        raise PICAttentionBackendUnsupported(
            "PIC packed attention positions must be [tokens] or [axes, tokens]"
        )
    if positions.shape[-1] < num_tokens:
        raise PICAttentionBackendUnsupported(
            "PIC packed attention positions do not cover query tokens"
        )
    if positions.shape[-1] > num_tokens_padded:
        raise PICAttentionBackendUnsupported(
            "PIC packed attention positions exceed padded token extent"
        )

    if not slot_mappings_by_group:
        raise PICAttentionBackendUnsupported(
            "PIC packed attention has no KV group slot mappings"
        )
    for group_id, slot_mapping in slot_mappings_by_group.items():
        if not isinstance(group_id, int):
            raise PICAttentionBackendUnsupported(
                "PIC KV group IDs must be integers"
            )
        if not isinstance(slot_mapping, torch.Tensor):
            raise PICAttentionBackendUnsupported(
                f"PIC slot mapping for group {group_id} is not a tensor"
            )
        if slot_mapping.ndim != 1 or slot_mapping.numel() != num_tokens_padded:
            raise PICAttentionBackendUnsupported(
                f"PIC slot mapping for group {group_id} has an invalid extent"
            )
        if slot_mapping.dtype not in (torch.int32, torch.int64):
            raise PICAttentionBackendUnsupported(
                f"PIC slot mapping for group {group_id} is not integer typed"
            )

    query_start_loc = torch.zeros(
        len(query_lengths) + 1,
        dtype=torch.int32,
        device=positions.device,
    )
    if query_lengths:
        query_start_loc[1:] = torch.tensor(
            query_lengths,
            dtype=torch.int32,
            device=positions.device,
        ).cumsum(0)

    return PICPackedAttentionRound(
        request_ids=ids,
        query_lengths=query_lengths,
        query_start_loc=query_start_loc,
        absolute_ranges=tuple((item.start, item.end) for item in ordered_ranges),
        positions_shape=tuple(positions.shape),
        num_tokens=num_tokens,
        num_tokens_padded=num_tokens_padded,
        kv_group_ids=tuple(sorted(slot_mappings_by_group)),
    )
