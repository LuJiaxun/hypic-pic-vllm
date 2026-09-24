# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for position-independent segment caching (PIC)."""

from typing import Literal

from pydantic import Field

from vllm.config.utils import config

PICMode = Literal[
    "addition",
    "transition",
    "transition_rope",
    "transition_rope_recompute",
]


@config
class PICConfig:
    """Configuration for HYPIC-style position-independent caching."""

    enabled: bool = False
    """Whether to initialize the PIC subsystem."""

    auto_enable: bool = True
    """Enable PIC for tokenized prompts containing the configured separator."""

    mode: PICMode = "transition_rope_recompute"
    """PIC composition and RoPE/seam policy."""

    separator: str = "<<PIC_SEP>>"
    """Text separator used to identify reusable prompt segments."""

    seam_sink: int = Field(default=8, ge=0)
    """Number of leading tokens recomputed at a reused segment boundary."""

    exclusive_batch: bool = True
    """Keep PIC prefill batches separate from the normal V1 path."""

    allow_fallback: bool = True
    """Fall back to the normal path for unsupported PIC requests."""

    max_cache_bytes: int | None = Field(default=None, ge=0)
    """Optional budget for independent PIC cache pools."""

    capture_live: bool = False
    """Capture completed live Mamba/GDN states into the PIC pool."""

    restore_live: bool = False
    """Restore materialized Mamba/GDN states before the next forward."""

    copyback_kv: bool = False
    """Copy full-KV PIC snapshots into vLLM's ordinary KV cache blocks."""

    zero_copy: bool = False
    """Request the experimental native-KV-slot PIC bridge."""

    single_request: bool = False
    """Request the experimental single-request skip/recompute prototype."""

    batch: bool = False
    """Request the experimental mixed-request PIC execution path.

    Stage 9-B keeps each request's native KV row and recurrent state isolated.
    The first implementation uses request-isolated microbatches for correctness;
    it is intentionally opt-in and does not change ordinary batching by default.
    """

    packed_batch: bool = False
    """Request the Stage 10-A packed hybrid mixed-batch prototype.

    This path keeps one InputBatch and one model forward per packed range round;
    it reuses the existing attention/GDN kernels and adds no Triton/CUDA code.
    """

    mooncake: bool = False
    """Enable the optional external native-KV provider bridge.

    This flag only permits registered providers to import native blocks.  It
    does not install or initialize Mooncake by itself; unavailable providers
    fall back at request scope.
    """

    debug: bool = False
    """Emit diagnostic ``[PIC-DEBUG]`` logs for the PIC control/data path."""

    def compute_hash(self) -> str:
        """Return a stable identity for PIC behavior-affecting settings."""
        from vllm.config.utils import get_hash_factors, hash_factors

        return hash_factors(get_hash_factors(self, ignored_factors=set()))
