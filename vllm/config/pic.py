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

    min_segment_tokens: int = Field(default=0, ge=0)
    """Shortest segment worth caching.

    A short segment loses most of its body to the seam and to block alignment,
    so importing it costs more than it saves.  Zero keeps every segment eligible;
    production workloads should set this to a few hundred tokens.
    """

    exclusive_batch: bool = True
    """Keep PIC prefill batches separate from the normal V1 path."""

    allow_fallback: bool = True
    """Fall back to the normal path for unsupported PIC requests."""

    max_cache_bytes: int | None = Field(default=None, ge=0)
    """Optional budget for independent PIC cache pools."""

    capture_live: bool = True
    """Capture completed live Mamba/GDN states into the PIC pool."""

    restore_live: bool = True
    """Restore materialized Mamba/GDN states before the next forward."""

    zero_copy: bool = True
    """Request the experimental native-KV-slot PIC bridge."""

    single_request: bool = True
    """Request the experimental single-request skip/recompute prototype."""

    mooncake: bool = True
    """Enable the optional external native-KV provider bridge.

    This flag only permits registered providers to import native blocks.  It
    does not install or initialize Mooncake by itself; unavailable providers
    fall back at request scope.
    """

    async_import: bool = True
    """Park a request in ``WAITING_FOR_REMOTE_KVS`` while its KV is imported.

    Disabled by default.  When enabled, other requests keep prefilling while one
    request waits for its external segments, instead of the import blocking the
    step it was scheduled with.

    The import is started before the step's forward and inspected afterwards,
    so the receive overlaps the rest of the batch instead of blocking it.
    """

    debug: bool = False
    """Emit diagnostic ``[PIC-DEBUG]`` logs for the PIC control/data path."""

    def __post_init__(self) -> None:
        # One switch for diagnostics. The flag can be set on the command line and the environment can
        # set it too; keeping them apart meant a run could turn on `HYPIC_PIC_DEBUG` and still see
        # only half the log, because the capture objects read this flag instead.
        if not self.debug:
            from vllm.v1.pic.env import pic_debug_enabled

            if pic_debug_enabled():
                object.__setattr__(self, "debug", True)
        if self.async_import and not self.mooncake:
            raise ValueError(
                "PIC async_import requires mooncake to be enabled: there is no "
                "external import to overlap otherwise"
            )

    def compute_hash(self) -> str:
        """Return a stable identity for PIC behavior-affecting settings."""
        from vllm.config.utils import get_hash_factors, hash_factors

        return hash_factors(get_hash_factors(self, ignored_factors=set()))
