# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional external native-KV import bridge for PIC.

The bridge deliberately contains no Mooncake import.  A connector can register
an adapter that imports an external object into vLLM-owned native KV blocks
and returns a :class:`PICNativeKVLease`.  Until such an adapter is registered,
external references fail closed and the affected request falls back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from vllm.v1.pic.native_kv import (
    PICNativeKVKind,
    PICNativeKVLease,
    PICNativeKVReference,
    PICKVPositionMode,
)


class PICExternalKVImportError(RuntimeError):
    """Raised when an external KV object cannot be safely imported."""


@dataclass(frozen=True)
class PICExternalKVImportRequest:
    """Validated metadata passed to an external KV provider."""

    provider: str
    external_key: str
    kv_cache_group_id: int
    block_size: int
    token_count: int
    canonical_start: int
    dtype: str | None
    layout: tuple[int, ...]


@dataclass(frozen=True)
class PICExternalKVImportResult:
    """Native blocks imported by a provider and pinned by a lease."""

    block_ids: tuple[int, ...]
    block_size: int
    token_count: int
    lease: PICNativeKVLease

    def __post_init__(self) -> None:
        block_ids = tuple(int(block_id) for block_id in self.block_ids)
        object.__setattr__(self, "block_ids", block_ids)
        if not block_ids or len(set(block_ids)) != len(block_ids):
            raise ValueError("external KV import must return unique native blocks")
        if self.block_size <= 0 or self.token_count <= 0:
            raise ValueError("external KV import has invalid layout")
        if self.lease.block_ids != block_ids:
            raise ValueError("external KV lease does not match imported blocks")


class PICExternalKVProvider(Protocol):
    """Adapter contract implemented by Mooncake or another KV provider."""

    name: str

    def import_native_kv(
        self, request: PICExternalKVImportRequest
    ) -> PICExternalKVImportResult:
        """Import one external object into native vLLM KV blocks."""


class PICExternalKVRegistry:
    """Worker-local registry for optional external native-KV providers."""

    def __init__(self) -> None:
        self._providers: dict[str, PICExternalKVProvider] = {}

    def register(self, provider: PICExternalKVProvider) -> None:
        name = str(provider.name).strip()
        if not name:
            raise ValueError("external KV provider name cannot be empty")
        if name in self._providers:
            raise ValueError(f"external KV provider already registered: {name}")
        self._providers[name] = provider

    def unregister(self, name: str) -> bool:
        return self._providers.pop(str(name), None) is not None

    def provider_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def import_reference(
        self,
        reference: PICNativeKVReference,
    ) -> PICNativeKVReference:
        """Resolve one external reference into a leased native reference."""
        if not reference.is_external:
            return reference
        provider_name = reference.external_provider
        external_key = reference.external_key
        if provider_name is None or external_key is None:
            raise PICExternalKVImportError(
                "external PIC reference is missing provider or object key"
            )
        provider = self._providers.get(provider_name)
        if provider is None:
            raise PICExternalKVImportError(
                f"external PIC provider is unavailable: {provider_name}"
            )
        request = PICExternalKVImportRequest(
            provider=provider_name,
            external_key=external_key,
            kv_cache_group_id=reference.kv_cache_group_id,
            block_size=reference.block_size,
            token_count=reference.token_count,
            canonical_start=reference.canonical_start,
            dtype=reference.dtype,
            layout=reference.layout,
        )
        try:
            result = provider.import_native_kv(request)
        except Exception as exc:
            raise PICExternalKVImportError(
                f"external PIC provider import failed: {provider_name}"
            ) from exc
        expected_blocks = (
            (reference.canonical_start + reference.token_count + reference.block_size - 1)
            // reference.block_size
        ) - (reference.canonical_start // reference.block_size)
        if result.block_size != reference.block_size:
            result.lease.release()
            raise PICExternalKVImportError(
                "external PIC provider returned a different block size"
            )
        if result.token_count != reference.token_count:
            result.lease.release()
            raise PICExternalKVImportError(
                "external PIC provider returned a different token count"
            )
        if len(result.block_ids) != expected_blocks:
            result.lease.release()
            raise PICExternalKVImportError(
                "external PIC provider returned an incomplete block span"
            )
        return PICNativeKVReference(
            kv_cache_group_id=reference.kv_cache_group_id,
            block_ids=result.block_ids,
            block_size=reference.block_size,
            token_count=reference.token_count,
            kind=PICNativeKVKind.SHARED,
            position_mode=PICKVPositionMode.CANONICAL_LOCAL,
            canonical_start=reference.canonical_start,
            source_token_start=reference.source_token_start,
            source_block_start=reference.source_block_start,
            dtype=reference.dtype,
            layout=reference.layout,
            lease=result.lease,
        )

    def import_references(
        self, references: Sequence[PICNativeKVReference]
    ) -> tuple[PICNativeKVReference, ...]:
        """Resolve a group-aligned reference tuple atomically."""
        imported: list[PICNativeKVReference] = []
        try:
            for reference in references:
                imported.append(self.import_reference(reference))
        except Exception:
            for reference in imported:
                reference.release()
            raise
        return tuple(imported)
