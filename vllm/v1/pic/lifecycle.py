# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Process-local PIC resource leases.

The scheduler owns the real vLLM ``BlockPool`` references, while the worker
owns live snapshots and request-local private KV views. This registry gives
those worker-owned resources one explicit ref-count protocol for cache-entry,
active-request, and in-flight execution references; it does not replace the
underlying allocators.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PICLeaseToken:
    """One ownership reference held by a PIC consumer."""

    token_id: int
    resource_key: Hashable
    owner: str
    kind: str


@dataclass
class _PICLeaseRecord:
    ref_count: int = 0
    retired: bool = False
    release_fn: Callable[[], None] | None = None


class PICLeaseRegistry:
    """Reference-count worker-owned PIC resources."""

    def __init__(self) -> None:
        self._records: dict[Hashable, _PICLeaseRecord] = {}
        self._tokens: dict[int, PICLeaseToken] = {}
        self._next_token_id = 1

    def acquire(
        self,
        resource_key: Hashable,
        *,
        owner: str,
        kind: str,
        release_fn: Callable[[], None] | None = None,
    ) -> PICLeaseToken:
        record = self._records.get(resource_key)
        if record is None:
            record = _PICLeaseRecord(release_fn=release_fn)
            self._records[resource_key] = record
        elif record.retired:
            raise RuntimeError(f"PIC resource is retired: {resource_key!r}")
        elif release_fn is not None:
            if record.release_fn is not None and record.release_fn is not release_fn:
                raise ValueError("PIC resource release callback changed")
            record.release_fn = release_fn

        token = PICLeaseToken(
            token_id=self._next_token_id,
            resource_key=resource_key,
            owner=str(owner),
            kind=str(kind),
        )
        self._next_token_id += 1
        record.ref_count += 1
        self._tokens[token.token_id] = token
        return token

    def release(self, token: PICLeaseToken) -> bool:
        """Release one token; repeated release is harmless."""
        live = self._tokens.pop(token.token_id, None)
        if live is None:
            return False
        if live != token:
            raise ValueError("PIC lease token does not match registry state")
        record = self._records.get(token.resource_key)
        if record is None or record.ref_count <= 0:
            raise RuntimeError("PIC lease registry ref-count underflow")
        record.ref_count -= 1
        self._finalize_if_unused(token.resource_key, record)
        return True

    def release_owner(self, owner: str) -> int:
        """Release all tokens held by one request or execution owner."""
        tokens = tuple(token for token in self._tokens.values() if token.owner == owner)
        for token in tokens:
            self.release(token)
        return len(tokens)

    def retire(self, resource_key: Hashable) -> bool:
        """Stop new acquisitions and finalize when no references remain."""
        record = self._records.get(resource_key)
        if record is None:
            return False
        record.retired = True
        self._finalize_if_unused(resource_key, record)
        return True

    def ref_count(self, resource_key: Hashable) -> int:
        record = self._records.get(resource_key)
        return 0 if record is None else record.ref_count

    def has_live_tokens(self, resource_key: Hashable) -> bool:
        return self.ref_count(resource_key) > 0

    def owner_tokens(self, owner: str) -> tuple[PICLeaseToken, ...]:
        return tuple(token for token in self._tokens.values() if token.owner == owner)

    def snapshot(self) -> dict[str, Any]:
        """Return bounded diagnostics without exposing mutable internals."""
        by_kind: dict[str, int] = {}
        by_owner: dict[str, int] = {}
        for token in self._tokens.values():
            by_kind[token.kind] = by_kind.get(token.kind, 0) + 1
            by_owner[token.owner] = by_owner.get(token.owner, 0) + 1
        return {
            "resources": len(self._records),
            "tokens": len(self._tokens),
            "by_kind": dict(sorted(by_kind.items())),
            "by_owner": dict(sorted(by_owner.items())),
        }

    def clear(self) -> None:
        """Release all callbacks and discard the registry."""
        for resource_key, record in tuple(self._records.items()):
            record.ref_count = 0
            record.retired = True
            self._finalize_if_unused(resource_key, record)
        self._tokens.clear()
        self._records.clear()

    def _finalize_if_unused(
        self, resource_key: Hashable, record: _PICLeaseRecord
    ) -> None:
        if not record.retired or record.ref_count != 0:
            return
        self._records.pop(resource_key, None)
        if record.release_fn is not None:
            record.release_fn()
