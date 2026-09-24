# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded worker-local observability for the PIC runtime."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from vllm.v1.pic.runtime import PICSingleRequestRuntimePlan


@dataclass
class PICMetrics:
    """Monotonic counters for one GPU worker's PIC activity.

    The counters are deliberately process-local.  The scheduler owns request
    admission and the worker owns native materialization, so combining these
    values across processes would make lease and pool diagnostics misleading.
    ``snapshot`` returns ordinary Python values suitable for logs or a future
    metrics exporter and never exposes mutable internal state.
    """

    _counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _fallback_reasons: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _lock: Lock = field(default_factory=Lock, repr=False)

    def _increment(self, key: str, value: int = 1) -> None:
        if value <= 0:
            return
        with self._lock:
            self._counts[key] += int(value)

    def record_lookup(self, *, hit: bool, matched_segments: int = 0) -> None:
        self._increment("lookup_total")
        self._increment("lookup_hits" if hit else "lookup_misses")
        self._increment("matched_segments", matched_segments)

    def record_plan(self, plan: PICSingleRequestRuntimePlan) -> None:
        self._increment("reused_ranges", len(plan.reused_ranges))
        self._increment(
            "reused_tokens",
            sum(item.token_count for item in plan.reused_ranges),
        )
        self._increment(
            "recompute_tokens",
            sum(item.token_count for item in plan.recompute_ranges),
        )

    def record_seam(self, token_count: int) -> None:
        self._increment("seam_tokens", token_count)

    def record_fallback(self, reason: str) -> None:
        normalized = str(reason).strip() or "unknown"
        with self._lock:
            self._counts["fallbacks"] += 1
            # Prevent an exception string containing request data from growing
            # this diagnostic map without bound.
            if normalized not in self._fallback_reasons and len(
                self._fallback_reasons
            ) >= 32:
                normalized = "other"
            self._fallback_reasons[normalized] += 1

    def record_materialization(self, *, kind: str, reused: bool = False) -> None:
        key = f"{kind}_kv_materialization_reused" if reused else (
            f"{kind}_kv_materialized"
        )
        self._increment(key)

    def record_capture(self, *, deduplicated: bool = False) -> None:
        self._increment(
            "capture_deduplicated" if deduplicated else "capture_stored"
        )

    def record_eviction(self) -> None:
        self._increment("evictions")

    def record_forward(
        self,
        *,
        kind: str,
        token_count: int,
        request_count: int = 1,
    ) -> None:
        self._increment("model_forward_calls")
        self._increment(f"{kind}_forward_calls")
        self._increment("forward_tokens", token_count)
        self._increment("forward_requests", request_count)
        if kind == "packed":
            self._increment("packed_rounds")
        elif kind == "single":
            self._increment("single_request_rounds")

    def record_decode_mapping(self, *, reused: bool) -> None:
        self._increment(
            "decode_mapping_reused" if reused else "decode_mapping_fallback"
        )

    def record_external_import(self, *, failed: bool = False) -> None:
        """Record optional external native-KV imports without unbounded labels."""
        self._increment(
            "external_kv_import_failed" if failed else "external_kv_imported"
        )

    def snapshot(
        self,
        *,
        lease_snapshot: dict[str, Any] | None = None,
        pool_capacity_bytes: int | None = None,
        pool_free_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Return a stable metrics snapshot with optional resource gauges."""
        with self._lock:
            result: dict[str, Any] = dict(sorted(self._counts.items()))
            result["fallback_reasons"] = dict(
                sorted(self._fallback_reasons.items())
            )
        if lease_snapshot is not None:
            result["leases"] = dict(lease_snapshot)
        if pool_capacity_bytes is not None or pool_free_bytes is not None:
            result["pool"] = {
                "capacity_bytes": pool_capacity_bytes,
                "free_bytes": pool_free_bytes,
            }
        return result
