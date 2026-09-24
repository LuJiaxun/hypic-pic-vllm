# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Position-independent segment-cache primitives."""

from vllm.v1.pic.cache import PICCachePlan, PICSegmentCache, PICSegmentEntry
from vllm.v1.pic.handles import PICHandle, PICHandleKind, PICHandlePool
from vllm.v1.pic.state import (
    PICAffineTransition,
    PICConvTransition,
    PICGDNTransition,
    PICLinearTransition,
    PICStateLayout,
    PICStateSpec,
    PICTransitionOperator,
    PICRequestStateBinding,
)
from vllm.v1.pic.gdn_transition import build_gdn_transition_operator
from vllm.v1.pic.execution import (
    PICExecutionPlan,
    PICExecutionRange,
    PICTransition,
    compile_execution_plan,
)
from vllm.v1.pic.pool import (
    PICPhysicalAllocation,
    PICPhysicalPool,
    PICSlotMapping,
    PICTensorRegion,
)
from vllm.v1.pic.snapshot import PICPhysicalSnapshot, PICSnapshotStore
from vllm.v1.pic.worker_plan import (
    PICWorkerCapabilities,
    PICWorkerPlan,
    PICWorkerUnsupported,
    build_worker_plan,
)
from vllm.v1.pic.materialization import PICMaterialization
from vllm.v1.pic.live_capture import (
    PICGDNTransitionCapture,
    PICLiveCaptureManager,
    capture_completed_mamba_segment,
    capture_transition_operator_segment,
    restore_matched_mamba_segment,
)
from vllm.v1.pic.kv_copyback import (
    PICKVLayerTarget,
    PICAttentionKVCacheCopyBack,
    build_kv_slot_mapping,
)
from vllm.v1.pic.native_kv import (
    PICLocalBlockSpan,
    PICKVPositionMode,
    PICNativeKVKind,
    PICNativeKVLease,
    PICNativeKVAllocation,
    PICNativeKVReference,
    PICNativeKVRequestMapping,
    PICNativeKVSlotPlan,
    PICRoPERerotationPlan,
    build_native_kv_slot_plan,
    gather_native_kv_slots,
    get_full_local_block_span,
    materialize_private_rope_key,
    rerotate_native_key,
)
from vllm.v1.pic.lifecycle import PICLeaseRegistry, PICLeaseToken
from vllm.v1.pic.metrics import PICMetrics
from vllm.v1.pic.external_kv import (
    PICExternalKVImportError,
    PICExternalKVImportRequest,
    PICExternalKVImportResult,
    PICExternalKVProvider,
    PICExternalKVRegistry,
)
from vllm.v1.pic.single_request import (
    PICSingleRequestPlan,
    build_single_request_plan,
)
from vllm.v1.pic.runtime import (
    PICRuntimeRange,
    PICSingleRequestRuntimePlan,
    build_single_request_runtime_plan,
)
from vllm.v1.pic.segmenter import (
    PICSegment,
    segments_from_ranges,
    split_text_and_tokenize,
    split_token_ids,
)

__all__ = [
    "PICCachePlan",
    "PICSegment",
    "PICSegmentCache",
    "PICSegmentEntry",
    "PICHandle",
    "PICHandleKind",
    "PICHandlePool",
    "PICStateLayout",
    "PICStateSpec",
    "PICAffineTransition",
    "PICConvTransition",
    "PICGDNTransition",
    "PICLinearTransition",
    "PICTransitionOperator",
    "PICRequestStateBinding",
    "build_gdn_transition_operator",
    "PICExecutionPlan",
    "PICExecutionRange",
    "PICTransition",
    "compile_execution_plan",
    "PICPhysicalAllocation",
    "PICPhysicalPool",
    "PICSlotMapping",
    "PICTensorRegion",
    "PICPhysicalSnapshot",
    "PICSnapshotStore",
    "PICWorkerCapabilities",
    "PICWorkerPlan",
    "PICWorkerUnsupported",
    "build_worker_plan",
    "PICMaterialization",
    "PICLiveCaptureManager",
    "PICGDNTransitionCapture",
    "capture_completed_mamba_segment",
    "capture_transition_operator_segment",
    "restore_matched_mamba_segment",
    "PICKVLayerTarget",
    "PICAttentionKVCacheCopyBack",
    "build_kv_slot_mapping",
    "PICKVPositionMode",
    "PICLocalBlockSpan",
    "PICNativeKVKind",
    "PICNativeKVLease",
    "PICNativeKVAllocation",
    "PICNativeKVReference",
    "PICNativeKVRequestMapping",
    "PICNativeKVSlotPlan",
    "PICRoPERerotationPlan",
    "build_native_kv_slot_plan",
    "gather_native_kv_slots",
    "get_full_local_block_span",
    "materialize_private_rope_key",
    "rerotate_native_key",
    "PICLeaseRegistry",
    "PICLeaseToken",
    "PICMetrics",
    "PICExternalKVImportError",
    "PICExternalKVImportRequest",
    "PICExternalKVImportResult",
    "PICExternalKVProvider",
    "PICExternalKVRegistry",
    "PICSingleRequestPlan",
    "build_single_request_plan",
    "PICRuntimeRange",
    "PICSingleRequestRuntimePlan",
    "build_single_request_runtime_plan",
    "segments_from_ranges",
    "split_text_and_tokenize",
    "split_token_ids",
]
