"""Fail if code or tests still reference a symbol this cleanup deleted.

The unbound-name checker cannot see attribute access, so a test that asserts on a
field removed from a dataclass passes compileall and only fails at runtime.  This
checker keeps the deletion list in one place and greps for it.

Usage:  python tools/pic_check_removed.py [repo_root]
"""

from __future__ import annotations

import pathlib
import sys

# symbol -> why it is gone
REMOVED: dict[str, str] = {
    # TransferEngine client (11E-C)
    "PICTransferEngine": "TE descriptor client removed",
    "PICTransferEngineError": "renamed to PICTransferError in pic/request_transfer.py",
    "PICTransferDescriptor": "TE descriptor client removed",
    "PICTransferHandle": "TE descriptor client removed",
    "PICTransferBatchHandle": "TE descriptor client removed",
    "pic_transfer_engine_enabled": "TE switch removed",
    "_pic_native_kv_transfer_mode": "route switch removed; Store only",
    "_pic_transition_transport_mode": "route switch removed; Store only",
    "register_native_kv_buffers": "TE buffer registration removed",
    "has_current_native_transfer_source": "TE manifest check removed",
    "release_transfer_sources_for_segment": "TE source registry removed",
    "_build_transition_transfer_layout": "TE layout removed",
    "_load_transition_operator_transfer_engine": "TE receive removed",
    "_receive_native_blocks_transfer_engine": "TE receive removed",
    "_coalesce_transfer_segments": "TE descriptor coalescing removed",
    "_count_contiguous_transfer_pairs": "TE descriptor coalescing removed",
    "_native_block_segments": "TE source descriptor builder removed",
    "_transfer_source_keys": "TE source registry helper removed",
    "_pic_native_kv_batch_descriptors": "TE descriptor budget removed",
    # PICNativeKVReceived fields (11E-C)
    "received.direct": "field removed from PICNativeKVReceived",
    "received.transfer_handle": "field removed from PICNativeKVReceived",
    "received.transfer_submitted_at": "field removed from PICNativeKVReceived",
    "received.transfer_segment_count": "field removed from PICNativeKVReceived",
    "received.transfer_batch_count": "field removed from PICNativeKVReceived",
    "received.compact_buffers": "field removed from PICNativeKVReceived",
    "received.layers": "field removed from PICNativeKVReceived",
    # manifest metadata (11E-C)
    "transfer_source_addresses": "manifest field removed",
    "transfer_source_block_ids": "manifest field removed",
    "transfer_source_region_ids": "manifest field removed",
    "transfer_layout": "manifest field removed",
    "transfer_session": "manifest field removed",
    "transfer_control_endpoint": "manifest field removed",
    "transfer_bootstrap_addr": "manifest field removed",
    "transfer_source_engine_id": "manifest field removed",
    # profile switches (11E-B)
    "HYPIC_PIC_PROFILE_DETAIL": "folded into the diagnostics switch",
    "HYPIC_PIC_PROFILE_FORWARD": "folded into the diagnostics switch",
    "HYPIC_PIC_PROFILE_MEMORY": "folded into the diagnostics switch",
    "HYPIC_PIC_PROFILE_MEMORY_SYNC": "folded into the diagnostics switch",
    "HYPIC_PIC_PROFILE_TIMELINE_SYNC": "folded into the diagnostics switch",
    "pic_profile_level": "replaced by pic_profile_enabled",
    # TE config environment (11E-C)
    "HYPIC_PIC_TRANSFER_ENGINE": "TE config removed",
    "HYPIC_PIC_ENGINE_ID": "TE config removed",
    "HYPIC_PIC_BOOTSTRAP_ADDR": "TE config removed",
    "HYPIC_PIC_TP_RANK": "TE config removed",
    "HYPIC_PIC_PP_RANK": "TE config removed",
    "HYPIC_PIC_TRANSFER_WORKERS": "TE config removed",
    "HYPIC_PIC_NATIVE_KV_BATCH_DESCRIPTORS": "TE config removed",
    "HYPIC_PIC_NATIVE_KV_TRANSFER_MODE": "route switch removed",
    "HYPIC_PIC_TRANSITION_TRANSPORT": "route switch removed",
}

# The removed module itself is checked by path, not by name.
REMOVED_FILES = ("vllm/v1/pic/transfer_engine.py", "tests/v1/pic/test_transfer_engine.py")

ALLOW = {
    # documentation of the removal is fine in these files
    "docs/",
    "patches/",
    "tools/pic_check_removed.py",
}


def main(root: str) -> int:
    base = pathlib.Path(root)
    problems: list[str] = []

    for rel in REMOVED_FILES:
        if (base / rel).exists():
            problems.append(f"{rel}: deleted file still present")

    for path in sorted(base.glob("vllm/**/*.py")) + sorted(base.glob("tests/**/*.py")):
        rel = path.relative_to(base).as_posix()
        if any(rel.startswith(prefix) for prefix in ALLOW):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for symbol, reason in REMOVED.items():
            if symbol in text:
                for lineno, line in enumerate(text.splitlines(), 1):
                    if symbol in line:
                        problems.append(f"{rel}:{lineno}: {symbol} ({reason})")

    if problems:
        for problem in problems:
            print(problem)
        print(f"\n{len(problems)} reference(s) to removed symbols")
        return 1
    print(f"clean: {len(REMOVED)} removed symbols, none referenced")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
