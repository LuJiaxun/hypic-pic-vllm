# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal A/B: per-layer store.get() vs one batch_get_into_multi_buffers().

Answers two questions for the same payload:

1. is the batched read faster than N single reads?  (native KV, 6 objects)
2. is one big store.get() as fast as the batch?  (if yes, transition could drop
   the host arena and the batch API entirely; if no, keep the batch)

    python3 tests/v1/pic/run_store_read_path_probe.py \
        --master 10.112.92.61:50088 --host 10.112.92.61 --reps 5

Default cases mimic the two PIC shapes: 6 objects / 53.5 MB (native KV) and
90 objects / 38.4 MB (transition). No torch, no pinned memory, no GPU target.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import time


def buffer_address(buffer: bytearray) -> int:
    return ctypes.addressof((ctypes.c_char * len(buffer)).from_buffer(buffer))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="10.112.92.61:50088")
    parser.add_argument("--host", default="10.112.92.61")
    parser.add_argument("--protocol", default="tcp")
    parser.add_argument("--segment-gib", type=int, default=8)
    parser.add_argument("--buffer-gib", type=int, default=2)
    parser.add_argument(
        "--cases",
        default="1:38.4,6:53.5,90:38.4",
        help="count:total_MB pairs; 1:X compares one store.get with a one-key batch",
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--keep", action="store_true", help="keep objects")
    args = parser.parse_args(argv)

    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    rc = store.setup(
        args.host,
        "P2PHANDSHAKE",
        args.segment_gib * 1024**3,
        args.buffer_gib * 1024**3,
        args.protocol,
        "",
        args.master,
    )
    print(f"setup rc={rc} protocol={args.protocol} reps={args.reps}", file=sys.stderr)
    if rc != 0:
        raise SystemExit(f"setup failed: {rc}")
    batch_get = getattr(store, "batch_get_into_multi_buffers", None)
    if not callable(batch_get):
        raise SystemExit("this Mooncake build has no batch_get_into_multi_buffers")

    print(
        "objects | total_MB | singles_ms | batch_ms | big_ms | "
        "singles_GBs | batch_GBs | big_GBs | batch_vs_big"
    )

    for case in args.cases.split(","):
        count_str, total_str = case.split(":")
        count = int(count_str)
        total_mb = float(total_str)
        object_bytes = int(total_mb * 1024**2 / count)
        blob = bytes(object_bytes)
        keys = [f"probe/{count}/{index}" for index in range(count)]
        for key in keys:
            assert store.put(key, blob) == 0
        # Same total bytes as one object: this is what the compat payload path
        # would read, and the cheapest possible store.get() variant.
        big_key = f"probe/{count}/big"
        assert store.put(big_key, blob * count) == 0

        buffer = bytearray(object_bytes * count)
        base = buffer_address(buffer)
        addrs = [[base + index * object_bytes] for index in range(count)]
        sizes = [[object_bytes] for _ in range(count)]

        # Registration question: the first attempt uses the raw (unregistered)
        # host buffer. If the Store rejects it, register and retry once; the
        # printout says which mode this deployment actually needs.
        needs_register = False
        result = batch_get(keys, addrs, sizes)
        if len(result) != count or any(int(item) < 0 for item in result):
            needs_register = True
            register_buffer = getattr(store, "register_buffer", None)
            if not callable(register_buffer):
                raise SystemExit(
                    "batch rejected the unregistered buffer and this build has "
                    f"no register_buffer: {[int(i) for i in result][:4]}"
                )
            status = register_buffer(base, object_bytes * count)
            result = batch_get(keys, addrs, sizes)
            if len(result) != count or any(int(item) < 0 for item in result):
                raise SystemExit(
                    f"batch failed even after register_buffer status={status}: "
                    f"{[int(i) for i in result][:4]}"
                )
        print(
            f"  {count} objects: registration_required={needs_register}",
            file=sys.stderr,
        )

        singles_ms: list[float] = []
        batch_ms: list[float] = []
        big_ms: list[float] = []
        for _ in range(args.reps):
            start = time.perf_counter()
            for key in keys:
                if len(store.get(key)) != object_bytes:
                    raise SystemExit("single get returned the wrong size")
            singles_ms.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            result = batch_get(keys, addrs, sizes)
            batch_ms.append((time.perf_counter() - start) * 1000.0)
            if len(result) != count or any(int(item) < 0 for item in result):
                raise SystemExit(f"batch get failed: {[int(i) for i in result][:4]}")

            start = time.perf_counter()
            if len(store.get(big_key)) != object_bytes * count:
                raise SystemExit("big get returned the wrong size")
            big_ms.append((time.perf_counter() - start) * 1000.0)

        single = statistics.median(singles_ms)
        batch = statistics.median(batch_ms)
        big = statistics.median(big_ms)
        size_mb = object_bytes * count / 1024**2
        print(
            f"{count:>7} | {size_mb:>8.1f} | {single:>10.1f} | {batch:>8.1f} | "
            f"{big:>6.1f} | {size_mb / single:>11.3f} | {size_mb / batch:>9.3f} | "
            f"{size_mb / big:>7.3f} | {batch / big:>12.2f}x"
        )

        if not args.keep:
            for key in keys:
                store.remove(key, force=True)
            store.remove(big_key, force=True)

    print(
        "\nsingles vs batch : batch clearly faster -> adopt the host-staging batch\n"
        "                   receive for native KV.\n"
        "batch vs big     : batch clearly faster -> keep the batch for transition.\n"
        "                   about the same       -> transition could use one\n"
        "                   store.get() and drop the arena/batch complexity."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
