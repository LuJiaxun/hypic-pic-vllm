#!/usr/bin/env python3
"""Minimal Mooncake TCP loopback and provider-lifecycle check.

Run this script in two terminals. It intentionally tests transfer mechanics and
the PIC provider lease contract separately from vLLM's native KV allocator.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import time

from mooncake.engine import TransferEngine

from vllm.v1.pic.external_kv import PICExternalKVImportRequest
from vllm.v1.pic.lifecycle import PICLeaseRegistry
from vllm.v1.pic.mooncake_provider import MooncakePICExternalKVProvider

DEFAULT_SIZE = 4096


def make_engine(address: str, metadata_server: str) -> TransferEngine:
    engine = TransferEngine()
    result = engine.initialize(address, metadata_server, "tcp", "")
    if result != 0:
        raise RuntimeError(f"TransferEngine.initialize failed: {result}")
    return engine


def target_main(args: argparse.Namespace) -> None:
    engine = make_engine(args.address, args.metadata_server)
    address = engine.allocate_managed_buffer(args.size)
    if address == 0:
        raise RuntimeError("allocate_managed_buffer returned 0")
    payload = bytes(index % 251 for index in range(args.size))
    result = engine.write_bytes_to_buffer(address, payload, len(payload))
    if result != 0:
        raise RuntimeError(f"write_bytes_to_buffer failed: {result}")
    print(f"target ready address={args.address} bytes={len(payload)}", flush=True)
    print(f"target checksum={hashlib.sha256(payload).hexdigest()}", flush=True)
    try:
        while True:
            time.sleep(1)
    finally:
        engine.free_managed_buffer(address, args.size)


def provider_check(size: int) -> None:
    transferred: list[tuple[int, ...]] = []
    released: list[tuple[int, ...]] = []
    provider = MooncakePICExternalKVProvider(
        lease_registry=PICLeaseRegistry(),
        block_resolver=lambda request: (101, 102),
        engine="tcp-loopback",
        transfer=lambda request, block_ids, engine: transferred.append(block_ids),
        release=lambda request, block_ids, engine: released.append(block_ids),
    )
    request = PICExternalKVImportRequest(
        provider="mooncake",
        external_key="tcp-loopback-object",
        kv_cache_group_id=0,
        block_size=size // 2,
        token_count=size,
        canonical_start=0,
        dtype="uint8",
        layout=(2, size // 2),
    )
    result = provider.import_native_kv(request)
    result.lease.release()
    if transferred != [(101, 102)] or released != [(101, 102)]:
        raise RuntimeError("provider transfer/release lifecycle mismatch")
    print("provider lease released", flush=True)


def initiator_main(args: argparse.Namespace) -> None:
    engine = make_engine(args.address, args.metadata_server)
    host_buffer = (ctypes.c_ubyte * args.size)()
    source = ctypes.addressof(host_buffer)
    result = engine.register_memory(source, args.size)
    if result != 0:
        raise RuntimeError(f"register_memory failed: {result}")
    try:
        destination = engine.get_first_buffer_address(args.target_address)
        if destination == 0:
            raise RuntimeError("target buffer is not visible")
        payload = bytes(index % 251 for index in range(args.size))
        ctypes.memmove(source, payload, len(payload))
        result = engine.transfer_sync_write(
            args.target_address, source, destination, args.size
        )
        if result != 0:
            raise RuntimeError(f"transfer_sync_write failed: {result}")
        ctypes.memset(source, 0, args.size)
        result = engine.transfer_sync_read(
            args.target_address, source, destination, args.size
        )
        if result != 0:
            raise RuntimeError(f"transfer_sync_read failed: {result}")
        expected = bytes(index % 251 for index in range(args.size))
        payload = ctypes.string_at(source, args.size)
        if payload != expected:
            raise RuntimeError("tcp loopback checksum mismatch")
        print("tcp loopback checksum matched", flush=True)
        provider_check(args.size)
    finally:
        engine.unregister_memory(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("target", "initiator"))
    parser.add_argument("--address", required=True)
    parser.add_argument("--target-address", default="127.0.0.1:12345")
    parser.add_argument("--metadata-server", default="127.0.0.1:2379")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.size % 2 != 0:
        raise ValueError("--size must be a positive even number")
    if args.role == "target":
        target_main(args)
    else:
        initiator_main(args)


if __name__ == "__main__":
    main()
