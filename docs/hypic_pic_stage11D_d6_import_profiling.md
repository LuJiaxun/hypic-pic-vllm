# Stage 11-D D6: PIC import profiling

This profiling mode does not change PIC behavior. It is enabled only when the
worker environment contains:

```bash
export HYPIC_PIC_PROFILE_IMPORT=1
```

Application-level SHA-256 verification is separate from profiling and is
disabled on the normal hot path. Mooncake's transport provides the transfer
integrity checks; PIC still publishes the checksum in its manifest for
diagnostics and compatibility. Enable the expensive verification explicitly
when testing corruption handling:

```bash
export HYPIC_PIC_VERIFY_CHECKSUM=1
```

For a performance run, leave `HYPIC_PIC_VERIFY_CHECKSUM` unset (or set it to
`0`). The profile record reports `checksum_enabled=false` and
`checksum_ms=0.000` in that mode. A correctness/corruption run should set it
to `1` and confirm that a modified payload is rejected.

The worker then emits `[PIC-PROFILE]` records for:

- `native-manifest`: Mooncake manifest `Get`;
- `native-layer`: per-layer payload `Get`, checksum, CPU decode, and GPU
  `index_copy_`;
- `native-total`: one native KV object import;
- `native-materialization`: request-private/public KV materialization,
  RoPE rerotation, and backend KV writes;
- `transition-apply`: hybrid transition application after a reuse range is
  selected;
- `transition-manifest` and `transition-total`: hybrid transition fetch and
  deserialization;
- `native-import`: provider-level resolver and transfer total.

The hybrid transition wire format is compact by default. Set
`HYPIC_PIC_COMPACT_TRANSITION=0` to force the legacy token-level payload for
an A/B compatibility run. With profiling enabled, the publisher emits
`transition-publish` with `payload_bytes`, `wire_version`, and `compact`; B's
`transition-total` repeats the received `payload_bytes`. Compare these values
before comparing deserialize time. The compact format stores the fixed-size
GDN linear operator plus its zero-start state, and only the convolution tail
needed to restore the boundary state; it does not serialize every segment
token's raw GDN inputs or the full convolution input sequence. Wire version 3
uses a small JSON tensor descriptor followed by contiguous tensor bytes, so it
does not invoke `torch.load`/pickle for compact payloads. Wire version 4 keeps
the same descriptor and pads every tensor offset to 16 bytes so the receiver can
upload the data region once and reinterpret it in place. Wire versions 1 and
2 remain readable through the legacy `torch.load` path.

## Test procedure

Run the same D6 workload on both machines, with the same prompt files and
request count. Capture A and B logs separately. Keep at least 20 warm-up
requests, then 100 measured requests; use 200 requests when checking the
wraparound regression.

For each B process:

```bash
export HYPIC_PIC_PROFILE_IMPORT=1
# Start vLLM with the normal Stage 11-D PIC command.
# Redirect the complete B log to /tmp/pic_b.log.
```

Summarize the logs outside or inside the container:

```bash
python3 tools/pic_import_profile.py /tmp/pic_b.log
python3 tools/pic_import_profile.py /tmp/pic_b.log --json > /tmp/pic_b_profile.json
```

Run the same commands for the single-card and three-card logs. Do not compare
only total request latency; compare the per-object and per-layer records.

## Required comparison table

Record these values for each machine:

| Metric | Meaning |
|---|---|
| `native-manifest.get_ms` | metadata lookup latency |
| `native-layer.get_ms` | remote payload transfer latency |
| `native-layer.checksum_ms` | CPU checksum cost |
| `native-layer.cpu_ms` | bytes-to-CPU-tensor cost |
| `native-layer.h2d_index_ms` | CPU-to-GPU copy plus block write |
| `transition-total.deserialize_ms` | hybrid operator decode cost |
| `transition-publish.payload_bytes` | A-side serialized hybrid payload size |
| `transition-total.payload_bytes` | B-side received hybrid payload size |
| `native-import.transfer_ms` | complete native import path |
| `native-materialization.materialization_ms` | canonical/public or request-private KV materialization |
| `native-materialization.rerotation_ms` | RoPE rerotation portion of materialization |
| `native-materialization.copy_ms` | backend KV write portion of materialization |
| `transition-apply.apply_ms` | hybrid state transition application |
| record count by `key`/`layer` | repeated Gets and duplicate imports |

Also collect Mooncake master `Get` and `GetBytes` counters, but do not compare
aggregate counts until they are split by worker/rank. A three-card run may
include several independent worker clients.

## Interpretation

- High `native-layer.get_ms`: network, Mooncake protocol, NIC/RDMA/TCP, or
  remote-store bottleneck.
- Low `get_ms` but high `checksum_ms` with `checksum_enabled=true`: the
  application checksum is CPU-bound; repeat the performance run with
  checksum verification disabled before optimizing the transport.
- Low `get_ms` and checksum but high `cpu_ms`: Python/CPU materialization is
  the bottleneck.
- High `h2d_index_ms`: PCIe, GPU synchronization, or per-layer copy overhead.
- High transition `deserialize_ms`: repeated transition payload decode.
- More records on the three-card machine: duplicated lookup/import across
  ranks or an avoidable repeated request path.

Do not add a persistent native KV payload cache before this measurement. The
next request may not reuse the object, so the first optimization target is the
one-shot import path: duplicate manifest Gets, serial per-layer Gets, checksum
placement, and unnecessary synchronization/copies.

## Transition representation selection (2026-10-21)

`transition-store-total` / `transition-total` now print `representation=`. Three
representations exist, and only the last one deserializes:

- `representation=store-host-staging` (**default**): one
  `batch_get_into_multi_buffers()` writes every state tensor into a single
  cached pinned host arena, then **one** H2D upload runs and each tensor becomes
  an aligned zero-copy view. `HYPIC_PIC_TRANSITION_RECEIVE=host` (or unset)
  selects it. This mirrors the portable part of Full Attention's Store contract
  — a registered destination plus one batch receive — without depending on
  Mooncake accepting GPU destination addresses.
- `representation=store-gpu-direct`: the same single batch call writes straight
  into per-tensor GPU buffers, the closest analogue of Full Attention's
  registered KV cache. Opt in with `HYPIC_PIC_TRANSITION_RECEIVE=gpu`. It falls
  back to host staging when the Store rejects the GPU destination, and stops
  offering GPU addresses after three consecutive rejections.
- `representation=compat-payload` (last resort): the ~38 MB compact payload is
  fetched and decoded. This is the only record that carries a non-zero
  `deserialize_ms`.

All three print `deserialize_ms=0.000` except the last one; the fixed-layout
records always report it as zero because nothing is decoded.

If the fallback keeps appearing, check the reason printed by
`PIC fixed-layout Store transition unavailable ... cause=` before touching
payload encoding. `register_buffer` returning a non-zero status used to abort
the fixed-layout path; it is now best-effort, matching Full Attention, because
Mooncake still copies into an unregistered destination. A single warning
(`Mooncake register_buffer failed for PIC buffer role=...`) marks that state.

The compact payload itself is now wire version 4: every tensor offset is padded
to 16 bytes so the receiver performs **one** H2D upload of the whole data region
and builds zero-copy aligned views. Version 3 payloads remain readable through
the previous per-tensor path. The new record

```text
[PIC-PROFILE] transition-deserialize wire_version=4 tensors=... data_bytes=...
              single_h2d=True h2d_ms=...
```

separates the single upload from the tensor-view construction. Compare
`transition-deserialize.h2d_ms` against the old `transition-total.deserialize_ms`;
`single_h2d=False` means a version-3 payload was received.
