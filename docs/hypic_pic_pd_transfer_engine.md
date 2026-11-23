# PIC P/D with Mooncake TransferEngine

This is the request-handoff path for hybrid (attention + GDN/Mamba) models. It is intentionally separate
from PIC's Mooncake Store segment cache:

- **PIC segment reuse** keeps using the Store and its segment/transition semantics.
- **P/D request handoff** uses Mooncake **TransferEngine for every payload**: attention KV, recurrent GDN
  state, and convolution state. No Store staging path is used by P/D.

## Recovery protocol

A normal Mooncake P/D request lets D recompute the last prompt token to obtain logits. Repeating that token
would also advance GDN state a second time, so a hybrid request uses a token-exact relay:

1. P prefills the original `N` prompt tokens and returns its first sampled token.
2. P sends the KV for those `N` tokens and the live GDN/convolution state after exactly those tokens
   (`h_N`).
3. D receives `prompt = P_prompt_token_ids + [P_first_token]`.
4. Only the original `N` tokens count as externally computed (`remote_prefill_tokens=N`). D forwards just
   the relayed token, starting from `h_N`, and samples the following token.

Thus D never recomputes the original prompt and never advances the same GDN token twice. The proxy also
emits P's first token as the first output chunk, because D consumes it as input and would otherwise start
its response at the second generated token.

For chat requests the proxy uses the internal `prompt_token_ids` request field, which bypasses the chat
template. Completion requests use their existing integer-prompt form. No detokenize/retokenize round trip
is allowed.

## Transfer metadata

`MooncakeXferMetadata` carries, in deterministic group/layer/state order:

- base addresses and per-block byte strides;
- page shapes, page strides and dtypes;
- group ids and Mamba block sizes;
- natural payload bytes.

`build_state_descriptors` computes each source/destination pointer as
`base + block_id * block_stride`, but copies only the natural bytes. Page padding is not payload.

A mismatch in layer order, shape, dtype, strides, or block size aborts the transfer. Non-contiguous state
pages are rejected at registration. Attention block IDs never enter the Mamba descriptor arithmetic.

## Reading the logs

These lines answer "did the state actually travel through TransferEngine":

```
PIC-PD state request=... prompt_tokens=7453 pages=36 bytes=19537920   (prefill side)
PIC-PD relay request=... external_tokens=7453 prompt_tokens=7454       (decode side)
KV Transfer metrics: Num successful transfers=9 ... Num failed transfers=0
```

`pages` counts transferred state pages: one convolution page and one recurrent page per GDN layer, so
36 pages means 18 layers were covered.

Two lines are **not** about P/D and are expected in a disaggregated run:

| Line | Meaning |
|---|---|
| `PIC Stage .../10-A capability check: capability=False reasons=['hybrid_state_restore_unavailable']` | PIC **segment reuse** is off. It needs a PIC cache pool and the Store; a P/D run disables PIC, so this is expected. |
| `PIC Mooncake store initialization failed ... MOONCAKE_CONFIG_PATH is not set` | Should no longer appear: the Store now initializes only when PIC is enabled. If it does appear, PIC was enabled and needs a config. |

P/D therefore needs **no** `MOONCAKE_CONFIG_PATH`, no `mooncake_config.json` and no `mooncake_master`, because
the Store is not on its data path. If you also want PIC segment reuse in the same deployment, enable PIC and
point `MOONCAKE_CONFIG_PATH` at a generated config (as `tools/pic_run.sh` does).

## Reading probe failures

`tests/v1/pic/run_pd_transfer_probe.py` bypasses system HTTP proxies on purpose: images here export
`http_proxy`, and a proxied `127.0.0.1` request comes back as a 403 HTML page instead of JSON. The launcher
probes `/health` first and `/v1/models` second, because the proxy implements only `/health`.

The probe requests `stream=false`. The proxy returns one JSON completion in that mode, merging the
prefiller's first token into its text and token IDs; streaming clients receive SSE instead. The probe also
accepts `data:` frames, so it remains compatible with a streaming-only proxy and with streaming vLLM
responses. This removes the earlier `JSONDecodeError` at char 0 caused by parsing SSE text as one JSON
document.

Shape C uses **chat with thinking disabled** by default, which is its semantic acceptance path. Use
`--completions` only for transport debugging; that raw endpoint does not disable thinking and is not a
valid `dog` semantic acceptance criterion. The relay's internal `prompt_token_ids` field belongs to
`ChatCompletionRequest` (and the batch request model), so P can accept the chat relay as well as the
completion relay.

The launcher leaves P, D and the proxy running after it exits, including on FAIL, so a failing request can
be reproduced and the live logs read. It prints their URLs and the cleanup command. Set
`PIC_PD_CLEANUP_ON_EXIT=1` to tear everything down on exit instead.

## Reuse probes

P/D success and cache reuse are separate questions. `D external 100%` means D consumed the KV produced by P
for this request; it does not prove P reused an older cache. The launcher prints that distinction on PASS.

Two independent probes measure the other paths:

```bash
# Local HBM prefix cache: exit 0 when prefix_cache_hits grows, 1 otherwise.
PREFILL_URL=http://127.0.0.1:8010 bash tools/pic_prefill_reuse_probe.sh

# Full experiment: restart only P with prefix caching enabled, then run that probe.
bash tools/pic_prefill_reuse_e2e.sh

# Or append that independent experiment to the P/D run without changing its verdict.
PIC_PD_CHECK_PREFILL_REUSE=1 bash tools/test_pd_transfer_engine.sh

# PIC segment reuse: skipped with exit 2 unless PIC and its Store are configured.
PIC_ENABLED=1 MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json \
  bash tools/pic_segment_reuse_probe.sh
```

`pic_prefill_reuse_probe.sh` compares `vllm:prefix_cache_queries/hits` before and after repeated identical
requests. `pic_segment_reuse_probe.sh` compares `lookup_hits`, `matched_segments`, `reused_tokens` and
`external_kv_imported` in the Prefill log. Both bypass system HTTP proxies and accept `PREFILL_URL`,
`PROMPT_FILE`, `MAX_TOKENS`, `REPEAT` and `JSON_OUT`.

## Store-backed reuse in a P/D deployment

`tools/pic_store_reuse_e2e.sh` starts a fresh Mooncake master, P and D with **MultiConnector**
(`MooncakeConnector` for P/D plus `MooncakeStoreConnector` for the shared pool), and the relay proxy. It then
sends the same prompt twice:

1. the first request publishes KV to the Store and completes the P/D handoff;
2. the second request must hit the Store on **P during prefill**, while D still receives the handoff.

The script requires Store evidence on P (`external_prefix_cache_hits` or `kvpool hit tokens`) and the expected
semantic word in the second response (`dog` by default, using chat with thinking disabled). It deliberately
does **not** require identical token sequences, because reuse is not assumed to be bit-exact. The Store path
is measured through `external_prefix_cache_*`, not the local `prefix_cache_*` counters. For hybrid Mamba
models, the Store sub-connector must also declare `mamba_cache_mode: align`, and P runs with prefix caching
enabled so reusable blocks are produced.

## PIC custom Store reuse in a P/D deployment

`tools/pic_pd_pic_reuse_e2e.sh` uses the existing PIC custom Store, not `MooncakeStoreConnector`.
`MooncakeConnector` is used only for the current-request P/D handoff. P runs with
`--pic-enable --pic-max-cache-bytes 4294967296 --pic-mooncake`; D stays PIC-disabled. By default it publishes
**publish_a, publish_b and publish_c in order**, waits for each native-KV and hybrid-state publication to be
confirmed in the Store, and then reuses `reuse.txt`. The second request must increase P's `lookup_hits`,
`matched_segments` and `reused_tokens`, and the second chat response must contain `dog` with thinking disabled.
`PUBLISH_FILE` can optionally replace the three-publish sequence with one file. Exact token equality is not required.

For a controlled failure reproduction, `PIC_MINIMAL=1` starts only P and sends two identical
`/v1/completions` requests without D or the proxy. It still starts P with `MooncakeConnector`; use
`PIC_MINIMAL=1 PIC_PURE=1` for a true colocated-PIC control run with no P/D connector at all. This starts
A and B with PIC, publishes on A and reuses on B, matching `pic_run.sh`. The script exports `PIC_DEBUG`, `HYPIC_PIC_DEBUG`,
`CUDA_LAUNCH_BLOCKING` and `TORCH_USE_CUDA_DSA` when supplied and prints CUDA/assert context on failure.
This branch keeps the **native HND layout on both P and D**, matching the standard Mooncake P/D path.
The HND gather fix indexes token slots without putting offsets on the head axis, and publish resolves the
K/V and block axes from the actual cache shape. There is no `preserve_backend_layout` or NHD-specific
transport decoupling. PIC's custom Store and the P/D handoff still operate independently: PIC performs
reuse and prefill on P, then the standard HND block/state transfer runs after the request is complete.

## Two-P isolation before full P/D

`tools/pic_two_p_reuse_e2e.sh` starts **two prefill-role services**, both with PIC custom Store and
`MooncakeConnector(kv_producer)`, but **no D and no proxy**. P1 publishes `publish_a/b/c`; after each
Store publication is confirmed, P2 reuses `reuse.txt`. Acceptance requires P2 `lookup_hits`,
`matched_segments`, `reused_tokens` and `external_kv_imported` to increase, transition application, no
fallback, and `dog` in the answer. This isolates producer-connector initialization from the full P/D handoff:
if this passes but full P/D fails, investigate the consumer/proxy path; if it fails, investigate the producer
side before touching the full P/D configuration.

For a native-crash bisection, `PIC_DISABLE_PD_REGISTRATION=1 bash tools/pic_two_p_reuse_e2e.sh` skips
`batch_register_memory` while keeping TransferEngine initialized. This is **diagnostic only** and not a real
P/D configuration. If the crash disappears, the conflict is in KV registration/address handling; if it
persists, it is in the coexistence of the TransferEngine and PIC Store clients themselves.
`PIC_DISABLE_TRANSFER_ENGINE=1` additionally skips `TransferEngine.initialize()`; use it only for the
next bisection step, never for a real P/D run. If publish/reuse then recovers, the conflict is in native
engine initialization rather than KV registration, layout or the PIC algorithm.

## Current boundaries

- P and D must use the **same TP size and the same state layout**. Heterogeneous TP is rejected rather than
  guessed.
- PP=1 is inherited from the native Mooncake connector.
- First validation should disable speculative decoding and local prefix reuse on D.
- `remote_prefill_tokens` is optional metadata for native clients; without it the original whole-prompt
  behavior is unchanged.
- The existing `--relay-first-token` proxy flag is opt-in so the ordinary attention-only proxy behavior is
  unchanged.

## Launch sketch

```bash
# P
vllm serve "$MODEL" --port 8010 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# D
vllm serve "$MODEL" --port 8020 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'

# Proxy: required for hybrid GDN/Mamba. It ships with this patch under tools/
# because a pip-installed vLLM has no examples/ directory. Use python3: these
# images have no `python` alias.
python3 tools/pic_pd_proxy.py \
  --prefill http://127.0.0.1:8010 --decode http://127.0.0.1:8020 \
  --relay-first-token --port 8000
```

Set `mooncake_protocol` in `kv_connector_extra_config` to `rdma` when the machines support it. `tcp` remains
a valid development transport, but it is not zero-copy. For the same launch sequence with readiness checks
and logs, use `PIC_MODEL=/path/to/model bash tools/test_pd_transfer_engine.sh`; it starts P, D and the
relay proxy, then runs the existing `run_pic_concurrency_probe.py` publish/reuse traffic and the shape-C
semantic/transport checks. Optional inputs are `PIC_TP`, `PIC_CUDA_P`, `PIC_CUDA_D`, `MOONCAKE_PROTOCOL`,
`PIC_PD_CONCURRENCY`, `PIC_PD_REQUESTS`, `PIC_PD_WARMUP`, and `PIC_PD_BASELINE_URL`. When the last one is
set, the probe additionally compares P/D and single-instance token IDs exactly, including the first token.

## Acceptance

1. P/D and a single instance produce identical token IDs for the same deterministic request.
2. D schedules only the relayed token, not the original prompt.
3. A GDN-sensitive multi-segment prompt remains correct; attention-only correctness is not enough.
4. Mamba state descriptors use the exact `h_N` page and natural byte counts.
5. Abort, timeout and metadata mismatch do not release a partially transferred request as ready.
6. After that, repeat at TP=2 with rank-for-rank state transfer before considering heterogeneous TP.
