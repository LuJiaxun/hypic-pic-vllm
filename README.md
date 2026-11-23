# Hybrid-Attention PIC for vLLM

Position-independent caching (PIC) for hybrid attention + recurrent models on
top of **vLLM 0.22.1**.

Prefix caching requires a cached segment to sit at the position it was cached
at. PIC removes that requirement: a prompt splits at `<<PIC_SEP>>`, and each
segment's **KV** and its **GDN recurrent state** are reused across requests even
when the segment lands at a different offset. Ordinary requests keep running on
the native vLLM path, untouched.

## Relationship to the original HYPIC implementation

The design is informed by the
[original HYPIC implementation for SGLang](https://github.com/redai-studio/HYPIC).
This is an independent vLLM implementation: the PIC ideas are adapted to vLLM's
scheduler, GPU worker, native KV pool, block table and attention metadata rather
than reusing the SGLang runtime.

这是基于 vLLM 0.22.1 的 HYPIC PIC 实现，面向混合注意力（注意力 + GDN/Mamba
递推）模型提供**位置无关**的 segment 复用；普通请求仍走 vLLM 原生路径。
设计参考 [SGLang 版 HYPIC](https://github.com/redai-studio/HYPIC)，但代码是
重新适配到 vLLM 的 scheduler / worker / native KV pool / block table /
attention metadata，不是移植。

## How it works

Two mechanisms, one per half of a hybrid model:

- **Attention KV.** Cached keys are re-rotated to the new positions, and the
  segment's blocks are mapped onto the new request's native KV slots. The
  numbering is defined once and shared by the exporter and the importer
  (`vllm/v1/pic/kv_slots.py`), so the attention kernel never sees a second
  cache layout.
- **Recurrent state.** A recurrent layer's output state is an affine function of
  its input state, so one segment reduces to **one matrix plus a bias**
  (`state @ M + c`). The matrix comes from the model's own prefill kernel
  (`chunk_gated_delta_rule_fwd_h`); reuse is a single matmul. No custom kernel
  and no hand-written recurrence.

Segments travel between processes through **Mooncake**, published under a key
derived from their tokens. The import overlaps the request's wait instead of
blocking the step it was scheduled with, and a failed validation never
publishes — a wrong transition must not spread.

## Highlights

- Non-prefix segment reuse: a segment cached at one offset is reused at another.
- Native vLLM KV slots and attention metadata; no external KV view for the kernel.
- GDN recurrent state carried as a linear operator produced by the model's own kernel.
- Publication span == reuse window (`[seam, length-1)`), so the two can never drift.
- Mooncake-backed store with async import while the request waits.
- Fail-closed everywhere: no recorded backend means an exception, not a silent fallback.

## Validation snapshot

Measured on 2× RTX 4090 (24 GB), Qwen3.5-2B, Mooncake 0.3.10, eager, one server
publishes and one reuses. Shape B (`system <<PIC_SEP>> doc <<PIC_SEP>> query`),
document of 6976 tokens, five rounds with the first discarded.

| Metric (c=4) | PIC on | PIC off |
| --- | ---: | ---: |
| Throughput (req/s) | **5.90** | 4.40 |
| Wall for 20 requests (ms) | **3387** | 4578 |
| p50 latency (ms) | 617 | 959 |
| p95 latency (ms) | 878 | — |

**+34% throughput** at c=4. Cross-check: `4578/3387 = 1.35` and `5.90/4.40 =
1.34`, so wall and throughput agree.

| Correctness | Result |
| --- | --- |
| Shapes a / b / c, nine criteria each | all pass |
| Semantic answer (shape c) | the word only the reused documents contain |
| `fallback_reasons` | `{}` on every request |
| Reuse alignment | 49/49 exact ranges |

PIC trades single-request latency for throughput: at c=1 the p50 is 367 ms
against 294 ms without PIC, because a lone request pays the import and the
matmul without anything to amortise them over. **PIC is a throughput feature.**

## Quick start

Start a Mooncake master, point the servers at it, then two servers — one that
publishes, one that reuses:

```bash
# a store, and a config file that names it
mooncake_master --rpc_port 50088 --client_ttl 3600 &
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json

# publisher on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3.5-2B \
  --served-model-name qwen3.5-2b --port 8000 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.5 --enforce-eager \
  --no-enable-prefix-caching \
  --pic-enable --pic-max-cache-bytes 4294967296 --pic-min-segment-tokens 512
```

The same command on GPU 1 with `--port 8001` is the reusing server. **One switch
is enough**: the bridge, the per-range loop, capture/restore, the external
store and the async import all follow `--pic-enable`. The two numbers are
deployment choices and are stated out loud.

Publish by sending the segment once; reuse by sending a prompt that contains it:

```bash
# publish: the document enclosed by separators
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen3.5-2b\",\"prompt\":$(jq -Rs . < publish.txt),\"max_tokens\":1}"

# reuse: the same document, now behind a system prompt, with a question
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen3.5-2b\",\"messages\":[{\"role\":\"user\",\"content\":$(jq -Rs . < reuse.txt)}],
       \"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
```

## Test commands

Everything the numbers above came from is in this repository; no external script
is needed.

```bash
# three prompt shapes, nine criteria each, plus a semantic check on the answer
bash tools/test_all.sh

# shape B, five rounds, median of the last four
bash tools/run_pic_test.sh perf

# the same fixtures, the same hardware, no PIC flags at all
bash tools/run_pic_test.sh off
```

Each shape's script starts its own servers, publishes, reuses and judges, so a
failure names the shape and the criterion:

```text
PASS A publish-matrix     a transition matrix was published
PASS A native-kv          native KV blocks were published
PASS B lookup-hit         the segment was found
PASS B transition-applied the transition was applied
PASS B executed           the PIC path ran
```

## Code map

- `vllm/v1/pic/segmenter.py` — split a prompt at the separator, hash the segments.
- `vllm/v1/pic/cache.py`, `runtime.py`, `single_request.py` — plan reuse per segment.
- `vllm/v1/pic/state.py`, `gdn_transition.py` — the recurrent state as an operator.
- `vllm/v1/pic/native_kv.py`, `kv_slots.py` — rerotation and KV slot mapping.
- `vllm/v1/pic/mooncake_kv.py`, `async_import.py` — the store and the import.
- `vllm/v1/worker/gpu_model_runner.py` — the range loop: reuse, recompute, reuse.
- `vllm/v1/core/sched/scheduler.py` — lookup, reserved blocks, parked imports.
- `tests/v1/pic/` — unit tests and the committed fixtures (three shapes, 17 files).
- `docs/` — design notes, verification and measurements.

Design notes and the rules a later change has to keep:
[docs/hypic_pic_patch_summary.md](docs/hypic_pic_patch_summary.md).

## Scope and limitations

- Single GPU, eager, text-only hybrid models. No CUDA Graph, deliberately.
- **Throughput, not latency**: c=1 is slower than without PIC.
- The c=4 tail (p95 ≈ 878 ms) comes from imports being serialised per store
  connection. Raising the worker count made p50 worse in measurement, so it
  stays at one.
- Documents above ~8000 tokens exceed 24 GB at c=4.
- A failed capture or validation does not publish; that request falls back to a
  full prefill, which is correct but unreused.
- This is an independent research branch, not an official vLLM feature, and it
  is not proposed to the vLLM upstream repository.

## TODO

- Overlap import with forward instead of parking the request.
- Fuse the transition matmul into the state update where profiling shows it pays.
- A real RDMA transport for the store; today it is a TCP object store.
- CUDA Graph for the recompute ranges, once the range loop is stable enough.
- Multi-request batching for PIC ranges. Two prototypes existed and were removed;
  the single-request path is the one that is measured.

## License and upstream

The codebase is derived from vLLM 0.22.1 and retains the upstream license and
notices. See [vllm-project/vllm](https://github.com/vllm-project/vllm).
