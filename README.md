# HYPIC PIC for vLLM

An experimental research implementation of HYPIC PIC on top of vLLM 0.22.1.
The project targets non-prefix segment reuse for hybrid-attention language
models while keeping ordinary requests on the native vLLM execution path.

中文：这是基于 vLLM 0.22.1 的 HYPIC PIC 研究实现，面向混合注意力模型提供
非前缀 segment 复用，同时保持普通请求继续使用 vLLM 原生执行路径。

## Relationship to the original HYPIC implementation

The design is informed by the original
[HYPIC implementation for SGLang](https://github.com/redai-studio/HYPIC).
This repository is an independent vLLM implementation: the PIC ideas are
adapted to vLLM's scheduler, GPU worker, native KV pool, block table and
attention metadata rather than reusing the SGLang runtime code directly.

中文：本项目的设计参考了
[SGLang 版本的 HYPIC 实现](https://github.com/redai-studio/HYPIC)。
这里不是直接复制 SGLang 代码，而是将 HYPIC 的 PIC 思路重新适配到 vLLM 的
scheduler、GPU worker、native KV pool、block table 和 attention metadata，形成
独立的 vLLM 版本实现。

## Highlights

- Hybrid GDN/Mamba recurrent-state and convolution-tail transitions.
- Native vLLM KV-slot mapping for full-attention layers.
- Seam-window recomputation and single-request skip/recompute.
- Mixed PIC and ordinary-request execution with request-local isolation.
- Persistent private KV materialization with lease/ref-count cleanup.
- Fail-closed fallback for unsupported or unsafe request conditions.
- External native-KV provider boundary for future Mooncake integration.

PIC ultimately uses vLLM native KV slots and attention metadata, so the normal
attention backend does not need a separate external KV view.

## Validation snapshot

The primary validation target is Qwen3.5-2B hybrid-attention, text-only mode,
single GPU and eager execution.

| Check | Result |
|---|---:|
| PIC unit tests | 75 passed |
| Correctness regression | passed; max log-probability difference about `2.12e-5` |
| Prefill speedup, seam sink 0 | about `1.38x` |
| Prefill speedup, seam sink 4 | about `1.37x` |
| Reused-token ratio | about `63%` |

These measurements represent a correctness-oriented prototype. Latency depends
on the model, GPU, batch shape, eager-mode overhead, seam size and cache state.

## Quick start

The implementation was validated from the official CUDA 12.9 image:

```text
vllm/vllm-openai:v0.22.1-cu129
```

Start the server with PIC controls enabled:

```bash
vllm serve /path/to/Qwen3.5-2B \
  --served-model-name qwen3.5-2b \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --pic-enable \
  --pic-auto-enable \
  --pic-zero-copy \
  --pic-single-request \
  --pic-capture-live \
  --pic-restore-live \
  --pic-max-cache-bytes 2147483648 \
  --pic-debug
```

Example request metadata:

```json
{
  "model": "qwen3.5-2b",
  "prompt": "left <<PIC_SEP>> reusable segment <<PIC_SEP>> right",
  "max_tokens": 4,
  "temperature": 0,
  "vllm_xargs": {
    "pic_enabled": true,
    "pic_separator": "<<PIC_SEP>>",
    "pic_seam_sink": 4,
    "pic_mode": "transition_rope_recompute"
  }
}
```

## Code map

- `vllm/v1/pic/`: segmentation, cache, transitions, native KV, runtime,
  lifecycle and external-provider interfaces.
- `vllm/v1/worker/gpu_model_runner.py`: capture, restore, skip/recompute and
  packed execution integration.
- `vllm/v1/worker/gpu_input_batch.py`: request-local native slot and batch
  metadata handling.
- `vllm/v1/core/sched/scheduler.py`: scheduler-side lookup and metadata flow.
- `tests/v1/pic/`: PIC unit and runtime tests.

完整的中英双语说明、限制和更多使用细节见：[HYPIC.md](HYPIC.md)。

## Scope and limitations

- Current validation focuses on single-GPU, eager, text-only hybrid models.
- Unsafe state layouts, cursor mappings or batch shapes fall back per request.
- `--pic-mooncake` is an external native-KV provider boundary; it does not
  bundle a Mooncake server, RDMA transport or remote provider.
- Remote KV import requires a separately registered provider implementation.
- This is an independent research branch, not an official vLLM feature and
  not a submission to the vLLM upstream repository.

## TODO

- Profile Python/Torch metadata, transition application, KV materialization and
  kernel-launch overhead on representative hybrid models.
- Implement and benchmark custom Triton/CUDA kernels for transition application,
  native-KV gather/scatter, slot mapping and seam-window execution.
- Fuse transition, state update and KV metadata preparation where profiling
  shows measurable benefit, while preserving the native vLLM attention path.
- Explore CUDA Graph compatibility for steady-state decode and packed mixed
  batches.
- Add a real Mooncake provider and run complete connectivity validation:
  provider registration, external native-KV import, cross-process transfer,
  lease/ref-count lifecycle, remote KV correctness, RDMA/Transfer Engine
  behavior and safe fallback when the provider is unavailable.
- Re-run correctness, mixed-batch isolation, lifecycle and acceleration
  benchmarks after each kernel or provider optimization.

## License and upstream

The codebase is derived from vLLM 0.22.1 and retains the upstream license and
notices. See the upstream project at
[vllm-project/vllm](https://github.com/vllm-project/vllm).
