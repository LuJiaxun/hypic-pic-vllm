# HYPIC PIC for vLLM

This repository contains a personal research implementation of HYPIC PIC on top
of vLLM 0.22.1. It explores non-prefix segment reuse for hybrid-attention
language models while keeping ordinary vLLM requests on the native execution
path.

The implementation is based on the official
`vllm/vllm-openai:v0.22.1-cu129` container and the vLLM baseline commit
`0decac0d96c42b49572498019f0a0e3600f50398`.

## What is implemented

- PIC segment parsing and non-prefix lookup.
- Hybrid recurrent-state capture and transition application for GDN/Mamba
  layers, including convolution tail state.
- Native vLLM KV-slot mapping for full-attention layers.
- Seam-window recomputation at reused-segment boundaries.
- Single-request skip/recompute execution.
- Mixed PIC and ordinary-request batch isolation.
- Persistent private KV materialization with lease/ref-count cleanup.
- Request-local fallback for unsupported or unsafe conditions.
- Metrics for lookup, reuse, recompute, materialization, fallback, leases and
  pool usage.
- An external native-KV provider boundary for future Mooncake integration.

The attention path is intentionally expressed through vLLM's native KV pool
and attention metadata. PIC requests therefore remain compatible with the
existing attention backend instead of requiring a separate external attention
view.

## Current validation scope

The primary validation model is Qwen3.5-2B in hybrid-attention, text-only mode,
on a single GPU with eager execution.

The latest locally validated Stage 10-C-6 results include:

| Check | Result |
|---|---:|
| PIC unit tests | 75 passed |
| Stage 9-A-3 correctness regression | passed; max log-probability difference about `2.12e-5` |
| Prefill speedup, seam sink 0 | about `1.38x` |
| Prefill speedup, seam sink 4 | about `1.37x` |
| Reused-token ratio | about `63%` |
| Repeated-request leak test | 30/30 passed |
| HTTP 400/500 or service errors | 0 |

These are correctness-prototype measurements, not a claim of peak production
performance. Latency depends on model, GPU, batch shape, eager-mode overhead,
seam size and cache warm-up state.

## Quick start

Start from the official vLLM CUDA 12.9 image:

```bash
docker run --gpus all --rm -it \
  -p 8000:8000 \
  -v /path/to/this/repository:/workspace/hypic-pic-vllm \
  vllm/vllm-openai:v0.22.1-cu129 bash
```

Inside the container, apply or copy the repository code according to the
container setup, then start the server with the PIC controls enabled:

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

PIC metadata is passed per request through `vllm_xargs`, for example:

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

## Important limitations

- The validated path is single-GPU, eager execution and text-only hybrid
  models. Multimodal requests are not the current validation target.
- Unsupported state layouts, unsafe cursor mappings and incompatible batch
  shapes fail closed at request level and continue with ordinary vLLM
  execution.
- `--pic-mooncake` currently provides an external native-KV provider boundary;
  it does not bundle a Mooncake server, RDMA transport or remote provider.
  Remote KV import requires a separately registered provider implementation.
- The reported speedups include Python/Torch metadata and materialization
  overhead. Kernel fusion and CUDA Graph optimization are separate future work.

## Code map

- `vllm/v1/pic/`: PIC segmentation, cache, transition, native-KV, runtime,
  lifecycle and external-provider interfaces.
- `vllm/v1/worker/gpu_model_runner.py`: worker-side capture, restore,
  skip/recompute and packed execution integration.
- `vllm/v1/worker/gpu_input_batch.py`: request-local native slot and batch
  metadata handling.
- `vllm/v1/core/sched/scheduler.py`: scheduler-side PIC lookup and execution
  metadata propagation.
- `tests/v1/pic/`: unit and regression tests for the PIC control and runtime
  paths.

## Positioning

This is an independent research branch built on vLLM. It is not an official
vLLM feature and is not submitted to the vLLM upstream repository. The code is
intended to make the HYPIC PIC execution path reproducible and inspectable;
please treat the current implementation as experimental.

---

# 中文说明

本仓库是在 vLLM 0.22.1 基础上实现的 HYPIC PIC 研究版本，目标是为混合注意力
模型提供非前缀 segment 复用能力，同时保持普通 vLLM 请求继续使用原生执行路径。

## 已实现能力

- PIC segment 切分、非前缀 lookup 和缓存管理；
- GDN/Mamba recurrent state、conv tail 的捕获与 transition 应用；
- full-attention 层使用 vLLM 原生 KV pool 和 native KV slot；
- seam window 重计算；
- 单请求 skip/recompute；
- PIC 请求与普通请求的混合 batch 隔离；
- private KV 持久化、lease/ref-count 和 eviction；
- 不安全或不支持场景下的 request-local fallback；
- lookup、复用、重计算、materialization、fallback、lease 和 pool 指标；
- 面向后续 Mooncake 接入的 external native-KV provider 接口。

核心设计是让 PIC 最终使用 vLLM 原生 KV slot 和 attention metadata。这样 attention
backend 不需要区分 PIC 请求和普通请求，也不需要维护一套独立的外部 attention view。

## 当前验证范围

当前主要在单卡、eager 模式下验证 Qwen3.5-2B hybrid-attention 纯文本请求。

Stage 10-C-6 的本地验证结果包括：

| 验证项 | 结果 |
|---|---:|
| PIC 单元测试 | 75 passed |
| Stage 9-A-3 正确性回归 | passed，最大 log-probability 差异约 `2.12e-5` |
| prefill 加速，seam sink 0 | 约 `1.38x` |
| prefill 加速，seam sink 4 | 约 `1.37x` |
| token 复用比例 | 约 `63%` |
| 重复请求泄漏测试 | 30/30 通过 |
| HTTP 400/500 或服务错误 | 0 |

这些数据是正确性原型的实测结果，不代表生产环境的理论峰值性能。实际延迟会受到
模型、GPU、batch 形状、eager 开销、seam 大小和缓存预热状态影响。

## 快速启动

基础镜像：

```bash
vllm/vllm-openai:v0.22.1-cu129
```

服务启动示例：

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

请求通过 `vllm_xargs` 传递 PIC 参数：

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

## 当前限制

- 当前验证范围是单卡、eager、纯文本 hybrid 模型；多模态请求不是本阶段目标；
- state layout、cursor mapping 或 batch shape 不安全时，会按 request 回退到普通 vLLM；
- `--pic-mooncake` 目前只是 external native-KV provider 接口，不包含 Mooncake
  server、RDMA transport 或远程 provider；
- 远程 KV import 需要额外注册 provider 实现；
- 当前性能包含 Python/Torch metadata 和 materialization 开销，kernel fusion 与
  CUDA Graph 属于后续优化方向。

## 代码路径

- `vllm/v1/pic/`：PIC 切分、缓存、transition、native KV、runtime、生命周期和
  external provider 接口；
- `vllm/v1/worker/gpu_model_runner.py`：worker 侧 capture、restore、skip/recompute
  和 packed execution；
- `vllm/v1/worker/gpu_input_batch.py`：request-local native slot 与 batch metadata；
- `vllm/v1/core/sched/scheduler.py`：scheduler 侧 PIC lookup 和执行元数据传递；
- `tests/v1/pic/`：PIC 控制面和 runtime 路径测试。

## 项目定位

这是基于 vLLM 的个人研究分支，不是官方 vLLM 功能，也不会直接提交到 vLLM
upstream。代码主要用于复现和研究 HYPIC PIC 的执行路径，当前版本仍属于实验性实现。
