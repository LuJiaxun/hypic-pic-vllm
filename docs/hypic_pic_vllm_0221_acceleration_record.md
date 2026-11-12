# HYPIC PIC vLLM 加速比记录

本文档记录当前 HYPIC PIC 在 vLLM 0.22.1-cu129 上的理论上限、已测端到端
结果和阶段边界。数据以服务器 GPU 实测报告为准，不把“跳过的 token 比例”
直接当作端到端延迟加速比。

## 1. 当前结论

在已经完成验证的 Stage 9-A/9-B/10-A/10-C 中：

- 理想 token-level 上限约为 **1.84–1.85×**；
- Stage 9-A/9-B 单请求实测约为 **1.34–1.42×**；
- Stage 10-A 在 4 并发、纯 prefill 条件下，相比 Stage 9-B 实测约为
  **1.078×**，即延迟下降约 **7.3%**；
- Stage 10-A 在 decode 负载下基本持平，没有观察到明显退化；
- Stage 10-C-4 已完成服务器验证：稳态 PIC decode 约 74.90 ms，普通 baseline
  108.63 ms，约 **1.45×**（延迟降低约 32%）；prefill 为 seam=0 **1.33×**、
  seam=4 **1.40×**。

**跨机（Mooncake TCP，无 RDMA）当前基线见第 21 节：PIC p50 271.1 ms，Full 193.0 ms，
0.718×。所有后续优化的比较都以第 21 节的配置和判定窗口为准。**

## 2. 理论加速比

### 2.1 Token-level 理想上限

一次已记录的 Stage 9-A 场景中：

```text
完整 forward token：约 2367
PIC forward token：约 1280–1284
跳过 token：约 1083–1087
跳过比例：约 45.8%–45.9%
```

在假设每个 forward token 成本完全相同、transition、seam、KV materialization、
scheduler、通信和 Python/Torch metadata 全部免费时，理想上限为：

```text
ideal speedup = full_tokens / pic_tokens
               ≈ 2367 / 1280–1284
               ≈ 1.84–1.85×
```

这只是计算量上限，不是实际端到端保证。实际系统还要支付：

- seam window forward；
- transition/state restore 或 apply；
- private KV materialization、gather/scatter 和 rerotation；
- scheduler、batch packing 和 request isolation；
- Python/Torch metadata 构造；
- GPU kernel launch、显存访问和同步开销。

因此，理想上限不能直接用来判断实现是否达到生产性能。

## 3. 已完成阶段的实测结果

### 3.1 Stage 9-A/9-B：单请求 PIC runtime

已验证的 Qwen3.5-2B、单卡、eager 场景中，PIC 请求相对完整重算的端到端
加速约为：

| 场景 | 实测加速 |
|---|---:|
| seam_sink=0 | 约 1.34× |
| seam_sink=4 | 约 1.42× |

这低于 1.84–1.85× 的 token-level 理想上限，主要原因是 transition、private KV
materialization 和 eager/Python 调度开销没有被跳过。

### 3.2 Stage 10-A：packed mixed batch 对 Stage 9-B

测试条件：Qwen3.5-2B、RTX 4090、单卡、eager、4 并发、复用段约 63%。

| 负载 | Stage 9-B | Stage 10-A | Stage 10-A / Stage 9-B |
|---|---:|---:|---:|
| 单请求，max_tokens=1 | 78.79 | 80.65 | 0.977× |
| 单请求，max_tokens=4 | 103.72 | 105.50 | 0.983× |
| 4 并发，max_tokens=1 | 177.53 | 164.65 | **1.078×** |
| 4 并发，max_tokens=4 | 270.05 | 272.26 | 0.992× |

解释：

- 4 并发纯 prefill 场景延迟从 177.53 降到 164.65，提升约 **7.3%**；
- decode 场景约 0.992×，可视为基本持平；
- 单请求场景略慢，说明 packed mixed-batch 优化主要收益来自并发调度，
  不是单请求 kernel 加速；
- 该结果符合 Stage 10-A 的 correctness-first、Python/Torch metadata 原型定位。

### 3.3 Stage 10-C：lifecycle fix 的单请求 prefill

服务器报告基于 Stage 10-C lifecycle-fix 提交 `a4bcf9aa32`，并手动补充
`Any` 导入及 `_evict_published_segment()` 的 `return True`。测试条件为
Qwen3.5-2B、单卡、串行单请求、`max_tokens=1`、每组 `n=20`，target
segment 为 1088 tokens，剩余请求为 640 tokens，总 prompt 为 1728 tokens。

| 场景 | mean | median | 相对完整重算加速 |
|---|---:|---:|---:|
| 完整重算 baseline | 91.25 ms | 91.77 ms | 1.00× |
| PIC，`seam_sink=0` | 68.26 ms | 68.13 ms | **1.34×** |
| PIC，`seam_sink=4` | 66.24 ms | 66.23 ms | **1.38×** |

该场景的 token-level 理想比例为 `1728 / 640 = 2.70×`，实际跳过
1088 tokens，skip ratio 为 63%。端到端差距主要来自 transition、private KV
materialization、seam 和固定 scheduler/HTTP 开销。`seam_sink=4` 比 `0` 快约
3%，并且最大延迟更稳定；在该测试条件下可作为默认实验配置。

这组数据证明了 lifecycle fix 版本没有因 snapshot lease/eviction 改动而损失
已有 PIC 加速，同时 30 轮压力测试无 `MemoryError`。但由于服务器测试使用的
是 `a4bcf9aa32` 加手动修复版本，不能直接替代最新累计 patch 的最终验收；
最新 patch 仍需完成同条件回归和性能复测。

后续完整验收报告进一步确认：单测为 `61 passed`，Stage 9-A-3 的
`seam_sink=4` 回归中 `max_logprob_abs_diff=1.87e-05`，HTTP 400/500 和
其他错误均为 0；首次 private KV materialization 20 次，后续复用 20 次，
并且 30 轮压力测试没有 `MemoryError`。因此该版本同时满足正确性、生命周期
稳定性和无明显性能退化三个条件。

### 3.4 Stage 10-C-2：allocation-time public/private slot ownership

Stage 10-C-2（commit `5e7b1dc6a2`）在相同单请求纯 prefill 条件下完成服务端
验收：62 个测试通过，Stage 9-A-3 的 `max_logprob_abs_diff=2.12e-05`，
HTTP 400/500 和其他错误均为 0。服务器观察到 22 次 public KV materialization、
41 次首次 private materialization 和 48 次 private materialization reuse。

| 场景 | Stage 10-C-1 | Stage 10-C-2 | 变化 |
|---|---:|---:|---:|
| `seam_sink=0` 加速比 | 1.34× | **1.40×** | +0.06× |
| `seam_sink=4` 加速比 | 1.38× | **1.38×** | 持平 |
| PIC mean latency，`seam_sink=0` | 68.26 ms | **65.22 ms** | -3.04 ms |

本阶段确认 aligned complete range 可以绑定 public native blocks，例如
`canonical=[0,1088)`、`source=[0,1088)`，同时保持 Stage 9-A-3 正确性和
private materialization reuse。该结果说明 allocation-time ownership 没有引入
性能回退，并在 `seam_sink=0` 场景有小幅收益；它仍不是 Triton/CUDA kernel
优化结果。

### 3.5 Stage 10-C-4：request-bound state 与 decode mapping reuse

Stage 10-C-4（commit `5ca0707881`）已在干净官方 Docker 中完成服务器验收：
68 个测试通过，Stage 9-A-3 回归 `max_logprob_abs_diff=2.12e-05`，
`PIC decode mapping reused=28`，decode mapping fallback 为 0，30 轮 leak test
全部通过，且没有 `MemoryError`、HTTP 400/500 或 traceback。

| 场景 | Stage 10-C-3 | Stage 10-C-4 | 变化 |
|---|---:|---:|---:|
| seam=0，串行 prefill | 1.35× | **1.33×** | -0.02× |
| seam=4，串行 prefill | 1.41× | **1.40×** | -0.01× |

decode 测试中，PIC 首次 cold decode 为 94.37 ms，warmup 后稳定 decode 约
74.90 ms；普通 baseline 为 108.63 ms。因此稳态 decode 加速约为
`108.63 / 74.90 = 1.45×`，即延迟降低约 32%。cold decode 包含首次路径开销，
不作为稳态加速比。强制制造 unsafe mapping/cursor 并验证单请求 fallback 的
可选测试尚未执行。代码结构确认该 fallback 是 request-local：mapping/cursor/state
不一致时仅调用 `_disable_pic_runtime(request_state)` 并设置该 request 的
fallback 标志，不修改其他 request 或全局运行状态；30 轮混合 leak test 也未发现
cross-request contamination。因此该可选异常注入测试不阻塞 Stage 10-C-4 验收。

## 4. 为什么实际值低于理论值

当前实现只减少了命中 segment 的主体 forward，并没有等比例减少所有端到端
成本。尤其是：

1. GDN/Mamba transition 仍需要执行；
2. seam 必须重新计算以适配当前 prefix；
3. full-attention 的 private KV 可能需要 materialize/rerotate；
4. Stage 9-B 使用 request-isolated microbatch，Stage 10-A 虽然改为 packed
   round，仍有 Python 侧组装和校验开销；
5. decode 阶段只有少量 token，metadata 和 launch 固定成本占比更高；
6. 当前没有新增 Triton/CUDA kernel 做 gather、transition 或 metadata fusion。

因此，当前 1.34–1.42× 是单请求 correctness runtime 的结果，1.078× 是
Stage 10-A 在并发纯 prefill 下相对 Stage 9-B 的结果；两者测试基线不同，不能
直接互相替代。

## 5. 当前阶段边界

已完成并有服务器结果：

```text
Stage 9-A：单 request skip/recompute              已验收
Stage 9-B：request-isolated batch runtime         已验收
Stage 10-A：packed mixed-batch prototype          已验收
Stage 10-B：packed attention metadata validation  已验收
```

当前代码状态：

```text
Stage 10-C-1：request native mapping + materialization 幂等控制
              已完成服务器验证
Stage 10-C-2：allocation-time public/private slot ownership
              已完成服务器验证
Stage 10-C-3：已完成服务器验证（30/30 leak test）
Stage 10-C-4：已完成服务器验证（decode mapping reuse）
Stage 10-C-5：已完成服务器验证（metrics、request-level fallback、30/30 leak）
              seam=0 1.40×，seam=4 1.41×
Stage 10-C-6：代码已生成，服务器验证待执行（external native-KV import/lease）
```

特别说明：上面的 Stage 10-C 数据来自服务器报告中的 lifecycle-fix 版本，
而不是最新累计 patch 的最终复验结果。后续应使用最新累计 patch，在相同
prompt、并发度、warmup 和重复次数下复测，确认 1.34–1.38× 结果没有变化。

## 6. 后续性能判断标准

后续每个阶段都应在相同条件下同时记录：

- prefill latency；
- decode latency；
- 单请求和并发吞吐；
- 实际 forward token 数和跳过 token 数；
- PIC hit rate、fallback rate、seam token 数；
- private KV materialization 次数；
- native block、lease 和显存使用量；
- kernel launch 数与 GPU 利用率。

只有在相同请求、并发度、模型、GPU、eager/graph 模式和 max token 配置下对比，
才能把后续结果与本记录中的 Stage 9-B/10-A 基线进行比较。

## 7. 后续 patch README 的强制要求

从下一份 Stage patch 开始，每个 patch 目录中的 `README.md` 都必须包含独立的
“加速比测试”章节，至少写明：

1. 对比基线是哪个 Git commit/Stage，以及是否为普通完整重算、Stage 9-B 或上一
   个 Stage；
2. 模型、GPU、单卡/多卡、eager 或 CUDA Graph、并发度、prompt/segment 构造、
   `max_tokens` 和 warmup 方法；
3. 可直接复制执行的服务启动命令、请求脚本、重复次数和日志提取命令；
4. prefill/decode latency、throughput、forward token 数、跳过 token 比例、
   PIC hit/fallback/seam，以及 speedup 计算公式；
5. 与上一阶段的结果对照表，并说明结果是加速、持平、退化还是尚未测量；
6. correctness、HTTP 错误、显存/lease 泄漏和性能数据分别报告，不能用功能
   通过替代性能结论。

如果本阶段只实现控制面或 correctness 原语，README 仍必须提供“性能不适用”
或“尚未进行服务器性能测试”的明确结论，并保留后续应执行的测试命令模板。

## 8. Stage 10-C-3：统一 lease/ref-count 生命周期（代码已生成）

Stage 10-C-3 的代码提交为 `7bd44df8fb`、生命周期修复提交为
`b7bfd7019e`、测试断言修复提交 `85d6e73217`，累计 patch 归档在：

```text
patches/hypic-pic-vllm-0221-cu129-stage10C-3/
```

本轮新增 worker-side `PICLeaseRegistry`，统一记录 cache-entry、active-request
和 in-flight execution 三类引用；public/private native KV allocation、live
snapshot manager、plan refresh、fallback、request finish 和 shutdown 均接入
对应释放路径。scheduler 进程中的真实 BlockPool lease 仍由 scheduler 单独管理，
没有伪造或直接释放 native block ID。

本轮仍未完成 10-C-5 监控指标和 10-C-6 Mooncake/kernel 优化。Stage 10-C-3
已在干净官方 Docker 中完成服务器验收：30/30 leak test
通过，Stage 9-A-3 `max_logprob_abs_diff=2.12e-05`，无 HTTP 400/500 或
MemoryError。

Stage 10-C-3 的可比性能结果为：

| 配置 | Stage 10-C-2 | Stage 10-C-3 | 变化 |
|---|---:|---:|---:|
| seam=0，串行 prefill | 1.40× | 1.35× | -0.05× |
| seam=4，串行 prefill | 1.38× | 1.41× | +0.03× |

该波动属于测试运行差异，不能宣称 lease 本身带来稳定加速；10-C-3 的主要
收益是生命周期稳定性和无 pool 泄漏。

Stage 10-C-4 已提交代码 `5ca0707881`，增加 request-bound state/decode cursor，
并已完成服务器 correctness、decode mapping reuse、性能和 leak 回归验证。

Stage 10-C-5 已提交代码 `ceadb44dad`，测试断言修复提交为 `db5c8e7251`，增加
worker-local PIC metrics 和 request-level fallback 计数。服务器验收为 71 passed、
metrics snapshot 正常、30/30 leak test，通过同条件性能测试得到 seam=0 1.40×、
seam=4 1.41×。该阶段不新增 CUDA/Triton kernel，主要收益是可观测性与故障隔离。

Stage 10-C-6 已提交代码 `24aac4de8d`，新增可选 external native-KV import/lease
协议和 `--pic-mooncake` 开关；未安装或未注册 provider 时 request-level fallback，
不改变普通路径。真实 Mooncake provider、跨进程 native block import 和性能数据
仍待服务器 Docker 准备后验证。

## 9. Stage 11E：本地 PIC 4k 性能结果（2026-10-15）

### 9.1 测试条件

| 项目 | 条件 |
|---|---|
| 机器 | 61 三卡，Xeon Silver 4210R，3×RTX 4090 |
| 模型 | Qwen3.5-2B |
| 配置 | `--gpu-memory-utilization 0.35`，`--max-model-len 8192`，`--max-num-batched-tokens 8192` |
| 执行模式 | `--enforce-eager`，profiling 全部关闭 |
| 请求 | 本地 PIC，4k 复用：4352 reused + 640 recompute，`max_tokens=1` |
| warmup | publish + 3 次 warmup + 10 次测量 |
| Patch | Stage 11E，含 PIC input preprocessing fast path |

### 9.2 核心结果

| 版本/场景 | Full p50 (ms) | PIC p50 (ms) | Full/PIC | 节省 (ms) |
|---|---:|---:|---:|---:|
| 旧 11E，input processor 约 40 ms | 193.8 | 174.1 | 1.113× | 19.7 |
| 新 11E，input preprocessing fast path | 193.6 | **136.6** | **1.418×** | **57.0** |
| 历史 10-C（参考） | ~91.25 | ~68.26 | ~1.34× | ~23.0 |

在当前严格相同的 4k、`max_tokens=1` 条件下，新 11E 的本地 PIC 达到
`1.418×`，超过历史 10-C 的约 `1.34×`。该结果是端到端 p50，Full/PIC
均为同一 eager 配置；仍需注意历史 10-C 的机器、warmup 和基线条件可能不同。

### 9.3 完整场景对照

| 场景 | Full p50 (ms) | PIC p50 (ms) | Full/PIC | 结论 |
|---|---:|---:|---:|---|
| 1k，`max_tokens=1` | 79.9 | 97.0 | 0.823× | PIC 慢 17.2 ms，复用段过短 |
| 4k，`max_tokens=1` | 193.6 | 136.6 | **1.418×** | PIC 节省 57.0 ms |
| 4k，`max_tokens=8` | 381.6 | 360.5 | 1.059× | decode 成本稀释 prefill 收益 |

`seam_sink=0` 与 `seam_sink=4` 的 4k、`max_tokens=1` 结果几乎一致：

| seam | Full p50 (ms) | PIC p50 (ms) | Full/PIC | 节省 (ms) |
|---:|---:|---:|---:|---:|
| 0 | 193.4 | 136.4 | 1.418× | 57.1 |
| 4 | 193.8 | 137.3 | 1.412× | 56.5 |

差异小于 0.5%，当前测量中 seam sink 不是主要性能因素。

### 9.4 40 ms input processor 开销的定位与修复

跨层 profiling 定位到旧版本的主要差异：

| 阶段 | 旧 PIC (ms) | Full (ms) | 新 PIC (ms) |
|---|---:|---:|---:|
| `engine.input_processor_total` | 40.2 | 0.7 | ~0.7 |
| `engine.scheduler_enqueue` | 0.3 | 0.3 | ~0.3 |
| `engine.scheduler_step` | 0.2 | 0.2 | ~0.2 |
| `engine.execute_model_entry` | 72.5 | 138.5 | ~72.5 |
| `engine.scheduler_update_from_output` | 0.1 | 0.1 | ~0.1 |

旧路径在 renderer 已经产生完整 prompt token IDs 后，
`_prepare_pic_request()` 又对每个文本 segment 调用 `tokenizer.encode()`，
造成约 40.2 ms 的重复 tokenization。新 11E 首先尝试在已有完整 token IDs
中定位 separator 的精确 token 序列，直接构造 segment IDs 和 ranges；若
separator 因 tokenizer 的上下文合并而无法精确定位，则回退到原有独立
per-segment tokenization，保持边界安全性。

因此，`engine.input_processor_total` 从 40.2 ms 降至约 0.7 ms，端到端 PIC
从 174.1 ms 降至 136.6 ms，减少 37.5 ms；剩余差距主要来自 PIC forward、
slot attach、range preparation 以及共同的 HTTP/API/output 固定开销。

### 9.5 新 11E 的 4k、`max_tokens=1` 拆解

| PIC 阶段 | 时间 (ms) |
|---|---:|
| `engine.input_processor_total` | ~0.7 |
| `engine.scheduler_enqueue` | ~0.3 |
| `engine.scheduler_step` | ~0.2 |
| `engine.execute_model_entry` / worker execute | ~72.5 |
| `engine.scheduler_update_from_output` | ~0.1 |
| Engine 小计 | ~73.8 |
| HTTP/API/output 未覆盖部分 | ~62.8 |
| 端到端 | 136.6 |

worker 内部稳态记录为：

| worker 阶段 | 时间 (ms) |
|---|---:|
| `execute_model.pic_slot_attach` | 7.8 |
| `execute_model.prepare_inputs` | 1.28 |
| `execute_model.attention_metadata` | 2.05 |
| `pic_range.make_isolated_input_batch` | 0.79 |
| `pic_range.attach_native_slots` | 0.23 |
| `pic_range.prepare_inputs` | 0.94 |
| `pic_range.attention_metadata` | 1.70 |
| `pic_range.preprocess_inputs` | 0.23 |
| `pic_range.forward_dispatch` | 50.9 |
| `pic_range.total` | 58.0 |
| `execute_model.pic_runtime` | 58.2 |
| `execute_model.total` | 74.7 |
| `sample_tokens.total` | 0.70 |

### 9.6 当前结论与剩余优化点

- 4k、`max_tokens=1`：本地 PIC 已从 `1.113×` 提升到 `1.418×`；
- 1k 复用段仍不足以抵消 PIC 固定开销，PIC 为 `0.823×`；
- `max_tokens=8` 时 decode 成本稀释 prefill 收益，PIC 为 `1.059×`；
- 最大的重复 tokenization 开销已经消除；
- 剩余重点是 `pic_slot_attach` 约 7.8 ms、range preparation 约 7 ms，
  以及 HTTP/API/output 约 54–63 ms 的共同固定开销；
- 本节是本地 PIC 结果，不包含 Mooncake TCP/RDMA 传输成本，不能直接代表
  跨进程 PIC 的最终延迟。

## 10. Stage 11E：Full/PIC 4k 细粒度耗时拆解（`max_tokens=1`）

本节记录同一份 4k 本地复用实验的完整拆解结果。测试条件为：机器 61
三卡、Xeon Silver 4210R、3×RTX 4090、Qwen3.5-2B，
`--gpu-memory-utilization 0.35`、`--max-model-len 8192`、
`--max-num-batched-tokens 8192`、`--enforce-eager`，profiling 关闭；请求为
`4352 reused + 640 recompute`，`max_tokens=1`。

### 10.1 端到端结果

| 条件 | p50 | p95 |
|---|---:|---:|
| Full 4k | 193.6 ms | ~200 ms |
| PIC 4k | **136.6 ms** | ~140 ms |
| 差值 | **-57.0 ms** | |
| 加速比 | **1.418×** | |

### 10.2 Engine 层对比

| stage | Full (ms) | PIC (ms) | 差值 |
|---|---:|---:|---:|
| `engine.input_processor_total` | 0.7 | 0.7 | 0 |
| `engine.scheduler_enqueue` | 0.3 | 0.3 | 0 |
| `engine.scheduler_step` | 0.2 | 0.2 | 0 |
| `engine.execute_model_entry` | 138.5 | 72.5 | -66.0 |
| `engine.scheduler_update_from_output` | 0.1 | 0.1 | 0 |
| Engine 小计 | 139.8 | 73.8 | -66.0 |
| HTTP/API/输出未覆盖 | ~53.8 | ~62.8 | +9.0 |
| 端到端 | 193.6 | 136.6 | -57.0 |

### 10.3 Full worker 拆解

Full 的 worker `execute_model.total` 为 138.5 ms，其中完整重算的 prefill
`forward_dispatch` 为 133.8 ms，处理 4992 token：

| stage | 时间 (ms) |
|---|---:|
| `execute_model.pic_slot_attach` | 0.05 |
| `execute_model.prepare_inputs` | 1.13 |
| `execute_model.schedule_and_padding` | 0.06 |
| `execute_model.mamba_preprocess` | 0.003 |
| `execute_model.restore_pic_states` | 0.04 |
| `execute_model.slot_mapping` | 0.07 |
| `execute_model.attention_metadata` | 1.96 |
| `execute_model.preprocess_inputs` | 0.21 |
| `execute_model.copyback_pic_kv` | 0.04 |
| `execute_model.pic_runtime` | 0.003 |
| `execute_model.forward_dispatch` | **133.8** |
| `execute_model.postprocess` | 0.18 |
| `execute_model.state_finalize` | 0.006 |
| `execute_model.total` | **138.5** |
| `sample_tokens.total` | 0.63 |

### 10.4 PIC worker 拆解

PIC 的有效 prefill 位于 `pic_range.forward_dispatch`，处理 640 个 recompute
token；外层 `execute_model.forward_dispatch` 仅为空跑：

| stage | 时间 (ms) |
|---|---:|
| `execute_model.pic_slot_attach` | **7.8** |
| `execute_model.prepare_inputs` | 1.28 |
| `execute_model.schedule_and_padding` | 0.04 |
| `execute_model.mamba_preprocess` | 0.003 |
| `execute_model.restore_pic_states` | 0.18 |
| `execute_model.slot_mapping` | 0.08 |
| `execute_model.attention_metadata` | 2.05 |
| `execute_model.preprocess_inputs` | 0.25 |
| `execute_model.copyback_pic_kv` | 0.045 |
| `pic_range.make_isolated_input_batch` | 0.79 |
| `pic_range.attach_native_slots` | 0.23 |
| `pic_range.mamba_preprocess` | 0.003 |
| `pic_range.prepare_inputs` | 0.94 |
| `pic_range.schedule_and_padding` | 0.023 |
| `pic_range.slot_mapping` | 0.055 |
| `pic_range.attention_metadata` | 1.71 |
| `pic_range.preprocess_inputs` | 0.24 |
| `pic_range.forward_dispatch` | **50.9** |
| `pic_range.total` | **58.0** |
| `execute_model.pic_runtime` | 58.2 |
| `execute_model.forward_dispatch` | 0.04（空跑） |
| `execute_model.postprocess` | 0.16 |
| `execute_model.state_finalize` | 0.006 |
| `execute_model.total` | ~74.7 |
| `sample_tokens.total` | 0.70 |

### 10.5 PIC 节省与额外开销

| 项目 | 时间 (ms) | 说明 |
|---|---:|---|
| Full prefill forward | 133.8 | 4992 token |
| PIC recompute forward | 50.9 | 640 token，attention 仍访问 4992 token KV |
| prefill 节省 | **-115** | 端到端收益主要来源 |
| transition-apply | +2.4 | compact transition 稳态开销 |
| `pic_slot_attach` | +7.8 | PIC 特有，Full 约 0.05 ms |
| `pic_range` preparation | +~7 | isolated batch 的二次准备 |
| HTTP/API/输出差值 | +~9 | PIC 比 Full 更高 |
| 其他 | +~0.8 | metadata、sample 等 |
| 净结果 | **-57** | PIC 端到端少 57 ms |

### 10.6 Forward 的非线性与优化优先级

640 token 的 PIC forward 为 50.9 ms，而 Full 4992 token 为 133.8 ms，
PIC 不是简单按 `640 / 4992` 线性缩放。这是因为 PIC 的 640 个 query token
仍需对 4992 个 KV token 执行 attention，单 query token 的代价更高。

当前优先级：

| 优先级 | 项目 | 当前开销 | 目标 |
|---|---|---:|---|
| 高 | `pic_slot_attach` | 7.8 ms | 拆分并降低 alloc/map/sync |
| 中 | `pic_range` preparation | ~7 ms | 减少重复 InputBatch、slot mapping 和 metadata 构造 |
| 中 | HTTP/API/output 差值 | ~9 ms | 继续用跨层 profiling 拆分 |

如果 `pic_slot_attach` 和 `pic_range` preparation 均能接近 Full 水平，
理论上 4k、`max_tokens=1` 的 PIC 端到端还可减少约 15 ms，加速比可能从
`1.418×` 提升到约 `1.55×`。该数值是按当前拆解的估算，不是已验证结果。

## 11. Stage 11E：跳过单请求 PIC 的废弃普通准备（2026-10-15）

针对上一节定位出的两个 PIC 特有开销，代码先做了一项低风险的结构优化：

- 单请求满足现有 Stage 9-A runtime gate，且 `copyback_kv` 关闭时，在状态恢复后
  直接执行已有的 `pic_range` runtime；
- 成功执行后跳过随后不会被使用的普通 `slot_mapping`、attention metadata、
  `_preprocess` 和 KV copyback 构造；
- batch PIC、`copyback_kv`、不支持的 runtime 以及普通请求仍走原路径；runtime
  失败时自动禁用 PIC runtime 并回退普通 forward。

这项改动不改变 `pic_range` 内部的 forward、transition 或 native KV materialization，
目标是消除 execute_model 中“先准备一次普通输入、再为 miss range 重建一次”的重复工作。
因此需要在同一机器、同一 warmup 和 `max_tokens=1` 条件下重新测量，不能把预期收益当作
已实测结果。

开启 `HYPIC_PIC_PROFILE_DETAIL=1` 后，`native-slot-attach` 还会输出 request 级细分：

```text
[PIC-PROFILE] native-slot-attach request=... slots=... \
  materialize_ms=... attach_map_ms=... request_ms=...
```

其中 `materialize_ms` 用于判断 7.8 ms 是否主要来自逐层 private KV
materialization/RoPE，`attach_map_ms` 用于判断 block-table attach 是否值得继续优化。
验证时同时记录端到端 p50/p95、`execute_model.total`、
`execute_model.pic_slot_attach`、`pic_range.*`，并回归 D5 wrong=0、D1 lease/pool
稳定性和普通请求无回归。

## 12. Stage 11E：本地 PIC 最终收益（profiling 全关闭，2026-10-16）

机器 61（三卡 RTX 4090，Xeon 4210R），模型为 Qwen3.5-2B。配置为
`--gpu-memory-utilization 0.35`、`--max-model-len 8192`、
`--max-num-batched-tokens 8192`、`--enforce-eager`。请求为 4k 复用：4352
token reused、640 token recompute，`max_tokens=1`。所有 profiling 和校验环境变量
均关闭，以下是最终端到端 p50：

| 条件 | Full p50 (ms) | PIC p50 (ms) | PIC/Full | 节省 (ms) |
|---|---:|---:|---:|---:|
| 1k，`max_tokens=1` | 77.7 | 93.6 | 0.830× | -15.9 |
| 4k，`max_tokens=1` | 191.8 | 133.0 | **1.442×** | **58.8** |
| 4k，`max_tokens=8` | 373.8 | 351.0 | 1.065× | 22.7 |

4k、`max_tokens=1` 的真实加速比从旧 11E 的 1.113× 提升到 1.442×，也超过
历史 10-C 的 1.34×。与 DETAIL 开启时的 1.418×相比，最终关闭 profiling 后的
1.442×才是性能结论。`seam_sink=0` 和 `seam_sink=4` 的差异小于 0.5%。

### 12.1 端到端收益来源

旧版本 PIC 的 `engine.input_processor_total` 约 40.2 ms，而 Full 约 0.7 ms；
pretok fast path 后 PIC 降至约 0.7 ms。该优化消除了对每个 text segment 的重复
`tokenizer.encode()`，直接复用 renderer 已生成的完整 prompt token IDs，并在找不到
separator 精确 token 序列时保留原有安全回退路径。

4k、`max_tokens=1` 的 worker 拆解仍显示：

| 阶段 | Full (ms) | PIC (ms) |
|---|---:|---:|
| `forward_dispatch` | 133.8 | 0.04（外层空跑） |
| `pic_runtime` / `pic_range.total` | 0.003 | 58.2 / 58.0 |
| PIC range forward | 133.8 | 50.9 |
| `pic_slot_attach` | 0.05 | 7.8 |
| `execute_model.total` | 138.5 | 74.7 |

PIC 省下约 115 ms 的 Full prefill，支付约 56 ms 的 transition、range preparation
和 PIC 固定开销，最终端到端净节省 58.8 ms。640 token 的 forward 不能按
`640/4992` 线性外推，因为 query 仍需访问 4992 token 的 attention KV。

### 12.2 当前收益边界

- 1k 复用段只有 1088 token，省下的 prefill 不足以覆盖 PIC 固定开销，因此 PIC
  仍慢 15.9 ms。
- `max_tokens=8` 时 decode 约 240 ms，Full/PIC 的 decode 成本接近，prefill 收益
  被稀释到 1.065×。
- 当前本地 PIC 的盈亏平衡点约为 2k--3k 复用段；这取决于模型、硬件和 decode
  长度，不能作为跨机器或 Mooncake 路径的保证。

下一阶段应进入 Mooncake 跨进程对照：同一 4k、`max_tokens=1` 条件分别测 Full、
Local PIC 和 Mooncake PIC，并单独报告 native KV/transition 传输成本。

## 13. Stage 11E：HTTP/API 未覆盖开销定位（2026-10-16）

本地 4k、`max_tokens=1` 结果中，worker timeline 已覆盖 PIC 约 69 ms，
但端到端请求仍有约 10 ms 的未覆盖差值。该差值不能直接归因于 JSON 或 API
代码，因为原有路由打点位于 FastAPI 请求解析之后、响应写回之前。

为拆开这段边界，11E 增加了 opt-in ASGI profiling middleware，仅在
`HYPIC_PIC_PROFILE_DETAIL=1` 时启用：

```text
api.asgi_request_total
api.asgi_response_start
api.asgi_response_write
api.handler_create_completion
```

结合既有的 `api.handler_total`、`api.response_serialize`、
`engine.get_output` 和 `engine.output_processor`，可以区分：

```text
FastAPI/decorator/request parsing
EngineCore output queue wait
detokenization/output processing
JSON/streaming response write
```

profiling 关闭时 middleware 直接透传，不改变正常请求路径。下一次实验应在
相同 Full/PIC、warmup、eager 和 `max_tokens=1` 条件下收集这些阶段，再决定
是否值得修改 HTTP/API 实现。

## 15. Stage 11E：separator tokenization 残余开销（2026-10-16）

复测发现 `HYPIC_PIC_PROFILE_FORWARD` 关闭后，PIC 的
`engine.input_processor_total` 仍约为 9.4 ms，而 fast-path 记录约为
0.7 ms。因此这部分不是 forward profiling 引入的同步开销。

代码对比显示，完整 prompt token IDs 已经复用，但每个请求仍会调用一次
`tokenizer.encode(separator)` 来得到 separator token IDs。该调用在 Qwen3.5-2B
上可能贡献大部分剩余 8--9 ms。修复方式是由 `InputProcessor` 缓存最近使用的
separator token IDs，再传给 `split_text_and_tokenize()`；仅保留一个最近值，避免
自定义 separator 导致无界缓存。没有精确 separator token 序列时仍回退到原有的
逐 segment tokenization，边界语义不变。

本次修复增加了 segmenter 单元测试，验证传入预计算 separator token IDs 时不会
再次调用 tokenizer。需要在真机上重新测量 warmed PIC 的
`engine.input_processor_total`，并确认 PIC segment ranges、命中、partial fallback
和普通 Full 请求无回归。

## 16. Stage 11E：pretokenized split 直接路径（2026-10-16）

separator token cache 后，`split_text_and_tokenize()` 仍会通过
`split_token_ids()` 构造完整的 `PICSegment` 对象，再重新展平 token IDs。该路径
会产生不需要的 tuple、segment hash 和列表分配。新增的 pretokenized 专用扫描路径
直接从已有 prompt token IDs 生成去 separator 的 token 列表和 ranges；公共
`split_token_ids()` 以及没有完整 token IDs 时的原始 tokenizer fallback 不变。

在本地 5000 token synthetic case 中，旧结构路径约 3.09 ms，新路径约 0.64 ms，
约 4.8 倍改善。该结果仅用于验证算法方向，真机验收仍看 warmed PIC 的
`engine.input_processor_total`、segment ranges、命中、partial fallback 和普通
Full 请求正确性。

## 17. Stage 11E：direct split 真机最终结果（2026-10-16）

机器 61（三卡 RTX 4090，Xeon 4210R），Qwen3.5-2B，
`--gpu-memory-utilization 0.35`、`--max-model-len 8192`、
`--max-num-batched-tokens 8192`，`--enforce-eager`，profiling 全关闭。

4k 复用、`max_tokens=1` 的 5 轮结果中，第 1 轮 Full p50=184.3 ms 为异常
偏低值；后 4 轮稳定。稳定中位数为：

| 条件 | Full p50 | PIC p50 | PIC/Full | 节省 |
|---|---:|---:|---:|---:|
| 4k mt=1 稳态 | 194.1 ms | 129.2 ms | **1.503×** | **65.2 ms** |
| 1k mt=1 | 77.7 ms | 93.6 ms | 0.830× | -15.9 ms |
| 4k mt=8 | 373.8 ms | 351.0 ms | 1.065× | 22.7 ms |

版本演进中，direct pretokenized split 将上一版 `135.3 ms / 1.422×` 改善为
`129.2 ms / 1.503×`。PIC 的 `engine.input_processor_total` 从约 9.4 ms
降至约 2.4 ms，PIC 只比 Full 多约 1.3 ms 的输入处理时间；剩余主要 PIC
特有开销是约 7.8 ms 的 `pic_slot_attach`。

4k 场景已经超过历史 10-C 的 1.34×，下一步应转入 Mooncake 跨进程对照：
Full、Local PIC、Mooncake PIC 三档保持相同 prompt、warmup、eager 和
`max_tokens=1`，单独报告 native KV 与 transition 传输成本。1k 仍处于
PIC 盈亏平衡点以下，4k mt=8 的收益则会被 decode 成本稀释。

### 17.1 最终本地复现命令

以下命令用于复现上述本地 4k 结果。先清理旧的 vLLM/EngineCore 进程，使用
`--enforce-eager`，关闭所有 profiling 和 checksum；随后执行一次 publish、
warmup，再执行 10 次 PIC reuse 请求，`max_tokens=1`。

```bash
pkill -9 -f "vllm serve"; pkill -9 -f "EngineCore"; pkill -9 -f "VLLM::EngineCore"
sleep 15

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HYPIC_PIC_PROFILE_FORWARD
unset HYPIC_PIC_PROFILE_IMPORT
unset HYPIC_PIC_PROFILE_DETAIL
unset HYPIC_PIC_PROFILE_TIMELINE_SYNC
unset HYPIC_PIC_VERIFY_CHECKSUM

vllm serve /data/eval_model/Qwen/qwen3.5-2b \
  --served-model-name qwen3.5-2b \
  --port 8000 \
  --enforce-eager \
  --pic-enable --pic-auto-enable \
  --pic-zero-copy --pic-single-request \
  --pic-capture-live --pic-restore-live \
  --pic-max-cache-bytes 2147483648 \
  --pic-debug \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.35
```

测试请求保持 4k 复用形状（4352 reused + 640 recompute），`max_tokens=1`；
报告 Full 和 PIC 的 p50/p95，并丢弃首次 Full warmup 异常值。该命令只用于
本地 PIC 基线；Mooncake 跨进程测试需要另外启动 master、A、B 三个进程。

## 18. Stage 11E：Mooncake native-slot-attach 作用域修复（2026-10-16）

Mooncake B 侧首次跨进程测试中，25/25 请求出现
`native_slot_attach_failed`，全部回退到 ordinary block-table execution，
实际 prefill 为 4992 token，PIC 复用没有生效。

根因是 materialization profiling 变量的函数作用域错误：
`slot_setup_ms`、`gather_ms`、`rotary_lookup_ms` 被初始化在
`_materialize_pic_public_kv()`，但在
`_materialize_pic_private_kv()` 的计时和 logger 中使用。profiling 关闭时，
private path 仍会执行 logger 参数求值，因此触发 `UnboundLocalError`。

修复已将三项变量初始化移动到 `_materialize_pic_private_kv()`，并删除 public
函数中无效的同名变量。该修复不改变 KV materialization 算法，只修复跨进程
private native-slot attach 的异常路径。修复后 Mooncake 验收必须确认：

```text
prefill_tokens=640
reuse body skipped
transition applied
fallback_reasons 不包含 native_slot_attach_failed
```

## 14. Stage 11E：API gap 与 private materialization 细分打点（2026-10-16）

为定位 PIC 与 Full 之间仍未解释的 API/EngineCore 差异，11E 在请求侧
generator 增加了两个 detail profiling 阶段：

```text
engine.request_output_wait
engine.generate_stream_total
```

前者直接包住每次请求从 `RequestOutputCollector` 取结果的等待，后者覆盖
请求侧 generator 从 `add_request` 到最终输出的完整时间。它们不依赖后台
`engine.get_output` / `engine.output_processor` 是否被批处理或日志过滤，能够
直接判断 endpoint gap 是否来自 EngineCore 输出队列等待。

同时，private native-KV materialization 的 `native-materialization` 记录增加：

```text
slot_setup_ms gather_ms rotary_lookup_ms rerotation_ms copy_ms materialization_ms
```

其中 `gather_ms` 对应从 native KV 中 gather canonical KV，
`rotary_lookup_ms` 对应查找 RoPE module，`rerotation_ms` 和 `copy_ms` 分别对应
RoPE 重排与 backend KV 写入。下一轮实验应在相同 Full/PIC、warmup、eager 和
`max_tokens=1` 条件下，开启 `HYPIC_PIC_PROFILE_DETAIL=1`、
`HYPIC_PIC_PROFILE_IMPORT=1`，重点收集这两组新增字段；最终延迟仍须在关闭
profiling 的独立运行中测量。

## 19. Stage 11E：Mooncake 跨进程 PIC 最终结果（2026-10-16）

机器 61（三卡 RTX 4090，Xeon 4210R），Qwen3.5-2B，Mooncake TCP，A 使用
GPU 0 publish，B 使用 GPU 1 reuse，4k、`max_tokens=1`。

| 条件 | p50 | 说明 |
|---|---:|---|
| Full（B） | 193.2 ms | 4992 token full prefill |
| Local PIC（A） | 129.2 ms | 无网络传输 |
| Mooncake PIC（B） | 258.1 ms | TCP 跨进程复用 |

本地 PIC 相比 Full 节省约 65.2 ms；Mooncake PIC 相比 Local PIC 增加
128.9 ms，因此相比 Full 净亏约 64.9 ms。报告中的 `0.749x` 是
`Full / Mooncake PIC = 193.2 / 258.1`；按延迟比 `Mooncake PIC / Full` 表示，
应为 `1.336x`。

### 19.1 Mooncake 传输拆解

| 项目 | 时间 | 数据 |
|---|---:|---:|
| transition payload Get | 43.1 ms | 38.4 MB |
| transition deserialize | 17.4 ms | 38.4 MB |
| transition total | 61.0 ms | compact payload |
| native KV Get | 38.7 ms | 53.5 MB，6 layers |
| native KV CPU | 2.8 ms | |
| native KV H2D/index | 13.7 ms | |
| native total | 56.4 ms | 136 blocks |
| transition apply | 1.9 ms | 与 Local PIC 一致 |

已知传输与加载小计约 119.3 ms，而 Local PIC 到 Mooncake PIC 的端到端差值为
128.9 ms，仍有约 9.6 ms 的 scheduler、request boundary 或其他未覆盖开销，
后续优化时不应全部归入 TCP Get。

### 19.2 功能结论与下一阶段

跨进程复用功能已通过：

```text
scheduler lookup HIT
reused=(0,) recompute=(1,)
Mooncake native KV imported, group=3, blocks=136
transition applied, range=[0,4352)
reuse body skipped
prefill_tokens=640
fallbacks=0
```

当前主要瓶颈是 TCP 传输和传输后的加载：transition Get、native KV Get、
transition deserialize 以及 native KV H2D/index copy。下一阶段按优先级评估
Mooncake 多连接/多流 TCP、native KV 与 transition 的异步重叠、deserialize
零拷贝或预分配、pinned/异步 H2D；RDMA 只有在硬件支持时再单独验证。

## 20. Stage 11E-B：三卡 TCP Store-pull 完整拆解（2026-10-21）

本节记录服务器提交的 11E-B Store-pull 实测报告。所有数字来自真机日志，未在本机
复测；与第 19 节（2026-10-16，11E 早期版本）不是同一份代码，不能混用。

### 20.1 测试条件

| 项目 | 条件 |
|---|---|
| 机器 | HOST_A，三卡（Xeon 4210R + RTX 4090） |
| 模型 | Qwen3.5-2B |
| 代码 | Stage 11E-B，Store-pull 默认 |
| 环境 | `HYPIC_PIC_NATIVE_KV_DIRECT=1`；未设置 `NATIVE_KV_TRANSFER_MODE` / `TRANSITION_TRANSPORT` / `TRANSFER_ENGINE`；`MC_STORE_MEMCPY=0`；`PROFILE_IMPORT=1` |
| 网络 | Mooncake `protocol: "tcp"`，**无 RDMA**（`ibv_devinfo` 无输出） |
| Store | `global_segment_size=8 GiB`，`local_buffer_size=2 GiB` |
| 请求 | 4k 复用形状，640 recompute token，`max_tokens=1` |
| 统计 | 10 轮，去掉首个冷启动后取中位 |
| 日志 | `/tmp/pic_mooncake_072244/B.log` |

### 20.2 端到端结果

| 指标 | Full | PIC |
|---|---:|---:|
| p50 | 193.0 ms | 274.3 ms |
| 加速比 | 1.0× | **0.708×** |
| PIC 额外开销 | — | **+81.3 ms** |

结论：PIC 计算侧省下 69.6 ms，但传输 + materialize 花掉 149.5 ms，净亏 81.3 ms。

### 20.3 完整耗时拆解（热路径中位）

| 阶段 | PIC (ms) | Full (ms) | 差 (ms) | 有效吞吐 |
|---|---:|---:|---:|---|
| native-receive（6 层 `store.get`） | 86.5 | — | +86.5 | 53.5 MB / 0.62 GB/s |
| forward 640 tok | 65.0 | 134.6（5000 tok） | −69.6 | 0.105 ms/token（Full 0.0269，3.9×） |
| 未归因（HTTP/调度/采样） | 59.8 | 58.4 | +1.4 | 与 Full 持平 |
| transition-transfer | 35.5 | — | +35.5 | 38.4 MB / 1.08 GB/s |
| native-materialize（H2D） | 23.5 | — | +23.5 | 53.5 MB / 2.8 GB/s（cpu 3 + h2d 19） |
| transition-apply | 4.0 | — | +4.0 | — |
| **合计** | **274.3** | **193.0** | **+81.3** | |

要点：

- 之前被当作“未归因 146 ms”的部分，实际是 `native-receive 86.5` 与
  `native-materialize 23.5` 没有被拆开统计，不是隐藏黑盒。
- PIC 的 640 token forward 是 0.105 ms/token，Full 是 0.0269 ms/token，小 batch
  的 query 效率差 3.9 倍；这 65 ms 不是传输造成的。
- `transition-apply` 只有 4.0 ms，与第 10 节的 2.4 ms 同量级，不是瓶颈。

### 20.4 Store 裸带宽实测（同一台机器，TCP）

脚本：`store_bw_test.py`；配置同上（TCP、
`global_segment_size=8 GiB`、`local_buffer_size=2 GiB`、`MC_STORE_MEMCPY=0`）。

| 对象大小 | put | get |
|---|---:|---:|
| 8 MB | 123.28 ms / 0.07 GB/s（首次冷启动） | 21.87 ms / 0.38 GB/s |
| 16 MB | 27.32 ms / 0.61 GB/s | 30.65 ms / 0.55 GB/s |
| 32 MB | 56.21 ms / 0.60 GB/s | 50.46 ms / 0.66 GB/s |
| 64 MB | 130.00 ms / 0.52 GB/s | 81.56 ms / 0.82 GB/s |

**Mooncake Store 在纯 TCP 下的原生带宽为 0.5–0.8 GB/s，get 峰值 0.82 GB/s。**

### 20.5 PIC 传输与 Store 原生带宽对照

| 场景 | 数据量 | 耗时 | 带宽 | 与 Store 裸 get 对比 |
|---|---:|---:|---:|---|
| Store 裸 get 64 MB | 64 MB | 81.6 ms | 0.82 GB/s | 基准 |
| PIC native-receive | 53.5 MB | 86.5 ms | 0.62 GB/s | 已达该路径上限 |
| PIC transition store get | 38.4 MB | 27.9 ms | **1.38 GB/s** | 高于裸测 |

关键结论：

1. **native-receive 的 0.62 GB/s 不是“6 层串行 get”造成的，而是 Store TCP 的硬
   上限**；裸测 get 也只有 0.55–0.82 GB/s。因此“把逐层 `store.get` 改成
   `batch_get_into_multi_buffers` 可省 47 ms”的 P0 假设**不成立，撤回**。
   **→ 2026-10-21 修正：这条撤回本身被 20.11 的 E2 数据推翻。** 单对象 get 的
   0.8 GB/s 只是**单次调用**的上限；transition 路径用一次批量读拿到 1.38 GB/s，
   说明批量/并发能突破单调用上限。“单调用上限 = TCP 硬上限”的推论作废，详见 20.11。
   **→ 2026-10-22 最终结论：跨进程实测（21.10）证明 batch 确实省 28–35 ms，
   “撤回”是错的。** 早期单进程探针没有设置 `MC_STORE_MEMCPY=0`，Mooncake 自动
   降级为 LOCAL_MEMCPY，测的是内存拷贝，因此把“固定开销”误判成了“传输上限”。
2. transition 的 1.38 GB/s 反而高于裸测 0.82 GB/s，说明固定布局/批量路径本身还有
   空间（可能是批量提交或对象已缓存），但该路径总量只有 38.4 MB。
3. PIC 传输合计 92 MB（native 53.5 + transition 38.4），按 0.7 GB/s 理论需
   131 ms，实测 114 ms，已经接近这条 TCP 链路的物理上限。
4. 274 ms 的构成：传输 114 ms（TCP 硬限）+ forward 65 ms（小 batch 效率）+ 
   materialize 23.5 ms（PCIe）+ apply 4 ms + 未归因 59.8 ms（与 Full 持平）。

### 20.6 优化路径与预估（按可行性排序）

| 优先级 | 路径 | 预估收益 | 前提 | 可行性 |
|---|---|---|---:|---|
| P0 | 换 RDMA | 传输 114 → 约 10 ms，省 100+ ms，PIC 可到约 170 ms，优于 Full | 需 RDMA 网卡（当前 `ibv_devinfo` 无输出） | 需硬件 |
| P1 | forward batch 合并 | 65 → 约 17 ms（理论），省约 48 ms，PIC 到约 226 ms | 需要 batch 内多 PIC 请求合并 forward | 中（架构改动） |
| P2 | 减少传输量 | 92 MB → 只传必要部分 | 需要改 KV 格式 | 高难度 |
| P3 | Store 调参 | 0.7 → 1+ GB/s，省 30–50 ms | 试调大 `global_segment_size` / `local_buffer_size` | 低风险，先试 |
| P4 | transition get 1.38 → 更高 | 有空间但总量小 | 查清为何比 native 快 | 低 |

情景结论：

- 若 P0（RDMA）可用：PIC 274 → 约 170 ms，优于 Full 约 23 ms，PIC 路线成立；
- 若只有 P1 + P3：PIC 约 196 ms，与 Full 持平；
- 若都不做：PIC 274 ms，慢于 Full 81 ms，不建议生产启用 PIC。

**2026-10-21 更新：P0 已确认不可用——机器没有 RDMA 网卡。** 因此上表 P0 行作废，
"只有 P1 + P3 时约 196 ms / 与 Full 持平"是**乐观上限**，实际还要扣掉 20.6.1 节算出的
结构性缺口。无 RDMA 条件下能翻正的路径见 20.6.1。

#### 20.6.1 无 RDMA 条件的盈亏分析（2026-10-21）

把 20.3 的数字换算成**每 token**，才能判断哪种硬件/数据条件能让 PIC 翻正。
基准：4352 reused token、640 recompute token。

| 项目 | 实测 | 折算每 reused token |
|---|---:|---:|
| 省下的计算 | 69.6 ms | **16.0 µs/token** |
| native KV 传输（53.5 MB @ 0.62 GB/s） | 86.5 ms | 19.9 µs/token（12.29 KB/token） |
| H2D materialize（53.5 MB @ 2.8 GB/s） | 19.0 ms | 4.4 µs/token |
| 固定开销（transition 35.5 + apply 4.0） | 39.5 ms | 9.1 µs/token（与复用长度无关） |

净结果：`16.0 − (19.9 + 4.4 + 9.1) = −17.4 µs/token`，乘以 4352 约为 −75.7 ms，
与实测 −81.3 ms 同量级（余下约 5.6 ms 来自未归因项与取整），说明拆解自洽。

**结构性结论：在 TCP + fp16 KV 下，PIC 每复用 1 个 token 要付 24.3 µs 传输+H2D，
却只省 16.0 µs 计算。这不是调优能补的差，是符号反了。**

盈亏条件（按每 token 计算）：

| 条件 | 需要的值 | TCP 可达性 |
|---|---:|---|
| 固定开销都保留（39.5 ms） | 传输 ≥ **4.9 GB/s** | 不可能（裸测 0.82） |
| 固定开销可忽略 | 传输 ≥ **1.06 GB/s** | 接近裸测上限，需调参 |

**已排除的方向**：KV 改 fp8、transition 降 bf16 这两个"减少字节"的方案**不做**（2026-10-22
决定），因此上表里涉及它们的推演只作为敏感度分析保留，不再是待办项。无 RDMA 时能做的
只有：把调用次数降下来（D3，已实现）、把固定开销摊薄（更长的复用段）、Store 调到 0.82。

反过来，**每 token 成本是线性等比的，单靠"复用更长的段"不能修复符号**：native KV
字节随 reused token 线性增长，省下的计算也线性增长，1.58× 的比值不变；只有固定的
39.5 ms 会被摊薄。

**短期最能止损的是一件事：按段准入。** 段短、或预计传输时间超过预计计算节省时，
直接按 miss 走本地重算（现有 fallback 通道即可承载）。这样 PIC 只在赢的段上生效，
不会再出现整机 0.708× 的结果。

### 20.7 待确认项（下一轮必须回答）

1. **`native-direct-total` 没有出现，只有 `native-materialize-total`**：需要确认
   native KV 的 direct Store 接收契约是否真的生效；如果没生效，说明仍在走
   `bytes → CPU clone → H2D → index_copy`，这与 20.3 的 cpu 3 ms + h2d 19 ms 一致。
   **2026-10-21 结论：不追这条。** 无 RDMA 的 TCP 部署下 `register_buffer` 实际失败
   （vLLM 自己在 `mooncake/store/worker.py:1142` 只记 error 后继续），Mooncake 仍按
   host staging 搬运，PIC 单独做直写路径既偏离参考实现也拿不到收益。因此
   `native-materialize` 的 cpu 3 + h2d 19 ms 视为既定成本，20.6.1 的 4.4 µs/token
   H2D 项保持不变。
2. **transition 本次拆解里没有 `deserialize` 项**：本次 38.4 MB 用 27.9 ms（1.38
   GB/s），而第 19 节同一 payload 是 61.0 ms（含 17.4 ms deserialize）。需要用
   `B.log` 里的 `transition-store-total` / `transition-total` 记录确认本次走的是
   固定布局批量接收还是 compact payload；在确认前，不能把两次的 transition 数字
   互相比较，也不能假定 deserialize 仍然存在。
3. **6 个 pytest 失败**（legacy 测试未随 Store-pull 契约更新）仍待修复。
4. 本次是 `max_tokens=1` 的纯 prefill 场景，decode 与并发场景未覆盖。

### 20.8 与第 19 节（2026-10-16）的差异说明

| 项目 | 第 19 节（11E 早期） | 本节（11E-B Store pull） |
|---|---:|---:|
| Full p50 | 193.2 ms | 193.0 ms |
| PIC p50 | 258.1 ms | 274.3 ms |
| native KV | Get 38.7 + cpu 2.8 + h2d 13.7 = 56.4 ms | receive 86.5 + materialize 23.5 = 110.0 ms |
| transition | Get 43.1 + deserialize 17.4 = 61.0 ms | transfer 35.5 ms（无 deserialize 项） |
| 未归因 | 约 9.6 ms | 59.8 ms（= HTTP/调度/采样，与 Full 持平） |

两节的 native 与 transition 数字不能直接相减：第 19 节把 6 层 get 记成 38.7 ms
而本节记成 86.5 ms，说明计时口径（是否含 Python/对象解析、是否含首次冷启动、
是否分 rank）不同。后续对比必须使用同一份 `[PIC-PROFILE]` 记录口径，并同时报告
`native-layer` 条数与 `native-*` 合计。

### 20.9 复现脚本

- Store 裸带宽：`store_bw_test.py`
- Bench：`mooncake_bench.sh`
- 基准日志：`/tmp/pic_mooncake_072244/B.log`

### 20.11 E2：Store 容量参数无效 + put/get 严重不对称（2026-10-21）

四组只改 Mooncake client `setup()` 的 `global_segment_size` / `local_buffer_size`，
每组跑 `store_bw_test.py`（GB/s）：

| 组 | segment | buffer | get 16 MB | get 32 MB | get 64 MB | put 16 MB | put 32 MB | put 64 MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 8 GiB | 2 GiB | 0.671 | 0.683 | 0.796 | 5.411 | 5.445 | 5.288 |
| S1 | 16 GiB | 2 GiB | 0.682 | 0.686 | 0.786 | 5.277 | 5.426 | 5.298 |
| S2 | 8 GiB | 4 GiB | 0.657 | 0.784 | 0.832 | 5.394 | 5.383 | 5.205 |
| S3 | 16 GiB | 4 GiB | 0.654 | 0.727 | 0.768 | 5.268 | 5.408 | 5.258 |

干净启动（master 全新、Mem Storage 0 B）复测 S0：get 0.739 / 0.658 / 0.768，与上表
无差异。

结论一：**这两个参数不参与传输路径，停止调参。** 四组 get 差异 < 10% 且无一致趋势
（S2 比 S3 快），put 四组完全一致；master 启动日志里根本没有 `global_segment_size`
字段（只有 `global_file_segment_size`），它们是 client `setup()` 参数，master 只记录
mount 大小。

结论二：**put 5.2–5.4 GB/s 与 get 0.65–0.83 GB/s 相差约 6–8 倍**，读写严重不对称。
这改变了此前的两个判断：

1. 第 20.4 节的 put 数字（16/32/64 MB 为 0.61/0.60/0.52 GB/s）与本节相差约 10 倍。
   两次都是 TCP、都是 `MC_STORE_MEMCPY=0`，差异只能来自冷/热状态或对象分配时机，
   因此**两次 put 数据不可混用**，引用时必须注明是否首次写入、Storage 是否为空。
2. 第 20.5 节“单对象 get 0.8 GB/s 就是 TCP 硬上限、批量无用”的推论作废：
   - 单次 `store.get(key)` 的上限 ≈ 0.8 GB/s；
   - PIC transition 用**一次批量读**（约 90 个对象）拿到 **1.38 GB/s**；
   - 说明限制在**单次调用**，不在链路本身。批量与并发都能突破它。

对 PIC 的直接含义（重算，传输部分按批量可达 1.3 GB/s 估）：

| 方案 | native receive | native materialize | transition | 其余 | 估算 p50 | vs Full |
|---|---:|---:|---:|---:|---:|---:|
| 当前（6 次串行 get，0.62） | 86.5 | 23.5 | 35.5 | 128.8 | 274.3 | 0.70× |
| native 改批量读（1.3） | 41 | 23.5 | 35.5 | 128.8 | **228.8** | 0.84× |

也就是说：**在 TCP 上单靠 D3（native KV 批量接收）就能把 0.70× 拉到 0.84×**，这是
本轮唯一采用的方向。KV 改 fp8、transition 降 bf16 这两个"再减字节"的方案**不做**
（2026-10-22 决定），所以上表不再向下延伸。

与“不做 native direct”的决定不冲突：这里的方案是 **native KV 也用 host staging
批量接收**（一次 `batch_get_into_multi_buffers` 写入 pinned host arena，再一次性
H2D，最后按 native block 做 index_copy），与 11E-C 给 transition 用的模式完全相同，
**不需要 GPU 目标注册、不新增 TE、不写 GPU 直写路径**。

但上表的 1.3 GB/s 假设**来自 90 个对象的 transition 路径，不能直接套到 6 个 layer
对象上**，必须先做受控测量（见 20.12）。若 6 对象批量只有 0.8–0.9，则本行收益从
−45 ms 降到 −10 ms 量级，结论要重算。

### 20.12 测量口径审计（2026-10-21）

E2 的 `store_bw_test.py` 有几处会让读数偏离“跨进程冷读带宽”，引用时必须注明：

| 细节 | 后果 |
|---|---|
| `put` 后立刻 `get` 同一个 key、同一进程 | get 命中的是刚写入的热数据（本机 staging/页缓存），不是冷读 |
| 每轮 `store.remove(key)` 再 `put` | put 含新对象分配开销，不是纯写带宽 |
| `store.put()` 立即返回，且 master/segment 都在本机 | 5.3 GB/s 是**本机 segment 写**（shared-memory 级），不能当作 TCP 读能力 |
| `store.get()` 返回 Python `bytes` | 含一次大对象分配 + 拷贝；PIC native 路径正是这个形态 |

因此“put 5.3 / get 0.8 相差 6 倍”**不能推成“链路空闲、读路径有缺陷”**。目前对
1.38 GB/s 这个离群值有三个候选解释，尚未区分：

1. **单次调用固定开销**：每次 `get` 都付一次 RPC turnaround，对象越多摊得越薄；
2. **Python `bytes` 物化**：`get` 返回 bytes（分配+拷贝），批量路径直接写目标缓冲；
3. **对象数与局部性**：native 是 6 个大对象，transition 是约 90 个小对象。

区分实验（脚本已入库，零改动、只测传输）：

```bash
python3 tests/v1/pic/run_store_read_path_probe.py \
  --master HOST_A:50088 --host HOST_A --reps 5
```

同一份总字节下对比两种读法，默认用 `6:53.5`（native 形状）和 `90:38.4`
（transition 形状）两组：

```text
A. N 次 store.get()                    -> N 个 Python bytes
B. 1 次 batch_get_into_multi_buffers   -> 一个预分配 bytearray
```

判据：

- **B 明显快于 A**：读路径受单次调用限制，native KV 应改成 host-staging 批量接收；
- **B ≈ A**：链路/服务端饱和，20.5 的“单调用上限”结论成立，批量不用做。

（pinned / GPU 目标两种变体暂不测：GPU 目标已决定不做，pinned 只影响 H2D，
不影响“接口谁快”的判断。）

## 21. 优化基线起点（2026-10-22 固化）

跨机 PIC 之后所有优化，都以本节配置和判定窗口为比较基准。**跑基线时必须同时记录
patch MD5 和镜像 tag**，否则前后对比没有锚点。

### 21.1 四组对照

| 组 | 配置 | PIC p50 | Full p50 | 加速比 | 轮间波动 | 稳定性 |
|---|---|---:|---:|---:|---|---|
| **COMPACT_TRANSITION（采用）** | `TRANSFER_ENGINE=0` + `NATIVE_KV_DIRECT=0` + `COMPACT_TRANSITION=1` + `MC_STORE_MEMCPY=0` | **271.1 ms** | 193.0 ms | **0.718×** | 263.9–275.1（±11 ms, 4%） | ✅ 最稳 |
| Store-pull 默认 | `NATIVE_KV_DIRECT=1` + fixed-layout | 274.3 ms | 194.4 ms | 0.708× | 249.7–285.5（±36 ms, 13%） | ⚠️ 波动大 |
| 旧 Store | `TRANSFER_ENGINE=0`，无 COMPACT | 287.1 ms | 194.4 ms | 0.677× | 285.7–296.0（±10 ms, 3.5%） | ✅ 稳但慢 |
| 旧 TE | `TRANSFER_ENGINE=1` | 404.3 ms | 194.8 ms | 0.482× | 385.0–430.6（±45 ms, 12%） | ❌ 最差最不稳 |

选择 COMPACT_TRANSITION 的理由：p50 最低、轮间波动最小（4% vs 13%）、日志完整
（`transition-store-total` + `transition-transfer` + `transition-apply` 都有，便于拆解）、
无 TE 残留 / 无 fallback / 无 OOM、`prefill_tokens=640` 与 `reuse body skipped` 齐全。

**为什么 `NATIVE_KV_DIRECT=0` 反而更稳**（待验证的推断）：`DIRECT=1` 时会先尝试
注册 native KV buffer 并走 GPU 目标批量接收，注册在某些轮次成功、某些轮次失败，
导致同一配置在不同轮次落到不同接收路径（±36 ms 的来源）。`DIRECT=0` 固定走逐层
`store.get()`，路径单一，所以稳定。**后续若要把 native 改成 host-staging 批量接收，
必须重新验证稳定性，不能只看 p50。**

### 21.2 固化配置

```bash
export HYPIC_PIC_TRANSFER_ENGINE=0
export HYPIC_PIC_NATIVE_KV_OVERLAP=1
export HYPIC_PIC_NATIVE_KV_DIRECT=0
export MC_STORE_MEMCPY=0
# Mandatory: without it every transfer opens a new TCP connection (see 21.13).
export MC_TCP_ENABLE_CONNECTION_POOL=1
export HYPIC_PIC_PROFILE_IMPORT=1          # 测基线时开；最终测速要关

unset HYPIC_PIC_TRANSITION_RECEIVE
unset HYPIC_PIC_PROFILE_DETAIL HYPIC_PIC_PROFILE_MEMORY HYPIC_PIC_PROFILE_MEMORY_SYNC
unset HYPIC_PIC_VERIFY_CHECKSUM HYPIC_PIC_BOOTSTRAP_ADDR HYPIC_PIC_ENGINE_ID
unset HYPIC_PIC_TP_RANK HYPIC_PIC_PP_RANK HYPIC_PIC_NATIVE_KV_BATCH_DESCRIPTORS
unset HYPIC_PIC_NATIVE_KV_TRANSFER_MODE HYPIC_PIC_TRANSITION_TRANSPORT
```

服务参数：`--pic-max-cache-bytes 2 GiB`、`--gpu-memory-utilization 0.35`、
`--enforce-eager`；prompt 4992 token（复用 4352 + miss 640）、`max_tokens=1`；
每组前重启 master/A/B；Mooncake `protocol: "tcp"`，`global_segment_size=8 GiB`、
`local_buffer_size=2 GiB`（E2 已证明这两个参数不影响带宽，8/2 即可）。

跑法（脚本在服务器固定路径，不进 patch 附件，避免与代码版本脱钩）：

```bash
bash pic_baseline_compact.sh [日志路径]
```

### 21.3 判定窗口

| 指标 | 基线值 | 容差 | 超差含义 |
|---|---:|---|---|
| Full p50 | 193.0 ms | ±5 ms | 环境/GPU 状态变化 |
| PIC p50 | 271.1 ms | 265–281 ms | 配置漂移或新回归 |
| 加速比 | 0.718× | 0.70–0.74 | — |
| `prefill_tokens=640` | 必须 | — | 若 4992 说明走了 Full，不算复用 |
| TE 残留（`native-transfer-submit` / `TransferEngine connected`） | 0 | — | >0 说明环境变量没清干净 |
| fallback / import failed | 0 | — | >0 说明走了兼容路径 |
| `native-materialize-total` | 23.5–26 ms | ±3 ms | — |
| `transition-store-total` | 22–24 ms | ±3 ms | — |
| `transition-apply` | 3–5 ms | — | — |

### 21.4 已知问题（不影响基线数值，但需记录）

- `store.remove()` 返回 `Del=0/16`，key 残留；不影响 get，也不影响 PIC 运行时；
- `Local segment descriptor not found`，启动时序问题，无害；
- `segment_already_exists`，同进程 `setup()` 内部 mount 两次，无害；
- 6 个 pytest legacy TE 用例失败（已在 11E-C 归档里修复，见对应 README）。

### 21.5 关于“未归因 142–149 ms”的校正

本组日志里只剩 `native-materialize` 23.5、`transition` 22.8 两段传输，看上去还有
142–149 ms “未归因”。**这不是新的黑盒**：它与 20.3 的拆解一致，

```text
native-receive ~86.5 ms（本轮没有单独打点）
HTTP/调度/采样 ~59.8 ms（与 Full 持平）
合计 ~146 ms  == 本轮的“未归因 142–149 ms”
```

所以优化目标应写成“**把 native-receive 压下来**”，而不是“压未归因”。下一步先确认
本轮日志里是否确实缺少 `native-receive-total` / `native-layer` 记录（`DIRECT=0` 时
逐层 `store.get` 仍应有 `native-layer` 打点）；若缺失，那是打点问题，先补打点再谈优化。

注意：`native-materialize-total` 的 23.5 ms **不含 `store.get`**，不要用它换算网络带宽
（详见 21.7 的路径表）。

### 21.6 基线不覆盖

- **11E-C（host staging + 无 deserialize）尚未在本基线下测量**：本节是 11E-B 代码上的
  基线，应用 11E-C 后必须重测一次，并确认 `representation=store-host-staging`；
- native 的“逐层 get vs 批量”对比（`run_store_read_path_probe.py`）尚未跑；
- decode、并发、多请求场景未覆盖；
- PROFILE_DETAIL 下的新增计时标签未验证。

### 21.7 基线环境下两条传输路径（定论）

环境：`TRANSFER_ENGINE=0`、`NATIVE_KV_DIRECT=0`、`
`MC_STORE_MEMCPY=0`、`TRANSITION_RECEIVE` 与 `TRANSITION_TRANSPORT` 未设置。

| 数据 | 接口 | 目标 | 日志记录 | 该记录的计时范围 |
|---|---|---|---|---|
| native KV | **逐层 `store.get()` × 6** | Python `bytes` | `native-layer`（每层一条）/ `native-receive-total` | 只含 `store.get` |
| native 落地 | — | GPU | `native-materialize-total` | `cpu_ms` = frombuffer+clone；`h2d_index_ms` = H2D+index_copy |
| transition/state | **1 次 `batch_get_into_multi_buffers`** | pinned host arena | `transition-store-total get_ms` | **只含 batch 调用，不含 H2D** |
| transition 落地 | 1 次 `host.to(device)` | GPU | **无独立打点** | — |

代码依据（`vllm/v1/pic/mooncake_kv.py`）：

- native：`receive_native_blocks()` 的 direct 门是
  `_pic_native_kv_direct_enabled() and group in _registered_native_kv`（:3210），
  `DIRECT=0` 时整段跳过；TE 门是 `self.transfer_engine is not None`（:3232），
  `TRANSFER_ENGINE=0` 时 `transfer_engine` 为 `None`，也跳过；于是循环落到
  `raw = self.store.get(member.key)`。所以 native **确实是普通逐层 get**。
- transition：`_load_transition_operator_store()` 的 gpu 门是
  `_transition_receive_mode == "gpu"`（:2604），未设置时默认 `host`（:253），
  因此走 `batch_get(keys, addrs, sizes)`（:2655），目标地址来自
  `_store_state_host_plan()` 的 pinned arena。

#### 两个“矛盾”的成因

1. **native 23.5 ms 与 TCP 0.8 GB/s 不矛盾**：23.5 ms 是 `native-materialize-total`
   （`cpu_ms` 4.3 + `h2d_index_ms` 19.4），**不包含 `store.get`**。53.5 MB / 23.5 ms
   ≈ 2.28 GB/s 是 CPU 拷贝 + H2D 的本地速度，与网络无关。`store.get` 的时间在
   `native-receive-total`（11E-B 实测 86.5 ms / 0.62 GB/s）。21.5 的“未归因
   142–149 ms”正是 `native-receive 86.5 + HTTP/调度 59.8`，与“native 走普通 get”
   完全自洽。
2. **`store-host-staging` 不是“普通 get”**：该标签描述的是**目标**（pinned host
   arena），不是接口。接口仍是 `batch_get_into_multi_buffers`，只是写进 host 而不是
   GPU；写完再做一次 H2D。所以“batch 接口”和“host-staging 标签”同时成立。

#### 先定路径、再测带宽（正确的操作顺序）

```bash
# 1) 定路径（必须与上表一致，否则先排查配置）
grep -c 'native-direct-total'    B.log    # 必须 0
grep -c 'native-layer'           B.log    # ≈ 6 × 请求数
grep -E 'native-receive-total'   B.log | head -3
grep -E 'transition-store-total' B.log | head -3   # representation=store-host-staging
grep -cE 'native-transfer-submit|TransferEngine connected' B.log   # 必须 0

# 2) 路径确认后，才谈“带宽是否可提升”
python3 tests/v1/pic/run_store_read_path_probe.py \
  --master HOST_A:50088 --host HOST_A --reps 5
```

如果 `native-receive-total` 在基线日志里**完全缺失**，那是打点问题而不是路径问题：
先补打点，再用它的值去对齐 86.5 ms。

#### transition 能否也改用 `store.get`？

可以，但当前数据不支持。两种“走 get”的代价不同：

| 变体 | 传输 | +H2D | 合计 | 相对当前 |
|---|---:|---:|---:|---:|
| batch（当前） | 22.8 ms | ~13 ms | **~36 ms** | — |
| 一个 38.4 MB 大对象 `get`（= compat payload） | 43–48 ms | ~13 ms | 57–61 ms | **+21–25 ms** |
| 逐 tensor `get`（约 90 个 0.43 MB 对象） | ≫48 ms | ~13 ms | 更差 | 更差 |

原因与 E2 的结论一致：瓶颈在**单次调用**而不是链路。`batch_get_into_multi_buffers`
在一次 C++ 调用里提交约 90 个 key，把每次调用的固定开销摊薄；`store.get` 一次一个
对象，每对象都付一次往返。

这个判断**可测**：`run_store_read_path_probe.py` 现在会额外读一个等大的单对象
（`big` 列），于是同一份数据上能直接比出 `batch vs big`：

```text
90 objects, 38.4 MB | singles X | batch 1.68 | big ~0.8   -> 保留 batch
90 objects, 38.4 MB | singles X | batch 1.68 | big ~1.6   -> 可改用 store.get 并简化
```

若 `big ≈ batch`，把 transition 换成一个 `get` 就值得：可以删掉 pinned arena、
`register_buffer`、`_store_state_host_plan` 和整套布局计算，维护面显著变小；
若 `big ≪ batch`（预计如此），保留 batch，`store.get` 仅作兼容 fallback（现状）。

注意：**“用 get”并不能省掉布局**，只是换了一种布局来源。

| 布局来源 | 内容 | 换 `store.get` 后 |
|---|---|---|
| wire v4 header（一个大对象里的 JSON：offset/dtype/shape/nbytes） | 每个 tensor 在哪、多大 | **必须保留**，它就是“不 deserialize”的依据 |
| per-tensor `store_layout`（约 90 个独立对象） | 为了能一次 batch 写入 arena | **可以删** |

也就是说：B 侧仍然是“一次 H2D + 零拷贝视图”（v4 现状），删掉的是 per-tensor 对象、
arena 地址计算、`register_buffer` 与 16B 对齐；A 侧也省一半写（现在 payload 和 90 个
对象各写一遍 38.4 MB）。

#### `batch_get_into_multi_buffers` 需要 `register_buffer` 吗？

目前没有定论，代码是 best-effort：

- `_register_store_buffer_best_effort()`（`mooncake_kv.py:1435`）失败只打一次 warning，
  不阻断；
- 真正决定成败的是 `batch_get` 返回负 status；
- 参考实现 `vllm/distributed/kv_transfer/.../mooncake/store/worker.py:1142` 在启动时
  注册 KV cache，失败只 `logger.error` 后照常调用该接口。

`run_store_read_path_probe.py` 现在会直接回答：首次调用用**未注册的 host buffer**，
失败则 `register_buffer` 后重试，并打印 `registration_required=True/False`。

- `False` → host 目标不需要注册，可以用最简路径（也就不必再 best-effort 注册 arena）；
- `True` → 注册是硬要求，那 11E-C 的 host arena 注册不能省，且要确认注册在无 RDMA
  的 TCP 上是否稳定成功（这会影响轮间波动，见 21.1）。

### 21.8 `store.get` 与 `batch_get_into_multi_buffers` 的底层差异（源码级）

依据本机 Mooncake 源码 checkout：`D:\WorkingDocs\CodexDirs\HYPIC\Mooncake`
（下文路径相对该目录）。**`store.get` 不是 batch 的封装**，两者是两条不同的 C++ 入口。

`store.get(key)`，每对象：

```text
mooncake-integration/store/store_py.cpp:283        MooncakeStorePyWrapper::get
  -> mooncake-store/src/real_client.cpp:1966       RealClient::get_buffer
  -> mooncake-store/src/real_client.cpp:1895       get_buffer_internal
       client_->Query(key)                         :1908   1 次元数据 RPC
       client_buffer_allocator->allocate()          :1940   分配客户端 buffer
       client_->Get(key, query, slices)             :1953   1 次传输
  -> mooncake-integration/store/store_py.cpp:300   pybind11::bytes(ptr, size)
                                                         整个对象拷进 Python bytes
```

`batch_get_into_multi_buffers(keys, addrs, sizes)`，N 个对象：

```text
mooncake-integration/store/store_py.cpp:2243       pybind 入口
  -> mooncake-store/src/real_client.cpp:3705       batch_get_into_multi_buffers
  -> mooncake-store/src/real_client.cpp:3738       _internal
       client_->BatchQuery(keys)                   :3765   1 次批量元数据 RPC
       按调用方地址建 slices（每 key 可多段）
       client_->BatchGet(keys, queries, slices)
         mooncake-store/src/client_service.cpp:994
         注释 “Submit all transfers in parallel”（:1025-1037）
         每 key 一个 TransferFuture，全部提交后统一 wait
  -> 返回 vector<int> 状态，不产生任何 Python 对象
```

对照：

| | `store.get` | `batch_get_into_multi_buffers` |
|---|---|---|
| 元数据 RPC | **每对象 1 次** `Query` | **每 N 个对象 1 次** `BatchQuery` |
| 传输 | 每对象 1 次 `Get`，**串行** | 一次 `BatchGet`，**所有 key 并发提交** |
| 数据落点 | 客户端 buffer → **Python bytes（全量拷贝）** | 直接写调用方地址，零拷贝 |
| 失败粒度 | 返回空 bytes | 每 key 一个状态码 |

这解释了 20.5/20.11 的全部现象：

- 6 层 native：`get` = 12 次 RPC + 6 次 Python 拷贝 → 86.5 ms / 0.62 GB/s；
- 90 对象 transition：`get` 会是 180 次 RPC；batch 是 1 次 BatchQuery + 一批并发传输
  → 22.8 ms / 1.68 GB/s；
- 单对象（probe 的 `1:38.4`）：差的是“1 次额外 RPC + 一次 38 MB Python 拷贝”，
  预计 5–10 ms，不会像 6/90 对象那样差 2 倍。

中间选项：`get_into(key, buffer, size)`（`real_client.cpp:2424`，pybind 也暴露）是
**单 key 零拷贝**，能省掉 Python bytes 那次拷贝，但 RPC 仍是每 key 2 次。因此对 90 对象
的 transition 无意义，对 6 层的 native 只省几毫秒。

结论：`get` 每对象两次 RPC 是结构性开销，不是调参能补的；block 粒度 + 一次 batch
（11E-D3）才是参考实现的形态。

### 21.9 Full / 本地复用 / 跨机复用三段对照（2026-10-22）

口径：4k 复用（4352 reused + 640 recompute）、`max_tokens=1`、`--enforce-eager`、
单请求。数据来源：Full 与本地列取第 10 节（DETAIL 开、分解完整）与第 17 节
（profiling 全关、129.2/194.1）；跨机列取 20.3 与 21.1。三处 Full 分别为
193.6 / 194.1 / 193.0 ms，差 <1 ms，属采样噪声。

#### 表 A：端到端 / forward / 其余

| | Full（完整重算） | 本地 PIC | 跨机 PIC（TCP） |
|---|---:|---:|---:|
| **端到端 p50** | **194.1 ms** | **129.2 ms** | **274.3 ms** |
| **forward（真实 prefill 计算）** | **133.8 ms** | **50.9 ms** | **65.0 ms** |
| **差值 = 框架/调度/HTTP + 传输** | **60.3 ms** | **78.3 ms** | **209.3 ms** |
| ├ 其中网络传输 | 0 | 0 | **145.5 ms** |
| └ 其中框架/调度/HTTP | 60.3 | 78.3 | 63.8 ms |
| **相对 Full 加速比** | 1.000× | **1.503×** | **0.708×** |
| 处理的 token 数 | 5000 | 640 | 640 |

自洽校验：

```text
Full      : 133.8 + 60.3                  = 194.1
本地 PIC  : 50.9 + 78.3                   = 129.2
            forward 省 82.9，框架多 18.0（pic_slot_attach 7.8 + range 二次准备 ~7
            + 其他），净省 64.9
跨机 PIC  : 65.0 + 63.8 + 145.5           = 274.3
            forward 省 68.8，传输多 145.5，框架多 5.4，净亏 82.1（实测 −81.3）
```

#### 表 B：跨机 PIC 的 209.3 ms 非 forward 构成

| 项 | ms | 占非 forward | 性质 |
|---|---:|---:|---|
| native KV 接收（6 × `store.get`，53.5 MB @ 0.62 GB/s） | 86.5 | 41% | **网络** |
| native KV 落地（cpu 4.3 + H2D 19.4） | 23.5 | 11% | 本地拷贝 |
| transition/state 传输（38.4 MB @ 1.08 GB/s） | 35.5 | 17% | **网络** |
| transition apply | 4.0 | 2% | 计算 |
| HTTP / 调度 / 采样 | 59.8 | 29% | 框架（Full 为 58.4） |
| 合计 | **209.3** | 100% | 其中 122.0 为网络传输 |

#### 表 C：forward 效率（PIC 的 forward 不是同比例缩小）

| | token 数 | forward ms | ms/token | 相对 Full |
|---|---:|---:|---:|---:|
| Full | 5000 | 133.8 | 0.0268 | 1.00× |
| 本地 PIC | 640 | 50.9 | 0.0795 | 2.97× |
| 跨机 PIC | 640 | 65.0 | 0.1016 | 3.79× |

640 个 query token 仍要对全部 4992 个 KV 做 attention，因此每 token 成本远高于
Full 的平均值；这是本地 PIC 只能到 1.503×（理论上限 1.84×）的原因。

**结论：三段里“框架/调度/HTTP”基本恒定（60–78 ms），本地复用把 forward 从 133.8
压到 50.9（省 82.9 ms）；跨机复用却在非 forward 里多付 145.5 ms 传输，是计算节省的
2.24 倍，所以净亏。优化传输（或减少传输字节）才是当前唯一能翻正的方向。**

### 21.10 跨进程读接口实测：`store.get` vs `batch_get_into_multi_buffers`（2026-10-22）

本节回答 21.7 里挂起的读路径问题。**关键前提：必须 `export MC_STORE_MEMCPY=0`。**
未设置时 Mooncake 会自动判定 “TCP-only, memcpy enabled” 并走 LOCAL_MEMCPY，单进程
探针测到的是内存拷贝（batch 2.7–3.0 GB/s 的假高值），不能用于比较。

方法：两个独立进程各自 `MooncakeDistributedStore().setup()` 连同一 master；A 只
`put` 并保持存活，B 在窗口内读。每条路径独立 warmup 5 次、测 10 次，报 first /
min / median / max。`segment=3 GiB`、`buffer=1 GiB`（两个 client 各 mount 8 GiB
会超总容量并 segfault）。

#### 结果（跨进程 TCP 稳态，单位 ms；GB/s 由 median 换算）

| case | 路径 | first | min | median | max | GB/s (median) |
|---|---|---:|---:|---:|---:|---:|
| **1:38.4**（1 对象 38.4 MB） | singles | 156.62 | 109.09 | 116.02 | 156.62 | 0.347 |
| | batch | 64.41 | 46.69 | 53.64 | 64.41 | **0.751** |
| | big | 92.69 | 51.11 | 58.92 | 92.69 | 0.683 |
| **6:53.5**（6 层 53.5 MB） | singles | 115.17 | 103.35 | 121.15 | 129.81 | 0.463 |
| | batch | 73.02 | 69.40 | 73.31 | 73.80 | **0.765** |
| | big | 74.53 | 72.65 | 75.68 | 76.25 | 0.741 |
| **54:38.4**（54 tensor 38.4 MB） | singles | 83.49 | 74.83 | 76.08 | 83.49 | 0.529 |
| | batch | 60.72 | 58.47 | 58.77 | 60.72 | 0.685 |
| | big | 54.21 | 52.42 | 52.74 | 54.21 | **0.763** |

换算倍数：

| case | batch / singles | batch / big |
|---|---|---|
| 1:38.4 | **2.16×**（median）/ 2.34×（min） | 0.91×（batch 快 9%） |
| 6:53.5 | **1.65×** / 1.49× | 0.97×（几乎相同） |
| 54:38.4 | 1.29× / 1.28× | 1.11×（big 快 11%） |

单进程（LOCAL_MEMCPY）对照：singles 0.78–1.46 GB/s、batch 2.70–3.07 GB/s、
big 0.57–0.77 GB/s——singles/batch 比跨进程高 2–4×，big 几乎一样，说明单进程读数
不能代表 TCP。

#### 结论

1. **native KV 改 batch 值得**：6:53.5 的 batch/singles = 1.49×（min）/1.65×（median）。
   注意探针的 singles 每层多一次 `buf[offset:]=raw` 切片拷贝，真实 native 路径没有
   这一步，因此按 0.672（min 比）保守缩放 86.5 ms → 约 58 ms，**预计省 28–35 ms**。
2. **transition 保持 batch，不改 big**：54:38.4 的 big 只快 6 ms（52.42 vs 58.47
   min），但 big 返回一整块 `bytes` 后还要按 54 个 tensor 的 offset/shape 切分
   （`torch.frombuffer` + reshape），而 batch 路径是零拷贝直写目标 buffer，切分成本
   会吃掉这 6 ms。
3. 跨进程 TCP 稳态 0.46–0.77 GB/s，与 20.4/20.11 的裸测一致；batch 相对 singles 的
   收益比单进程小（1.29–2.34× vs 2–4×），因为传输时间占大头，能省的固定开销占比变小。
4. **20.5 的“撤回 batch 化”结论正式作废**：传输上限确实存在（batch 也只有
   0.77 GB/s），但 singles 的 103–121 ms 里有约 34 ms 是 6 次调用的固定开销，
   batch 能省掉它。“省 47 ms”估算偏高，实际 28–35 ms。

#### 对实现的约束（本地代码核对）

- `native-receive-total` **是存在的**（`vllm/v1/pic/mooncake_kv.py:3286`），测试报告
  里 grep 的是 `native-total`（不存在的名字）。但该记录只打 `total_ms`，没有
  `get_ms` 聚合，建议补 `n_gets / per_get_median / per_get_max`；
- `DIRECT=0` 时 **H2D 本来就要付**（`native-materialize` 的 23.5 ms 里含 19.4 ms
  H2D），所以改 batch **不会新增 H2D 成本**，是纯收益；
- batch 到 host staging 后 `received.direct` 必须保持 **False**，否则
  `materialize_native_blocks` 的 `if received.direct: return`（:3722）会跳过
  H2D + index_copy；
- 目标地址无特殊对齐要求：探针用 `ctypes.addressof(bytearray)` 即可，pinned arena
  的 16B 对齐只用于 dtype 视图，不是 Mooncake 的要求；
- 不建议复用 transition 的 arena：两者生命周期与 key 不同，且 53.5 MB 与 38.4 MB
  尺寸不同；应按 external key 缓存并带引用计数。

### 21.11 TCP 下「GPU 直写」并不零拷贝（源码核对）

参考实现把 `batch_get_into_multi_buffers` 的目标写成 KV cache 的 GPU 地址，看起来是
零拷贝直落 HBM。但在无 RDMA 的 TCP 下，**transport 内部会退化为 host staging**：
`mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp`——

```cpp
39:   const static size_t kDefaultBufferSize = 65536;     // 64 KiB 分块
183:  bool is_cuda_memory = isCudaMemory(addr);
185:  if (is_cuda_memory) dram_buffer = new char[buffer_size];   // pageable host buffer
191:  asio::async_read(socket_, asio::buffer(dram_buffer, buffer_size));
219:  cudaMemcpy(addr + offset, dram_buffer, transferred_bytes, cudaMemcpyDefault);  // 同步 H2D
```

即每 64 KiB：分配 pageable buffer → recv → 同步 `cudaMemcpy` H2D。53.5 MB 对应约
818 次，38.4 MB 对应约 587 次。

对照 PIC 的 host staging：

| | 原生（TCP，GPU 目标） | PIC host staging |
|---|---|---|
| 中转 buffer | `new char[64K]`，pageable | pinned arena |
| 粒度 | 64 KiB × N | 整块一次 |
| H2D | 每 chunk 一次同步拷贝 | 一次大拷贝（2.8 GB/s 实测） |
| 额外 CPU | ~818 次 malloc/free 与同步等待 | 1 次 |

结论（三条，均影响后续决策）：

1. TCP 下「GPU 直写」不是零拷贝，只是把 H2D 搬进 C++；
2. **现有 host staging 是比原生路径更优的实现**，不是妥协——pinned + 一次大拷贝；
   由此 `transition-store-total` 的 `representation=store-host-staging` 是正确默认；
3. `store-gpu-direct` 在 TCP 上大概率**更慢**，此前"direct 可省 H2D"的假设作废。
   `HYPIC_PIC_TRANSITION_RECEIVE=gpu` 仅在将来出现 RDMA 时才值得重测。

### 21.12 11E-C native 批量接收验收（2026-10-22）

patch `B90EFAD5CAB38F81517605D0A35D2BF6`，三卡 / TCP / `MC_STORE_MEMCPY=0`、
`PROFILE_IMPORT=1`。三轮各 10 次取中位，RUN 1 离群（forward 79.4、transition 34.0），
RUN 2/3 收敛且几乎重合（211.3 / 210.2，差 1.1 ms）。

| 指标 | 11E-B | 11E-C（无 native batch） | 11E-C + native batch | 变化 |
|---|---:|---:|---:|---:|
| PIC p50 | 274.3 | 263.4 | **210.7** | −52.7 |
| 加速比 | 0.708× | 0.733× | **0.922×** | +0.189 |
| Full p50 | 193.0 | 194.8 | 194.3 | ~0 |
| native receive | 86.5 | 86.5 | **49.2** | −37.3 |
| native materialize | 24.6 | 24.6 | **7.9** | −16.7 |
| transition transfer | 35.5 | 41.6 | 41.0 | ~0 |
| transition apply | 4.0 | 3.0 | 3.3 | ~0 |
| PIC forward | — | 64.2 | **71.2** | **+7.0** |
| 未归因 | — | 130 | 38.1 | −92 |

功能门：130/130 请求走 `mode=batch`，`native-receive-layer`/`native-materialize-layer`/
compat payload / fallback 全为 0；pytest 118 passed。

#### 计时口径修正（重要，影响所有历史数字）

batch 分支最初漏了 `_pic_profile_sync`，`h2d_ms` 只测到 enqueue：
显示 6.8 ms，而真实拷贝时间漂到了 `PIC forward`（当时显示 79.61）。补齐同步后
`h2d_ms=7.6`、forward 回到 69.48。**凡是比较 11E-C 前后的 forward/native-materialize，
必须确认那一版是否已含同步**，否则会得出"scatter 拖慢 forward"的错误结论
（实测 `scatter_ms=0.8`，撑不起 5 ms 级别的差异）。

#### pinned 目标 buffer 的价值

同一份数据、同一个接口：跨进程探针用 `ctypes.addressof(bytearray)`（pageable）读
6 × 9.35 MB 需 **69.4 ms**，应用里读进 `pin_memory` arena 只要 **约 44–49 ms**。
差距 ~1.6×，并且使 H2D 达到 **7.0 GB/s**（53.5 MB / 7.6 ms）。
结论：目标 buffer 是否 pinned 与"选哪个接口"同等重要，这是 host staging 优于
直写 GPU 之外的第二个理由。

#### 仍未解释的 7 ms

PIC forward 64.2 → 71.2，而同期 Full forward 稳定 135.4–135.5，**不是会话漂移**。
在澄清以下两点前不要动代码：

1. 一次 53.5 MB 的 device temp 分配是否改变了 caching allocator 状态；
2. 所有稳态数字都在 `PROFILE_IMPORT=1` 下测得，两次 `cuda.synchronize` 可能在 forward
   前制造流水气泡。需要 `NATIVE_KV_BATCH=0/1` × `PROFILE_IMPORT=0/1` 的 2×2 对照。

### 21.13 Mooncake TCP 连接池默认关闭（必须设置，2026-10-22）

任何 11E-C 及之后的 patch 在跑 A 组前**必须**：

```bash
export MC_TCP_ENABLE_CONNECTION_POOL=1
```

必须在进程启动前设置（transport 构造时读一次，之后不再读）。

#### 症状

| 现象 | 值 |
|---|---|
| Round 1 | 正常（约 300 ms） |
| Round 2 起 | 崩溃（1600 ms 至 4.8 s） |
| `TIME_WAIT` | 0 → 23663（Round 1）→ 43248（Round 2） |
| 日志 | 数万条 `TcpTransport::getConnection failed ... Cannot assign requested address`，随后 `-800 TRANSFER_FAIL` |
| `ulimit -n` 调到 65535 | **无效**（不是 fd 耗尽，是 ephemeral port 耗尽） |
| VmRSS / swap / GPU 显存 | **稳定**（不是内存泄漏） |
| patch 版本 | 8B81202E、B90EFAD5、F6A70B10、54AD8DD0 **全部崩**，与 Python 代码无关 |

#### 根因（源码）

`mooncake-transfer-engine/include/transport/tcp_transport/tcp_transport.h:113`

```cpp
bool enable_connection_pool_ =
    false;  // Use MC_TCP_ENABLE_CONNECTION_POOL=1 to enable connection pool
```

池关闭时 `getConnection`（`tcp_transport.cpp:695-712`）每次 `asio::connect` 新建 socket，
用完即弃、不归还。每个传输对象一条连接：PIC 4k 请求约 6（native）+ 54（transition）
+ 1（manifest）= 61 条。一条连接走完占一个端口 60 秒（TIME_WAIT = 2×MSL，内核硬编码，
`tcp_fin_timeout` 管不到）。

稳态 TIME_WAIT ≈ 连接速率 × 60 s，而默认端口池 28k（32768–60999）对应可持续速率仅
约 466 条/s。11E-C 把 p50 从 271 推到 210（+30% 请求速率），再加上任何一次失败重试，
就会越线。这就是"同一台机器以前能跑、现在四个 patch 全崩"的原因：**以前是刚好没越线**。

#### 打开池之后

按 `(host, port)` 复用连接（`tcp_transport.cpp:715-771`，含 `in_use` 标记与空闲清理），
TIME_WAIT 降到几十，`Cannot assign requested address` 归零。实测加该变量后连续三轮无异常。

#### 开关删除进度（11E-E，逐步制）

11E-E 第一轮一次删了 9 个开关，引入 `layers=0` 空转与两个 NameError，已整体回退。
现在改为**一次只删一个**，每个 step 单独出一份从零完整 patch，并在真机上跑
R2（pytest）+ R4（跨进程 `native-receive-total`）确认不退化后才进入下一步。

**删除原则（2026-10-23 人工确认）：只删我们自己加的代码；原生 TransferEngine 及其
入口、配置项一律保留。** 因此原"11E-E5 删 11 个 TE 接线"的做法作废。

| 环境变量 | 状态 | 类别 |
|---|---|---|
| `HYPIC_PIC_TRANSITION_KEEP_PAYLOAD` | **A1 已删** | 胶水 |
| `HYPIC_PIC_COMPACT_TRANSITION` | **A2 已删** | 胶水 |
| `HYPIC_PIC_NATIVE_KV_OVERLAP` | 待删 A3 | 胶水 |
| `HYPIC_PIC_TRANSITION_RECEIVE` | 待删 A4 | 胶水 |
| `HYPIC_PIC_NATIVE_KV_BATCH` | 待删 A5 | 胶水 |
| `HYPIC_PIC_NATIVE_KV_DIRECT` | 待删 A6 | 胶水 |
| `HYPIC_PIC_PROFILE_DETAIL` | 待删 B1 | 胶水 |
| `HYPIC_PIC_PROFILE_FORWARD` | 待删 B1 | 胶水 |
| `HYPIC_PIC_PROFILE_MEMORY` | 待删 B1 | 胶水 |
| `HYPIC_PIC_PROFILE_MEMORY_SYNC` | 待删 B1 | 胶水 |
| `HYPIC_PIC_PROFILE_TIMELINE_SYNC` | 待删 B1 | 胶水 |
| `HYPIC_PIC_TRANSFER_ENGINE` | **保留** | 原生入口 |
| `HYPIC_PIC_ENGINE_ID` | **保留** | 原生配置 |
| `HYPIC_PIC_BOOTSTRAP_ADDR` | **保留** | 原生配置 |
| `HYPIC_PIC_TP_RANK` | **保留** | 原生配置 |
| `HYPIC_PIC_PP_RANK` | **保留** | 原生配置 |
| `HYPIC_PIC_STATE_WORKERS` | **保留** | 原生配置 |
| `HYPIC_PIC_TRANSFER_WORKERS` | **保留** | 原生配置 |
| `HYPIC_PIC_NATIVE_KV_BATCH_DESCRIPTORS` | **保留** | 原生配置 |
| `HYPIC_PIC_NATIVE_KV_TRANSFER_MODE` | **保留** | store/p2p 路由 |
| `HYPIC_PIC_TRANSITION_TRANSPORT` | **保留** | store/p2p 路由 |
| `HYPIC_PIC_PROFILE_IMPORT` | 保留 | 诊断 |
| `HYPIC_MOONCAKE_KEY_PREFIX` | 保留 | 多实例隔离 |
| `HYPIC_PIC_NATIVE_ARENA_CACHE_BYTES` | 保留 | 预算 |
| `HYPIC_PIC_VERIFY_CHECKSUM` | 保留 | 校验 |

合计：**删 11 个（已完成 2）、留 14 个**。`vllm/v1/pic/transfer_engine.py` 与上游
`vllm/distributed/kv_transfer/kv_connector/mooncake` 全程不删。

**交付形态**：每个 step 给的是**从零完整 patch**（打干净 docker），另附 delta 便于
在上一 step 的树上增量验证；cumulative 不可打在已打补丁的树上——`patch --forward`
会把整个文件判为 "previously applied" 并跳过。

#### 对历史数字的影响

271.1 / 210.7 两个基线都是在**池关闭**下测的，因此它们包含了每请求约 60 次 TCP 三次握手
的开销。打开池后 PIC p50 预计**更低**，基线必须重新测量；在此之前不要把新旧数字混比。

#### 排查清单（下次再遇到）

```bash
sysctl net.ipv4.tcp_tw_reuse net.ipv4.tcp_fin_timeout net.ipv4.ip_local_port_range
cat /proc/net/sockstat                    # TIME_WAIT 数量
ss -tan | awk 'NR>1{print $1,$5}' | sort | uniq -c | sort -rn | head
```

### 21.14 2x2 前置条件与当前最优配置（2026-10-22）

`HYPIC_PIC_NATIVE_KV_DIRECT=0` + `MC_TCP_ENABLE_CONNECTION_POOL=1` 是当前最优配置，
10 轮全跑完：

| 指标 | 值 |
|---|---|
| PIC p50 | 209–220 ms（9 轮中位 **约 215 ms**） |
| 加速比 | 0.875–0.931×（中位 **约 0.90×**） |
| Full p50 | 192.8–195.4 ms |
| `TIME_WAIT` 峰值 | **1750**（无池时为 43248） |
| `Cannot assign requested address` | **0**（无池时 28528） |
| native-materialize-total(mode=batch) | 7.7–7.9 ms（与历史 7.8 一致） |
| transition-apply | 3.0–3.2 ms |
| PIC forward | 65–74 ms（与历史 71.2 一致） |

#### direct 与 host staging 的实测差

| native 路径 | 耗时 |
|---|---:|
| `DIRECT=1`（GPU direct，`native-direct-total.get_ms`） | **70–74 ms** |
| `DIRECT=0`（`native-receive-batch` 49 ms + `native-materialize-batch` 7.8 ms） | **56.8 ms** |

差 13–17 ms，与 21.11 的源码结论一致：TCP 下 GPU 目标退化为 64 KiB 分块的
pageable 暂存 + 同步 `cudaMemcpy`。因此 **`HYPIC_PIC_NATIVE_KV_DIRECT` 默认值改为 0**
（RDMA 部署再显式设 1），并且 `mooncake_kv` 在 import 时用
`os.environ.setdefault("MC_TCP_ENABLE_CONNECTION_POOL", "1")` 兜底连接池，
显式设 0 时仍然尊重用户选择。

#### 2×2 矩阵（全部使用 `MC_TCP_ENABLE_CONNECTION_POOL=1`、`DIRECT=0`）

| 组合 | `NATIVE_KV_BATCH` | `PROFILE_IMPORT` | 用途 |
|---|---|---|---|
| A | 1 | 1 | 已测：215 ms / 0.90× |
| B | 1 | 0 | 去 profile 同步气泡后的真实值 |
| C | 0 | 0 | 逐层 get 的干净对照 |
| D | 0 | 1 | 交叉验证 |

`PROFILE_IMPORT=0` 之前失败（`external_native_kv_import_failed`）主要是无连接池导致的
端口耗尽，不是数据竞态；但 profile 关闭时确实没有 `cuda.synchronize` 排空 H2D，
因此本版给 arena 加了 upload `torch.cuda.Event`：租借时若上一个上传未完成
（`event.query()` 为 False），改用一次性 buffer，**不阻塞**、也不写正在被读取的内存。

### 21.15 2×2 对照结果与新验收基线（2026-10-22）

patch `3A09990D9540E52B0ED5EE15E6012CA7`（默认 `DIRECT=0` + 连接池 + PROFILE=0
竞态防护 + 熔断），三卡 TCP，每组 10 轮取 9 轮中位。

| 组合 | `NATIVE_KV_BATCH` | `PROFILE_IMPORT` | PIC p50 | ratio | 判定 |
|---|---|---|---:|---:|---|
| A | 1 | 1 | 约 215 ms | 约 0.90 | 当前采用（可观测） |
| B | 1 | 0 | **214.2 ms** | **0.908** | 与 A 差 <1 ms → profile 气泡可忽略 |
| C | 0 | 0 | 约 278 ms | 约 0.68 | 逐层路径慢 65 ms |
| D | 0 | 1 | 跳过 | — | 与 C 同量级，无信息量 |

功能门（A/B/C 全过）：`prefill=130 / reuse=130 / apply=130 / fallback=0 / compat=0 /
native_layer=0 / TE=0 / OOM=0 / cannot_assign=0`。

结论：

1. **`PROFILE_IMPORT` 的两次 `cuda.synchronize` 只值 <1 ms**，A≈B。保留 `PROFILE=1`
   以获得 `native-*-total` 观测，不影响性能判断；
2. **`NATIVE_KV_BATCH=1` 是这轮唯一的大项**：A/B 相对 C 省 65 ms；
3. `DIRECT=0` + 连接池是稳定性前提（21.13/21.14），已设为默认。

#### batch 相对逐层的拆解（含一处口径提醒）

| 段 | 逐层 | batch | 差 |
|---|---:|---:|---:|
| native receive | 约 86.5 ms（6×`store.get`） | 约 49 ms（1×batch） | −37.5 |
| native materialize | 约 24.6 ms（逐层 H2D） | 约 7.8 ms（1 次 H2D + scatter） | −16.8 |
| PIC forward | 约 64 ms | 约 71 ms | **+7** |
| native 净差 | | | **−47** |

**口径提醒**：观测到的 A/B 与 C 的总差是 65 ms，而上面三行只解释 47 ms。差异来自
`forward=64` 这个数是**历史逐层会话**测得的，不是 C 组当轮实测。因此 18 ms 的残差
在拿到 C 组的 `PIC forward` 实测值之前**不可归因**，不要凭这张表下"batch 省 65 ms
全靠 native"的结论。

#### 当前验收基线（替换 21.12 的 210.7）

| 项 | 值 |
|---|---|
| 配置 | `TRANSFER_ENGINE=0`、`NATIVE_KV_DIRECT=0`（默认）、`NATIVE_KV_BATCH=1`、``MC_STORE_MEMCPY=0`、`MC_TCP_ENABLE_CONNECTION_POOL=1`（默认）、`PROFILE_IMPORT=1` |
| PIC p50 | **214–215 ms** |
| Full p50 | 194.5 ms |
| ratio | **0.90×** |
| 对比 11E-B | 274.3 → 214.5，**−22%** |
| TIME_WAIT 峰值 | 1750 |

#### 剩下的空间（按实测排序，均已接近上限）

| 段 | 当前 | 可动空间 | 说明 |
|---|---:|---|---|
| native receive | 49 ms | 小 | TCP 0.7–0.77 GB/s 的带宽上限 |
| transition-store-total | 30 ms | 最多约 5 ms | 探针同字节数曾到 25 ms |
| transition-transfer | 41 ms | 需先拆清 11 ms 的额外部分 | 含 store-total 30 |
| PIC forward | 65–71 ms | 小 | batch 比逐层多 5–7 ms，是 host staging 的固有代价 |

**判断**：5 ms 级别的优化不会改变"TCP 下 0.90×"这个结论；能翻转符号的只有
RDMA（11E-F）。因此性能工作到此冻结，下一步转入稳定性（p95/p99）与 11E-E 清理。

### 21.16 3A09990D 基线复核（2026-10-23，R1–R6 全过）

patch `3A09990D9540E52B0ED5EE15E6012CA7`（`11E-C-3A09990D-baseline.patch`），
三卡 / TCP / 容器 `hypic_vllm-openai-v0.22.1-cu129`。

| 检查 | 结果 |
|---|---|
| R2 静态门 | `pytest tests/v1/pic` **148 passed, 0 failed**（3 个无害 warning） |
| R3 环境 | 只有 `MC_STORE_MEMCPY=0` + `HYPIC_PIC_PROFILE_IMPORT=1`，无 stale export |
| R1 性能 | PIC p50 **210.9 / 212.7 / 211.5 ms**，ratio **0.914 / 0.915 / 0.919**（Full 192.9–194.7） |
| R4 `native-receive-total` | **39**（39 个请求各一次真实 Store 传输） |
| R4 `native-receive-batch` | `layers=6 bytes=53477376 get_ms=29.4–44.4` |
| R4 `prefill_tokens` | PIC **640 × 39**，Full **5000 × 39** |
| R4 `transition` | `store-host-staging=39`，`compat-payload=0`，`store-gpu-direct=0` |
| R4 `forward` | PIC `forward_ms=51–73`（首条 Full 946 ms 是 Triton JIT 冷启动） |
| R4 `lo` 流量 | 累计 86.9 GB，39 × 91.9 MB ≈ 3.58 GB 的传输在其中 |

结论：**跨进程复用成立，基线可信**。与 §21.15 的 209–220 ms / ~0.90× 一致。

`pic_check_unbound.py` 未能执行（patch 只导出 `vllm/ tests/ docs/`，`tools/` 不在其中），
这是导出范围导致的，不是检查失败。

#### 由此确立的删除纪律

1. **一次只删一个开关**，每个 commit 单独出一份 patch；
2. 每份都跑 R2 + R4（`native-receive-total ≥ 1`、`prefill_tokens=640`、
   `store-host-staging`、`compat-payload=0`）与一轮 PERF（PIC 210–215 ms / 0.90–0.92×）；
3. 任一项退化立刻停，回退到上一个 commit；
4. 11E-E 第一轮的教训：一次删 9 个 → `layers=0` 空转 → 417 ms。

删除顺序（按"删了也不改变默认路线"的先后）：`TRANSITION_KEEP_PAYLOAD` →
`COMPACT_TRANSITION` → `TRANSITION_RECEIVE` → `NATIVE_KV_BATCH` →
`NATIVE_KV_TRANSFER_MODE` → `TRANSITION_TRANSPORT` → `NATIVE_KV_OVERLAP` →
`NATIVE_KV_DIRECT` → `TRANSFER_ENGINE`。

### 21.17 A4+A6 封版（2026-10-23，新基线）

patch `9BF2AC19818A36481DBCAD806F098B7A`（1,275,987 字节，75 文件，commit `4ab43b330e`），
上游 base `0decac0d`，纯净环境仅设 `MC_STORE_MEMCPY=0` + `HYPIC_PIC_PROFILE_IMPORT=2`。

| 检查 | 结果 |
|---|---|
| 性能（重跑） | PIC p50 **212.5 / 212.9 ms**，ratio **0.913 / 0.921**（Full 194.4–195.6） |
| pytest | **142 passed**，3 条 `torch.frombuffer` 只读 buffer 警告（非失败） |
| 跨进程 | `native-receive-total=39`、`transition-store-total=39`（`store-host-staging`、`tensors=54`、`bytes=38412288`） |
| 复用 | `external_kv_imported=39`、`reused_tokens=169728`、`recompute_tokens=24960`、`forward_tokens=219960` |
| 输出一致 | max_tokens=10 时 PIC 与 Full 逐 token 一致 |
| 无回退 | `fallback_reasons={}`、`compat-payload=0`、`mode=layer=0`、`native-direct-total=0`、`store-gpu-direct=0` |
| 首轮 222–235 ms | 定位为首轮冷启动 + 单请求尖刺（同请求 native 87 ms 与 apply 91.7 ms 并发），重跑不复现，非 A6 回归 |

删除面：A4 去 `TRANSITION_RECEIVE`（含 GPU tier 与 `_store_state_device_targets`），
A6 去 `NATIVE_KV_DIRECT`（含 `_receive_native_blocks_direct` 与 direct 描述符）。
开关总数 **25 → 18**。

已知（非缺陷）：`PROFILE_IMPORT=2` 不打开 `forward` 打点——该记录仍由独立的
`HYPIC_PIC_PROFILE_FORWARD=1` 控制（`gpu_model_runner.py:300`），B 组合并后 `=2`
才会自动包含。需要时显式 `export HYPIC_PIC_PROFILE_FORWARD=1`。

### 21.18 11E-E-B 封版（2026-10-23，新基线：两态打点）

patch `B2F6EFD670EB3E7302DBA3D8501BB381`（1,278,919 字节，76 文件，commit `fed63e3901`），
base `0decac0d`，含 A1–A6 与 B 组。

**打点收敛为两态**：`HYPIC_PIC_PROFILE_IMPORT` 未设/`0`/非数字 → 关闭（生产）；
任意正整数 → 全开（分析，含 `forward_ms` 与阶段同步）。5 个细分开关
（`PROFILE_DETAIL`/`FORWARD`/`MEMORY`/`MEMORY_SYNC`/`TIMELINE_SYNC`）删除后不再读取。
开关总数 **25 → 14**。

| 检查 | 结果 |
|---|---|
| 生产档 `=0` 性能 | PIC p50 **209.4 / 210.1 / 211.8 / 211.9 / 212.1 ms**，ratio **0.911–0.926**（Full 193.2–195.0） |
| 生产档 gate | `PIC-PROFILE=0`、`timeline=0`、`forward=0`、`metrics snapshot=130`、`PIC-DEBUG=2081` |
| pytest | **142 passed**（3 条 `torch.frombuffer` 只读 buffer 警告） |
| 复用 | `external_kv_imported=65→66`、`reused_tokens=282880`、`recompute_tokens=41600`、`forward_tokens=366600`、`lookup 65/65` |
| 无回退 | `fallback_reasons={}`、`compat-payload=0`、`native-receive-layer=0` |
| 输出一致 | max_tokens=10 时 PIC 与 Full 逐 token 相同 |
| 分析档 `=1` | 分段打点齐全：native receive 45.1–55.2、transition store 24.0–31.7、H2D 7.6–8.4、apply 3.7–6.2、forward(PIC) 51–72 / (Full) 132–214 |

#### 关键教训：bench 脚本覆盖了 profile 变量

此前"`=0` 仍 5426 条打点、PIC 230–244 ms"的结论**作废**。真因是
`mooncake_bench.sh` 第 47 行写死 `export HYPIC_PIC_PROFILE_IMPORT=2`，覆盖了父 shell 的 `=0`，
于是"关闭档"实际跑的是采集档。脚本已改为
`export HYPIC_PIC_PROFILE_IMPORT=${HYPIC_PIC_PROFILE_IMPORT:-0}`。

**此后每次 bench 前必查**：

```bash
grep -n 'export HYPIC_PIC_' mooncake_bench.sh
```

任何写死的 PIC 变量都可能让"对照实验"失效。

### 21.19 11E-E-C1b 封版（2026-10-26，累计 patch）

patch `7595C6ECA598F1F45FE5F5A3B4940A01`（1,203,062 字节，84 文件，commit `42845d8b2c`），
从上游 `0decac0d` 单份累计打入（规则见 stage_plan §5.1）。

| 检查 | 结果 |
|---|---|
| 指纹 | 5 个文件行数 + MD5 全部与 README 一致 |
| 静态门 | compileall 0、unbound clean（308 文件）、removed clean（49 符号）、TE 模块已删 |
| pytest | **127 passed**（新增 `test_async_import` 9 + `test_async_import_config` 4） |
| 守卫 | 默认 False；`mooncake=False` → ValueError；`mooncake=True` → NotImplementedError（1c 未落地） |
| 关档 `=0` | PIC **213–216 ms**、Full 193.2–195.1、ratio **0.900–0.912**；`PIC-PROFILE=0`、`PIC-MEMORY=0`、`async import=0` |
| 正确性 | `external_kv_imported=65`、`reused=282880`、`recompute=41600`、`forward_tokens=366600`、`fallback_reasons={}`、`pool.free_bytes=2 GiB` |
| 并发探针 | c=1 **205.07 ms**（C1a 211.9，−3.2%）、c=2 **455.32**（C1a 482.4，−5.6%） |
| 显存 | `=1` 连跑 5 轮，used 14889 MiB 不变、allocated 峰值 13.98 GiB、cleanup 后 9.22 GiB，无泄漏 |

已知问题（非 C1b 引入）：

1. c=2 偶发 allocator 碎片 OOM（`torch.empty_like` 20 MiB），单请求 4992-token GDN
   prefill 峰值约 4.76 GiB + 碎片；5 轮不复现。建议 `--max-num-batched-tokens 4096`，
   并避免 A/B 两个实例共用同一张卡。
2. `=1` 下出现 `prefill_tokens=4992` 的全量 forward（lookup miss），属于**首次发布**
   的预期行为，不是缺陷。
3. `=0` PIC 213–216 比 C1a 的 211–213 高 2–4 ms，但同轮 c=1 探针为 205.07（更快），
   判断为共卡噪声，1c 落地后统一复测。

### 21.20 W1 理论上限（按耗时拆解计算）

数据全部取自本记录已测的分项（batch 路径，单机 TCP，PIC 640-token forward）：

| 项 | 耗时 | 说明 |
|---|---:|---|
| native-receive | 47.6 ms | 53.5 MB / 6 层 |
| transition | 26.3 ms | 38.4 MB |
| H2D materialize | 7.8 ms | 53.5 MB 索引拷贝 |
| **通信合计** | **81.7 ms** | 91.9 MB/请求 |
| forward | 70.9 ms | `prefill_tokens=640` |
| prepare/postprocess/sample | 24.0 ms | forward 之外 |
| **计算合计** | **94.9 ms** | |
| 固定开销 | 34.4 ms | 由 c=1 p50 211 ms 反推（HTTP/API/engine） |

### 模型

单请求串行：`211.0 ms = 81.7 + 94.9 + 34.4`。

流水后，一个请求的通信可以躲在别的请求的计算后面，每请求下限为
`固定 + max(通信, 计算)`。**GPU 是串行的**，并发只隐藏通信，不缩短任何请求自己的计算，
所以 c≥2 的上限相同：

```
上限 = (通信 + 计算 + 固定) / (max(通信, 计算) + 固定)
     = (81.7 + 94.9 + 34.4) / (94.9 + 34.4)
     = 211.0 / 129.3
     = 1.63x
```

| 口径 | 上限 |
|---|---|
| 只看变量部分 | **1.86×** |
| 含固定开销（真实口径） | **1.63×** |
| 吞吐上限 | **10.5 rps**（计算受限；链路可到 12.2 rps） |

关键转折：batch 路径把通信从 137.9 ms 降到 81.7 ms 之后，瓶颈**从通信变成计算**
（94.9 > 81.7）。老 get 路径下上限只有 1.22×（通信受限），batch 路径才把上限抬到 1.63×。

### 与实测对账（c=2）

| 项 | 值 |
|---|---:|
| 关档 p50 | 393.6 ms（每请求 196.8） |
| 开档 p50 | 312.6 ms（每请求 156.3） |
| 实测倍数 | **1.26×** |
| 理想每请求 | 129.3 ms |
| 差距 | 27.0 ms/请求 ≈ 54 ms/2请求 |

差距的三段来源：

1. **重复 inline 导入 ≈ 26 ms**：adopt 未生效，park 时导入一次、promote 后又导入一次，
   第二次落在关键路径上（transitions 26.3 ms 完全对得上）。
2. **同一时刻只有一个请求 park**：为防"全 park → 空步不派发"死锁，保留了"最后一个可运行"，
   于是重叠窗口只有对方一次 forward 的时长。
3. **固定开销 34.4 ms 不可隐藏**。

> 注意：上表的「PIC 独有准备 11.7 ms」取自分析档（`HYPIC_PIC_PROFILE_IMPORT > 0`），其中 `_materialize_pic_private_kv` 内部有
> 7 处 `torch.cuda.synchronize`（`gpu_model_runner.py:419`），数字包含同步停顿。
> 生产档的真实成本需单独测量，见 stage_plan §7.7；若生产档只有 1–2 ms，
> 则 1.63× 的上限不需要因这 11.7 ms 而上调。

修掉第 1 段后预期 ≈ 1.36–1.40×；要接近 1.63× 必须解决第 2 段（放开并发 park 数），
而要突破 1.63× 只能降计算（减少 forward token），W1 本身到此为止。

### 21.21 11E-E-C1c 以 B 组验收（2026-10-28）

patch `A10B00A58CF3B39C99224453B72A3DE6`（1,290,526 字节，89 文件，commit `b2fb26ca00`，
base `0decac0d`）。容器 `hypic_vllm-openai-v0.22.1-cu129`，RTX 4090 ×3，Mooncake
0.3.10.post2，TCP，`--enforce-eager`，`HYPIC_PIC_STATE_WORKERS=4`。

#### B 组（同 prompt 复用，c=2）— 主验收：通过

| 指标 | 关档 | 开档 | 加速 |
|---|---:|---:|---:|
| p50 | 393.6 ms | **229.3 ms** | **1.72×** |
| tput | 5.13 rps | **8.22 rps** | **1.60×** |
| p95 | — | 253.6 ms | — |
| 成功/失败 | 20/0 | **20/0** | — |

判据（14 项全绿）：`PARK=SUBMIT=ADOPT=PROMOTE=DELIVER=38`、`FAILED=0`、`TIMEOUT=0`、
`skipped=no_destination=0`、`empty_refs|empty_spans=0`、`requires a transition handle=0`、
`parked_result=True=38`、`Traceback/EngineDead=0`、时序检查
`clean: 38 submit, 38 ready, 0 failed, 38 adopt`。

`no_result=46` 属**设计内**：这些是 `skip park reason=last_runnable` 的请求（为防"全 park →
空步不派发"死锁，每步保留一个可运行），它们没有 park 因而没有结果，随后走 inline 导入并正常完成。

单请求：Full 193.5 ms / PIC 208.7 ms（0.927×），符合"c=1 不 park、无重叠"的设计预期。

#### A 组（20 个不同 prompt，c=2）— 触发两个新 bug，记为 C1d

A 组 20 组 prompt 各带 `<<PIC_SEP>>`，prefix 长度 4967–5015 token，publish 全部 200，
capture 正常；随后 reuse 阶段出现：

| # | 现象 | 位置 | 性质 |
|---|---|---|---|
| 1 | `ValueError: native KV block span mismatch: expected 119, got 135` | `mooncake_kv.py:1103 publish_native_blocks` ← `_materialize_pic_public_kv` ← `_capture_completed_pic_states` | 软错误，被捕获，仅 capture 失败 |
| 2 | `ValueError: PIC transition slicing is unsupported for this state kind` | `state.py:816 PICTransitionOperator.slice` ← `_operator_for_range` ← `apply_transition` ← `apply_matched_mamba_transition` | **硬崩溃**，EngineCore 退出，17/20 HTTP 500 |

B 组没触发的原因：固定 prompt 长度让 seam 区间恰好等于整段（`state.py:813` 的全段分支），
且 block 数与 token 对齐；A 组长度在 4967–5015 之间浮动，两个前提都不成立。

## 22. C1c 性能结论：同源热路径对照（2026-10-30 固化）

> 本节数字来自第一次同源对照。之后修掉了 warmup 污染，并在 §23 重测——**引用请用 §23**。


### 22.1 为什么必须"同源 + 热路径"

本轮之前的每一个"A 组慢"的结论都倒掉了，原因只有两条：

| 错误 | 后果 |
|---|---|
| 基线来自不同轮次 | A 组关档一度引用 201.8 ms（疑似 c=1 数据），与 B 组关档 393.6 ms 相差 1.95 倍 |
| 用冷启动数据对比热路径 | A 组 c=1 曾报"开档慢 41%"，复测热路径只有 209.6 vs 214.8 ms（**开档更快**） |

现在的口径：**同一实例、同一批 prompt、同一并发、先关档再开档、每格跑两次只取第二次。**

### 22.2 八个格子（PROFILE_IMPORT=0，生产档）

| 组 | 并发 | 档 | wall | p50 | tput |
|---|---|---:|---:|---:|---:|
| A | 1 | 关 | 4347.2 ms | 214.8 ms | 4.60 rps |
| A | 1 | 开 | 4181.7 ms | 209.6 ms | 4.78 rps |
| A | 2 | 关 | 3796.7 ms | 370.6 ms | 5.27 rps |
| A | 2 | 开 | 2866.8 ms | 255.2 ms | 6.98 rps |
| B | 1 | 关 | 3960.9 ms | 196.7 ms | 5.05 rps |
| B | 1 | 开 | 4453.1 ms | 221.3 ms | 4.49 rps |
| B | 2 | 关 | 3847.4 ms | 380.3 ms | 5.20 rps |
| B | 2 | 开 | 2694.1 ms | 245.1 ms | 7.42 rps |

### 22.3 收益

| 对比 | p50 比 | tput 比 | 结论 |
|---|---:|---:|---|
| A 组 c=2 开/关 | **0.689×** | **1.324×** | PIC 快 24–31% |
| B 组 c=2 开/关 | **0.644×** | **1.427×** | PIC 快 36–43% |
| A 组 c=1 开/关（热路径） | 0.976× | 1.039× | 持平略快 |
| B 组 c=1 开/关（热路径） | 1.125× | 0.889× | 同段单请求略慢 11% |

并发收益（c=2/c=1 makespan）：开档 A **0.425**、B **0.605**；关档 A 0.782、B 0.971。

### 22.4 口径与限制

| 项 | 说明 |
|---|---|
| 硬件 | 3× RTX 4090、Xeon 4210R，TP=1，`--enforce-eager` |
| 传输 | Mooncake 0.3.10.post2，**TCP，无 RDMA** |
| 形状 | A 组 20 个不同 4.8K doc；B 组同一 doc × 20 |
| PIC 不做合批 | 一次 forward 只跑一个请求，因此并发收益**只**来自"导入被别人的 forward 覆盖" |
| 冷启动 | 开档冷启动比关档多约 1.2 s（park 首次初始化），一次性开销 |
| 未解释 | 冷启动两次运行的延迟均**逐请求上升**（225→360 / 248→421），热路径平稳；已加 `[PIC-TRACE] finish free_blocks=` 待查 |

### 22.5 一句话

> **在同一实例、同一负载、热路径、生产档下：c=2 时 PIC 把 p50 降到关档的 0.64–0.69、吞吐提到 1.32–1.43 倍；c=1 持平。收益来自异步导入被并发 forward 覆盖，而不是更快的单请求。**


## 23. 收益复测与“对象数”诊断（2026-10-30 晚固化）

### 23.1 这一轮修掉的测量缺陷

| 缺陷 | 造成的假象 | 修法 |
|---|---|---|
| warmup 发 `prompts[:5]` | 前 5 个测量请求命中已读对象 | warmup 改用 publish prompt + B 探针 doc |
| 冷启动与稳态混在一起 | “c=1 开档慢 41%”，热路径其实快 2–4% | 只报 warmup 之后的稳态 |
| 底线计数整轮累计 | 单个格子没复用也看不出来 | 每格记录日志区间并单独计数 |

### 23.2 八个格子（`--enforce-eager`，PROFILE=0，引擎已预热）

| 组 | 并发 | 档 | wall | p50 | tput |
|---|---|---:|---:|---:|---:|
| A | 1 | 关 | 3816.6 ms | 190.2 ms | 5.24 rps |
| A | 1 | 开 | 7616.1 ms | **388.8 ms** | 2.63 rps |
| A | 2 | 关 | 2981.5 ms | 290.4 ms | 6.71 rps |
| A | 2 | 开 | 2721.6 ms | **250.5 ms** | 7.35 rps |
| B | 1 | 关 | 4092.2 ms | 203.4 ms | 4.89 rps |
| B | 1 | 开 | 4616.9 ms | 230.9 ms | 4.33 rps |
| B | 2 | 关 | 2915.7 ms | 282.1 ms | 6.86 rps |
| B | 2 | 开 | 2848.8 ms | 260.3 ms | 7.02 rps |

| 对比 | p50 比 | tput 比 | 结论 |
|---|---:|---:|---|
| A c=1 | 2.044× | 0.502× | 首次读暴露，**净亏** |
| A c=2 | 0.863× | **1.095×** | 赢 9.5% |
| B c=1 | 1.135× | 0.885× | 亏 11% |
| B c=2 | 0.923× | 1.023× | 赢 2% |

### 23.2b A 组 c=1 开档的两种口径（按交替 doc 刷新）

`388.8 ms` 是“20 个不同 doc、每个都首次读”的口径，其中含约 260 ms 的对象级首次开销。
把对象数压到 2（doc0/doc1 交替，仍然每请求真传）后重测：

| 口径 | 对象数 | p50 | wall | tput | 说明 |
|---|---:|---:|---:|---:|---|
| A c=1 开（distinct） | 20 | 388.8 ms | 7616.1 ms | 2.63 rps | 含对象级首次开销 |
| **A c=1 开（alternate2）** | **2** | **210.0 ms** | **4140.8 ms** | **4.83 rps** | 去掉对象数饱和 |
| A c=1 关（对照） | — | 190.2 ms | 3816.6 ms | 5.24 rps | — |

对应比值：

| 对比 | p50 比 | tput 比 |
|---|---:|---:|
| alternate2 / 关档 | **1.104×** | **0.922×** |
| distinct / 关档 | 2.044× | 0.502× |

**读法**：同一次复用、同一份代码、同样 20 次真传，只把对象数从 20 降到 2，
p50 就从 388.8 回到 210.0。所以：

- `210.0 ms` 才是“PIC 单请求本身”的成本（≈ 关档 + 20 ms）
- `388.8 ms` 里多出的 **178.8 ms** 是“每个新对象一份资源”的开销

跑法（只跑这一格）：

```bash
ONLY="A c=1 on" A_MODE=alternate2 bash tools/pic_ab_gain_probe.sh
```

### 23.3 关键结论一：c=2 已经是“步数受限”，不是“链路受限”

```
A c=2 开档 wall = 2721.6 ms
每步 ≈ 136 ms（p50 250 ≈ 2 步）
20 步 × 136 = 2720  ← 与实测 2721.6 一致
```

**导入已被完全藏住**（若链路受限，wall 会随 Σ导入/并发 缩放，实测是 N×step）。因此：

| 并发 | 预期 wall | 说明 |
|---|---:|---|
| c=2 | 2720 | 实测 2721 ✅ |
| c=4 / c=8 | **仍是 2720** | PIC 一步一个请求，步数固定 |

**提高并发不能继续增加掩盖机会。** 想更快只能降 step 或合批。

### 23.4 关键结论二：首次读的额外开销由“对象数”决定

三个实验，传输次数都是 20 次、每次都是真传：

| 实验 | 对象数 | p50 | 形状 |
|---|---:|---:|---|
| A distinct（20 个不同 doc） | 20 | **388.8 ms** | 前 2 个 279/254，第 3 个起 381–419 |
| A alternate2（2 个 doc 交替） | 2 | **210.0 ms** | 186–228，平稳 |
| B（同 doc × 20） | 1 | **230.9 ms** | 平稳 |

由 `A = 开档 − 关档 + (forward_关 − forward_开)` 反推每请求导入：

| 场景 | 导入 | 相对链路满速（83 ms） |
|---|---:|---:|
| 20 个不同对象 | **≈ 260 ms** | 3.1× |
| 1–2 个对象 | **≈ 90 ms** | 1.1× |

**总量几乎相同（1.87 GB vs 1.83 GB），所以不是带宽、也不是“连续传输降速”。** 唯一变量是对象数。

`seq` 形状（前两个便宜、第三个起变贵并保持）说明是**按对象建立的资源**，而不是“每请求固定成本”。

### 23.5 嫌疑定位：每对象一份 pinned staging arena

`vllm/v1/pic/mooncake_kv.py` 的转移算子落在一个 pinned host arena 上：

```python
cache = entry.setdefault("host_staging", {})      # entry 的 key 是对象
host = torch.empty(total, dtype=torch.uint8, pin_memory=on_cuda)   # 38 MB 锁页
cache["host_registered"] = self._register_store_buffer_best_effort(...)
```

| 事实 | 后果 |
|---|---|
| arena 大小 = 该对象自己的字节数 | 20 个 doc 长度不同 → 20 个尺寸 → **无法复用 torch 的 pinned 缓存块** |
| 请求结束 `users==0` 即释放 | 下一请求重新 pin + register |
| 同 doc / alternate2 | 尺寸固定或只有两种 → 命中缓存 → 便宜 |

**这把“接收缓冲池只有 2–3 个”的现象解释成“尺寸不同导致无法复用”，根因在我们代码里。**

已加打点（`[PIC-PROFILE] staging-arena bytes=... previous=... alloc_ms=... register_ms=...`），
跑一格 A c=1 开档（PROFILE=1）即可判定：

| 观察 | 结论 | 修法 |
|---|---|---|
| `alloc_ms` 单次 30–100 ms | 我们的 pinned 分配是主因 | 固定尺寸 arena（或 N=STATE_WORKERS 份轮转），只 pin/register 一次 |
| `register_ms` 单次 30–100 ms | 传输层注册是主因 | 与 Mooncake 确认注册缓存策略 |
| 两者都 < 5 ms | 在 store 侧 | 转查 `native-receive get_ms` |

若证实在 arena：A c=1 开档 p50 预期 **389 → ~220 ms**（与 B 持平），c=2 也会小幅下降。

### 23.5b 修复：固定尺寸的 staging pool（已实现，待实测确认）

把“每对象一份、尺寸各异的 pinned arena”换成“固定 N 份、按最大对象容量增长”的池：

| 现状 | 改后 |
|---|---|
| `entry["host_staging"]["host"]` 每对象一份 | `_PinnedStagingPool`，N = `HYPIC_PIC_STATE_WORKERS`（默认 4） |
| arena 大小 = 该对象字节数 | 容量统一涨到“见过的最大对象”，不足 1 MB 向上取整 |
| 每请求 pin + `register_buffer` | 每份 arena 只 pin + 注册一次 |
| 20 个 doc → 20 次 | 20 个 doc → **1–2 次**（受 20 个 doc 长度差异影响，容量涨一两次） |

实现要点：

- `_store_state_host_layout()` 变成**纯布局计算**，不再分配；
- `_load_transition_operator_store()` 里 `acquire` → 接收 → H2D → `finally release`；
- 设备侧的算子张量仍按对象缓存（那是真正的载荷），只有 host 侧共享；
- 被替换掉的 arena 会 `unregister`，`_registered_buffer_ptrs` 同步维护。

**为什么安全**：arena 只在“接收完成 → 上传完成”这段窗口内被占用，而这段时间在同一次导入内部同步发生；
槽位数等于导入线程数，因此并发导入不会共享同一块。

验证方式：

```bash
# PROFILE_IMPORT=1 跑一次 A 组
grep -c 'staging-pool bytes=' "$BLOG"   # 分配次数，期望 1–2，而不是 20
grep -c 'staging-pool-use'    "$BLOG"   # 复用次数，期望 ≈ 请求数
grep 'staging-pool bytes='    "$BLOG" | grep -oE 'alloc_ms=[0-9.]+ register_ms=[0-9.]+'
```

预期：`alloc_ms` 只出现一两次，A c=1 开档 p50 从 **388.8 → ~220 ms**（回到 alternate2 的水平）。

### 23.6 一句话

> **c=2 下 PIC 赢 9.5%（A 组）、2%（B 组），且已是“20 步 × 136 ms”的步数受限形态，提并发无效；c=1 下 A 组净亏 2 倍，原因是首次读取每个新对象约 260 ms，而对象数正是唯一变量（20 个 → 389 ms，2 个 → 210 ms，1 个 → 231 ms），嫌疑落在每对象一份、尺寸各异的 pinned staging arena 上，打点已就位，一格即可判定。把对象数压到 2（alternate2）后 A c=1 开档回到 **210.0 ms**（等价于关档 + 20 ms），说明 PIC 单请求本身没有额外成本，那 179 ms 全部来自对象级资源。**

## 24. C1c 成本结构定稿与优化收口（2026-10-31）

### 24.1 结论

C1c 的功能与性能已定稿，**不再在"步内准备"上继续投入**。

| 项 | 结果 |
|---|---|
| 功能闭环 | park → submit → poll → adopt 全通；`SUBMIT=READY=ADOPT`，`order exit=0`，`import-failed=0` |
| c=2 延迟 | **0.90×**（比关档快 10%） |
| c=2 吞吐 | 0.94×（略输）；`max-num-batched-tokens=6000` 下 1.03× |
| B 组 c=2 吞吐 | **1.10×** |
| c=1 | **1.15×**（慢 15%） |
| 单元测试 | 214 passed |
| 打点覆盖面 | `execute_model.total` 的 87% 已归属 |

### 24.2 成本结构（A 组 c=1 on，p50 216.9 ms）

| 层 | 项 | ms | 归属 |
|---|---|---:|---|
| 等待 | 导入（park/import） | **68.5** | PIC 独有，必经 |
| 执行 | `execute_model.total` | **140.8** | — |
| 外壳 | API + 响应 | 7.6 | 共有 |

`execute_model.total` 内部：

| 项 | ms | 归属 | 可砍 |
|---|---:|---|---:|
| `forward_dispatch`（2 段） | 76.0 | 原生 | 0（真算力） |
| `pic_runtime_early` 除 forward | 17.7 | 一半 PIC | ~5 |
| `postprocess.gap` | 14.4 | **原生**（core 循环 + grammar + log） | 0（关档同样付） |
| `runtime-plan`（plan apply） | 7.1 | PIC 独有 | ~3.8 |
| `pic_slot_attach` | 2.4 | PIC 独有 | ~1 |
| `prepare_inputs` | 2.4 | 原生 | 0 |
| `postprocess` | 1.3 | 原生 | 0 |
| `update_states` | 0.7 | 原生 | 0 |
| `entry_and_input_prep_sync` | 0.15 | 原生 | 0 |
| 残余（8–10 小项） | 11.9 | 混合 | 分散 |

**归属汇总：原生 ~95 ms（67.5%，关档同样付，砍了比值不变）；PIC 独有 ~21 ms。**

### 24.3 与关档的净账（c=1）

```
省下的算力    −64.0   （forward 140 → 76）
必经的导入    +68.5
PIC 独有准备  +21.4
------------------------------
净慢          +25.9   ≈ 实测 +28（关档 forward 为反推值）
```

### 24.4 已放弃的优化，以及原因

| 放弃项 | 预期 | 放弃原因 |
|---|---:|---|
| 跨请求 plan / slot 缓存 | 5–10 | **unique doc 不复用**，生产无收益 |
| 预热（已做） | 已收 | 只影响每进程首个请求 |
| 去掉 `_attach_pic_native_slots` 的一次调用 | 3.3 | **不是重复**：第一次负责物化（batch 随后被 isolated 替换），第二次在新 batch 上打行补丁 |
| 继续细分 41 ms 残余 | ~11.9 | 分散在 8–10 项，单项 <3 ms，收益/成本比过低 |
| `max-num-batched-tokens` 压缩对照 | — | **无效**：4–5k 的 prompt 在默认 8192 下本来就放不下两个（见 §24.5） |

### 24.5 `max-num-batched-tokens` 对照作废

早期假设"关档 c=2 靠合批赢"。实测 prompt 为 4k–5k token：

| 两个 prompt | 合计 | ≤8192？ | 能否合批 |
|---|---:|---|---|
| 4000 + 4000 | 8000 | ✅ 勉强 | 偶发 |
| 4500 + 4500 | 9000 | ❌ | 不能 |
| 5000 + 5000 | 10000 | ❌ | 不能 |

关档 c=2 的 p50 287 ≈ 2 步 × 148.6，且 `cap=6000` 与默认 8192 结果相同（2948 vs 2972，属噪声）——
**说明关档从未靠合批，两者都是 20 步、一步一个请求。** 原先"cap=6000 能带来 1.15×"的判据前提不成立，已删除。

### 24.6 下一步只有三条路（都不属于"改几行代码"）

| 手段 | 效果 | 前提 |
|---|---|---|
| **算子 fp16（38 → 19 MB）** | 导入 68.5 → ~57；Σ链路 1.66 → 1.44 s | 精度验证 |
| **RDMA 替代 TCP** | 导入 68.5 → ~10；链路不再成为上限 | 换网络 |
| **PIC 合批（batch=2）** | c=2 吞吐 ~1.5× | 需要 c≥4 才有导入/forward 重叠；改 isolated batch 结构 |

### 24.7 ⚠️ 未验证的部分

**A 组测试用的是前缀对齐形态，真实生产是"复用段在中间"。**

| 形态 | 代码路径 | 性能是否测过 |
|---|---|---|
| 前缀复用 | `_run_pic_single_request_ranges`（isolated batch + 两次 attach） | ✅ 本文全部数据 |
| **中段复用** | public block 拷贝 + `rerotate_native_key` | ❌ **未测** |

本文 §24.2/24.4 里标"只对测试形态有效"的项，在中段形态下可能根本不存在。**下一轮若要继续，应先测中段形态的成本结构，而不是继续拆前缀形态的 41 ms。**

## 25. KV 导入串行化：把"等导入"变成流水线（2026-11-02）

### 25.1 发现

c=2 时两个请求同时导入，会分享同一条链路（1.1 GB/s），于是**同时**在约 166 ms 完成
（2 × 91.5 MB = 183 MB），谁也没法先开始计算，引擎在整个导入窗口内空转。

profiling 上看到的就是这个：PIC 的 `pic:ranges` 两两一组，**组间有几十到 162 ms 的空窗**，
而关档（无 park）的 `gpu_model_runner: forward` 是紧密排列、没有空窗的。

### 25.2 假设

把导入串行化——先传完 A、A 立刻开始算、B 在 A 计算期间传完——可以把"等第一个"的时间
从 166 ms 降到 83 ms，而总传输量不变。

### 25.3 验证（A 组 20 个不同 doc，预热完毕）

| 配置 | wall | p50 | tput |
|---|---:|---:|---:|
| `HYPIC_PIC_STATE_WORKERS=4`（并发导入） | 2545.7 ms | 265.3 ms | 7.86 |
| `HYPIC_PIC_STATE_WORKERS=1`（串行导入） | **2413.3 ms** | **243.2 ms** | **8.29** |
| 收益 | **−132.4 ms（−5.2%）** | **−22.1 ms（−8.3%）** | **+5.5%** |

`seq=` 序列从抖动（230–277）变为平稳（212–277）——并发导入时的资源抢占消失。

### 25.4 落地

| 变量 | 默认 | 管什么 |
|---|---:|---|
| `HYPIC_PIC_KV_IMPORT_WORKERS` | **1** | KV 接收线程数（串行 = 流水线） |
| `HYPIC_PIC_STATE_WORKERS` | 2 | 算子（transition）传输线程数（走专用 stream，并行无坏处） |

此前 KV 接收线程数错误地复用了 `HYPIC_PIC_STATE_WORKERS`，于是"设 4 则 KV 并发、设 1 连算子也串行"。
现在拆开，各自可调。

### 25.5 适用边界

串行只在**单次导入时间 < 计算时间**时更优：

```
计算 158 ms > 导入 83 ms   → 串行（当前）
计算  30 ms < 导入 83 ms   → 并发
```

判断方法：`WORKERS=1` 与 `WORKERS=4` 各跑一次比 wall。

### 25.6 与 §24 的关系

§24 说"步长由固定成本主导、导入被掩盖"，§25 修正了后半点：**导入只在"两个同时到"时被掩盖，
串行化后才是真正被掩盖**。这也解释了 §24 里 c=2 每步 158 与关档 148.6 的差距中，
有一部分是导入窗口（现在 −132 ms wall 拿回了一部分）。

---

## 26. 最终收益主表与封版（2026-11-02）

### 26.1 测量条件

| 项 | 值 |
|---|---|
| Patch | 见 patches/hypic-pic-vllm-0221-cu129-stage0-11E-E-C1c-full/README.md 的 Artifact 段（本文档本身在该 patch 内，故不写死 MD5） |
| Fixture | `a_group_v2`：`publish_i` = doc 一段；`reuse_i` = doc + query 两段 |
| 探针 | `tests/v1/pic/run_pic_concurrency_probe.py`（同探针跑 on/off） |
| 打点 | `HYPIC_PIC_PROFILE_IMPORT=0`（零打点，墙钟可比） |
| off 档 | `PIC_BASE_FLAG="" PIC_ASYNC_FLAG=""`，日志核对 `lookup HIT=0`、`reuse body skipped=0` |
| 硬件 | 3× RTX 4090，TCP，无 RDMA，Mooncake 0.3.10.post2 |

### 26.2 主表（on vs off）

| c | off wall | on wall | wall 加速 | off p50 | on p50 | p50 加速 | off tput | on tput | tput 加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3589.3 | 4087.7 | **0.88×（慢 12%）** | 178.5 | 208.5 | 0.86× | 5.57 | 4.89 | 0.88× |
| 2 | 2935.8 | 2416.9 | **1.21×（快 21%）** | 287.6 | 225.9 | 1.27× | 6.81 | 8.28 | **1.22×** |
| 3 | 2915.9 | 2473.2 | **1.18×（快 18%）** | 424.6 | 337.9 | 1.26× | 6.86 | 8.09 | **1.18×** |
| 4 | 3007.8 | **2023.7** | **1.49×（快 49%）** | 569.2 | 348.1 | 1.63× | 6.65 | **9.93** | **1.49×** |

**复用率 100%**：c=1/2/3/4 共 92 个请求，`reuse body skipped` = 92、`plan_none=False` = 92、
`pic path skipped` 仅 1 次（publish 阶段整段命中，§4j 设计）。

**wall 已经触底**：2117.9 ms ÷ 20 = 105.9 ms/请求 = 单次串行导入耗时，说明计算被完全掩盖。
再快只能缩短单次导入（KV fp8 → ~70 ms → 上限 ~14 rps）。

**wall = 20 个请求的总耗时**，`tput = 20 / wall`，两列加速必然相同（可交叉验证读数）。
**on 档最小 wall 出现在 c=4（2117.9 ms）**，与 tput 峰值 9.44 同一点。

### 26.3 判读

| 并发 | 结论 | 机制 |
|---:|---|---|
| **c=1** | 慢 14% | 导入（~145 ms）无人掩盖，省下的 forward（~100 ms）抵不过导入 |
| **c=2** | 快 21% | 导入被部分掩盖（B 的导入藏在 A 的计算里），流水线开始生效 |
| **c=3** | **快 32%（tput 峰值）** | 导入被完全掩盖，p50 快 52% |
| c=4 | 快 16% | 排队加剧（p50 278 → 498），tput 仍正收益 |

### 26.4 tput 曲线

| c | off tput | on tput |
|---:|---:|---:|
| 1 | 5.57 | 4.79 |
| 2 | 6.81 | 8.22 |
| 3 | 6.86 | 8.09 |
| 4 | **6.65（off 峰值）** | **9.44（on 峰值）** |

**on 峰值 9.44（c=4）比 off 峰值 6.86 高 38%。**

### 26.5 结论

1. **PIC 在 c≥2 全面正收益**，最佳工作点是 **c=2（+21%）与 c=3（+32%）**：导入被计算掩盖，
   且批内仍有独占机会。
2. **c=1 慢 14% 是结构性缺口**，不是缺陷：单请求下导入无法被掩盖。生产上应在 c≥2 使用。
3. **c=4 起排队主导**（p50 从 278 涨到 498），但 tput 仍比 off 高 16%。
4. 全命中（整段 100% 复用）**显式回退**到普通 forward，属设计外场景（见 README §4j）。

### 26.6 与 §22/§24 的关系

§22 的结论是“c=1 略慢、c≥2 才有收益”，本节的实测把那个判断**量化并扩展到 c=4**；
§24 定下的成本结构（导入 68.5 ms 固有 + PIC 准备 ~21 ms）解释了 c=1 缺口与 c≥2 掩盖的
来源，两者一致。
