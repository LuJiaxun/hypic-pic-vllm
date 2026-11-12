# HYPIC PIC → vLLM 0.22.1-cu129 阶段迁移总计划

本文档是 HYPIC PIC 迁移到 vLLM 的阶段边界、代码路径、验证标准和归档规则。
以后生成或审查新的 stage patch 时，必须以本文档为准。

## 0. 计划冻结与增量修改规则

本计划是冻结的人工确认计划，未经人工明确同意不得修改。后续如果需要
调整阶段顺序、阶段职责、Python/Triton/CUDA 分层、验证标准或新增阶段，必须
先暂停当前工作并向人工说明原因、影响和备选方案；只有得到明确同意后，才可
修改本文档和继续实现。

每次开始新的 Stage 前，必须先完整读取本文档，并检查当前 Git HEAD、上一阶段
的验证结果和工作区状态。新 Stage 的代码必须建立在**之前已经验证通过的
Stage 代码之上**，采用增量修改；不得为了生成新 Stage 而从固定基线重新开始、
重写或丢弃之前 Stage 的实现。之前阶段已经验证过的代码必须原样保留，除非
发现明确 bug 且经过人工同意后单独修复并回归验证。

新 Stage 的实现过程必须继续建立在上一阶段已验证代码之上；但 Stage 归档只
导出一份累计代码 patch，从固定初始 commit
`0decac0d96c42b49572498019f0a0e3600f50398` 到当前 Stage，便于从干净镜像重建
和审计完整修改。累计 patch 的导出格式不改变代码实现必须增量进行的要求。

每个 Stage 的归档物必须包含：

1. 从固定初始 commit `0decac0d96c42b49572498019f0a0e3600f50398` 到当前
   Stage 的一份累计代码 patch；
2. 与 patch 放在同一目录中的 README，明确写出基线、上一阶段 commit、当前
   Stage commit、修改文件、累计 patch 应用方法、是否涉及 CUDA/C++/Triton、
   验证命令和回滚方法。

不再要求或生成“上一阶段 → 当前 Stage”的增量归档 patch。

后续任何 Stage 的代码修改、patch 生成或 README 编写，都必须先读取本文件，
确认当前 Stage 在冻结路线中的位置后才能开始。本文档本身属于人工冻结文件；
如需修改其内容，必须先询问人工并得到明确同意，不能自行更新计划。

Git 版本规则：

- 每个 Stage 的源代码和测试代码都必须同步记录到 Git；README、patch 归档文件
  不作为必须同步进 Git 的内容，可按需单独保存或导出；
- 每个 Stage 至少形成一个清晰的 Git commit，commit message 必须明确包含
  `Stage 0`、`Stage 1-A`、`Stage 7-B` 等对应阶段编号；
- 新 Stage 开始前，必须从 Git 查看上一个已验证 Stage 的 commit、diff 和文件
  路径，以该 commit 的代码为增量修改依据；
- 禁止绕过 Git 直接覆盖已验证 Stage 的代码；如果需要修复历史 bug，必须在
  commit message 和 README 中说明修复原因，并重新执行受影响阶段的回归验证。

修订记录（人工批准后生效）：

| 日期 | 修订 | 批准方式 | 影响章节 |
|---|---|---|---|
| 2026-10-21 | Store-only 收敛：TransferEngine 退场，改由 Store pull 承担全部数据面；新增 11E-D1/D2 | 人工明确要求“围绕 store 改 plan” | 11E 实施边界、B0-A、B1、B2、B2-A、Stage 11E-TP、执行顺序 |
| 2026-10-22 | 数据面全面对齐 vLLM full-attention 原生形态：native 与 transition 都用 **block 粒度 key** + 同一条 `batch_get_into_multi_buffers`；新增 11E-D3 与“读路径实测门” | 人工明确要求“参考 full attention 原生 vllm 的路径走” | 11E-D3、执行顺序 |

## 1. 固定基线

```text
镜像：vllm/vllm-openai:v0.22.1-cu129
源码 commit：0decac0d96c42b49572498019f0a0e3600f50398
CUDA：cu129
```

当前分支：

```text
hypic/pic-vllm-0.22.1-cu129-g0decac0d96c
```

所有 Stage 累计 patch 都必须说明：

1. 固定初始 commit 和当前 Stage commit；
2. 当前 Stage 之前的代码来自已验证的历史 Stage；
3. 没有把后续 Stage 的代码提前带入；
4. 是否修改 CUDA、C++、Triton 或模型 forward。

## 2. 总体阶段顺序

```text
Stage 0
  ↓
Stage 1-A：handle 生命周期 + scheduler → worker 元数据
  ↓
Stage 1-B：worker → scheduler 回传 + PIC/non-PIC batch 隔离
  ↓
Stage 2：Mamba/GDN recurrent state、conv tail layout 描述
  ↓
Stage 3：transition 和 range execution plan
  ↓
Stage 4：physical PIC pool、非连续 slot mapping
  ↓
Stage 5：worker-side skip/recompute 安全门控
  ↓
Stage 6-A：worker runtime wiring、capability gate
  ↓
Stage 6-B：physical snapshot/restore adapter
  ↓
Stage 6-C：state-layout-aware snapshot bridge
  ↓
Stage 7-A：worker live Mamba/GDN state capture
  ↓
Stage 7-B：真实 Mamba/GDN state restore
  ↓
Stage 8-A：copy-back attention KV 原型
  ↓
Stage 8-B：zero-copy PIC attention slot
  ↓
Stage 9-A-pre：native KV slot bridge（SGLang-style logical→physical mapping）
  ↓
Stage 9-A：单 request skip/recompute
  ↓
Stage 9-B：batch 混合执行
  ↓
Stage 10：Triton/CUDA 性能优化与生产化
```

Stage 0–5 是控制面、描述层和安全原语的逐步构建；它们不等价于最终的
HYPIC GPU 加速已经完成。真实非前缀复用仍需要后续 attention backend 和
模型 forward 集成。

## 3. 各阶段职责与代码路径

### Stage 0：PIC 控制面

目标：在不改变 vLLM 原有执行路径的前提下，完成 prompt 分段、hash、metadata
lookup、非前缀复用计划和请求级 feature flag。

关键路径：

```text
vllm/config/pic.py
vllm/config/vllm.py
vllm/engine/arg_utils.py
vllm/v1/engine/input_processor.py
vllm/v1/pic/segmenter.py
vllm/v1/pic/cache.py
vllm/v1/request.py
vllm/v1/core/sched/scheduler.py
```

必须保持：

- 普通请求不创建 PIC plan；
- vLLM 原有 prefix cache、KV block allocation、forward 不变；
- embedding/mixed prompt 回退普通路径；
- 不修改 CUDA/C++/Triton。

验证：

- CLI `--pic-*` 参数可见；
- segmenter/cache 单测通过；
- 普通请求开启 PIC 全局配置后输出逐字一致；
- PIC 请求能进入 scheduler 并返回 HTTP 200；
- 无 CUDA/C++/Triton 文件变更。

当前状态：**已验证通过**。服务器报告中的 Stage0 8 项验收全部通过。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage0/
```

### Stage 1-A：handle 生命周期与 scheduler → worker 元数据

目标：建立 PIC 控制面 handle 和跨 scheduler/worker 的元数据传递，但不分配
真实 GPU storage。

关键路径：

```text
vllm/v1/pic/handles.py
vllm/v1/pic/cache.py
vllm/v1/pic/__init__.py
vllm/v1/core/sched/output.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/worker/gpu_model_runner.py
```

实现：

- `PICHandleKind`、`PICHandle`、`PICHandlePool`；
- handle retain/release/clear；
- `PICCachePlan.matches` 携带命中的 segment entry；
- `NewRequestData` 将 PIC metadata 传给 worker；
- worker 保存 metadata，但普通执行路径不消费这些字段。

验证：

- patch 只修改 Python；
- `compileall` 通过；
- handle smoke test 通过；
- Stage0 单测仍通过；
- 普通请求和 PIC 请求均 HTTP 200；
- 无 scheduler crash 或 traceback。

当前状态：**已验证通过**。服务器验收记录明确标记为 Stage1-A。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage1/
```

### Stage 1-B：worker → scheduler 回传与 batch isolation

目标：补齐 PIC metadata 的反向回传通道，并在配置打开时隔离 PIC 与非 PIC
请求，避免两类请求进入同一个执行 batch。

关键路径：

```text
vllm/v1/pic/materialization.py
vllm/v1/outputs.py
vllm/v1/core/sched/scheduler.py
vllm/v1/pic/__init__.py
```

实现：

- `PICMaterialization` worker → scheduler metadata；
- scheduler 校验 request、segment index 和 segment hash；
- 合法 materialization 更新 scheduler-side PIC metadata cache；
- `pic-exclusive-batch` 开启时，PIC/non-PIC 请求不混合调度；
- 不改变普通 batch 的默认行为，配置关闭时保持原流程。

明确不包含：

- worker 自动生成真实 GPU handle；
- attention KV block copy；
- Mamba/GDN state restore；
- worker-side skip/recompute。

验证：

- 构造合法 materialization 能更新对应 cache entry；
- unknown request、错误 segment index、错误 hash 被丢弃并记录 warning；
- exclusive batch 开启时两类请求不会混合；
- exclusive batch 关闭时原有调度行为不变；
- 普通请求/PIC 请求服务 smoke test 继续通过。

当前状态：**独立 Stage1-B patch 已在服务器完成服务端验收**。普通、PIC 首次、PIC 重复以及普通/PIC 并发请求均返回 HTTP 200，Stage0/Stage1-A 回归通过。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage1B/
```

### Stage 2：hybrid recurrent state / conv tail layout

目标：从 vLLM 的 `MambaSpec` 读取 authoritative shape/dtype，建立不同
Mamba/GDN/conv backend 的 state descriptor。

关键路径：

```text
vllm/v1/pic/state.py
vllm/v1/pic/__init__.py
```

布局：

```text
LINEAR     → recurrent state
MAMBA1     → conv tail + recurrent state
MAMBA2     → conv tail + recurrent state
SHORT_CONV → conv tail
GDN_ATTN   → conv tail + recurrent state
```

未知 backend 必须显式失败，不能猜测 state 顺序。

明确不包含：

- GPU allocation；
- state snapshot/restore；
- attention KV copy；
- CUDA/Triton kernel；
- forward skip。

验证：

- state layout import/compile 通过；
- attention-only 配置无 state；
- 真实 `MambaSpec` fixture 的 shape/dtype/kind 数量一致；
- 不改变普通 vLLM 服务行为。

当前状态：**服务器验证通过**。state-layout smoke test、Stage0/Stage1 回归、
普通请求和 PIC 请求均通过，未引入额外显存开销。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage2/
```

### Stage 3：transition 和 range execution plan

目标：将 Stage0 的 segment match 编译成非前缀的 reuse/recompute ranges，
并描述 segment seam/transition，不修改 vLLM 原有 scalar
`num_computed_tokens` 语义。

关键路径：

```text
vllm/v1/pic/execution.py
vllm/v1/request.py
vllm/v1/core/sched/output.py
vllm/v1/core/sched/scheduler.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/pic/__init__.py
```

验证：

- segment 0/2 reuse、segment 1 recompute 等非前缀组合生成正确 range；
- seam token 和 transition handle 绑定正确；
- 空 prompt、全 miss、全 hit、最后 segment 不提前复用均正确；
- worker 未支持 range execution 时必须回退普通路径。

当前状态：**服务器验证通过**。Stage0/1/2 回归、普通请求和 PIC 请求均通过，
range-plan smoke test 通过，显存与 Stage2 基本一致，未引入额外显存开销。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage3/
```

### Stage 4：physical PIC pool 与非连续 slot mapping

目标：提供独立于 vLLM `BlockPool` 的物理存储原语，为未来 PIC snapshot、
restore 和 non-contiguous slot mapping 提供可测试的基础。

关键路径：

```text
vllm/v1/pic/pool.py
vllm/v1/pic/handles.py
vllm/v1/pic/__init__.py
```

实现：

- 独立 byte-addressable tensor pool；
- allocation/free/release；
- tensor snapshot/restore；
- logical position → physical slot mapping；
- 多 allocation、绝对 offset、碎片回收。

明确不包含：

- 将 vLLM attention KV cache 自动拷贝进该 pool；
- 修改 block table；
- 让 FlashAttention/FlashInfer/Triton 消费 PIC slots；
- 修改模型 forward。

验证：

- CPU tensor snapshot/restore round-trip；
- 多 allocation 的绝对 offset 正确；
- 非连续 slot mapping 正确；
- release 后 free bytes 恢复；
- capacity、dtype、shape、device 错误被拒绝；
- GPU allocation smoke test 单独验证显存行为。

当前状态：**服务器验证通过**。Stage0–3 回归、普通/PIC 请求、physical pool
和非连续 slot mapping 验证通过；本阶段仍未接入 attention/KV backend。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage4/
```

### Stage 5：worker-side skip/recompute 安全门控

目标：只有在 physical handles 和 range execution backend 都明确可用时，
才生成 skip/recompute worker descriptor；否则回退普通 vLLM 路径。

关键路径：

```text
vllm/v1/pic/worker_plan.py
vllm/v1/pic/execution.py
vllm/v1/pic/cache.py
vllm/v1/pic/__init__.py
```

验证：

- 无 physical handle 时 `allow_fallback=True` 返回 ordinary fallback；
- `allow_fallback=False` 显式抛出 unsupported；
- skip/recompute positions 完整覆盖 prompt 且不重叠；
- range 不完整、顺序错误、长度不匹配时拒绝执行。

当前状态：**服务器验证通过**。Stage0–4 回归、worker gate smoke、普通/PIC
请求均通过，显存保持 21127 MiB，未引入额外显存开销。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage5/
```

### Stage 6-A：worker runtime wiring 与 capability gate

目标：把 Stage 5 的 worker descriptor 接入 `GPUModelRunner` 的 request state，
但只有明确注册的 worker backend 才能声明支持 range execution。默认 capability
关闭时，PIC hit 必须记录 fallback 并继续原始 vLLM forward。

关键路径：

```text
vllm/v1/pic/worker_plan.py
vllm/v1/pic/__init__.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/worker/gpu_model_runner.py
```

明确不包含：

- attention backend 或 block table 修改；
- physical snapshot/restore；
- 模型 forward skip/recompute；
- CUDA/C++/Triton 源码。

验证：

- `PICWorkerCapabilities` 默认关闭；
- 新请求和 streaming request 都能得到 worker descriptor 或 fallback 标记；
- 未注册 backend 时普通/PIC 请求路径不变；
- compileall、Stage5 gate smoke 和普通/PIC 服务回归通过。

当前状态：**服务器验证通过**。Stage5 回归、普通/PIC 请求、worker capability
gate 和 fallback 行为均通过，未改变原始 forward 路径。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage6A/
```

### Stage 6-B：physical snapshot/restore adapter

目标：把 Stage 4 physical pool、Stage 2 state kind 描述和 PIC cache metadata
组合成可独立测试的 snapshot/restore adapter。调用方显式提供 tensor，adapter
不读取 vLLM live KV cache，也不改变模型执行。

关键路径：

```text
vllm/v1/pic/snapshot.py
vllm/v1/pic/pool.py
vllm/v1/pic/state.py
vllm/v1/pic/cache.py
tests/v1/pic/test_segmenter.py
```

实现：

- full-KV、recurrent、transition、conv-tail 分类型 snapshot；
- 将 snapshot handle 发布到 `PICSegmentCache`；
- 从 physical pool restore 到调用方 tensor；
- capture 失败时释放已分配的部分 handle；
- snapshot hash、token range 和 restore tensor shape/dtype 由 pool/cache 校验。

明确不包含：

- 从 vLLM live KV cache 自动抓取 tensor；
- Mamba/GDN live state index 接入；
- attention block table 或 attention backend 修改；
- 模型 forward skip/recompute；
- CUDA/C++/Triton 源码。

当前状态：**patch 已生成并完成 Stage 6-A 基线的静态叠加检查、compileall；
服务器验证待执行，且必须在 Stage 6-A 服务器验证通过后再应用。**

当前归档使用最终单 patch：只包含 `vllm/v1/pic/snapshot.py` 和
`vllm/v1/pic/__init__.py`，不包含 `tests/`，以兼容服务器中的 Stage0 测试基线。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage6B/
```

### Stage 6-C：state-layout-aware snapshot bridge

目标：使用 Stage 2 的 `PICStateLayout`，按 `(group_id, state_index)` 显式组织
recurrent state 和 conv tail tensor，并在进入 Stage 6-B snapshot store 前校验
shape、dtype、state kind，避免不同 Mamba/GDN backend 的隐式顺序错位。

关键路径：

```text
vllm/v1/pic/snapshot.py
vllm/v1/pic/state.py
tests/v1/pic/test_segmenter.py
```

明确不包含：

- vLLM live Mamba/GDN state 读取；
- 自动 state index 读写；
- attention backend、block table 或模型 forward 修改；
- CUDA/C++/Triton 源码。

当前状态：**单 patch 已生成并完成 Stage 6-B 基线叠加检查，依赖 Stage 6-B，服务器验证待执行**。

Stage 6-C 的单 patch 同时包含两部分：

- `vllm/v1/pic/snapshot.py` 的功能增量；
- 从容器实际 Stage 0 版 25 行测试文件到当前累计测试版的整体替换，hunk 为
  `@@ -1,25 +1,211 @@`，不依赖源码仓库中漏进容器的历史测试增量。

归档：

```text
patches/hypic-pic-vllm-0221-cu129-stage6C/
```

### Stage 7-A：worker live Mamba/GDN state capture

目标：在显式打开 `--pic-capture-live` 且配置
`--pic-max-cache-bytes` 后，从 GPU worker 当前已完成的 PIC segment 边界读取
Mamba/GDN recurrent state 和 conv tail，复制到 worker-owned `PICPhysicalPool`，
并通过已有 `PICMaterialization` 通道返回 scheduler。

关键路径：

```text
vllm/v1/pic/live_capture.py
vllm/v1/pic/snapshot.py
vllm/v1/pic/state.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/mamba_utils.py
vllm/config/pic.py
vllm/engine/arg_utils.py
```

本阶段的安全边界：

- 默认 `capture_live=False`，普通请求和旧 PIC 路径不分配额外 pool；
- 只在当前 worker step 恰好结束于 segment 边界时 capture；
- 只处理单 worker rank，pipeline parallel 暂不合并跨 rank state；
- capture 失败只记录 warning，不影响普通 forward；
- 只 capture，不 restore，不修改 attention backend，不执行 skip/recompute；
- 不新增 CUDA/C++/Triton kernel，复用 vLLM 已有 state storage 和 copy 路径。

验证：

- `PICLiveCaptureManager` 能发布 materialization 并释放 snapshot；
- `pool.py` 多 allocation absolute offset 回归通过；
- 默认配置显存和普通请求行为不变；
- 打开 capture 后，hybrid state 在完整 segment 边界生成有效 handle；
- scheduler 能接收并写入 PIC metadata cache；
- server 仍无 traceback 或 scheduler crash。

当前状态：**代码已生成，静态检查和服务器验证通过**。

### Stage 7-B：真实 Mamba/GDN state restore

目标：把 Stage 7-A 保存的真实 PIC snapshot 恢复到当前请求的 live state，
使请求可以从指定 segment 继续执行。

关键路径：

```text
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/mamba_utils.py
vllm/v1/pic/snapshot.py
vllm/v1/pic/live_capture.py
vllm/v1/core/sched/scheduler.py
```

本阶段优先复用 vLLM 已有 state-copy 逻辑，不新增 Triton/CUDA kernel，也不
接入 attention KV zero-copy。必须校验 state index、shape、dtype、group 映射，
并保持 restore 失败时回退普通 forward。

验证：

- restore 前后 recurrent state、conv tail、transition state 一致；
- restore 后生成结果与完整重算一致；
- 多 request 不互相覆盖；
- 普通请求和未启用 PIC 的请求行为不变。

当前状态：**代码已生成，静态检查和服务器验证通过**。Stage 7-B 已完成真实
Mamba/GDN recurrent state 与 conv tail restore 的服务器验证；仍未接入 attention
KV 复用或 forward skip/recompute。

### Stage 8-A：copy-back attention KV 原型

目标：将 PIC snapshot 中的 attention KV 拷贝回 vLLM 原有 KV cache block，
继续使用原有 block table、slot mapping 和 attention kernel，先完成 correctness
prototype。

关键路径：

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/worker/block_table.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/pic/kv_copyback.py
vllm/v1/attention/backends/*
vllm/v1/attention/ops/*
```

边界：

- 允许一次 copy-back；
- 不直接让 attention kernel 读取 PICPhysicalPool；
- 不引入 zero-copy slot；
- 先限制单卡、eager、单 request；
- 如果现有 kernel 无法表达目标布局，记录为后续 Stage 8-B 问题，不提前混入。

当前状态：**代码已生成，静态检查通过，服务器验证待执行**。当前实现只在
显式 `--pic-copyback-kv` 打开、命中 full-KV snapshot 且目标 backend 支持现有
`do_kv_cache_update` 时执行；默认仍回退普通 forward，不包含自动 live KV capture。

### Stage 8-B：zero-copy PIC attention slot

目标：让 attention backend 直接读取 PIC physical slot，消除 Stage 8-A 的
copy-back。该阶段是 Stage 8-A 的直接延续，先完成单卡、单 request、eager mode
下的 attention slot correctness，再进入执行层的 skip/recompute。

可能涉及：

```text
vllm/v1/attention/backends/*
vllm/v1/attention/ops/*
vllm/v1/worker/block_table.py
vllm/v1/worker/gpu_input_batch.py
```

只有在现有 attention kernel 无法表达第二套物理存储时，才新增 Triton/CUDA
代码；新增 kernel 必须单独编译和回归，不与 Stage 8-A 混合。

当前状态：**原 external PICPhysicalPool full-KV view 已删除；native KV slot
bridge 控制面统一在 Stage 9-A-pre 实现。真实 worker attention attach 仍待
Stage 9-A-pre 后续增量接入。**

### Stage 9-A-pre：native KV slot bridge

目标：采用与 SGLang PIC 相同的存储契约，让 PIC segment 最终落在 vLLM
原生 KV pool 的 block/slot 上。attention backend 不识别 PIC 类型，只消费
当前 request 的 logical token position 到 native physical KV slot 的映射。

但 native slot 复用必须区分 KV 的位置语义：对于 position-independent KV，命中
segment 可以直接共享 native slots；对于已经带绝对 RoPE 的 K，不能让不同绝对
位置的 request 直接共享同一份 K，必须采用 SGLang 的 canonical public K 加
request-private rerotated K 方案，或者安全回退到 copy-back。

核心抽象：

```text
PIC segment entry
  ├── logical token range [start, end)
  ├── per-KV-group native KV reference
  │    ├── shared native slots（position-independent）
  │    └── canonical public + private rerotated slots（RoPE）
  ├── KV cache group/layout metadata
  ├── native block lease / ref-count
  └── recurrent state reference

request logical position
  → position-aware native block/slot mapping
  → existing block_table / attention metadata
  → existing FlashAttention/FlashInfer kernel
```

参考 SGLang 的语义：

```text
req_to_token[request_id, token_position] = physical KV slot
```

在 vLLM 中对应为：

```text
request block_table[logical_block] = native KV block_id
slot_mapping[logical_token] = native KV slot
```

命中 segment 时，PIC 只需要把已保存的 native block/slot 映射装配到当前
request；不能把 PIC 当作 attention 能识别的第二套 KV 存储。这样 attention
kernel 继续读取原生 `kv_cache`，无需新增 Triton/CUDA kernel。

对于带 RoPE 的 full attention，native-slot bridge 的正确路径是：

```text
cache time:
  segment K/V → public native slots
  K 保存为 canonical/local position

hit time:
  public K → rerotate 到当前 request 的 absolute position
  → request-private native slots
  V 可共享或复制到 private slots
  attention 只读取当前 request 可见的 native slots
```

因此，“zero-copy”在本阶段的准确含义是：不再使用独立 PIC byte pool 作为
attention storage，并复用原生 attention kernel；它不保证 RoPE K 在每个 request
上都完全没有 slot materialization。

关键路径：

```text
vllm/v1/core/block_pool.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/worker/block_table.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/attention/backend.py
vllm/v1/attention/backends/*
vllm/v1/pic/cache.py
vllm/v1/pic/handles.py
vllm/v1/pic/native_kv.py
```

本阶段必须完成：

- PIC entry 保存**每个 KV cache group**的 native KV reference，而不是只保存
  独立 byte pool handle；
- native reference 明确标注 `shared`、`canonical_public` 或 `private` 语义；
- 当前 request 能装配非连续 native block table，并区分 logical block 与
  kernel block；
- attention metadata 能生成对应 `slot_mapping` 和绝对 logical positions；
- native KV block 在 PIC entry 或 request-private lease 存活期间被 pin/ref-count，
  不能被 allocator 回收；
- 普通 request 完全沿用原始 block allocation 和 attention 路径；
- 不同 KV cache group、block size、KV dtype/layout 不匹配时显式 fallback；
- segment 尽量按 native KV block 对齐；非对齐边界不能直接覆盖共享 block，
  必须使用 edge-private block、canonical/private 方案或回退；
- RoPE K 的 canonical→absolute rerotation 不能修改共享 public slots；
- 为后续 Mooncake store 保留统一的 KV reference/import 边界。

Mooncake 集成边界：

- Mooncake 可以作为 native KV block 的远端来源；
- 远端 KV 进入 vLLM native KV pool 后，仍使用同一个 block/slot mapping 和
  position-aware reference 类型；
- PIC lookup、local native KV 和 Mooncake restore 不应分别定义 attention 格式；
- 如果 Mooncake 返回的是外部 GPU pointer，而不是 native block，需要先建立
  import/registration 或 copy-back 适配，不能直接伪造 block ID；
- Mooncake 传输的是 canonical public KV 时，目标 request 仍需执行本地
  rerotation/private materialization，不能把远端 block 直接当作当前 request 的
  absolute-position K。

明确不包含：

- single-request miss-only forward；
- recurrent state 的 range composition；
- batch 混合执行；
- CUDA/C++/Triton 新 kernel；
- 通过伪造 `num_computed_tokens` 表示非前缀命中。

验证：

- native KV block 非连续映射的 block table/slot mapping 单测；
- PIC 引用期间 block 不会被回收，release 后 ref-count 恢复；
- 使用现有 attention metadata/backend 读取 shared native KV 的 smoke test；
- 两个不同 absolute position 的 request 命中同一 segment 时，验证 public K
  不被修改，private K rerotation 后 attention 结果正确；
- 非对齐 segment 边界触发 edge-private 或 fallback，而不是覆盖共享 block；
- 普通请求和 PIC miss 请求行为不变；
- PIC hit、native KV hit 和 fallback 三条路径的映射结果一致；
- 未启用 native-slot capability 时安全回退普通路径。

当前状态：**已实现 native KV reference、canonical/local slot plan、BlockPool
lease、RoPE private-materialization 原语及单测；尚未接入真实 worker 的
BlockTable 写入、attention forward 或 skip/recompute。旧的 external
PICPhysicalPool full-KV view 不保留。**

### Stage 9-A：单 request skip/recompute

目标：对单个 request 执行真实的非前缀 reuse/recompute，例如
`reuse → recompute → reuse → recompute`，只计算 miss segment。

本阶段建立在 Stage 8-B 和 Stage 9-A-pre 已验证的 native KV slot bridge 之上；Stage 8-A
的 copy-back 仅作为 correctness 对照和 fallback 路径保留。

关键路径：

```text
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/attention/backends/*
```

限制：单卡、单 request、eager mode、无 speculative decoding、无 CUDA graph。
验证 skip 后输出、transition state 和完整重算一致；任何不安全条件都回退
普通 forward。

#### 9A-1：Transition capture/composition

关键路径：

```text
vllm/v1/pic/live_capture.py
vllm/v1/pic/snapshot.py
vllm/v1/pic/state.py
vllm/v1/worker/gpu_model_runner.py
```

实现：segment C 执行时计算并保存 `T_C`、zero-start end-state，以及 conv tail
和 transition 辅助信息。先使用 Torch/Python 实现，不修改 CUDA/Triton。

验证：

```text
T_C(zero_state) == segment 独立执行后的 end_state
T_C(state_A) == 在 state_A 上完整执行 segment C 的 end_state
```

#### 9A-2：Hybrid single-request execution

关键路径：

```text
vllm/v1/worker/gpu_model_runner.py
vllm/v1/pic/runtime.py
vllm/v1/pic/execution.py
```

执行流程：

```text
miss segment A：正常 forward，得到 state_A
reuse segment B：应用 T_B(state_A)，装配 B 的 native KV，不执行 B 的主体 token，
                 只执行 seam window
miss segment C：从 B 之后的 state 继续正常 forward
```

要求：seam 必须真正执行；复用 segment 不能直接 restore 原请求旧 state；必须
应用当前 prefix 下的 `T_B`；full-attention KV 继续使用 native slot/copy-back；
不安全条件继续 fallback。

#### 9A-3：端到端单请求验收

固定场景：第一次请求 `A <<PIC_SEP>> B <<PIC_SEP>> C` 保存 B 的 KV、`T_B` 和
zero-start state；第二次请求 `X <<PIC_SEP>> B <<PIC_SEP>> Y` 命中 B。

必须同时验证：lookup HIT、transition applied、seam forward、B 主体 token 不
进入普通 forward、实际 forward token 数减少、PIC 输出与完整重算逐 token 一致、
Qwen3.5 GDN/Mamba state 一致，以及普通请求、无 transition、非对齐 segment
均安全 fallback。

最终对照：

```text
完整重算：A + B + C 全部 forward
PIC：     A forward → T_B(state_A) → B seam forward → C forward
```

比较最终 logits、生成 token、recurrent state、full-attention KV 读取结果和
实际 forward token 数。`transition_state_handle` 必须代表可应用的 `T_C`，不能
只是普通 tensor snapshot。在此完成前，当前代码仍属于“复用计划和安全回退原型”，
不能称为完整 Stage 9-A。

当前状态：**Stage 9-A-1、9A-2、9A-3 已完成并通过服务器验收**。

验收范围为 Qwen3.5-2B hybrid 纯文本、单卡、单 request、eager mode、无
speculative decoding、无 CUDA Graph。已验证 transition capture/composition、
GDN/Mamba state、native/private KV、seam forward、reuse body skip、miss-range
forward、普通路径 fallback、重复请求稳定性和 PIC pool 不持续增长。

最终修复提交为 `28e3ea72f8`，修复了 transition capture 的 model-order 与
`kv_cache_groups` group-major state-order 错位。Stage 9-A 的 batch、多 request
和生产化能力不包含在本次结论中，下一步进入 Stage 9-B。

### Stage 9-B：batch 混合执行

目标：支持普通 request、多个 PIC request，以及不同 request 的独立 reuse range
在同一 batch 中执行。

验证：

- batch 内 slot 不串；
- 每个 request 的 range 独立；
- scheduler 与 worker 状态一致；
- 某个 PIC request fallback 不影响普通 request。

当前状态：**Stage 9-B 初版已完成服务器验收**。代码建立在已通过的
Stage 9-A `28e3ea72f8` 代码之上，最终修复提交为
`0b8ef63cd7`，不得回退或重写 Stage 9-A 的单 request 路径。

本轮实现边界：

- 新增全局实验开关 `--pic-batch`，默认关闭；
- scheduler 在 `--pic-batch` 打开时允许 PIC/non-PIC 混合调度，不改变默认的
  `--pic-exclusive-batch` 隔离行为；
- 每个 request 独立保存 runtime plan、native KV row、transition cursor 和
  recurrent state，不把多个 request 的 range 扁平合并；
- native KV materialization 按 request row 执行，单个 request attach 失败只回退
  该 request；
- 第一版执行采用 request-isolated microbatch：每次模型调用只给一个 request
  非零 token，其余 batch row 为零。这是正确性和状态隔离原型，不是最终吞吐优化；
- multimodal、speculative decoding、pipeline parallel、pooling、prompt logprobs
  以及不满足 live-state/native-KV 条件的 request 保持普通 forward 或安全回退。

因此，本轮通过只能证明混合 batch 的 request 级隔离和回退语义，不能宣称已经完成
packed/fused batch 性能优化；后者属于 Stage 10。

关键路径：

```text
vllm/config/pic.py
vllm/engine/arg_utils.py
vllm/v1/core/sched/scheduler.py
vllm/v1/pic/batch.py
vllm/v1/pic/worker_plan.py
vllm/v1/worker/gpu_model_runner.py
```

本轮验证：

- `--pic-batch` 可见且默认关闭；
- 两个 PIC request 的 range plan、native slot 和 state cursor 相互独立；
- 普通 request 混入时不改变其原生路径；
- 单个 PIC attach/runtime 失败时只对该 request 回退；
- decode/chunked-prefill 混合时安全回退原生 vLLM 路径；
- Stage 9-A-3 的单 request 测试继续通过；
- 普通请求输出、PIC 输出、slot/state 隔离和重复请求稳定性均通过。

Stage 9-B 的边界仍然是 correctness-first 的 request-isolated microbatch：
每个 request 使用临时一行 `InputBatch`，不是 SGLang 式 fused/packed batch。
普通 request 在同一调度环境中保持原生路径，但不宣称已经和 PIC request 在同一
model forward 中融合执行。真正的 packed/fused mixed batch 进入 Stage 10。

### Stage 10：性能优化与生产化

Stage 10 建立在 Stage 9-A-3 和 Stage 9-B 初版验收通过的代码之上，目标是把
request-isolated correctness prototype 演进为真正的 fused mixed batch，并在
确认 correctness 后再做 kernel 和生产化优化。本节的 10-A～10-C 是 Stage 10
内部实施包，不改变对外的 Stage 编号，也不允许跨包提前宣称完成。

#### Stage 10-A：packed hybrid mixed-batch execution

目标：在一次真实 model forward 中同时执行普通 request、PIC request 和不同
reuse range，并同时维护 GDN/Mamba recurrent state、conv tail 和 transition，
移除 Stage 9-B 的临时 one-row `InputBatch`。

主要路径：

```text
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/pic/batch.py
vllm/v1/pic/runtime.py
vllm/v1/pic/live_capture.py
vllm/v1/pic/state.py
vllm/model_executor/layers/mamba/*
vllm/v1/attention/backend.py
```

实现要求：

- 每个 request 保留独立的 range cursor、logical position、query length 和
  logits index；
- batch 内构造 packed `num_scheduled_tokens`、`query_start_loc`、positions 和
  slot mapping；
- reuse segment 不生成 query，miss/seam segment 进入同一次 forward；
- 普通 request 直接复用 vLLM 原生 row；
- 不再用 zero-token row 模拟 request isolation；
- 每个 request 的 recurrent/conv/transition state 在 packed round 中保持独立；
- reuse、seam 和 miss range 的执行顺序不能跨 request 合并或错位；
- 不满足 packed 条件时按 request 或整批安全回退。

本实施包只使用 Python/Torch metadata 和 vLLM 现有 attention/GDN kernel，
**不新增或修改 Triton/CUDA kernel**。

验收：

- 同一个 `execute_model` 中出现 PIC + 普通 request；
- 两个 PIC request 使用不同 reuse range；
- 三类 request 的输出分别与独立运行逐 token 一致；
- 不同 request 的 block table、slot mapping、logits index 不串扰；
- 无 shape mismatch、无重复 forward、无重复 KV/state 写入。

补充验收：

- packed 与单 request 逐层 recurrent/conv state 一致；
- 最终 logits、生成 token 和 full-attention KV 一致；
- 任一 request fallback 不影响其他 request；
- 重复压力测试中 pool、native lease 和 state block 不增长泄漏；
- 单卡、eager、无 speculative decoding 先通过后，才扩展 decode/chunked-prefill。

#### Stage 10-B：attention backend 与 kernel 路径

目标：确认现有 FlashAttention/FlashInfer/GDN backend 是否已经能够消费
packed PIC metadata；只有不能表达时才修改 Triton/CUDA。

决策顺序：

1. Python/Torch 构造正确的 block table、slot mapping、absolute positions；
2. 复用现有 attention kernel 完成 correctness；
3. 对比 native KV layout、NHD/HND、RoPE K、V、GDN metadata；
4. 只有现有 kernel 无法表达非连续 slot 或 packed query 时，才新增最小的
   Triton/CUDA kernel。

若新增 kernel，必须单独记录：源码路径、编译命令、产物版本、fallback 开关、
单测和 GPU 回归；不能把未编译的 kernel 改动混入普通 Python patch。

验收：

- shared/native/private KV reference 读取结果一致；
- 非连续 block、边界 seam、RoPE rerotation 正确；
- kernel 读写没有跨 request 越界；
- 无 kernel 时仍能安全回退 Python/普通 attention 路径。

#### Stage 10-C：性能优化与生产化

只有 Stage 10-A～10-B correctness 通过后才执行：

本阶段参考 SGLang PIC 的 native KV pool 和 request-to-token 映射设计，但不直接
复制 SGLang 的代码。目标是把当前“每轮临时构造 metadata/materialize KV”的
正确性原型，演进为 request 生命周期内持久的 native slot 映射和缓存租约。

#### 10-C SGLang 对齐 TODO

> **2026-10-23 归并说明**：Stage 10-C 与 §11E-D3-b（block 粒度 key）是**同一项工作**
> 的两个来源——10-C 从"SGLang 式 logical→native 映射"切入，D3-b 从"每 block 一个
> key"切入。两者合并为 **W4（见 §7）**，且**不带来带宽收益**（§11E-D3-b 已实测：
> 1/6/54 key 的 batch 带宽都在 0.68–0.77 GB/s，粒度不是瓶颈），收益在复用能力、
> 跨段去重与 eviction 粒度。实施顺序排在 W3 之后（W4 本身排在 W3 之后）。

10-C 固定拆分为以下六个连续子阶段，不使用字母子编号。每个子阶段都在前一个
已验证子阶段的 Git 代码上增量实现，并单独
验证、提交和导出累计 patch。

1. **10-C-1：request 级 logical-to-native 映射（性能）**

   建立类似 SGLang `req_to_token` 的映射：

   ```text
   request logical token position → vLLM native KV block/slot
   ```

   映射应属于 request state，而不是只存在于一次 `execute_model` round。prefill、
   seam、miss forward 和 decode 必须复用同一份映射，避免每轮重新生成临时
   `slot_mapping`。

2. **10-C-2：KV allocation 阶段完成 public/private slot 分配（性能 + 正确性）**

   将 public canonical KV、request-private KV 和 edge-private block 的分配前移到
   native KV allocation/attach 阶段：

   - position-independent KV 可以共享 public native slots；
   - 带绝对 RoPE 的 K 使用 public canonical slot 加 request-private rerotated slot；
   - 非对齐 segment 的边界使用 edge-private slot 或安全 fallback；
   - attention 仍只读取 vLLM 原生 KV pool。

   目标是消除每个 packed round 重复 private KV materialization，同时避免修改
   shared public K。

3. **10-C-3：private KV 持久化与统一 lease/ref-count/eviction（性能 + 稳定性）**

   private KV 在 request 的 prefill→decode 生命周期内保持有效，不能在每个
   forward round 结束后重新 materialize。为 public native KV、private KV 和
   state block 建立统一生命周期管理：

   - PIC entry、active request、in-flight execution 分别持有引用；
   - block 被引用时禁止 BlockPool 回收；
   - request 结束、取消、fallback 和 cache eviction 都必须释放引用；
   - 重复 lookup 只增加引用，不重复分配 snapshot 或 private block。

   需要覆盖长时间重复请求、不同 request 共享 segment、取消请求和显存压力下的
   eviction，确保 pool、native lease 和 state block 不单调增长。

4. **10-C-4：transition/state 绑定与 decode 复用（正确性 + 稳定性）**

   为每个 request 维护独立的：

   - recurrent state block；
   - conv tail block；
   - transition cursor/operator；
   - logical segment/range cursor。

   禁止通过全局临时 tensor 或 batch 位置推断 state 所属 request，防止 packed
   batch 和 decode 过程中发生跨 request 覆盖。

   decode 不得因为进入单 token 阶段而退回全量 materialization 或普通 KV 路径。
   每个 decode step 应直接使用 request 已持有的 logical-to-native mapping；只有
   mapping、RoPE 位置或 state 不安全时才按 request fallback。

5. **10-C-5：生产监控与 request-level 安全 fallback（稳定性）**

   增加 hit/miss、reuse token、seam token、private materialization、fallback、
   lease 数量、eviction、pool 使用量和 kernel launch 指标。任何 lease、layout、
   position、state 或 backend 不一致都按 request 回退，不影响同 batch 其他请求。

   本阶段还要覆盖普通请求零回归、PIC request 单独 fallback、同 batch 其他请求
   继续执行，以及长时间运行无 pool/lease/state 泄漏。

实施顺序固定为：

```text
10-C-1 request logical-to-native mapping
  → 10-C-2 allocation-time public/private slot lease
  → 10-C-3 private KV persistence + lease/ref-count/eviction
  → 10-C-4 request-bound state + decode reuse
  → 10-C-5 production monitoring + request-level fallback
```

本阶段仍不改变普通请求的原生 block allocation 和 attention 路径；所有优化必须
保留显式开关和 request-level fallback。10-C 的完成标准不是“加入了某个 kernel”，
而是：持久 native mapping 正确、重复 materialization 显著减少、decode 不退化、
长时间运行无 pool/lease/state 泄漏，并在普通请求上零回归。

性能指标必须同时记录普通基线、Stage 9-B prototype 和 Stage 10 版本：

- prefill/decode latency；
- end-to-end throughput；
- 实际跳过 token 数；
- PIC hit rate、fallback rate；
- 显存和 native block 使用量；
- kernel launch 数、GPU 利用率和通信开销。

Stage 10-A 和 Stage 10-B 已完成服务器验收，分别验证了 packed mixed-batch
执行以及 packed attention metadata 的安全校验与 fallback。Stage 10-C-1 至
Stage 10-C-5 已完成验证。

Stage 11 从当前 Git 工作线继续增量实现，目标是完成真实 Mooncake native KV
持久化和 PIC 非前缀复用。Mooncake 源码、TransferEngine 和 Store 不修改；
Mooncake 只负责 KV object 的存储和传输，KV payload 采用 vLLM 原生 KV block
格式。当前 Git 工作线中已有的 Mooncake provider、native-KV reference/lease
接口和 fail-closed 开关，作为 Stage 11-A 的起始代码；它们尚未单独宣称完成，
必须与真实 native KV 持久化一起验证。

#### Stage 11-A：原生 KV block 持久化

目标：直接保存 vLLM native KV pool 中的 block 数据，保留原生 dtype、layout、
KV head、head size 和 block size；定义 Mooncake object key、metadata 和 external
reference，并在首次请求完成后持久化 public/private/edge KV。

验证：KV block 可写入和读取；metadata 与 vLLM KV cache 配置一致；相同 segment
可去重；失败时不影响普通请求；不修改 Mooncake 或 attention kernel。

#### Stage 11-B：远程 KV 导入与 native block 绑定

目标：从 Mooncake 读取 native KV block，写入当前请求由 vLLM 分配的 native block，
复用原生 block table、slot mapping 和 attention kernel，并建立本地 lease、
ownership 和释放机制。远端 block ID 不跨进程复用，不伪造 block ID。

验证：跨进程读取一致；native KV 逐 block 校验一致；lease、释放和 fallback 正确；
多请求之间无 block 串扰。

#### Stage 11-C：真实非前缀 PIC 复用

目标：完成以下链路：

```text
Mooncake lookup
→ native KV import
→ block table / slot mapping attach
→ transition/state restore
→ seam forward
→ miss segment forward
→ decode
```

支持任意 segment 起点、完整 native block 复用、非对齐边界的 edge-private block
或 seam 重算，并正确处理 public KV、private KV、RoPE、绝对位置以及 hybrid 模型
的 recurrent state、conv tail 和 transition。

验证：`X + B + Y` 成功命中 `B`；forward token 数减少；logits、生成结果和 hybrid
state 与完整重算一致；普通请求、request-level fallback 和多请求并发不受影响。

#### Stage 11-D：端到端稳定性与性能验收

目标：验证真实 Mooncake 跨进程 KV 复用、重复请求、并发、取消、eviction 和
fallback，并记录远程传输开销、命中率、复用 token、seam token 和实际加速比。

验收：100～1000 次重复请求无 pool、lease、state 泄漏；普通请求零回归；
remote KV lookup/import/attach 全链路通过；PIC 输出与完整重算一致；性能结果与
本地 PIC、Stage 10-C 和完整重算基线对比记录。

##### Stage 11-D 性能优化实施计划（TCP、native KV 主路径）

Stage 11-D 的功能链路已经进入性能优化阶段。当前目标不是继续增加新的
correctness 分支，而是在现有“canonical native KV → request-private KV → 必要的
materialization/RoPE rerotation → attention”主路径上，降低 TCP、CPU 反序列化、
H2D 和逐 layer 调用的固定开销。

当前硬件没有 RDMA 网卡，Mooncake 使用 TCP；因此本计划不把 RDMA 收益写入当前
验收结果，只保留为后续硬件条件满足时的对照项。当前也不新增“位置相同且对齐时
直接写最终 native KV buffer”的小众快速旁路：该场景覆盖面有限，会引入额外的
position/ownership/slot 分支，不能作为主优化方向。非前缀位置的正确性仍以
request-private materialization 和必要的 RoPE rerotation 为准。

已确认的性能基线（Stage 11-D D6）：

| 项目 | 单卡 i9-14900KF | 三卡 Xeon 4210R | 结论 |
| --- | ---: | ---: | --- |
| Full | 111～112 ms | 296～312 ms | GPU/CPU 平台差异需单独记录 |
| PIC（compact transition） | 约 131 ms | 约 431 ms（1k）/516 ms（4k） | 当前 PIC 额外开销主要在 native KV |
| compact transition payload | 约 38.4 MB | 约 38.4 MB | 不随复用长度增长 |
| native KV（1k → 4k） | — | 13.4 MB → 53.5 MB | 按 token 线性增长 |
| native KV restore | — | 22.7 ms → 70.0 ms | TCP Get + H2D/index copy |

实施顺序固定如下；每一项都必须先在 Docker 真机上测量，再修改代码，并单独
commit、回归和刷新累计 patch。

1. **P0：冻结正确性与性能基线**

   固定 checksum 关闭、compact transition 开启、相同 prompt/tokenizer、相同
   `max_tokens`、warmup 和请求次数。分别记录 Full、PIC 首次、PIC reuse 的
   TTFT、端到端 latency、forward token、reused token、payload bytes、native
   blocks、输出一致性和 fallback。所有后续优化必须同时通过 D5 200+ 次
   reuse、D1 长时稳定性和普通请求零回归。

2. **P1：补齐 native KV 细粒度 profiling**

   将 native KV 路径拆成并记录：manifest/metadata lookup、object key 构造、
   Mooncake Get 等待、CPU buffer 接收、bytes view/clone、H2D、index copy、
   materialization、RoPE rerotation、transition apply 和总耗时。区分“调用次数”
   与“传输字节数”，特别检查同一 request 是否重复读取 manifest 或重复构造
   reference。没有 profiling 证据的优化不得进入主线。

3. **P2：消除 request 内重复 metadata Get**

   lookup 阶段已经得到的 native KV manifest、layer key、block range 和 payload
   descriptor，应通过 request-local reference 传给 import 阶段复用。只消除同一请求
   的重复元数据访问，不建立跨请求 payload 缓存，也不假设下一次请求一定复用
   同一 segment。连接失败时必须保留 request-level fallback，并记录 retry/fallback
   原因。

4. **P3：先对齐 Full Attention 的 Mooncake KV transfer 路径**

   P3 不直接在 PIC 现有的 `store.get() → bytes → clone → H2D → index_copy_()`
   路径上做局部优化，而是先把 Full Attention 已验证的实现拆解成可复用的接口契约。
   必须确认并记录以下内容：

   - Full Attention 使用的 KV block/layout 描述、layer 顺序和目标 block 映射；
   - `batch_get_into`、`batch_get_into_multi_buffers` 或等价批量接口在当前
     Mooncake 版本中的实际可用性；
   - 预注册 host/GPU buffer、地址列表、大小列表和 buffer 生命周期；
   - TCP 模式下是否启用了 Mooncake 的 CUDA/GPU-aware transfer 能力；
   - Full Attention 的传输完成通知、同步点、异常处理和 request/block lease
     释放方式。

   输出一份 Full Attention 与 PIC 的差异表，未完成接口对齐前不得进入 PIC
   native KV 改码。当前 Mooncake 版本不支持某个接口时，必须保留已有串行 fallback，
   不能改变普通请求或 PIC 的正确性。

5. **P4：把 Full Attention 的批量传输机制迁移到 PIC native KV**

   在 P3 契约确认后，将 PIC native KV 从“每层一个独立 `store.get()`”改为与
   Full Attention 一致的 block-oriented 批量接收：

   ```text
   PIC native KV manifest/reference
     → Full Attention 风格的 layer/block transfer description
     → batch Get
     → 预分配/预注册目标 buffer
     → native KV layout materialization
   ```

   第一版只替换传输和接收层，不改变 canonical KV、request-private KV、RoPE
   rerotation、checksum、lease 和 request-level fallback 语义。验收重点是：

   - Get 调用次数和 batch size；
   - payload bytes 不变时的有效吞吐；
   - native KV checksum、block table、slot mapping 和输出一致性；
   - Mooncake API 不可用时是否正确回退到旧路径。

6. **P5：复用 Full Attention 的 buffer、同步和异步生命周期**

   在 P4 批量传输正确后，继续迁移 Full Attention 的 host/GPU buffer 管理和完成
   通知机制，避免 PIC 自己重新设计一套传输生命周期。目标是：

   - native KV Get 与 transition payload Get/deserialize 并行；
   - native KV 接收与不依赖 native KV 的 descriptor/metadata 准备重叠；
   - 由 request-owned state 管理 host buffer、CUDA stream、future 和 lease；
   - attention 只在所有依赖的 transfer/materialization 完成后执行。

   验收时必须分别报告 Get、CPU 接收、H2D、index/materialization、transition
   deserialize、overlap 后总耗时，不能只比较端到端 latency。

7. **P6：沿用 Full Attention 的直接 buffer 经验，减少 PIC 中转**

   在保持非前缀位置正确性的前提下，优先复用 Full Attention 已使用的 registered
   buffer、连续目标布局和 GPU-aware 接收能力，逐步消除 PIC 中的：

   ```text
   Python bytes
     → CPU clone
     → 临时 tensor
     → H2D
     → index_copy_
   ```

   不能为了减少一次 copy 而把 canonical KV 错当作目标绝对位置的最终 KV。对于
   request-private KV，仍必须执行 canonical KV → materialization → 必要的 RoPE
   rerotation。位置不同、非对齐和 edge-private 场景继续使用正确性优先的路径。

8. **P7：PIC 特有的 transition/linear-attention 优化**

   Full Attention 的 block transfer 经验只能直接覆盖 native KV，不能直接替代
   GDN/Mamba 的 recurrent state、conv tail 和 transition operator。因此在 P4～P6
   稳定后，单独优化 PIC transition：

   - native KV batch Get 与 transition payload Get 并行；
   - 保持 compact binary wire format；
   - 复用 `frombuffer`/连续 tensor view，避免 `torch.load/pickle` 和额外 clone；
   - 减少 JSON metadata 解析和逐 tensor `.to(device)` 的串行开销；
   - 拆分并记录 payload Get、deserialize、state materialization 和 transition
     apply 时间。

   这一阶段不得反向修改 Full Attention native KV 的 correctness 路径。

9. **P8：最后才评估 PIC 自己的数据量和 TCP 优化**

   只有 P2～P7 完成，并证明传输仍是主要瓶颈后，才评估 block packing、压缩、
   delta、TCP 并发 Get、socket buffer 或 Mooncake 参数。第一优先级仍是继承
   Full Attention 的已验证实现，不重复发明新的传输协议。

   数据量优化必须分别证明：

   - payload bytes 确实下降；
   - CPU 压缩/解压开销没有抵消收益；
   - native KV dtype/layout 和非前缀 materialization 语义不变；
   - 跨请求 delta 不被错误假设为可复用；
   - 失败时仍能 request-level fallback。

   当前没有 RDMA 网卡时，不把 `protocol=rdma` 作为可用方案；RDMA 仅作为未来
   硬件对照实验。TCP 多流也必须先证明 Mooncake Full Attention 路径支持并能在当前
   版本启用，再进入 PIC 方案。

性能验收口径：PIC 单请求 TTFT 不要求必然低于 Full，因为 PIC 必然包含远程
lookup/import/transition；主要指标改为重复请求吞吐、节省的 forward/prefill token、
GPU 利用率、每请求传输字节和 amortized cost。同时保留 PIC/Full latency，作为
额外开销指标。每个优化子阶段都要给出：旧版本、当前版本、Full 基线、1k/4k
复用长度、单卡/三卡 TCP 环境、warmup、重复次数、wrong/fallback 数和日志提取
结果。

Stage 11 的固定依赖顺序为：

```text
11-A 原生 KV 持久化
  ↓
11-B 远程 KV 导入与 native block 绑定
  ↓
11-C 非前缀 PIC 复用与 hybrid state
  ↓
11-D 稳定性、性能和完整验收
```

每个子阶段都必须基于上一阶段已验证 Git commit 增量修改，并生成从官方初始
commit 到当前子阶段的累计 patch 和 README。初始实现不修改 Triton/CUDA；只有
profiling 证明 vLLM 原生 kernel 无法满足需求时，才另行增加最小 kernel，并单独
验证编译产物、开关和 fallback。

### Stage 11E：P3-P6 合并实施计划（当前执行版本）

#### 最终架构收敛：Mooncake Store pull + 统一 request 生命周期

本阶段不再把 PIC native KV 误当成 PD 分离场景的 TransferEngine push。PIC
是共享 KV cache 场景，默认采用 Full Attention 已验证的 Mooncake Store pull：
启动时注册 KV buffer，请求到达后构造 block descriptor，再由 Store 的 C++
`batch_get_into_multi_buffers()` 直接写入已注册目标。Python 只负责控制面和
descriptor，正常传输过程不应返回大块 Python `bytes`。

最终 PIC 架构统一为一个 `PICRequestTransfer`，但保留两种数据专用执行器：

```text
PICRequestTransfer
├── NativeKVTransfer
│   └── Store batch Get -> registered native KV buffer
│
└── TransitionStateTransfer
    └── Store fixed-layout batch Get -> registered state buffers
```

**强制架构决策：native KV 和 transition/state 的实时共享缓存拉取均优先使用
Mooncake Store 的 C++ batch/direct-buffer 数据面。** 不再逐层 Python `Store.get()`；
state 与 native KV 只在传输后的数据处理阶段区分。TransferEngine 只作为明确的
legacy P2P/PD 对照路径，不参与默认 PIC 共享缓存路径。

Store direct transfer 传输的是已注册 destination address、object key 和 length；
因此 state fast path 的 manifest 只需发布 key、dtype、shape、byte length 和
ABI/version，不需要跨进程裸 GPU source pointer 或 endpoint。Store bytes/H2D 只保留
为目标布局不支持、API 不可用或 fixed-layout ABI 校验失败时的 fallback。

两条执行器统一以下生命周期：

- request transfer handle；
- ready/error 状态；
- cancel/timeout；
- lease 和 buffer 生命周期；
- request-level fallback；
- forward 前统一 barrier；
- profiling 和端到端等待统计。

但不能强行共用数据处理函数：native KV 是固定 block ABI、可变 block 数量；
transition/state 是 matrix、zero_state、conv state 等固定 ABI buffer 加上请求级
descriptor。二者都不应在 ready 前被 attention 或 state apply 读取。

native KV 的目标路径：

```text
注册目标 KV buffer
    -> 构造 key/dst/length block descriptor
    -> Store batch_get_into_multi_buffers
    -> native_kv_ready
```

transition/state 的目标路径：

```text
固定 state ABI：dtype/shape/stride/layer layout
    -> Store fixed-layout tensor batch Get
    -> transition_ready
    -> state.matmul(matrix) + zero_state
```

当前 38 MB 级别 compact payload 的 deserialize 不是 state 的固有要求，而是
当前 Store 存储表示的结果。固定布局 fast path 下，只解析很小的 descriptor，
不再执行大 payload 的 `torch.frombuffer`、CPU tensor 重建和应用层 H2D。

必须保留兼容路径：Store direct API 不可用、ABI/dtype/shape 校验失败或传输错误时，
回退到现有 Mooncake Store compact payload + deserialize。Store fixed-layout fast path
与 bytes fallback 必须共享同一套 ready/error、fallback、lease 和 forward barrier。

注意：Store pull 不依赖 A 进程在线，manifest 和对象可持久化复用；TCP 下仍受
Mooncake staging/CPU 网络栈带宽限制，但避免了当前 TE padded-page 的额外字节和
P2P 控制面开销，必须用相同 payload 做基准对比。

本阶段的实施边界因此确定为：

```text
A：native KV 传输契约和 direct-receive 基础
  -> B：真正 TransferEngine + 统一 request transfer + native KV/state 异步传输
  -> C：剩余 materialize/clone/H2D/index_copy 低拷贝优化
```

> **【2026-10-21 已作废】** 上述 B 包的 TransferEngine 数据面不再实施；A 已按
> Store 落地，C 保留。Store-only 的最终边界与 11E-D1/D2 见
> “2026-10-21：Store-only 收敛，TransferEngine 退场”。

#### 实施状态修正（2026-10-19）

#### 实施方向修正（2026-10-21）

由于三卡 TCP 实测显示 TransferEngine padded-page P2P 路径明显慢于 Store
compact pull，且其语义属于 PD 分离而非本项目共享 KV cache，以下 11E-B 实现以
Store pull 为默认主路径。后文 2026-10-19 的 TransferEngine 主路径描述仅保留为
历史设计/显式对照模式；以本节和“Store-pull 路径收敛”节为准。

> 2026-10-21 进一步收敛：TE 不再作为 PIC 的一部分维护，将按 11E-D1/D2 删除。
> 见“2026-10-21：Store-only 收敛，TransferEngine 退场”。

- `HYPIC_PIC_NATIVE_KV_DIRECT=1` 默认启用 Store direct buffer registration；
- `HYPIC_PIC_NATIVE_KV_TRANSFER_MODE=store` 默认，`page|compact` 仅 legacy P2P；
- `HYPIC_PIC_TRANSITION_TRANSPORT=store` 默认，`p2p` 仅 legacy state P2P；
- transition/state 新 manifest 使用 `store_layout`，固定 tensor 通过一次 Store
  batch Get 写入 GPU，正常路径 `deserialize_ms=0`；
- bytes/H2D/deserialize 仍保留为布局/API/ABI 失败时的 request-level fallback。

Stage 11E 当前只完成了 A 包的基础代码，A 不能被表述为“已经完成 Full
Attention 的 TransferEngine 数据面”。A 的已完成范围是：

- native KV buffer registration 入口；
- block-oriented descriptor；
- Store `batch_get_into_multi_buffers()` 的 opt-in direct-receive 尝试；
- direct path 不可用时回退到旧的 receive/materialize 路径；
- native KV 的完成状态、校验和基础 profiling。

A 当前仍使用 Mooncake Store API 作为实际数据源，真正的
`TransferEngine.batch_transfer_async_read/write()` C++ 数据面尚未接入；PIC
transition/state 的 TransferEngine 路径也尚未实现。A 的当前任务是先在 Docker 真机
验证 descriptor、目标地址、TCP staging/H2D 和 correctness，验证前不改变默认路径。

下一步直接进入 B 包，B 才是本阶段的主要实现阶段：

| 包 | 当前状态 | 下一步必须完成的内容 |
|---|---|---|
| A | 基础代码已写，真机验证待完成 | 保持 Store fallback，确认 native KV direct contract |
| B | 尚未完成 | 接入 TransferEngine C++ 数据面，同时管理 native KV 和 state |
| C | 尚未开始 | 在 B 稳定后减少 materialize/clone/H2D/index_copy |

B 包内部顺序固定为：

```text
B0：PICRequestTransfer 统一 handle/ready/error/fallback/lease
  -> B1：native KV TransferEngine batch/async transfer
  -> B2：state fixed-layout TransferEngine batch/async transfer
  -> B3：scheduler WAITING_FOR_REMOTE_KVS 和统一 forward barrier
```

B1/B2 必须使用同一个 request-level transfer 生命周期，但保留各自的 descriptor
和数据处理器。只有源进程在线且 source buffer 已注册时，TransferEngine 才是实时
主路径；源进程退出、buffer 不可访问或 ABI 校验失败时，才使用 Store fallback。

C 包不再负责决定传输协议，也不再新增独立的 state 传输实现；C 只处理 B 完成后的
残余 CPU clone、H2D、index_copy、materialize 和 buffer reuse 优化。

本阶段的目标不是只把 `store.get()` 放入线程，而是逐步对齐 Full Attention
Mooncake 的四个能力：

```text
registered target buffer
    -> block-oriented batch Get
    -> ready/error completion
    -> scheduler-level waiting
```

PIC 的数据分为两条依赖链：

```text
native KV cache
    -> receive
    -> materialize/attach
    -> attention 读取

compact transition payload
    -> receive
    -> CPU deserialize
    -> GPU state.matmul + zero_state
    -> hybrid forward
```

两条链都必须有明确的 ready/error 状态；任意一条失败都必须 request-level
fallback，禁止读取旧 KV block 或未完成的 hybrid state。

#### 已完成基础：实现包 A0，P3/P4 生命周期骨架

当前状态：已实现并通过已有 correctness 回归，但尚未等价于 Full Attention
的 direct-buffer 数据路径。

已完成内容：

- request-owned native KV import handle；
- pending/ready/error 状态和完成事件；
- receive/materialize 生命周期拆分；
- lease-owned result、失败回收和 request-level fallback；
- 同步热路径通过 `complete_inline()` 保留，避免无重叠场景产生额外线程切换；
- `HYPIC_PIC_NATIVE_KV_OVERLAP=0` 可回退到已验证同步路径。

当前明确未完成：

- Mooncake 目标 buffer 注册；
- Full Attention 风格的 block descriptor；
- `batch_get_into_multi_buffers()` 或等价批量 direct-buffer API；
- scheduler-level `WAITING_FOR_REMOTE_KVS`；
- 去除 PIC 自己的 CPU clone、H2D 和 `index_copy_`。

#### 实现包 A：P3/P4 传输层对齐

本包合并 P3-a 和 P4-a，目标是一次性完成 Full Attention 风格的目标 buffer、
block descriptor 和 batch/direct receive。两项必须作为一个传输层改动一起验证，
避免先引入 descriptor、再用另一套接口实现接收。

##### A1：目标 buffer 注册和 block descriptor

在改动传输逻辑前，先确认当前 Mooncake 版本的实际 API 和 TCP 行为：

- native KV 每层 cache 的 base address、block size、dtype、layer/block 映射；
- `register_buffer()` 的 host/GPU/pinned 支持和生命周期要求；
- 目标地址是否允许跨请求复用，以及 lease 释放时机；
- TCP 模式下是否仍存在 Mooncake 内部 staging；“direct buffer”只表示应用层
  不再产生 Python bytes，不等同于网卡零拷贝直达 GPU。

输出统一的 `PICNativeKVBlockDescriptor`，至少包含 key、layer、block、目标地址、
大小、request/lease 标识。API 不可用时必须保留旧同步路径。

验收：descriptor 与 vLLM native KV layout 一致；目标 block 不重叠；失败、超时、
取消和 lease 回收不会留下旧数据可读。

##### A2：Full Attention 风格 batch/direct receive

将 PIC native KV 的当前路径：

```text
逐 layer store.get()
    -> CPU bytes
    -> clone/H2D/index_copy
```

改为：

```text
block descriptors
    -> batch_get_into_multi_buffers(keys, dst_addrs, sizes)
    -> ready/error
```

第一版只改变 native KV 接收层，不改变 canonical KV、request-private KV、RoPE
rerotation、slot mapping 和 fallback 语义。对于非前缀位置，仍然必须进行必要的
private materialization；不能把 canonical KV 误当成最终 request KV。

验收：Get 次数、batch size、bytes、CPU 接收、H2D、index/materialization 分别打点；
输出、logits、state、block table 和 checksum 与旧路径一致；Mooncake API 不可用时
自动回退。

A 包当前代码已实现为 opt-in 候选路径：
`HYPIC_PIC_NATIVE_KV_DIRECT=1` 时尝试注册 native KV storage 并调用
`batch_get_into_multi_buffers()`；默认值仍为 `0`。注册失败、目标布局不连续、
Mooncake API 不存在或开启诊断 checksum 时，自动回到旧的 receive/materialize 路径。
Docker 环境中的真实 GPU 地址、TCP staging/H2D 行为和输出正确性仍需测试 agent
验收；在该验收完成前，不把 direct path 设为默认。

#### 实现包 B：P5 全链路请求级异步

本包合并 P5-N、P5-T 和 P5-S。native KV、compact transition 和 scheduler waiting
必须作为一个请求级状态机一起实现；否则只能得到“后台传输但 execute_model 仍阻塞”的
局部重叠，不能对齐 Full Attention。

##### B0：统一 PIC request-level transfer 生命周期

B 包的最终架构不是为 Full Attention native KV 和 PIC transition/state 各写一套
互不关联的异步路径，而是建立统一的 `PICRequestTransfer` 请求级生命周期，再在
内部保留两种数据专用执行器：

```text
PIC request import
        │
        ├── native KV receiver
        │     ├── Mooncake Get / batch-direct receive
        │     ├── 写入目标 native KV buffer
        │     └── native_kv_ready/error
        │
        └── transition/state receiver
              ├── Mooncake payload Get
              ├── CPU deserialize
              └── transition_payload_ready/error

统一 forward barrier：
native KV ready + transition ready
        ↓
必要的 materialize/attach
        ↓
state.matmul(matrix) + zero_state
        ↓
PIC forward
```

两条执行器不能强行共用同一个数据处理函数：native KV 是按 block 描述、可直接
写入已注册目标 buffer 的数据；transition/state 是 compact matrix、zero_state、
conv state 等 payload，需要 deserialize 后再进行 GPU state apply。统一的部分必须
包括 request transfer handle、ready/error 状态、取消/超时、lease 释放、fallback 和
forward 前等待点；数据接收、解析和 materialize 仍按各自格式实现。

因此 A 包只先建立 native KV 的 buffer registration、block descriptor 和
batch/direct receive 基础能力；B 包必须把 native KV 与 transition/state 纳入同一个
request-level asynchronous import 生命周期。后续是否真正并行执行 native Get、
transition deserialize 和 GPU materialize，要根据 CPU/GIL、内存带宽和 GPU stream
竞争的实测决定，但不能退回到两个互不感知的 ready/fallback 机制。

##### B0-A：控制面必须复用 Mooncake connector 的 bootstrap/ZeroMQ 契约

> **【2026-10-21 已作废】** Store-only 收敛后 PIC 没有 TE 控制面；本节的
> bootstrap/ZeroMQ/`batch_transfer_sync_write` 设计不再实施。原文仅作历史记录，
> 见“2026-10-21：Store-only 收敛，TransferEngine 退场”。

PIC 不再维护自定义 TCP descriptor server。控制面统一采用 Full Attention
Mooncake connector 已有的形态：bootstrap 的 `/register`、`/query` 负责 worker
发现，ZeroMQ ROUTER/DEALER 负责交换 request-local descriptor，TransferEngine
负责真正的数据面 `batch_transfer_sync_write`。ZeroMQ 只承载 source key、目标
session、GPU 地址、length、rank 和 ready/error 元数据，不承载 native KV/state
payload。

native KV 和 transition/state 共用这套控制面与 request lifecycle，但保留不同
descriptor/data processor。旧 manifest、bootstrap 不可用或 TransferEngine 失败时，
仍按 request-level fallback 使用 Store；不得把旧自定义 TCP 协议继续作为新 manifest
的默认路径。

验收要求：任一子传输未 ready 或失败时，PIC attention 不得读取目标 KV block，
state 不得 apply；必须触发 request-level fallback，并释放两类 transfer handle 和
lease。

##### Stage 11E-TP：全部优化完成后的 TP≠1 适配

> **【2026-10-21 需重新设计】** Store-only 收敛后没有“每 rank TransferEngine
> endpoint / 裸 pointer”这回事。本节的**目标**（rank 对齐、ready 聚合、scheduler
> 成对清理）仍然有效，但**寻址方式**改为 Store key 前缀：每个 rank 用自己的
> `tp_rank` 前缀发布对象，接收端按本地 rank 生成 key 列表，一次
> `batch_get_into_multi_buffers` 完成本 rank 接收，再聚合成 request-level ready。
> 下列第 1 条按此修订，其余条款保持。

TP≠1 不作为 B1/B2/B3 或 C/P6 的前置条件。本阶段先固定在 TP=1 上完成并
验证 Store batch receive、transition/state、scheduler waiting、materialize 和
TCP/低拷贝优化；只有这些优化的性能与 correctness 基线稳定后，才拆出 TP
适配分支。这样可以避免同时引入传输路径变化、GPU buffer 变化和跨 rank
同步问题，导致无法判断性能回退来源。

进入 Stage 11E-TP 时，不能把 TP=0 rank 的对象 key、单 rank 就绪状态直接当成
整个请求的完成状态，必须补齐 tensor-parallel 多 rank 语义。

适配范围固定为以下四部分：

1. **每个 TP rank 独立寻址**：对象 key 必须带 `tp_rank`、KV group/layer 标识；
   每个 rank 只接收自己前缀下的对象，禁止跨 rank 复用 key 或地址。manifest 必须
   记录 `tp_rank`、dtype/shape/layout 版本和 byte length；接收端按本地 rank 的
   KV group/layer 生成自己的 key 列表与目标地址。
2. **publish/import 的 rank 对齐**：A 侧不能只发布 rank 0 的 native KV 或
   state。每个 TP rank 发布自己的 descriptor，B 侧必须验证 rank 数、TP rank、
   KV group、layer 集合、block 数、dtype、shape 和 token range 完整一致；缺任一
   rank 或 descriptor 不完整时，整个 request 级 fallback 到 Full/Store 路径。
3. **统一 ready/error 聚合**：`PICRequestTransfer` 先得到本 rank 的 native
   KV/state ready，再通过 TP group 的同步/完成通知聚合成 request-level ready。
   任一 rank transfer、state apply、CUDA event 或 lease 失败，都必须传播到
   所有 rank，并释放所有已成功取得的 handle/lease；不能出现部分 rank 进入
   attention、部分 rank fallback 的分叉状态。
4. **scheduler 对齐**：B3 的 `WAITING_FOR_REMOTE_KVS` 只能在所有 TP rank
   的 native KV 和 transition/state 完成后清除。scheduler 必须保存 request
   的 rank-aggregate transfer ID，取消、超时、重试和 preemption 时按 rank
   成对清理，不能只检查本地 worker 的 ready 标志。

TP 适配的实现顺序（位于 P3-P9 全部稳定之后）：

```text
TP-1  rank-local Store key schema / buffer registry / descriptor
  -> TP-2  native KV 每 rank batch receive
  -> TP-3  transition/state 每 rank fixed-layout receive
  -> TP-4  rank-aggregate ready/error + lease cleanup
  -> TP-5  scheduler WAITING_FOR_REMOTE_KVS promotion
```

TP≠1 验收至少包括 TP=2 和实际部署的 TP 配置；每组都要做 Full 对照、
4k reuse、partial/non-prefix correctness、D5 200+ 次 `wrong=0`、D1 1000
次 lease/pool 稳定性、普通请求回归和 descriptor 缺失/单 rank 失败 fallback。
同时记录每 rank 的 key 前缀、bytes、batch count、native materialize、state
apply 和总延迟，确认没有重复传输或只传 rank 0 的假命中。只有 TP 适配通过后，
Store batch receive 的性能数据才可以作为正式结果。

##### B0-NKV：PIC native KV public block 与 descriptor 组织对齐 Full Attention

本子阶段只调整 PIC native KV 的传输组织，不修改 vLLM 原生 `BlockPool` 的分配
算法，也不新增“强制连续 block”接口。原生 Full Attention 使用普通
`BlockPool.get_new_blocks()`，再按 source/destination block group 判断是否连续；
不连续时保留多个 descriptor。因此 PIC 必须复用这一语义，不能通过排序 block ID
伪造物理连续，也不能把 A 侧 block ID 直接用于 B 侧。

当前 PIC 的 public block 仍由 PIC scheduler 申请和记录，但底层 block 继续来自
vLLM 原生 `KVCacheManager -> BlockPool`。本阶段的目标是让 PIC 的 descriptor
组织接近 Full Attention，同时保持原生 block 生命周期不变：

```text
PIC scheduler
  -> BlockPool.get_new_blocks()
  -> request-local public native KV blocks
  -> layer/region transfer plan
  -> source/destination concurrent grouping
  -> coalesce only when both physical ranges are contiguous
  -> bounded batch_transfer_sync_write()
  -> one request-level ready/error barrier
```

实施顺序：

1. **冻结并记录现状**：记录 allocator block IDs、kernel block IDs、destination
   addresses、连续段数量、coalesce 后 descriptor 数量、batch 数量和
   `native-transfer-total`，确认 1632 的构成和每层分布。
2. **重构 descriptor 数据结构**：从“全量扁平的 layer/KV/block 地址数组”改为
   按 layer/KV region 保存 source/destination block group；采用 Full Attention
   的双端连续判断，只有 source 和 destination 同时连续时才合并。
3. **保留非连续安全路径**：B 的 block 不连续时不改写 block ID、不强行搬动
   BlockPool，只保留多个合法 descriptor；按 layer/region 建立传输批次，避免
   一次提交 1632 个 descriptor。
4. **保留批次保护开关**：
   `HYPIC_PIC_NATIVE_KV_BATCH_DESCRIPTORS` 仅作为 TransferEngine 的安全上限和
   回滚工具，不作为减少 descriptor 总量的最终方案。所有批次由同一个
   ready/error handle 等待，任一批次失败都触发 request-level fallback。
5. **保持生命周期不变**：继续使用现有 PIC public allocation、zeroing、lease、
   adoption 和 release 逻辑；TransferEngine 完成前禁止 attach/attention 读取，
   fallback 时释放所有未采用的原生 block。
6. **Full Attention 风格直写**：新 manifest 按 layer 保存有序的 K/V block
   source regions；接收侧使用 request-local 原生 BlockPool block 的真实地址，
   仅在 source 和 destination 同时连续时合并，随后由一个 request-level
   TransferEngine batch 直接写入 native KV。非连续目标保留多个合法 descriptor，
   不增加 staging buffer、不做额外 `index_copy_`，也不修改原生 BlockPool。
   `HYPIC_PIC_NATIVE_KV_BATCH_DESCRIPTORS` 仅作为 descriptor 上限和
   request-level Store fallback 保护；验收时记录实际 coalesce 后 descriptor
   数量和 transfer 长尾。

验收指标：

- 普通 vLLM 请求无回归；
- PIC native KV block ID、lease、zeroing 和释放无增长；
- descriptor 数量按 layer/region 可解释，不能出现错误合并；
- 200+ 次跨进程 reuse `wrong=0`；
- 1000 次 reuse 后 lease/pool 稳定；
- `native-transfer-total` 的 p50/p95 和长尾不因组织改动恶化；
- Store fallback、TransferEngine 失败和 manifest 缺失均保持 request-level fallback。

以下方案明确不纳入本子阶段：

- 修改原生 `BlockPool` 增加强制连续分配；
- 通过排序或重编号伪造物理连续 block；
- 把 A 侧 block ID 当作 B 侧 destination block ID；
- 提前适配 TP≠1；
- 继续以 256/512 调参替代 descriptor 组织重构。

##### B0-NKV-CANONICAL-LEASE：消除 PIC 第二套 public block

当前 PIC 在同一个原生 `BlockPool` 上同时存在普通 request blocks 和 PIC
public blocks。这会让普通请求与 PIC 分配互相穿插，导致接收侧物理地址碎片化，
并把 PIC 自己的 block 生命周期、清零、lease 和释放逻辑叠加到 Full Attention
已经验证过的原生路径上。

新设计不修改 KV cache layout，也不强制 `BlockPool` 连续分配，不通过排序 block ID
伪造物理连续性，而是引入 PIC canonical lease：

```text
exact-position prefix (segment.start == canonical_start == 0)
    ordinary request native KV blocks
        -> PIC canonical lease
        -> publish the same physical source regions

shifted/interior segment
    canonical public KV copy
        -> request-private materialization
        -> RoPE rerotation when target position differs
```

实现边界：

1. exact-position prefix 直接记录请求已经填充的 native block，并由 scheduler
   在 materialization 到达后调用 `PICNativeKVLease.from_block_pool()` 增加一份
   canonical 引用；请求结束只释放 request 引用，PIC eviction 再释放 canonical 引用。
2. exact-position 路径不再由 `_ensure_pic_public_blocks()` 申请第二套 public block，
   Mooncake source descriptor 直接描述 request-owned native block。
3. shifted/interior 路径暂时保留现有 canonical public copy 和 private
   materialization/rerotation，不能把 request-absolute KV 误标成 canonical-local KV。
4. B 侧仍然使用本地普通 request blocks 作为 destination；A/B block ID 不跨进程复用，
   descriptor 仍按 Full Attention 的真实 source/destination 连续性合并。
5. state/transition 不与 native KV 共用数据格式，但继续共用 `PICRequestTransfer`
   的 ready/error、lease、fallback 和 forward barrier。

本子阶段已实现 exact-position prefix 的安全 alias；scheduler-level
`WAITING_FOR_REMOTE_KVS`、shifted segment 的 canonical lease 完整替换、TP!=1
rank 聚合仍属于后续 B3/Stage 11E-TP，不在本次 alias 变更中提前混入。

验收必须包含：

- alias capture 后 request block pool 不额外减少一套 public block；
- request 结束后 canonical lease 仍保护 source block，eviction 后才回收；
- exact-position logits/tokens 与旧方案一致；
- shifted/non-prefix 仍走 private materialization，输出与 full recompute 一致；
- 200+ 次跨进程 reuse、1000 次 lease/pool soak、manifest/fallback 和普通请求回归。

##### B1：native KV 请求级异步

> **【2026-10-21 部分作废】** “batch/direct receive”已按 Store 语义实现（11E-C：
> 一次 `batch_get_into_multi_buffers` + best-effort `register_buffer`）；TE 的
> P2P push 变体不再实施。请求级异步本身仍有效，并入 B3。

当前已完成 receive/materialize split，且 overlap 默认开启：

```bash
unset HYPIC_PIC_NATIVE_KV_OVERLAP       # 默认开启
export HYPIC_PIC_NATIVE_KV_OVERLAP=0    # 回退同步路径
```

下一步把后台阶段从“后台拉取 payload”推进到 Full Attention 风格的 batch/direct
receive。后台 receiver 只负责传输和校验；主 GPU stream 在确认 ready 后完成必要的
private materialization/attach。

##### B2：compact transition 异步获取和解析

> **【2026-10-21 修订】** 11E-C 之后默认路径不再有 `deserialize`：transition 走
> 固定布局 Store batch receive（`store-host-staging`，`deserialize_ms=0.000`）。
> 本节描述的“receiver 线程 Get+deserialize / 主线程 H2D+apply”拆分只剩
> **传输与应用解耦**这一半意义，且属于 B3 的范畴；compact payload 仅作为最后兜底。

compact transition 不是普通 attention KV，而是 GDN/Mamba 状态恢复数据。当前路径：

```text
transition store.get()
    -> CPU deserialize
    -> H2D/准备 tensor
    -> state.matmul(matrix) + zero_state
```

需要拆为：

```text
后台 receiver：Mooncake Get + CPU deserialize
        -> transition_payload_ready/error

主线程/GPU stream：H2D + state.matmul + zero_state
        -> transition_applied/error
```

`state.matmul()` 不放入普通 CPU receiver 线程；它应提交到明确的 CUDA stream，
通过 CUDA event 通知 forward。native KV ready 和 transition applied 都完成后，PIC
forward 才能读取依赖数据。

验收必须分别报告 payload Get、CPU deserialize、H2D、transition apply、event wait；
不能把 transition payload 当作 Full Attention KV block 直接复用 batch-buffer 接口。

##### B2-A：固定布局 TransferEngine state fast path

> **【2026-10-21 已被取代】** 固定布局 state fast path 已按 **Store** 落地
> （11E-C）：`store_layout` + 一次 `batch_get_into_multi_buffers` 写入 pinned host
> arena，再一次性 H2D，零 deserialize。本节要求的“固定 ABI + 只传可变 descriptor”
> 已满足，但传输实现不再使用 TransferEngine。

当前 compact payload 的大规模 deserialize 不是 transition/state 的固有要求，而是
当前存储表示的结果。对于模型结构固定的 state，应参考 Full Attention 的 block ABI，
把 matrix、zero_state、conv state 的 dtype、shape、stride 和 layer layout 提升为
固定的 state ABI；请求之间只通过 descriptor 表达可变部分：

```text
state version / token_count / active layers
matrix、zero_state、conv_tail 的 byte length
source/destination buffer address
```

A 侧保持这些 raw state buffers 存活并注册，B 侧预分配并注册目标 buffer，使用
Mooncake TransferEngine 的 batch/async transfer 直接搬运原始 tensor bytes：

```text
A matrix / zero_state / conv_tail
        -> TransferEngine batch transfer
        -> B pre-allocated state buffers
        -> ready event
        -> state.matmul(matrix) + zero_state
```

fast path 不再执行 38 MB 级别的 payload deserialize、`torch.frombuffer`、CPU tensor
重建和应用层 H2D；仍需解析很小的 descriptor，并保留 GPU 上的 state apply。可变的
token range、conv tail 实际长度和 active layer 数量，采用 descriptor/byte length
表达，不要求每个请求重新定义完整 tensor schema。

固定布局路径必须保留兼容 fallback：A 进程不可达、TransferEngine 不支持目标内存、
ABI/version/dtype/shape 校验失败或传输错误时，回退到现有 Mooncake Store compact
payload + deserialize 路径。fast path 和 fallback 必须共享同一个 ready/error、
request-level fallback、lease 和 forward barrier，不能让 PIC 读取未完成或旧的 state。

该项属于 **B2/P5 transition/state 传输实现**，不是 P6 的单纯低拷贝优化；P6 只在
B2-A 稳定后继续消除 native KV 和 state materialize 的残余 CPU clone、H2D 和
index copy。

##### B3：scheduler-level waiting

当前 PIC 主要在 `execute_model()` 内等待；这只能把 native KV 与当前请求的
transition/input preparation 重叠，不能充分实现“其他请求继续 forward”。

需要将 PIC 请求接入与 Full Attention 等价的请求级状态机：

```text
发现远程 native KV/transition
    -> WAITING_FOR_REMOTE_KVS
    -> 其他请求继续 forward
    -> native KV ready && transition applied
    -> 重新进入 runnable queue
    -> PIC attention/forward
```

要求：

- ready 前绝不读取目标 KV block；
- receive、deserialize、GPU apply 任一失败都触发 request-level fallback；
- 请求取消、超时、重试时释放 transfer handle 和 lease；
- 不改变普通请求和当前已验证的同步 PIC 路径。

#### 实现包 C：P6 direct-buffer materialize 和低拷贝路径

在实现包 A/B 稳定后，删除或绕过 PIC 层的：

```text
Python bytes -> CPU clone -> temporary tensor -> H2D -> index_copy_
```

目标是让 Mooncake 的目标地址就是可供 PIC native KV 使用的 block。对需要
RoPE rerotation 或非对齐位置的 request-private KV，保留必要的 materialization；
P6 不能为了减少一次 copy 破坏非前缀正确性。

验收：native KV 的 CPU clone、H2D、index_copy 下降或消失；ready/error、D5 200+
次 reuse、D1 1000 次稳定性、普通请求和 partial/non-prefix correctness 全部通过。

#### P3-P6 总体验收

必须同时验证：

- Full Attention 无回归；
- PIC native KV 和 transition 均有明确 ready/error；
- overlap 开关开/关均可运行；
- 200+ 次跨进程 reuse `wrong=0`；
- 1000 次 reuse lease/pool 不增长；
- 不同 `gpu-memory-utilization` 下无 block wrap 读旧 KV；
- import 失败、manifest 缺失、transition 缺失时 request-level fallback；
- 对比 native Get、CPU receive、deserialize、H2D、materialize、transition apply、
  scheduler wait 和端到端耗时。

#### 2026-10-21：Store-pull 路径收敛（替代 TE 默认主路径）

上一版 B 将 TransferEngine push 作为 PIC 默认数据面，但该语义对应 PD
disaggregation，不是本项目的共享 KV cache pull。当前实现和后续验收统一改为
参考 Full Attention 的 Mooncake Store 路径：

```text
native KV:
    publisher Store.put(compact per-layer payload)
        -> receiver registered native KV storage
        -> batch_get_into_multi_buffers()
        -> native_kv_ready

transition/state:
    publisher Store.put(fixed-layout tensor objects)
        -> receiver registered GPU state tensors
        -> one batch_get_into_multi_buffers()
        -> state.matmul(matrix) + zero_state
```

这次收敛的代码契约：

- native KV 的默认 `HYPIC_PIC_NATIVE_KV_DIRECT` 为 `1`，启动时注册完整 native
  KV storage；接收时先尝试 Store `batch_get_into_multi_buffers()`，不再因为本地
  TransferEngine 可用而自动切换到 P2P push；
- 新 native KV manifest 标记 `transfer_mode=store`，不再发布 page-stride TE
  source descriptor，因此不会把 padded page 误当作共享缓存的传输 payload；
- transition/state manifest 增加 `store_layout`，每个 matrix、zero_state 和
  conv tail 都有独立 Store key、shape、dtype、nbytes；接收端一次 batch pull
  到 GPU tensor，直接构造 operator，正常路径不读取 compact payload，也不做
  `torch.load`/`torch.frombuffer` 大块 deserialize；
- 旧 `transfer_layout`、`HYPIC_PIC_NATIVE_KV_TRANSFER_MODE=page|compact` 和
  `HYPIC_PIC_TRANSITION_TRANSPORT=p2p` 只用于兼容和对照实验，不是默认路径；
- Store direct API、目标布局或 fixed-layout state ABI 不可用时，才回退到已有
  bytes/H2D 或兼容 payload 路径；失败仍是 request-level fallback，不能读取未
  ready 的 native KV 或 state；
- 当前版本仍不改 vLLM 原生 BlockPool，不伪造物理连续 block，也不把 PIC 的
  canonical/shifted RoPE materialization 语义改成 Full Attention 的前缀别名。

因此后续性能对比必须至少记录：Store native `get_ms`、native materialize 的
`cpu_ms/h2d_index_ms`、transition Store batch Get、`deserialize_ms`（固定布局应为
0）、transition apply、forward 及端到端延迟。不能再把 TE padded-page 的字节量和
Store compact payload 直接混为同一条基线。

#### 2026-10-21：Store-only 收敛，TransferEngine 退场

人工确认：PIC 的数据面**只保留 Mooncake Store pull**，TransferEngine（下称 TE）
不再作为 PIC 的一部分维护，并按 11E-D1/D2 两个阶段从代码中移除。

判定依据：

- 语义不符：TE push 是 PD 分离语义（源进程在线、源 buffer 注册、request-local
  descriptor），PIC 是共享 KV cache pull（对象持久、发布方可下线）；
- 硬件不符：当前机器无 RDMA 网卡，TE 的收益前提不成立；
- 实测已死：11E-B 的默认配置（`HYPIC_PIC_NATIVE_KV_TRANSFER_MODE=store`、
  `HYPIC_PIC_TRANSITION_TRANSPORT=store`）下 TE 分支不执行，只留下 6 个
  legacy 用例持续报红；
- 维护成本：TE 约占 `transfer_engine.py` 1212 行中的 1100 行、
  `mooncake_kv.py` 中 700–900 行分支，以及 6 个 `HYPIC_PIC_*` 环境变量。

本决定取代下列章节的 TE 设计（这些章节保留原文仅作历史记录，**不再实施**）：

```text
##### B0-A（bootstrap/ZeroMQ 控制面）       -> 作废，Store 无控制面
##### B1/B2/B2-A（TE 数据面与 state fast path）-> 由 Store batch receive 取代
##### Stage 11E-TP（每 rank TE endpoint）    -> 需按 Store key 前缀重新设计
```

Store-only 之后的 PIC 数据面只有一条：

```text
publisher  : Store.put(native KV layer objects / transition fixed-layout objects)
receiver   : 一次 batch_get_into_multi_buffers
             -> native KV：直接写入请求自有 native block
             -> transition：pinned host arena + 一次 H2D + 零拷贝视图
             -> ready -> attention / state.apply
```

与本节前一版（“一次 batch pull 到 GPU tensor”）的差异：无 RDMA 时 GPU 目标地址的
`register_buffer` 实际会失败，因此 transition 的默认目标改为 **pinned host arena**，
接收后再做一次 H2D；`store-gpu-direct` 只保留为显式对照开关，不是默认路径。native
KV 仍写入请求自有 native block（它本来就是 GPU 上的 KV cache，无需中转）。

##### 11E-D1：停止维护 TE（小改动，先做）

- 删除 6 个 TE 专属用例，使 E0 门禁转绿；
- `mooncake_kv.py` 的 TE 分支保留但标记为历史，命中时只打 warning，不作为
  fallback 依据；
- **必须同时补齐 Store 侧 eviction**：`remove_native_blocks()` 与
  `remove_transition()` 目前没有任何调用方，`on_evict` 只挂了 TE 的
  `release_transfer_sources_for_segment`。若只删 TE 而不接 GC，Mooncake 上的
  payload/tensor 对象/manifest 会永久堆积。验收：1000 次 eviction 后对象数持平。

##### 11E-D2：删除 TE 代码（独立提交）

顺序固定：

```text
1. 把 PICRequestTransfer / PICAsyncOperationHandle 抽到
   vllm/v1/pic/request_transfer.py（Store 路径正在使用，不能随模块一起删）
2. 删除 PICTransferEngine / descriptor / handle / push executor 与 6 个
   HYPIC_PIC_* 环境变量
3. 删除 mooncake_kv.py 中的 TE 分支与 _published_* 状态
4. manifest 保留 transfer_layout 字段的“解析容忍”：旧 A 发布的 manifest 由新 B
   读取时必须忽略而不是报错；只停止生产与消费
5. 更新 gpu_model_runner.py 的 on_evict 与 2 处 has_current_native_transfer_source
6. 回归：E0 全绿、200 次 reuse wrong=0、1000 次 eviction 对象数持平、
   普通请求零回归
```

11E-D1/D2 不改变 Store 的传输语义，不改 `BlockPool`、KV layout、RoPE
materialization 和 attention backend；TP≠1 在 D2 之后单独设计。

##### 11E-D3：block 粒度 + 统一 batch 读取（对齐 Full Attention 原生路径）

人工确认的方向：**PIC 的 native KV 与 transition/state 都改为对齐 vLLM
full-attention 前缀复用的原生形态**，即

```text
每个 block 一个 Store key
  -> 一次 batch_get_into_multi_buffers(keys, addrs, sizes)
  -> 直接写入目标（native：请求自有 native block；state：arena）
```

与参考实现的现状对照：

| | vLLM full-attention 前缀复用 | PIC native（当前基线） | PIC transition（11E-C） |
|---|---|---|---|
| 对象粒度 | 每 block 一个 key | 每 layer 一个 key（6 个） | 约 90 个 tensor 对象 |
| 读接口 | `batch_get_into_multi_buffers` | `store.get` × 6 | `batch_get_into_multi_buffers` |
| 目标 | GPU KV block（启动时注册） | Python bytes → 再拷进 GPU | pinned host arena → 一次 H2D |
| 与参考实现 | — | **偏离**（`NATIVE_KV_DIRECT=0` 造成） | 接口对齐、粒度未对齐 |

参考实现的位置（审阅用）：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`
`store/worker.py:842`（读取调用）、`:756-783`（按 block 造 key/addr/size）、
`:1109-1176`（目标注册）、`store/data.py:99`（每层地址列表）、
`store/connector.py:243/265`（load 在 `get_finished` 中发起）、
`store/scheduler.py:89`（lookup 只查元数据）。

源码级依据（Mooncake checkout `D:\WorkingDocs\CodexDirs\HYPIC\Mooncake`）：
`store.get(key)` 每对象要付 **1 次 `Query` + 1 次 `Get`**，再把整个对象拷成 Python
`bytes`（`store_py.cpp:283` → `real_client.cpp:1895/1966`）；而
`batch_get_into_multi_buffers` 是 **1 次 `BatchQuery` + 一批并发 `BatchGet`**、直接
写调用方地址、不产生 Python 对象（`store_py.cpp:2243` → `real_client.cpp:3738` →
`client_service.cpp:994`）。因此 6 层用 `get` 是 12 次 RPC、90 对象用 `get` 是 180
次 RPC，这是结构性开销，不是调参能消除的。详见
`hypic_pic_vllm_0221_acceleration_record.md` 21.8。

实施边界：

- native 与 transition 的 key 都改成 **block 粒度**，因此 **publish 侧格式要改**
  （一个 block 一个对象），比"只换读接口"大；manifest 需要记录 block 映射；
- 接收侧沿用同一套 `(keys, addrs, sizes)` 语义：一个 key 可以对应多个目标地址
  （层/平面），这正是 `multi_buffers` 的用法；
- 不修改原生 `BlockPool`、KV layout、RoPE materialization、attention backend；
- 任一 batch status < 0 仍走 request-level fallback，不读取半完成的 block。

**前置门（必须先做）**：用 `tests/v1/pic/run_store_read_path_probe.py` 实测
`store.get` 与 `batch_get_into_multi_buffers` 谁快，并回答"`store.get` 是否只是
单 key 的 batch 封装"：

```text
--cases 6:53.5     native 形状（6 个大对象）
--cases 90:38.4    transition 形状（90 个小对象）
--cases 1:38.4     单对象：singles 列 = 1 次 store.get，batch 列 = 1 个 key 的 batch
```

判据：

- `1:38.4` 里 `singles ≈ batch` → `store.get` 与单 key batch 成本相同（很可能共用
  同一底层取数逻辑），"换接口"的收益只能来自**减少调用次数**；
- `1:38.4` 里 `singles ≫ batch` → `get` 是另一条更慢的路径（多一次 host 拷贝），
  换 batch 收益更大；
- `6:53.5` 里 `batch ≫ singles` → native 值得改成 block 粒度 batch；
  否则只保留 transition 的 block 粒度改造，native 维持逐层 `get`。

在实测结果出来之前，**不改读接口，也不改 publish 格式**。

**前置门结果（2026-10-22，已通过，详见 acceleration_record 21.10）**：跨进程
TCP 实测要求 `MC_STORE_MEMCPY=0`（否则走 LOCAL_MEMCPY，读数不可用），并要求
`MC_TCP_ENABLE_CONNECTION_POOL=1`（Mooncake 的 TCP 连接池默认关闭，不设会因每次传输
新建连接而耗尽 ephemeral port，见 acceleration_record 21.13）。

| case | batch / singles | batch / big |
|---|---|---|
| 6:53.5（native） | **1.49×（min）/ 1.65×（median）** | 0.97× |
| 54:38.4（transition） | 1.29× | 1.11×（big 快 6 ms） |

据此确定 D3 的范围：

1. **native KV 改 batch，目标是 pinned host staging**（不做 GPU 直写）。理由：batch
   在 6 层上省 28–35 ms，且 `DIRECT=0` 现在本来就要付 H2D，改批量**不新增 H2D 成本**。
   落地要求：`received.direct` 必须保持 `False`，新增 host-staging 分支走一次 H2D +
   按 layer `index_copy`；arena 按 external key 缓存、计引用，并受
   `HYPIC_PIC_NATIVE_ARENA_CACHE_BYTES`（默认 512 MiB）预算约束，不复用 transition 的
   arena。开关为 `HYPIC_PIC_NATIVE_KV_BATCH`（默认 1）。
2. **transition 保持不变**（继续 `batch_get_into_multi_buffers`，不改 big）：big 只快
   6 ms，但多一次 38.4 MB `bytes` 物化 + 54 个 tensor 切分，成本会吃掉收益。
3. **publish 侧的 key 粒度暂不动**：本次先用"逐层 key + 一次 batch 调用"拿到收益；
   改成 vLLM 的"每 block 一个 key"需要 publish 格式改造，单独作为 D3-b 评估。

**D3 验收结果（2026-10-22，patch `B90EFAD5`，三卡 TCP）**：通过。

> **历史记录（2026-10-23 注）**：本节提到的 `NATIVE_KV_DIRECT`、`NATIVE_KV_BATCH`
> 两个开关已在 11E-E 中删除，当前只有 Store 一条通路 + `PROFILE_IMPORT` 两态打点。
> 数字仍然有效（是 Store 批量读的实测），但**不要再按这两个变量做配置**。

> 追加（同日）：在 `DIRECT=0` + 连接池下 10 轮稳定，PIC p50 约 215 ms / 0.90×，
> TIME_WAIT 峰值 1750。2×2 对照（21.15）确认 `PROFILE_IMPORT` 气泡 <1 ms、
> `NATIVE_KV_BATCH=1` 相对逐层省 65 ms。**性能工作在此冻结**，转入 p95/p99 稳定性
> 与 11E-E 清理；能翻转符号的只有 RDMA（11E-F）。`NATIVE_KV_DIRECT` 默认值已改为 0，
> 连接池由 `mooncake_kv` 在 import 时 `setdefault` 兜底（见 21.14）。

| 指标 | 11E-B 基线 | 11E-C（未含 native batch） | 11E-C + native batch |
|---|---:|---:|---:|
| PIC p50 | 274.3 ms | 263.4 ms | **210.7 ms**（RUN2/3 均值） |
| 加速比 | 0.708× | 0.733× | **0.922×** |
| native receive | 86.5 ms | 86.5 ms | **49.2 ms** |
| native materialize | 24.6 ms | 24.6 ms | **7.9 ms** |
| fallback | — | 0 | **0**（130/130 走 batch） |
| pytest | — | 4 failed | **118 passed** |

稳态拆解：native receive 49.2 + materialize 7.9 + transition 41.0（store 30.2）+
apply 3.3 + forward 71.2 + 未归因 38.1 = 210.7 ms。

**遗留问题（下一阶段必须查，不要当成噪声）**：PIC forward 从 64.2 升到 **71.2**
（+7 ms），而同期 Full forward 稳定在 135.4–135.5，**不是会话漂移**。候选原因只有两个，
需要 `HYPIC_PIC_NATIVE_KV_BATCH=0/1` 同会话对照 + `PROFILE_IMPORT=0/1` 交叉才能定：

1. batch 路径一次 53.5 MB 的 device temp 分配改变了缓存分配器状态，间接影响 forward；
2. profile 的两次 `cuda.synchronize` 在 forward 前制造流水气泡（当前所有稳态数字都是
   `PROFILE_IMPORT=1` 下测的，与 274.3/263.4 基线的口径不同）。

在两点澄清之前，**不要**按"batch H2D 直写 KV cache"去改：那是 GPU 直写路径，TCP 下
transport 退化为 64 KiB 分块同步拷贝（acceleration_record 21.11），会更慢。

##### 11E-D3-b：block 粒度 key（可选；改的是复用能力，不是带宽）

> **2026-10-23 归并**：本节与 Stage 10-C 合并为 **W4（见 §7）**。**没有性能收益**——
> 实测 1/6/54 key 的 batch 带宽都在 0.68–0.77 GB/s（acceleration_record 21.10），
> 粒度不是带宽瓶颈；收益在复用能力、跨段去重与 eviction 粒度。W4 排在 W3 之后，
> 且实施前必须先回答 §6.4 的三个问题。

现状差距：

| | PIC（per-layer key） | vLLM（per-block key） |
|---|---|---|
| 4k 段的 key 数 | 6（每层一个，9.3 MB/个） | 272（每 block 一个，约 197 KB/个） |
| 命中粒度 | 整段命中/整段 miss | 任意 block 子集 |
| 跨段去重 | 不能（重叠段各存一份） | 能（同 hash 即同 block） |
| 与普通 prefix cache 共享 | 不能 | 同一批 block |

实测 1/6/54 key 的 batch 带宽都在 0.68–0.77 GB/s（acceleration_record 21.10），
**粒度不是带宽瓶颈**；改为 block 粒度不会更快，收益在复用能力与 eviction 粒度。

硬前提：

1. **canonical（位置无关）block hash**。vLLM 的 block hash 是前缀链式的（从 position 0
   累积），同一内容在不同前缀下 hash 不同，不能直接用于非前缀复用；PIC 需要只按 block
   内容计算的 hash，与既有 canonical public KV + private rerotation 模型一致。
2. **只适用于 native KV**。GDN/Mamba 的 recurrent state 与 transition operator 是整段
   前缀的函数，不可按 block 分解，transition 保持 segment 粒度。
3. publish 需按 block 切，对象数 6 → N_blocks（4k/16 = 272），需要批大小上限与
   eviction 粒度调整。

附带收益：key 自带语义后可以**去掉 manifest**，改用 `batch_is_exist` 探活（vLLM 的做法，
`store/worker.py:1385`；元数据另用 `batch_get_replica_desc`，`:277`），每请求省一次
`store.get`。

不在本轮做的理由：当前场景是整段命中，用不到部分命中/去重；先拿 D3 的 28–35 ms。
4. 补打点：`native-receive-total` 已存在（`mooncake_kv.py:3286`），但只有 `total_ms`，
   需补 `n_gets / per_get_median / per_get_max`，否则无法复核批量收益。

##### 11E-E（进行中）

**进度（2026-10-22）**：第一批完成，env 24 → 15。

| 批次 | 内容 | commit |
|---|---|---|
| E1 | 删逐层 `store.get` 回退 + `NATIVE_KV_BATCH` + `PICNativeKVReceived.layers` | `e1d303f7c0` |
| E3 | 5 个 profile 开关合并进 `HYPIC_PIC_PROFILE_IMPORT`（0/1/2） | `9b37eb727e` |
| E4 | 删 Store GPU-direct transition tier + `TRANSITION_RECEIVE` | `b459caf00f` |
| E2 | 删 compat payload（publish/load + 序列化器 + manifest 字段）+ `COMPACT_TRANSITION`、`KEEP_PAYLOAD` | `d31556d68d` |

剩余 15 个 env 中，11 个属于 TransferEngine（`TRANSFER_ENGINE`/`ENGINE_ID`/
`BOOTSTRAP_ADDR`/`TP_RANK`/`PP_RANK`/`STATE_WORKERS`/`TRANSFER_WORKERS`/
`NATIVE_KV_TRANSFER_MODE`/`TRANSITION_TRANSPORT`/`NATIVE_KV_BATCH_DESCRIPTORS`/
`NATIVE_KV_DIRECT`），按人工指示**只删我们加的接线**：保留
`vllm/v1/pic/transfer_engine.py` 模块与上游 `kv_connector/mooncake`，留待 RDMA 复用。

> **2026-10-23 决定反转（重要）**：上面的"保留 TE 留待 RDMA"已作废。由于 TE 的 p2p
> 路径在 TCP 下比 Store 批量读更慢（64 KiB 分块同步拷贝，21.11），且要求 KV cache
> 按严格布局注册，而 **RDMA 本身属于 Store 客户端的 `protocol`，不依赖我们这套
> 描述符**——因此 11E-C 已删除 `vllm/v1/pic/transfer_engine.py` 与上述 9 个配置变量。
> 当前开关总数为 **5**，见 §7.3。上游 `kv_connector/mooncake` 一行未动。

##### 11E-E：环境变量与冗余路径清理（原说明）

原则：**一条数据通路 + 一个诊断开关**。任何取数失败都走 request-level fallback
（整段本地重算），不再为每种失败保留一条"更慢但能跑"的并行路径。

时机：11E-D1/D2 完成、且 D3（native 批量 host staging）在真机验收通过之后。清理前
不得删除当前仍在验收中被引用的开关。

当前 24 个 `HYPIC_PIC_*` 变量的处置：

| 分类 | 变量 | 处置 |
|---|---|---|
| 保留 | `HYPIC_PIC_PROFILE_IMPORT` | 保留为唯一诊断开关（`[PIC-PROFILE]` 记录） |
| 保留 | `HYPIC_MOONCAKE_KEY_PREFIX` | 保留（多租户/共享 Store 命名空间） |
| 保留 | `HYPIC_PIC_NATIVE_ARENA_CACHE_BYTES` | 保留（pinned 内存上限，生产可调） |
| 保留 | `HYPIC_PIC_VERIFY_CHECKSUM` | 保留（损坏注入测试用，默认关） |
| 转正后删除 | `HYPIC_PIC_NATIVE_KV_BATCH` | 成为唯一路径后删除 |
| 转正后删除 | `HYPIC_PIC_TRANSITION_KEEP_PAYLOAD` | payload 彻底移除后删除（同时 `COMPACT_TRANSITION=0` 变为不支持） |
| 转正后删除 | `HYPIC_PIC_NATIVE_KV_OVERLAP` | B3 落地后异步成为唯一路径，删除 |
| 删除 | `HYPIC_PIC_NATIVE_KV_DIRECT` | GPU 直写路径退场（TCP 下 transport 内部仍是 64 KiB 分块的同步 H2D，见 acceleration_record 21.11，直写不更快） |
| 删除 | `HYPIC_PIC_TRANSITION_RECEIVE` | 只保留 host staging；仅在将来出现 RDMA 时才需要重新引入 |
| 删除 | `HYPIC_PIC_NATIVE_KV_TRANSFER_MODE`、`HYPIC_PIC_TRANSITION_TRANSPORT` | TE 退场 |
| 删除 | `HYPIC_PIC_TRANSFER_ENGINE`、`HYPIC_PIC_ENGINE_ID`、`HYPIC_PIC_BOOTSTRAP_ADDR`、`HYPIC_PIC_TP_RANK`、`HYPIC_PIC_PP_RANK`、`HYPIC_PIC_STATE_WORKERS`、`HYPIC_PIC_TRANSFER_WORKERS`、`HYPIC_PIC_NATIVE_KV_BATCH_DESCRIPTORS` | TE 退场（D2） |
| 合并 | `HYPIC_PIC_PROFILE_DETAIL`、`HYPIC_PIC_PROFILE_FORWARD`、`HYPIC_PIC_PROFILE_MEMORY`、`HYPIC_PIC_PROFILE_MEMORY_SYNC`、`HYPIC_PIC_PROFILE_TIMELINE_SYNC` | 合并进 `HYPIC_PIC_PROFILE_IMPORT` 的分级（如 `=2` 开 detail） |

代码路径收敛：

| 环节 | 现在 | 目标 |
|---|---|---|
| native KV receive | 4 条：TE / GPU direct / host batch / 逐层 `store.get` | 1 条：host batch（失败→整段重算） |
| transition receive | 4 条：TE p2p / store GPU direct / store host staging / compat payload | 1 条：store host staging |
| transition publish | 2 条：compact payload + store objects | 1 条：store objects |
| eviction | TE source unregister 已退场；Store 对象需接 `remove_native_blocks`/`remove_transition` | 1 条：按 external key 删 Store 对象 |
| manifest | payload 字段仅剩兼容 | 只留 `store_layout` + token/range 元数据 |

验收：

- `grep -rn 'os.getenv("HYPIC_PIC' vllm/` 只剩白名单里的键；
- 单测在**不设置任何 HYPIC 环境变量**的情况下全绿（现在有 6 个 legacy 用例依赖 env 才通过）；
- 文档里所有"开关"章节同步删除，README 的基线脚本同步精简。

##### 11E-F（未来扩展，不排期）：接入 RDMA 网卡

触发条件：集群出现 RDMA 网卡（mlx5 等）且 Mooncake 的 TE 能走 `rdma` 协议。
届时 §21.9/§21.11 的前提全部改变——TCP 的 0.63 GB/s 与 64 KiB 分块 staging 都不再适用。

**不需要改的**（数据模型与接口保持不变）：

- key schema、manifest、publish 格式、`batch_get_into_multi_buffers` 调用方式；
- 一条数据通路的原则（11E-E）继续成立，只是"唯一通路"从 host staging 换成 direct；
- 失败语义仍是 request-level fallback。

**需要改的四处**（都是策略层，现有机制已具备；**2026-10-23 按 11E-C 之后的代码重写**）：

| # | 位置（当前代码） | 现状 | RDMA 下的改法 |
|---|---|---|---|
| 1 | `mooncake_kv.py` 新增 `register_native_kv_buffers` | **已随 TE 删除，需重写一个精简版** | 启动时一次性对每个 KV cache 层 `store.register_buffer(base, nbytes)`（K/V-first 布局拆段，参考 `store/worker.py:1151-1166` 的 `outer_dims` 判定），失败即**整进程**禁用 direct，不做逐请求重试 |
| 2 | 路由决策（新增，按 `MooncakeStoreConfig.protocol`） | 现在只有 arena 一条路 | 启动期探测一次：`protocol == "rdma"` 且注册全成功 → direct；否则 host staging。**决策只做一次**，避免 §21.1 的路径漂移（±36 ms 就是逐请求判定造成的） |
| 3 | `receive_native_blocks` / `_receive_native_blocks_host_batch`(:1889 附近) | 只写 pinned arena | direct 时目的地址改为 `base + block_id * page_len`（GPU），仍用同一个 `batch_get_into_multi_buffers`；arena 仅在回退路径使用 |
| 4 | `materialize_native_blocks` | 一次大 H2D + 按层 `index_copy` | direct 时**只做 stream 同步与租约释放**，不再 H2D/index_copy；`h2d_index_ms` 打点降为 0，用于确认真的零拷贝 |

**预期收益**（按 §21.9 表 A 的口径）：

| 传输耗时 | PIC p50 | 加速比 |
|---:|---:|---:|
| 145.5 ms（当前 TCP） | 274.3 | 0.708× |
| 10 ms（单口 RDMA 量级） | ~139 | **~1.39×** |
| 5 ms（双口 / GPUDirect） | ~134 | ~1.44× |

即 RDMA 是把 0.718× 翻正的**根本手段**；TCP 下能做的只有减少调用次数（D3 已完成）
与摊薄固定开销。字节级方案（KV fp8、transition bf16）**已排除，不做**。

**风险与验证**：

1. **路径漂移**：必须在启动期定死 direct/host，不要在请求里重试（§21.1 已观测到 ±36 ms）；
2. **注册失效**：CUDA graph capture、显存池扩容后地址会变，需要在 capture 之后注册；
3. **GPUDirect RDMA**：NIC 直写 HBM 需要 `nvidia-peermem`/dma-buf 支持，否则退化为
   H2D staging——这时的收益只剩"pinned + 一次大拷贝"，与现在同构；
4. 验收看 `h2d_ms≈0`、`native-receive-batch` 的 `get_ms`、以及 §21.9 表 A 的传输列。

##### 11E-E 删除原则（人工约束，2026-10-23）

**只删我们自己加的代码；原生的东西一律不动。**

> **2026-10-23 二次修订**：下表"原生配置/路由选择"两行的**保留**已被人工推翻——
> 人工判断 TE p2p 路径不快、且对 KV cache 布局要求高，指示"从 KV cache 共享场景中
> 删除"，于是 11E-C 删除了 `vllm/v1/pic/transfer_engine.py` 与 9 个配置变量。
> **RDMA 能力不受影响**（属于 Store 客户端的 `protocol`）。下表保留原样作为历史，
> 现行规则以本段为准：**只保留 Store 通路 + 5 个开关**。

| 类别 | 定义 | 处理（历史） | 现行 |
|---|---|---|---|
| 原生能力 | Mooncake 的 TransferEngine 实现、上游 `vllm/distributed/kv_transfer/kv_connector/mooncake` | 保留 | **上游保留；`vllm/v1/pic/transfer_engine.py` 已删** |
| 原生配置 | `TRANSFER_ENGINE`、`ENGINE_ID`、`BOOTSTRAP_ADDR`、`TP_RANK`、`PP_RANK`、`STATE_WORKERS`、`TRANSFER_WORKERS`、`NATIVE_KV_BATCH_DESCRIPTORS` | 保留 | **已删**（`STATE_WORKERS` 保留，属请求级异步） |
| 路由选择 | `NATIVE_KV_TRANSFER_MODE`、`TRANSITION_TRANSPORT` | 保留 | **已删**，只有 Store |
| 我们的胶水 | `KEEP_PAYLOAD`、`COMPACT_TRANSITION`、`TRANSITION_RECEIVE`、`NATIVE_KV_BATCH`、`NATIVE_KV_OVERLAP`、`NATIVE_KV_DIRECT`、5 个 `PROFILE_*` | 删除 | 已删 |

历史记录：本轮删除总数 **20** 个（11 个胶水 + 9 个 TE 配置），开关 **25 → 5**。

##### 交付纪律（人工约束，2026-10-23，长期有效）

**改代码一律用编辑工具逐处改，不用脚本批量写文件。** 原因有三条：

1. 界面要能看到 diff——脚本写文件不会产生变更事件，评审时只能靠事后 `git diff`；
2. 脚本的字符串锚点容易匹配错位置（本轮已发生两次：误删 778 行、误删 `load_transition_operator` 的初始化行）；
3. 逐处改会自然产生小 diff，出错时 `git stash` / `git revert` 的粒度更可靠。

| 场景 | 允许的方式 |
|---|---|
| 修改 `vllm/`、`tests/` 下的代码 | **只能用编辑工具**，逐处替换并确认 |
| 修改 `docs/`、`patches/` 下的 Markdown | 脚本或编辑工具均可 |
| 批量生成 patch / README / 统计 | 脚本（只读仓库，不写源码） |

**每次交付前必须自检并展示**：

```bash
git diff --stat HEAD -- vllm tests          # 改了哪些文件、多少行
git show HEAD --no-color -- vllm tests      # 关键 hunk（贴进回复）
python3 tools/pic_check_unbound.py vllm/v1 tests/v1/pic   # 必须 clean
python3 -m compileall -q vllm/v1 tests/v1/pic             # 必须 0
```

`pic_check_unbound.py` 已升级为**按作用域**检查（函数、方法、类体），能抓
"删了局部初始化但仍被引用"这类 `compileall` 抓不到的错误。

##### git 备份（强制，2026-10-23，长期有效）

**每一次改动都立刻提交，工作区不允许留未提交的代码。**

| 规则 | 具体要求 |
|---|---|
| 一次逻辑改动 = 一个 commit | 不把多个阶段塞进同一个未提交工作区；改完即 `git add` + `git commit` |
| 提交前必过两个静态门 | `pic_check_unbound.py vllm/v1 tests/v1/pic` clean、`compileall` 为 0 |
| 半成品不得留在工作区 | 未完成的实验放 `git stash` 或 `wip/*` 分支，主工作区保持干净 |
| 每个 patch 必须可复现 | 交付时同时给出 **commit hash + patch MD5**；能从该 commit 重新导出同一份 patch（MD5 一致） |
| 归档目录 | `patches/<stage>/` 内含 `<name>.patch` + `README.md` 两个文件，不放第三方文件 |
| 备份分支命名 | `backup/<stage>-<topic>-<yyyymmdd>`，用于回滚与前缀对比 |
| 历史件不动 | 已验收的 patch（如 `3A09990D`、`B90EFAD5`）原样保留，供回退与对账 |

每次提交后自检：

```bash
git status --porcelain -- vllm tests docs   # 必须为空
git log --oneline -3                        # 确认这次改动的 commit
```

配套的 `patches/bisect/` 保留各阶段的累计 patch，用于二分定位；删除任何历史件前，
先把它的 MD5 记入本文件对应小节的表格。

##### 打点说明（强制，2026-10-23，长期有效）

**每一份 patch 的 README 必须包含"打点说明"，缺这一节视为未完成交付。**
原因：A2 的 README 没有这一节，测试侧设了 `PROFILE_IMPORT=2`（当时门控是
`== "1"`）导致所有打点被关，被误判成"KV 没导入"，白跑一轮。

README 固定七节，顺序不变：

```markdown
## 本 patch 改了什么      （一句话，说清删了什么、新增了什么）
## 删了什么              （逐条：变量名 / 文件 / 改前 / 改后 + 自查 grep + 完整 diff）
## 打点说明              （关键字表 + 层级说明 + 正确性判据，缺一不可）
## 环境变量              （本 patch 需要哪些，默认值是什么）
## 验证命令              （命令 + 期望输出）
## 已知问题              （本 patch 的坑，测试前必读）
## Artifact              （MD5 / 大小 / commit / base）
```

**"删了什么"这一节是硬要求**（2026-10-23 追加）：测试侧反馈"README 只写改了哪个
判断、看不出删了什么"。因此每份 patch 必须包含：

| 要素 | 要求 |
|---|---|
| 删除清单表 | 每一行给出 **被删变量名 / 文件 / 改前代码 / 改后代码**，一行一处，不合并 |
| 删除的 import | 单独列一张小表，说明因为删了哪个判断而不再需要 |
| 新增文件 | 完整源码贴进 README（不是摘要） |
| 自查命令 | 每个被删变量给一条 `grep`，期望输出 0 |
| 完整 diff | 把 `git show <commit> -- vllm tests` 的全文贴进 fenced `diff` 块，不剪裁 |

原因：测试 agent 只能从 README 推断改动范围，摘要式描述会让它反复猜。
"完整 diff"一节允许 README 变长，这是可接受的成本。

"打点说明"三张表的具体要求：

| 表 | 必填列 | 用途 |
|---|---|---|
| 关键字表 | 关键字、层级、字段、含义、**正确性判据** | 让测试侧一眼判断"这条记录出现/数量对"是否等于正确 |
| 层级说明 | `PROFILE_IMPORT` = 0 / 1 / 2 各打开什么 | 避免再因层级不匹配而误判"打点全 0" |
| 正确性判据 | 打点路径、metrics snapshot 字段、`lo` 流量、A/B 输出对比 | 打点被关时仍有替代判据 |

`lo` 流量必须给出期望量级（每次复用 native 53.5 MB + transition 38.4 MB ≈
91.9 MB；N 次复用 ≈ N × 91.9 MB），因为它是唯一与日志无关的硬证据。
metrics snapshot 必须列出 `external_kv_imported`、`reused_tokens`、
`recompute_tokens`、`lookup_hits/lookup_total`、`fallback_reasons`、
`forward_tokens` 增量这六个正确性字段。

##### 11E-E5（已作废）：删掉我们加的 TE 接线

> 作废原因见上一节：原生 TransferEngine 及其配置项必须保留。E5 不再单独执行，
> 转为 A 组（胶水开关）与 B 组（profile 合并）两步走。

> **2026-10-23 再次反转**：人工推翻"保留 TE"的决定，E5 的内容以 **11E-C** 重新执行
> ——`vllm/v1/pic/transfer_engine.py`（1212 行）、TE 配置/路由 9 个变量、以及下表
> 第 1–3 行的 TE 分支全部删除；上游 `kv_connector/mooncake` 保持一行未动。

保留 `vllm/v1/pic/transfer_engine.py` 与上游 `kv_connector/mooncake`，留给 11E-F 的 RDMA 复用。

| # | 原计划 | 历史现状 | 11E-C 实际 |
|---|---|---|---|
| 1 | 删 `receive_native_blocks` 的 TE 分支 | 不删（原生通路） | **已删** |
| 2 | 删 `_receive_native_blocks_transfer_engine` | 不删 | **已删** |
| 3 | 删 publish 的 p2p 分支与 `_build_transition_transfer_layout` | 不删 | **已删** |
| 4 | 删 `NATIVE_KV_DIRECT` 及直写分支 | 照删（属胶水） | 已删 |
| 5 | 删 11 个 env | 改为只删 A/B 组的 11 个胶水开关 | **共删 20 个**（11 胶水 + 9 TE） |


##### 11E-G：通信与 prefill 异步（对齐 full attention）

人工定义：单个请求自己的 prefill 必须等自己的 KV，但**它的传输不能挡住其他请求的
prefill**。并发是本阶段的前提（已确认一定会做）。

#### 上游已有的机制（不必自造）

| 环节 | 位置 | 现状 |
|---|---|---|
| 等待态 | `vllm/v1/request.py:358` `WAITING_FOR_REMOTE_KVS` | 已存在 |
| 调度器置态 | `scheduler.py:884-888`（`load_kv_async` 为真时置态并跳过本步） | 已存在 |
| 等待期跳过 | `scheduler.py:2287`、`:2583` | 已存在 |
| 完成上报 | `scheduler.py:2616-2632` 消费 `KVConnectorOutput.finished_recving` | 已存在 |
| worker 侧 hooks | `kv_connector_model_runner_mixin.py:58/85/109` | 已存在 |
| 连接器元数据通道 | `SchedulerOutput.kv_connector_metadata` | 已存在 |

结论：把 PIC 做成一个 **load-only 的 KV connector** 即可，调度语义完全复用上游。

#### 实现方案

新增 `vllm/v1/pic/remote_kv_connector.py`，实现 `KVConnectorBase_V1` 的 load-only 子集：

```text
调度器侧
  get_num_new_matched_tokens(request, num_local) -> (external_tokens, load_kv_async=True)
  update_state_after_alloc(request, blocks, num_external)   # 记录 plan + 目标 block
  build_connector_meta(scheduler_output) -> PICLoadMetadata # req_id -> keys/plan
  request_finished(request, block_ids)                      # 清状态

worker 侧
  start_load_kv(forward_context, **kwargs)  # 把 metadata 里的请求提交到线程池，立即返回
  get_finished(finished_req_ids) -> (finished_sending, finished_recving)
                                            # 非阻塞轮询；完成的就地 materialize(7.8ms)
```

衔接点：`_set_pic_runtime_plan` 在异步模式下**不再同步 import**，只把 plan 记到
`request_state`；`start_load_kv` 用现有的 `pic_mooncake_object_store` 提交
`receive_native_blocks` + `load_transition_operator`，`get_finished` 里再做
`materialize_native_blocks`（主线程、仅 7.8 ms）。

#### 开关与回退

`HYPIC_PIC_ASYNC_IMPORT=0`（默认）→ 保持今天的同步 import，行为不变；
`=1` → 走连接器路径。两种模式共用同一套 store 调用，便于 A/B。

#### 预期与判据

| 场景 | 现状 | 异步后 |
|---|---|---|
| 并发 1 | 214 ms | **不变**（自己的 KV 仍要等） |
| 并发 2 | 新请求的 90 ms 挡住另一个请求 | 90 ms 被对方 compute 填充 |
| 并发 4 | 每步可能整批停 90 ms | 吞吐上限约 +40% |

判据：**p50 不变、吞吐上升**；若 p50 变差则说明等待态没有正确放行。

#### 7.11 PIC 池分配器优化（推迟，非阻塞）

### 现状

`PICPhysicalPool` 是自写的字节 arena：

| 机制 | 实现 | 问题 |
|---|---|---|
| 分配 | first-fit 扫 `_free_ranges` | 碎片化；有总量但拼不出一块就失败 |
| 对齐 | 256 B | 无 |
| 回收 | 手动排序 + 相邻合并 | 无引用计数的统一视图 |
| 上限 | `--pic-max-cache-bytes`（整池） | **没有单条上限** |

**这次 1.82 GB/条的 per-token 分配，直到把 4 GB 池子挤爆、出现 `evictions=18`
才暴露**——整池上限拦不住"单条过大"，只会表现为"用着用着开始淘汰"。

### 三段优化（按风险从低到高）

| # | 改动 | 风险 | 收益 | 优先级 |
|---|---|---|---|---|
| ① | **单条大小上限 + 告警**（如 > 256 MB 直接报错） | **0**（不动分配/传输路径） | 把 per-token 类回归变成显式错误 | **先做** |
| ② | **best-fit + 引用计数** | 低 | 碎片下降；淘汰/释放语义统一 | 次之 |
| ③ | **改用 KV `BlockPool` 的 2 MB 大页** | 中 | 显存预算统一（计入 `gpu_memory_utilization`）、页级回收、少一次 57 MB 拷贝 | 有时间再做 |

### ③ 的性能核算（已评估，可接受）

| 环节 | 影响 | 量级 |
|---|---|---|
| Mooncake 发布/导入 | 描述符从 1 条变成 N 条 | **+1–3%**（KV 本来就是按 native block 分批传） |
| 物化（逐 token 拷贝） | 不变 | 0 |
| 算子 apply | 跨页需先 gather 成连续 | 38 MB / 1.5 TB/s ≈ **25 µs**（占 p50 0.004%） |
| 净收益 | 同实例复用可**直接引用** public 页，省一次 57 MB 拷贝 | -1 次拷贝 |

页大小建议：KV 用 544 token 的 allocator 页（每层约 557 KB），GDN 状态/矩阵用 **2 MB 大页**
（一张矩阵正好一页，避免跨页 gather）。

### 推迟理由

1. 压缩修复后，单条 entry 从 1.82 GB 降到 57 MB，20 条 1.13 GB < 4 GB，溢出问题已消失；
2. A 组的延迟瓶颈在 **B 侧远端导入**（53 MB KV + 38 MB 算子），换分配器不影响这条路径；
3. 现在的容量是 bench 手动给 4 GB，生产化时再统一预算更合适。

### 重启条件（满足任一即做）

| 条件 | 观察方式 |
|---|---|
| 生产部署需要把 PIC 计入 `gpu_memory_utilization` | 启动参数评审 |
| 出现"free 总量够但分配失败" | `pool-exhausted` 的 `free_bytes` vs `largest_free` |
| 单条 entry 体积再次异常增长（> 256 MB） | 新加的告警 |
| 需要避免同实例复用的那次 57 MB 多余拷贝 | profile 中 materialize 占比偏高 |

## 风险

1. block 必须在 load 前分配并在等待期间持有（`update_state_after_alloc` 的块不能被回收）；
2. 抢占（preemption）与等待态交叉时的 `num_computed_tokens` 修正；
3. fallback 语义：等待期失败必须回到"整段重算"，且不能重复计入 metrics；
4. CUDA graph capture 与 connector hook 的执行顺序。

#### 低优先级：P7-P9

P7-P9 不属于 Full Attention 已验证的主路径，本轮不阻塞 P3-P6：

- **P7：** PIC 特有的逐 layer ready、transition/linear-attention 进一步流水化；
- **P8：** PIC 自行探索的 TCP 多连接、多流、socket buffer 和 Mooncake 参数；
- **P9：** PIC payload 压缩、delta、block packing 和无效状态裁剪。

只有 P3-P6 完成并证明主路径仍受对应问题限制时，才单独评估 P7-P9；不得为了
这些实验改动 Full Attention 的 correctness 路径。

当前执行顺序（**2026-10-23 更新：D1/D2/E/C 已完成，剩余工作见 §7**）：

```text
已完成：读路径实测门 -> D3（native batch）-> D1（停维护 TE）
        -> D2（删 TE 代码）-> E（开关收敛 25 -> 5）-> C（清理与漏删修复）
                ||
剩余（见 §7）：W1 调度器级异步 -> W2 TP>1
                -> W3 RDMA 直收（待硬件）-> W4 block 粒度
```

B3（scheduler-level waiting）现在**是 W1**：它是当前唯一能在**无新硬件**前提下
提升并发吞吐的改动，因此不再"按需插入"，而是排在最前，其后是 W2（TP>1）。
单请求场景下它仍没有收益（见 §7 的收益口径）。

## 4. 后端集成边界

Stage 0–7-B 的通过只证明控制面、状态描述、snapshot 原语、live state capture
和 live state restore 可以安全运行，不能宣称已经产生 HYPIC 的最终 GPU 性能
收益。Stage 8-A 到 Stage 10 必须按上面的顺序逐项验证。

Stage 8-A、Stage 8-B 和 Stage 9-A 优先使用 Python 与 vLLM 现有 kernel 完成
correctness prototype；Stage 8-B、Stage 9-B 或 Stage 10 如果发现现有 attention kernel 无法
表达 PIC slot，才引入 Triton/CUDA，并单独建立编译产物和回归验证。

## 5. Stage 累计 patch 与 README 标准

每个 Stage 的 patch 目录必须只包含一份累计代码 patch 和一份 `README.md`：

```text
hypic-pic-vllm-0221-cu129-stage0-<STAGE>/
├── hypic-pic-vllm-0221-cu129-stage0-<STAGE>.patch
└── README.md
```

累计 patch 必须从官方基线
`0decac0d96c42b49572498019f0a0e3600f50398` 生成到当前 Stage，不可叠加到已经
包含 PIC 修改的环境。面向 Linux 验证统一使用 LF 行尾，并在 README 中给出：

1. 基础镜像、源码基线 commit、Python 命令、vLLM 安装目录和 tests 目录；
2. patch 类型（累计）、应用命令、修改文件数和是否涉及 CUDA/C++/Triton；
3. 从 Stage 0 到当前 Stage 的完整功能清单；
4. 本 Stage 的新增/修改文件、CLI 参数、API、验证边界和明确不包含的功能；
5. 干净基线前提、`vllm` 与 `tests` 目录备份、应用后 hunk/`.rej`/`.orig` 检查；
6. `compileall`、准确的 pytest 通过数、CLI 参数检查和不依赖 `tests/` 的 smoke
   test；
7. 普通请求、PIC 首次/重复请求、普通+PIC 并发，以及本 Stage 新开关开/关两种
   服务验证；固定输出、显存预期和日志观察点；
8. 从备份目录恢复或重新启动官方镜像的回滚命令。
9. 加速比测试方法和结果记录：必须明确上一阶段基线、模型/GPU/并发度、
   prefill/decode 配置、请求构造、warmup、重复次数、计时口径、forward token
   统计和日志提取命令，并同时报告 latency speedup、throughput speedup、实际
   跳过 token 比例以及理论 token-level 上限。每个 Stage README 都必须把本阶段
   结果与上一阶段结果放入对照表；没有性能收益或仅完成 correctness 时，也必须
   明确记录为“未测出收益/不适用”，不能省略该章节。
10. 代码静态检查必须包含未定义名称检查：如果验证环境提供 Ruff，执行
    `ruff check --select F821` 覆盖本 Stage 修改的 Python 包和测试文件；如果
    没有 Ruff，使用等价的 pyflakes/静态检查并在 README 中说明。F821 只能发现
    未定义名称，不能替代 compileall、pytest、函数调用参数审计和端到端服务
    smoke test；这些验证仍然必须执行。

每次服务器验证应先确认 patch 应用于干净基线，再执行静态验证和服务验证；测试
基线若与 patch 不一致，必须在累计 patch 中完整替换到当前 Stage 的测试版本，不能
依赖容器中残留的中间测试文件。

### 5.1 累计 patch 规则（强制，2026-10-26）

**每次交付只给一份从上游 base 打的累计 patch，不再提供任何"增量 patch"。**

原因：2026-10-26 的 C1b 增量在测试容器上 dry-run 失败，两个报错同源——

```text
Hunk #6 succeeded at 2123 (offset -476 lines)
can't find file to patch at input line 43 / No file to patch
```

`-476` 恰好等于"上游 `scheduler.py`（2337 行）vs 已打 C1a 的版本（2813 行）"的行数差，
说明目标树里 C1a 的调度器改动并不存在（C1a-full 很可能打在了**已打过旧 patch 的容器**上，
`patch --forward` 把整文件的 hunk 当成"已应用"跳过）。增量 patch 一旦叠加在错误基线上，
失败是连锁的：某个文件被跳过后，后续 `diff --git` 头会被当作上一文件的 hunk 内容。

#### 规则

| # | 规则 |
|---|---|
| 1 | 交付物是**一份** `hypic-pic-vllm-0221-cu129-stage0-<STAGE>.patch`，从 `0decac0d` 打；不提供 delta |
| 2 | README 必须含**打之前**与**打之后**的指纹表（指定文件的 `行数 + MD5`），让测试侧先验"这棵树是不是干净基线" |
| 3 | 目标必须是**全新镜像**；若指纹不匹配，换 `docker run`，**不要**在已打过的树上叠加 |
| 4 | 打完必须验 `md5sum` 与指纹表一致、`find . -name '*.rej'` 为空，再跑 pytest 与 bench |
| 5 | 每份 patch 的 README 记录 `MD5 / 字节数 / 文件数 / +add -del / commit / base` |
| 6 | 旧的增量 patch 目录保留作历史，但在 README 里注明"被哪份累计 patch 取代" |

#### 生成命令（固定）

```bash
cd <repo>
git add -N docs/<untracked>.md tests/v1/pic/run_*.py      # 未跟踪文件
git diff --no-color --output <patch> 0decac0d96c42b49572498019f0a0e3600f50398 \
    -- vllm tests docs tools
git reset -q -- docs/<untracked>.md tests/v1/pic/run_*.py
```

只包含 `vllm tests docs tools` 四个路径；`patches/` 自身不进 patch。

#### 指纹固定下来的 5 个文件

| 文件 | 作用 |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | PIC 调度器状态机（1b 主战场） |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput.pic_async_imports` |
| `vllm/config/pic.py` | `async_import` 开关与守卫 |
| `vllm/v1/pic/mooncake_kv.py` | Store 数据通路 |
| `vllm/v1/pic/async_import.py` | 异步导入簿记 |

上游 base 里后两个文件**不存在**（`config/pic.py` 是 PIC 新增、`async_import.py` 是 1a 新增），
README 里标注"上游无此文件，属正常"，避免测试侧把"文件不存在"误判为脏树。

### 5.2 PIC 共享块与租约的引用计数（2026-10-28）

C1c 的最后一组问题是"同一 segment 被多个请求同时导入"引发的生命周期问题，分两层修，语义
与 vLLM 的 KV block 管理对齐：

| 层 | 规则 | 实现位置 |
|---|---|---|
| **目的地** | 一个 `external_key` 只对应一套 block（provider `_resource_block_ids`），所有请求导入到同一套 canonical blocks | `_ensure_pic_public_blocks` + `_pic_public_shared_blocks` |
| **租约** | `retire` 在 `ref_count > 0` 时只记 `retire_pending`；最后一个 token 释放才 finalize | `pic/lifecycle.py` |
| **共享块计数** | 缓存持 1 份 + 每个请求持 1 份；holders 归零才 `free_blocks` | `_pic_public_shared_holders` |
| **计数归属** | `Block.ref_cnt` 由 `BlockPool` 独占（`get_new_blocks` +1、`free_blocks` -1），PIC 不得改动 | 同上 |

**为什么共享块要缓存持一份**：PIC 的 `pic_cache` 只是 `seg_hash -> block id` 的索引，块池并不知道
它。若不持引用，最后一个请求释放后块会回到 free queue 被别的请求写入，索引就指向别人的 KV。
这一份引用是"把 seg_hash 注册进块池哈希表"的替代物。

**淘汰与释放是两个事件**：淘汰（worker `_published_lru` → materialization `evicted=True` →
`_evict_pic_segment`）只把缓存那份引用减掉；有请求仍持有则仅减计数，holders 归零才
`free_blocks`。这与 vLLM "淘汰只摘哈希、块在用时不动"一致。

**已知缺口（下一步）**：共享集占用的字节**未计入 `max_cache_bytes`**。Lru 按 snapshot 预算淘汰，
钉住的 canonical blocks 不参与，长跑多 segment 时实际占用可超预算。补齐方案：记
`_pic_public_shared_bytes` 与总量，超预算时按插入顺序淘汰 holders==0 的共享集（有持有者标记
pending，最后一个释放时回收），约 20 行。

**验证口径**（这轮跑完看）：

| 日志 | 期望 |
|---|---|
| `released shared public blocks` | 出现（请求侧归零释放） |
| `evicted shared public blocks` | 出现（淘汰侧归零释放） |
| `pool.free_bytes` 首尾 | 一致或只差当前活跃 segment 的一套 |
| `FAILED request=` | 0 |
| `ADOPT` | = `PARK` |
| `first_import=False` | > 0（共享复用而非重复传输） |


## 6. Stage 11E-D：Store 接收路径的三条候选（2026-10-23 记录，未实施）

11E-C 之后 PIC 只剩 Store 一条通路。本节把"接收端还能怎么优化"的事实、候选与
判据固定下来；**本节内容尚未改任何代码**。

> **2026-10-23 决定：O1 不再单独做，并入 W3（RDMA）。** 原因是核查代码后发现
> `_receive_native_blocks_host_batch` 本来就是"每层一个连续区间、共 6 个 key"，
> 没有可合并的碎片；而且 6 个层对象在发布端各自 `put`、源地址不相邻，
> `can_merge` 的"源与目的同时相邻"永远不成立。**注册 arena 并保留的收益为 0**，
> 只剩 master 多挂 segment、占用网卡 MR 预算、暴露可写段三类成本。因此 O1 缩为
> "**能否把 GPU 显存注册给 Store**"的一次性探测，作为 W3 的第 0 步，与 RDMA 一起做。

### 6.1 事实基线（已核源码，勿重复争论）

| 事实 | 依据 |
|---|---|
| **arena 是我们自己的**，上游 Full Attention 没有等价物 | `vllm/distributed/.../mooncake/store/` 全目录无 `pin_memory`/`host_staging`/`cudaHostAlloc` |
| 上游 Full 直接写 GPU：启动时注册 KV cache，接收时传 GPU block 地址 | `store/worker.py:1133-1149`（`register_buffer(data_ptr, nbytes)`）、`store/data.py:111-116`（`base + block_id * page_len`）、`store/worker.py:842`（`batch_get_into_multi_buffers`） |
| TCP 传输层对 GPU 目标逐 64 KiB 中转 | `tcp_transport.cpp:39`（`kDefaultBufferSize = 65536`）、`:183-186`（`new char[buffer_size]`）、`:217-220`（每块同步 `cudaMemcpy`） |
| RDMA 传输层**没有** staging 代码 | `rdma_transport.cpp` 无 `cudaMemcpy`/`is_cuda_memory` |
| 相邻区间会合并成一次传输 | `transfer_engine_impl.cpp:1128-1133`（`merge_requests_` 默认 true）、`:941-968`（同一对已注册 buffer 内、源与目的相邻才合并）、`:996-1021`（key 取自 `findBuffer`） |
| Store 的 `register_buffer` 会同时挂载 segment | `real_client.cpp:2280`、`client_service.cpp:2110-2135`（`MountSegment`） |
| TE 支持 VRAM 段（`cuda:N`） | `te_backend.cpp:130+`（`seg_type == "VRAM"`） |

实测（TCP，三卡 4090）：GPU 直写 **70–74 ms**，pinned arena + H2D **49 + 7.8 ms**。
结论：**合并减少的是 task 数，不减少 64 KiB 的中转次数**，所以 TCP 下 arena 仍然快。

### 6.2 O1：注册 arena 以启用合并（**已并入 W3，不单独实施**）

**核查结论（2026-10-23）**：`_receive_native_blocks_host_batch` 本来就是"每层一个
连续区间、6 个 key × 1 个区间"，没有可合并的碎片；而且 6 个层对象在发布端各自
`put`，源地址不相邻，`can_merge` 要求的"源与目的同时相邻"不成立。**所以注册 arena
不减少任何 task，收益为 0。**

| 原判据 | 更正 |
|---|---|
| task 234 → 6 | ❌ 现在是 6 个 task，不是 234（234 是已删的 TE 逐块路径） |
| `native-receive` 49 → 45–47 | ❌ 不会变化：49 ms 是 53.5 MB / 1.02 GB/s 的带宽时间 |
| 注册成功与否 | ✅ 仍值得知道，但**要探测的是 GPU 张量**（W3 的前提），不是 host arena |

**保留成本**（因此不做）：master 多挂 segment（每个 arena 一段，512 MiB 预算最多 9 段）、
占用网卡 MR 预算（W3 注册几十 GB KV cache 时可能因此失败）、arena 变成别人可写的
远端段（静默数据损坏风险）、淘汰路径多一次 `UnmountSegment` RPC。

**改为**：W3 的第 0 步做一次性探测——取一块 KV cache 层张量
`store.register_buffer(ptr, nbytes)`，打印返回值后立即 `unregister_buffer`，
不进入任何数据路径。

### 6.3 O2：RDMA 直收（把 arena 降级为回退）

做法与上游一致，不新造 API：

1. 启动时对每个 KV cache 层 `store.register_buffer(base, nbytes)`（K/V-first 布局拆段，
   照 `worker.py:1151-1166` 的 `outer_dims` 判定）；维护 `direct_ready`。
2. `receive_native_blocks` 在 `direct_ready` 时把目的地址换成
   `base + block_id * page_len`（GPU），走同一个 `batch_get_into_multi_buffers`。
3. `materialize_native_blocks` 在该模式只做 stream 同步与租约释放，不再 `index_copy_`。
4. 路由**按协议自动选择**，不新增环境变量：

```text
protocol == "rdma" 且 register_buffer 全成功  -> 直收（无 arena、无 H2D、无 scatter）
否则（tcp / 注册失败 / 非 CUDA）              -> 现有 arena 路径
```

前提与风险（缺一不可）：

| 项 | 要求 |
|---|---|
| GPUDirect RDMA | 主机需 `nv_peer_mem` 或 DMA-BUF；容器需 `/dev/infiniband` 与 `IPC_LOCK` |
| `MountSegment` 副作用 | 注册即把 KV cache 挂为可写 segment，需确认 master 的 segment 数量/大小限制 |
| 对齐 | `ibv_reg_mr` 需页对齐；确认 cudaMalloc 基址与 `nbytes` 的处理 |
| 同步 | RDMA 写完成 ≠ CUDA stream 可见；必须在 forward 前确认 batch_get 返回并同步 stream |
| 收益不确定 | 直写是分散小块；RDMA 带宽不足时可能不如"大块传输 + 一次 H2D" |

验证顺序：先在 RDMA 机器跑 Mooncake 自带
`transfer_engine_bench --seg_type=VRAM`，再用**只读探测**（注册一次、打印返回值与
耗时、不启用直收）确认能力，最后才改接收路径。

### 6.4 D3：PIC 的 KV cache 改为 block 方式管理（研究项，先出结论再动代码）

本节记录的问题是：**PIC 是否应像 Full Attention 那样，以 block 为第一等对象管理
KV cache（block 粒度寻址、租约、发布与接收）**，而不是现在"按层整块 + index_copy"。

支持（倾向做）：

| 理由 | 说明 |
|---|---|
| 与上游同构 | Full 的发布键、接收地址、注册区间都是 block 粒度；PIC 跟随后可以让两条路径共用地址计算、合并规则与注册生命周期 |
| 合并的前提 | 只有落在同一已注册 buffer 内的相邻区间才会合并；block 粒度布局天然满足 |
| 直收的前提 | RDMA 直收要把数据直接落到 block，block 粒度是必要条件 |
| 发布更省 | 现在按层整块 `put`（每层一个对象），block 粒度可以只发未命中的 block，减少无效字节 |

反对（要先量）：

| 理由 | 说明 |
|---|---|
| 对象数量 | block 粒度意味着 key 数 × block 数（一次复用 64 个 block × 6 层 ≈ 384 个对象），RPC 与元数据开销可能反噬 |
| 前缀语义 | PIC 的复用单位是 segment（含 GDN 状态），纯 block 化会破坏"segment 原子性"，需要额外的映射层 |
| 迁移成本 | `publish_native_blocks`/`receive_native_blocks`/`materialize_native_blocks` 与清单格式都要改，需一轮完整回归 |

先要回答的三个问题：

1. Store 侧单次 `BatchQuery` + `BatchGet` 的对象数上限与耗时曲线（384 vs 6 个对象）；
2. block 粒度下 GDN 状态（transition）如何保持与 KV block 的一致性（同一 segment 内原子）；
3. 发布端能否只发差异 block（需要 PIC segment 与 block 的映射稳定）。

判据：只有在"O2 已落地且 receipt 仍受对象/RPC 数限制"时才进入 D3 实施；否则结论
写回本节，不改代码。

### 6.5 执行顺序（2026-10-23 修订，权威顺序见 §7）

```text
W1（调度器级异步；无新硬件，当前唯一能提并发吞吐的改动）
  -> W2（TP>1：按 rank 分片的 Store key 前缀）
  -> W3（RDMA 直收，**后移**：现在没有 RDMA 网卡）
  -> W4（block 化管理，**后移**：无带宽收益，且需先答 §6.4 三个问题）
```

O1 已并入 W3（见 §6.2）；O2 未落地前不得改 `receive_native_blocks` 的默认路径。

## 7. 剩余工作（2026-10-23 定稿，权威）

本节是剩余工作的唯一清单；上文各 Stage 段落只作为背景与判据，冲突时以本节为准。

| 编号 | 名称 | 前置 | 状态 | 估算 |
|---|---|---|---|---|
| **W1** | 调度器级异步（原 11E-G / §B3） | 无（纯软件） | **待做，下一项** | 2–3 天 |
| **W2** | TP>1（原 11E-TP，**key 加 rank 分片**） | 无（当前 per-layer schema + rank 后缀） | 待做，第二项 | **1.5–2 天** |
| **W3** | RDMA 直收（原 11E-F / O2，**含注册能力确认**） | **RDMA 网卡**（当前没有） | 后移 | 2–3 天 |
| **W4** | block 粒度管理（原 11E-D3-b + Stage 10-C） | W3 结论 + §6.4 三个问题 | 后移（无性能收益） | 3–5 天 |
| ~~W5~~ | ~~私有 KV 物化前移~~ | — | **已取消**（§7.7：GPU 段接受串行） | 0 |
| ~~W6~~ | ~~PIC 与普通请求同 batch 融合执行~~ | — | **已取消**（§7.8：prefill 不组 batch） | 0 |
| **W7** | C1d-A：按 reuse 区间发布矩阵（每段一个，不 chunk、不逐 token） | 无 | 待做，下一项 | **0.65 天** |
| **W8** | C1d-B：publish 的 block span 计算与选取一致（A 组软错） | 无 | 待做 | 0.5 天 |

> **原 W6（验收）已删除**：验收不是一次性阶段——每份 patch 的交付流程本来就要求
> correctness 五层 + bench + pytest（见 §5 与各 patch README 的"正确性判据"）。
> 各工作项的完成判据见 §7.2，不再单列验收项；canonical lease 的检查并入 W1 的判据。

> **原 W1（注册 arena）已取消**：核查发现当前接收路径本来就是 6 个请求、无碎片可合并，注册 arena
> 收益为 0。原 O1 缩为"能否把 GPU 显存注册给 Store"的一次性探测，并入 **W3 第 0 步**。
> 详见 §6.2。

### 7.1 顺序与理由

```text
W1 -> W2 -> W3（等硬件）-> W4
```

| 决定 | 理由 |
|---|---|
| **W3 后移** | 当前环境无 RDMA 网卡；它能把 0.71× 翻正（§11E-F 预估 1.39×），但在 TCP 上做的任何"直收"都退化为 64 KiB 分块同步拷贝，反而更慢（21.11 实测 70–74 ms vs 49+7.8 ms） |
| **W4 后移且降级为"能力工作"** | 实测 1/6/54 key 的 batch 带宽都在 0.68–0.77 GB/s（21.10），**粒度不是带宽瓶颈**；收益只在复用能力、跨段去重、eviction 粒度。不要按"能提速"立项 |
| **W1 最先** | 不需要新硬件，且是"通信与 prefill 异步"这一原始目标的直接实现；单请求无收益，**并发**下把接收（约 50 ms/请求）移出关键路径 |
| **W2 紧随** | TP>1 不是优化而是**多卡正确性前提**（当前 key 无 rank，TP>1 会静默错配）；组屏障由框架的 `KVOutputAggregator` 提供，不需要我们实现，详见 §7.5 |
| **原 W1（注册 arena）取消** | 现在已是 6 个请求、无碎片可合并，注册 arena 收益为 0；且**注册能力已被原生 vLLM 证明可用**（`store/worker.py:1142` 对本机 KV cache 调 `register_buffer`，Full Attention 在本环境正常运行且日志无 `register_buffer failed`），不必单独探测 |

### 7.2 每项的完成判据

| 编号 | 判据 |
|---|---|
| W1 | 并发（≥4 请求）下 PIC p50 下降且单请求不退化；`WAITING_FOR_REMOTE_KVS` 只在 import 未完成时出现；任一接收失败仍走 request-level fallback 且 metrics 不重复计数；canonical lease 在 eviction 下不丢 source block |
| W2 | TP=2/4 下 key 前缀不冲突、跨 rank 不重复传输、输出与 TP=1 一致；单卡 p50 不回退 |
| W3 | `h2d_index_ms == 0`、`native-receive` 12 ms 量级、ratio ≥1.2、无 GPUDirect 时自动回退 host staging 且行为与当前 Store 路径一致 |
| W4 | 先答 §6.4 三问；若对象数导致 RPC 反噬（384 对象比 6 对象慢），结论写回 §6.4 且不动代码 |

### 7.3 不在剩余工作内（已排除）

| 项 | 原因 |
|---|---|
| TE p2p 路径、GPU 直写、逐层回退、payload 兼容层、重叠、逐层 batch | 11E-E 已删；`transfer_engine.py` 与 9 个配置变量已移除 |
| 5 个旧 `PROFILE_*` 开关 | 已并入 `PROFILE_IMPORT` 两态 |
| KV fp8 / transition bf16 等字节级压缩 | §21.9 已排除：改变数值语义，收益不足以抵消风险 |
| P7–P9（多连接、socket 调参、payload 压缩） | 未证明主路径仍受其限制，不立项 |

### 7.4 W1 设计：对齐 Full Attention 的异步加载

**参考实现（上游，已核对源码位置）**：

| 步骤 | 上游做法 | 位置 |
|---|---|---|
| 1 | scheduler 问 connector："这请求能复用多少 token、是否异步" → `(count, is_async)` | `base.py:454`；mooncake 实现 `mooncake_connector.py:548-586`（`return count, True`） |
| 2 | 需要异步时：先分配 block，再 `update_state_after_alloc`（connector 在此**启动**接收），然后 `status = WAITING_FOR_REMOTE_KVS`、`step_skipped_waiting.prepend_request()`、`continue`——**本步不调度它的 forward**，其它请求照常执行 | `scheduler.py:868-903` |
| 3 | worker 每步用 `connector.get_finished(finished_req_ids)` 报告完成了哪些请求，以及失败集合 | `kv_connector_model_runner_mixin.py:110`、`scheduler.py:2632` / `:2811` |
| 4 | scheduler 把完成的请求从 `WAITING_FOR_REMOTE_KVS` 提升回 `WAITING`（成功则 `cache_blocks`；失败则按已算 token 回退或释放 block） | `_update_waiting_for_remote_kv` `scheduler.py:2545-2577`、`_try_promote_blocked_waiting_request` `:2579-2594` |

关键点：**"等待"发生在 scheduler，不在 worker 的 execute 里**。所以一个请求等 KV 时，同一个 batch 里其它请求的 forward 照常推进。

#### PIC 现状与此改造

今天 PIC 在 `execute_model` 内部完成“提交接收 → **立即等待** → 物化 → apply transition → forward”：

```text
gpu_model_runner.py:2078  request_transfer.submit_state(...)
                     :2083  request_transfer.wait_state(...)      <- 这里阻塞整个 batch
                     :2116  request_transfer.wait_native(...)    <- 同样阻塞
```

即“提交后马上等”，接收时间完全串在关键路径上。改造分五步：

| # | 改动 | 说明 |
|---|---|---|
| 1 | PIC 在 scheduler 侧回答“可复用多少 token + 异步” | 复用现有 `pic_cache` / `pic_mooncake_reference_index` 的命中结果，返回 `(reused_tokens, True)` |
| 2 | `update_state_after_alloc` 里只**提交**接收 | 把 `submit_state`/native batch 提交移到这一步；`PICRequestTransfer` 存进 worker 侧 `pending[req_id]`，不在 execute 里等 |
| 3 | `get_finished` 报告完成/失败 | 轮询 `pending` 的 `wait(timeout=0)`；成功 → `finished_recving`；异常 → `failed_recving` 并记录原因 |
| 4 | execute 路径只做“已就绪”部分 | 请求被提升后再执行 H2D、`index_copy`、transition apply；这些仍必须在 forward 之前完成 |
| 5 | 失败语义不变 | 任一接收失败 → request-level fallback（整段重算），`num_computed_tokens` 回退，metrics 不重复计数 |

#### 与 Full Attention 的差异（PIC 特有的三件事）

| 差异 | 影响 |
|---|---|
| PIC 的复用单位是 **segment**（canonical KV + GDN transition），不是纯 token 前缀 | `get_num_new_matched_tokens` 只能按“整 segment 可复用”回答；跨 segment 的部分复用仍走 private materialization |
| forward 之前还要 **apply transition / 恢复 state** | “KV ready” 的定义要包含 state 就绪，否则 forward 会用错状态；这段放在被提升后的第一次 execute 里 |
| hybrid 模型有 Mamba state block | 与 KV block 一起分配、一起等；`update_state_after_alloc` 要同时拿到两类 block |

#### 判据

| 项 | 期望 |
|---|---|
| 状态机 | `WAITING_FOR_REMOTE_KVS` 只在“已提交、未完成”期间出现；提升后能正常 forward |
| 并发收益 | ≥4 并发（或 39 请求连续场景）下 PIC p50 明显下降，逼近 `max(接收, forward) + apply`，而不是两者相加 |
| 单请求不退化 | 单请求 p50 仍在 209–213 ms |
| 失败回退 | 注入接收失败（如删 key）后：请求整段重算、`fallback_reasons` 记录、metrics 只计一次 |
| 抢占交叉 | 等待态被抢占时 block 不被回收，`num_computed_tokens` 修正正确 |

#### 风险

1. block 必须在 load 前分配并在等待期间持有（`update_state_after_alloc` 的块不能被回收）；
2. 抢占与等待态交叉时的 `num_computed_tokens` 修正；
3. fallback 必须回“整段重算”且不能重复计入 metrics；
4. CUDA graph capture 与 connector hook 的执行顺序；
5. **不要**在等待期持有 GPU 显存（arena 在物化阶段才租），否则并发请求会互相挤占 pinned 内存。

> **与 TP 的关系**：W1 不需要为 TP>1 设计“组屏障”。框架的 `KVOutputAggregator`
> 按 `expected_finished_count`（默认 world_size）做倒计时，所有 worker 都报
> `finished_recving` 才让 scheduler 提升请求（`kv_connector/utils.py:73-85`）。
> 所以 W1 与 W2 解耦，按 W1 → W2 顺序做即可。

### 7.5 W2 设计：TP>1 的 key 分片与组屏障（已核上游源码）

**先纠正一个前提**：PIC 现在根本不能跑 TP>1。`vllm/v1/pic/*.py` 里
`tp_rank`/`pp_rank` 出现 **0 次**，key 只有

```python
# mooncake_kv.py:2303
external_key = f"segment/{segment.seg_hash.hex()}/group-{int(group_id)}"
```

而 `seg_hash` 只由 token 内容算出、与 rank 无关。所以 TP=2 时两个 rank 写同一个
key、互相覆盖，读的时候各 rank 都拿到同一份 manifest，但期望的是各自的 head 分片
→ **静默的 KV 错配**。W2 是**多卡正确性的前置条件**，不是性能项。

#### 上游怎么分片（照抄这个格式）

```python
# store/data.py:30-35  KeyMetadata 的字段
model_name, tp_rank, pcp_rank, dcp_rank, pp_rank, group_id

# store/data.py:58-67  PoolKey.to_string()
f"{model_name}@tp_rank:{tp_rank}@pcp{pcp_rank}@dcp{dcp_rank}"
f"@pp_rank:{pp_rank}@group:{group_id}@{chunk_hash}"
```

`tp/pcp/dcp/pp` 四个并行维度都进 key（PCP/DCP 是上下文并行）。PIC 目前只需要
`tp_rank`，但格式上留出位置，避免以后再加一次 schema 迁移。

#### 组屏障不用我们造：`KVOutputAggregator` 已经提供

```python
# kv_connector/utils.py:54-85
self._expected_finished_count = expected_finished_count   # 默认 world_size
remaining = remaining_count_dict.get(req_id, self._expected_finished_count)
remaining_count_dict[req_id] = remaining - 1
if remaining_count_dict[req_id] == 0:      # 所有 worker 都报过才算完成
    finished_set.add(req_id)
```

- 计数可被 connector 覆盖：`base.py:613 get_finished_count()`；
- 还能按请求动态调整：`utils.py:100-109`（`kv_output.expected_finished_count`）。

所以 **W1（异步）不需要为 TP 专门设计“组屏障”**：每个 rank 通过
`kv_connector_output.finished_recving` 报自己完成，聚合器自然会等全组。两者不再耦合，
可以按 W1 → W2 的顺序做。

#### 两个可以直接借鉴的小技巧

| 上游做法 | 位置 | 用途 |
|---|---|---|
| `rotation = tp_rank % len(key_list)`，按 rank 旋转 key 列表 | `store/worker.py:777` | 让不同 rank 从不同位置开始读，摊平热点 |
| `load_mask_per_group`：按 rank/group 决定哪些 chunk 要载入 | `store/worker.py:754` | 对应“每个 rank 只导入自己的分片” |

#### 改造清单与估算（修正后 ≈1.5–2 天）

| # | 改动 | 估算 |
|---|---|---|
| 1 | key 加 rank：`segment/{hash}/rank-{tp}/group-{g}`（格式参照 `PoolKey`，由 **worker 侧**用自己的 rank 拼，调度器保持 rank 无关） | 0.5 天 |
| 2 | manifest 增 `tp_rank`/`tp_size`，加载时校验与本地 rank 一致，不符 → request-level fallback | 0.25 天 |
| 3 | ~~组屏障~~ | **0**（框架已提供） |
| 4 | 复用前确认**全组** key 存在（缺任一 → 全组重算），并沿用 W1 的 canonical lease 防 evict 竞态 | 0.5 天 |
| 5 | TP=2 验证（3 卡环境可做；TP=4 不行） | 1 天 |

#### 风险（修正后）

| 风险 | 级别 | 对策 |
|---|---|---|
| rank 读到别人的分片（静默错配） | 中 | key 带 rank + manifest 校验 rank/形状/dtype |
| 部分 rank 命中、部分 miss | 低 | 复用前“全组 key 存在”检查 + 组屏障由聚合器保证 |
| check 与 read 之间被 evict | 中 | canonical lease 在使用期钉住（W1 判据已含） |
| TP 下每 rank 字节数变 1/TP | — | TP=2 时接收约 25 ms，比 TP=1 更快；Full 同样受益，ratio 不一定变 |

### 7.6 W1 工作计划（调度器级异步）

分三个阶段，每阶段单独提交、单独验证；**任何一阶段失败都能整体回退到当前 Store 路径**
（因为开关 `async_import` 默认关闭，见阶段 1）。

#### 阶段 1：开关 + 状态机骨架（接收仍同步，1.5 天）

原"阶段 0（开关骨架）"与"阶段 1（状态机）"合并：单独提交一个默认关闭、无人使用的
开关没有价值，还会留下一条"半接线"的路径。

| 项 | 内容 |
|---|---|
| 目标 | ① 把"是否异步"变成显式开关，默认关闭；② 开关打开时让请求走 `WAITING_FOR_REMOTE_KVS`；③ **接收仍是同步的**，所以本阶段不产生性能变化，只验证状态机 |
| 改动 | ① `PicConfig` 增 `async_import: bool = False`（放在 `mooncake` 之后）<br>② 新增 `vllm/v1/pic/async_import.py`：`PICAsyncImportManager`（`start / poll_finished / materialize / abort`，本阶段内部仍是同步调用）<br>③ `SchedulerOutput` 增 `pic_async_imports`（请求 id + 计划摘要）<br>④ scheduler：命中外部 segment 且开关开时，分配 block 后置 `RequestStatus.WAITING_FOR_REMOTE_KVS` 并跳过本步（照 `scheduler.py:868-903`）<br>⑤ 新增 `self._pic_finished_recving: set[str]`，在 `update_from_output` 末尾从 worker 输出收集完成集合<br>⑥ 复用 `_try_promote_blocked_waiting_request`（`scheduler.py:2579`），把 PIC 完成集合并入提升条件 |
| 验证（关） | `async_import=False` 时**行为与今天逐字节一致**：pytest 114 通过、bench 209–213 ms / 0.91–0.93×、`[PIC-DEBUG]` 打印 `async_import=off` |
| 验证（开） | ① 日志可见 `WAITING_FOR_REMOTE_KVS → WAITING → RUNNING`；② 输出与今天一致（10-token A/B）；③ p50 与基线差 **<2 ms**（因为仍同步）；④ `max_num_seqs=1` 注入抢占时 block 不被回收 |
| 回退 | 关开关；或整体 revert 本阶段的提交 |

#### 阶段 2：真正的异步（把 wait 从 execute 挪走，1 天）

| 项 | 内容 |
|---|---|
| 目标 | 提交与等待分离：请求等 KV 时其它请求照常 forward |
| 改动 | ① worker：`_prepare_pic_mooncake_destinations` / `begin_import_references` / `submit_state` 从 `execute_model` 的 import 块里抽出，改由 `PICAsyncImportManager.start()` 在"本请求被调度之前的一步"调用；② `gpu_model_runner.py:2083` 的 `wait_state` 与 `:2116` 的 `wait_native` 改为 `poll()`（`timeout=0`），未完成则**不放行 forward**、把请求留在等待态；③ 完成时一次性做 H2D + `index_copy` + transition apply（保持"KV ready 含 state ready"） |
| 验证 | ① 并发 ≥4 时 PIC p50 下降、单请求不退化；② `native-receive-total` 与 `metrics snapshot` 不变（字节量不变）；③ 高并发下不出现 pinned 内存暴涨（等待期不租 arena） |
| 回退 | 开关回到关；或只回退本阶段的 worker 改动（阶段 1 的状态机保留也不影响正确性） |

#### 阶段 3：失败与边界（0.5 天）

> **2026-10-23 修订**：阶段 3 的失败路径**不需要集群级故障注入**——三条失败路径里有两条
> 用单测即可（fake store 抛错、`max_num_seqs=1` 制造抢占），第三条（abort）用客户端超时。
> 注入只在需要确认真实 store 故障的端到端行为时才值得做，而单测覆盖的是同一段代码。

| 项 | 内容 |
|---|---|
| 目标 | 失败语义与今天严格一致 |
| 改动 | ① 接收失败 → `failed_recving`，请求整段重算、`num_computed_tokens` 回退、`fallback_reasons` 记一条、metrics 只计一次；② 等待态被抢占：解绑 transfer、释放 block；③ 请求 abort：`discard()` 清理 pending |
| 验证 | ① **失败路径由单测覆盖**（fake store 抛错 → request-level fallback），**不依赖集群级故障注入**；② 抢占交叉用 `max_num_seqs=1` + 同时 2 个请求验证（也无需注入）；③ `fallback_reasons` 恰好 1 条；④ `pool.free_bytes` 首尾一致（无泄漏）；⑤ pytest 全通过 |

#### 总验收（每阶段都跑）

| 项 | 判据 |
|---|---|
| correctness 五层 | `external_kv_imported` / `reused_tokens` / `recompute_tokens` / 10-token A/B / `fallback_reasons={}` 全过 |
| 性能 | 单请求 p50 209–213 ms 不退化；并发场景给出提升幅度（作为 W1 的收益记录） |
| 静态门 | `compileall` 0、`pic_check_unbound` clean、`pic_check_removed` clean、pytest 114 |

#### 主要风险与对策

| 风险 | 对策 |
|---|---|
| 等待态与抢占/abort 交叉导致 block 泄漏 | 阶段 1 先用"同步接收 + 状态机"把这条跑通，再进阶段 2 |
| 完成信号误判（未就绪就 forward） | `poll()` 只有返回完成才放行；完成前请求不进入 batch |
| pinned 内存被等待中的请求占满 | 等待期不租 arena；arena 只在物化窗口租借 |
| CUDA graph capture 与 hook 顺序 | 阶段 2 不改 capture 路径，hook 在 `execute_model` 之前完成 |
| 阶段 2 收益不明显 | 先用"39 请求连续 + 并发 4/8"两组数字确认，再决定是否继续优化 |

#### 阶段 1 执行记录与 1b/1c 的精确改动点（2026-10-23）

**1a 已完成**（commit `1da3a1303b`）：`vllm/v1/pic/async_import.py` 提供
`PICAsyncImportSpec` 与 `PICAsyncImportManager`（`start/complete/fail/abort/
take_finished/take_failed/in_flight/is_ready`），配 8 个单测；不接线、无行为变化。
配置开关**故意延后**到 1b，避免提交一个无人读取的开关。

**1b（scheduler 侧）**，按下面的顺序改，每处都是小 diff：

| # | 文件:位置 | 改动 |
|---|---|---|
| 1 | `vllm/config/pic.py:86` 之后 | 加 `async_import: bool = False`，docstring 说明"开启后请求会在 `WAITING_FOR_REMOTE_KVS` 等待，接收不阻塞同 batch 其它请求" |
| 2 | `vllm/v1/core/sched/output.py:59` 之后 | `SchedulerOutput` 加 `pic_async_imports: tuple[PICAsyncImportSpec, ...] = ()` |
| 3 | `scheduler.py:2111`（`_ensure_pic_public_blocks` 之后，**早于请求进入 `update_states`**） | 若 `async_import` 且 `request.pic_cache_plan.external_matches` 非空且该请求尚未导入：置 `RequestStatus.WAITING_FOR_REMOTE_KVS`，把 `PICAsyncImportSpec(request_id, tuple(external segment indices))` 存入 `self._pic_pending_imports`，**不** `_enqueue_waiting_request`。实测 import 发生在 worker 的 `execute_model.update_states`（70.8 ms），所以 park 必须更早，否则拦不住 |
| 4 | `scheduler.py:199` 附近 | 加 `self._pic_finished_recving: set[str]` 与 `self._pic_pending_imports: dict[str, PICAsyncImportSpec]` |
| 5 | `scheduler.py` 构造 `SchedulerOutput` 处 | 把 `_pic_pending_imports` 的值放进 `scheduler_output.pic_async_imports`，并清空（每 request 只发一次） |
| 6 | `scheduler.py:2623`（`update_from_output` 的 connector 段之后） | 从 `model_output.pic_finished_imports` 收集完成 id 写入 `_pic_finished_recving`；失败集合走 `_update_requests_with_invalid_blocks` |
| 7 | `scheduler.py:2583`（`_try_promote_blocked_waiting_request`） | 在既有的 connector 分支之前加 PIC 分支：`request_id in _pic_finished_recving` → 置 `WAITING`、`cache_blocks`、`_pic_finished_recving.discard` |

**1c（worker 侧）**：

| # | 文件:位置 | 改动 |
|---|---|---|
| 1 | `vllm/v1/outputs.py:286` 之后 | `ModelRunnerOutput` 加 `pic_finished_imports: tuple[str, ...] = ()` |
| 2 | `gpu_model_runner.py` 的 `execute_model` 开头（取 `scheduler_output` 之后） | 遍历 `scheduler_output.pic_async_imports`：按 spec 建一个**临时**导入上下文（用 spec 里携带的 `pic_cache_plan` 摘要 + 已分配 block ids，不建 `CachedRequestState`），调用现有的 `_prepare_pic_mooncake_destinations` / `begin_import_references` / `submit_state` / 等待，**同步完成**后记入 manager 并汇总到 `pic_finished_imports` |
| 3 | `gpu_model_runner.py:2150` 之后的异常处理 | 失败时调用 `manager.fail(request_id, reason)`，由 1b 的第 6 条转成 request-level fallback |
| 4 | 全文件 | `async_import=False` 时**不进入任何新代码**：`scheduler_output.pic_async_imports` 为空，行为与今天逐字节一致 |

**1b/1c 共同的前置决定（已定 A 方案）**：spec 携带与该请求 `NewRequestData`
相同的 PIC 载荷（`pic_segments` / `pic_cache_plan` / `pic_public_block_ids`），
worker 侧用一个临时结构完成导入，不依赖 `CachedRequestState`。这样"被 park 的请求"
不需要先被调度，也就不需要 vLLM 支持"0 token 调度"。

**安全约束**：1b 单独上线时开关必须默认关闭，且 `async_import=True` 但 worker
侧不支持时必须**启动即报错**（而不是把请求永久 park）。1c 落地后该限制解除。

#### 阶段 1 参考实现对照：Full Attention 前缀复用怎么走完一整轮（2026-10-23）

上游 store 连接器的完整流程（`.../mooncake/store/scheduler.py`）：

| 步 | 函数 | 做什么 |
|---|---|---|
| 1 | `get_num_new_matched_tokens`（`:78-121`） | `num_external_hit_tokens = client.lookup(token_len, request.block_hashes)`；命中数是**块对齐的前缀长度**；若整段都命中，则**故意留下一个 sub-block 尾巴**给采样（`:91-97`）；记 `LoadSpec(vllm_cached_tokens, kvpool_cached_tokens, can_load=False)`，返回 `(need_to_allocate, self.load_async)` |
| 2 | `update_state_after_alloc`（`:123-156`） | 记录 `(request, local_block_ids)`；**断言** `num_external_tokens == kvpool_hit - vllm_cached`（`:144-154`）；置 `can_load=True` |
| 3 | `build_connector_meta`（`:158-…`） | 对 `scheduled_new_reqs` 里带 `LoadSpec` 的请求，用已分配的 block ids 生成 key/addr/size，塞进 meta；同时清理 finished/preempted 的 `load_specs` |
| 4 | worker `start_load_kv` / `get_finished` | `batch_get_into_multi_buffers(keys, addrs, sizes)`；完成后由 `get_finished` 报告 |
| 5 | scheduler `_update_waiting_for_remote_kv`（`:2545-2577`） | 提升请求、`cache_blocks(request, request.num_computed_tokens)`；整段命中时把 `num_computed_tokens = num_tokens - 1` 以保留采样 |

**能照抄的（3 条）**：

1. **park/promote 状态机**：`WAITING_FOR_REMOTE_KVS` + `step_skipped_waiting` + 全组完成才提升；
2. **每个请求只发一次 meta**：`load_specs.pop(request_id)` 保证不重复下发；
3. **留尾巴给采样**：整段命中时要保留至少一个 token 不给复用（块边界对齐）。

**不能照抄的（3 条，PIC 特有）**：

| 上游语义 | PIC 的现实 | 后果 |
|---|---|---|
| 外部命中数 = **从 `num_computed_tokens` 开始的连续前缀** | PIC 的复用单位是 segment，**可以是非前缀**（中间有 recompute） | **不能用 `get_num_new_matched_tokens` 表达 PIC 的复用**——PIC 继续用自己的 `pic_execution_plan`（skip/recompute 区间） |
| 提升后把 `num_computed_tokens` 推进到命中数 | PIC 的 token 会计由自己的计划决定，vLLM 只看到 recompute 段落 | park/promote **不改变 token 会计**，只延后该请求开始 forward 的时机；这正是阶段 1 想要的 |
| `lookup(token_len, block_hashes)` 用**前缀链式** block hash | PIC 需要**位置无关（canonical）**的段 hash | 复用匹配继续走 PIC 自己的 `PICMooncakeReferenceIndex`，不用 vLLM 的 `block_hashes` |

**结论（阶段 1 的实现口径）**：

```text
只借用状态机与完成聚合，不借用 token 会计与前缀 hash。
scheduler：PIC 计划里存在 external_batch 的请求 -> park 到 WAITING_FOR_REMOTE_KVS
worker ：同步完成导入后报告 finished
scheduler：提升回 WAITING，下一次 schedule() 仍按 PIC 自己的 skip/recompute 计划调度
```

这样既不会篡改 vLLM 的 `num_computed_tokens` 语义，也不需要为 PIC 伪造"前缀命中"。

#### 阶段 1 实测事实（2026-10-23，C1a 验收 + 并发基线）

**C1a 通过**：pytest 123、env=5、`async_import` 未接线、生产档 PIC 211.0–212.7 / ratio
0.917–0.922、五层正确性全过。

**import 的确切位置**（与设计假设不同，必须记录）：

```text
EngineCore.execute_model.total 165.9 ms
  ├─ update_states              70.8 ms   ← import 在这里
  │    └─ runtime-plan build_ms=68.056
  │         ├─ native-manifest        0.002  (2368 B)
  │         ├─ native-receive-batch  47.648  (53.5 MB)
  │         ├─ transition-manifest    2.718  (11897 B)
  │         └─ transition-store-total 26.311  (38.4 MB)
  │    ├─ native-slot-attach           2.012
  │    ├─ native-materialization       0.856
  │    └─ transition-apply             3.939
  └─ pic_runtime_early          86.0 ms
       └─ forward_dispatch      71.1 ms
```

即：**import 在 `execute_model.update_states` 的 runtime-plan 阶段同步完成**，
不是"每请求单独提交后等待"。这决定了 1b 的 park 必须发生在**请求进入
`update_states` 之前**（否则它已经在阻塞这一整步了）。

**park 粒度**：请求级。实测 `segments=2`、`reused=(0,)`、`recompute=(1,)`、
`external_matches=[0]` —— 每请求只有 **1 个** external 段。

**恢复时机**：`capture candidate ... scheduled=4992 computed=0 completed_end=4992`
说明 PIC 一步算完 640 token（`[4352,4992)`），park 后恢复时整段调度，不需要多步。

**并发基线**（`run_pic_concurrency_probe`，`before.json`）：

| concurrency | p50 | 相对 c=1 |
|---:|---:|---:|
| 1 | 211.9 ms | 1.00× |
| 2 | 482.4 ms | **2.28×** |
| 4 | 687.5 ms | **3.24×** |

吞吐 4.70 → 4.15 → 5.78 rps 基本不涨 → **瓶颈就是 update_states 里的串行 import**。
这是 W1 的验收基线：目标 c=2 从 482 降到 ~220、c=4 从 687 降到 ~250–300。

**212 ms 的构成**（`PROFILE_IMPORT=1` 中位数）：

| 段 | ms |
|---|---:|
| native-receive-total | 47.6 |
| transition-store-total | 26.3 |
| transition-apply | 3.3 |
| native-materialize | 7.8 |
| forward（PIC 640） | 70.9 |
| **小计** | **155.9** |
| execute_model 其他（prepare/postprocess/sample 等） | 24 |
| 引擎层 | 24 |
| HTTP 层 | 33 |

→ 即使 import 完全被盖住，**单请求下限约 155.9 + 24 = 180 ms**；串行 bench 下 W1 无收益。

**故障注入的两个事实**（回答测试侧的问题）：

| 问题 | 事实 |
|---|---|
| scheduler lookup 走本地表还是 Store？ | **走 Store**：`PICMooncakeReferenceIndex.lookup` 直接调 `object_store.load(external_key)`（`mooncake_kv.py:2303-2307`）。所以 `remove` 返回 **-704 OBJECT_NOT_FOUND** 说明**删的是不存在的 key**，HIT 当然不变 |
| A 会重新 publish 吗？ | **会，且是无条件的**：11E-C 删掉了 `has_current_native_transfer_source` 这道"已是最新就不发"的判断，`publish_native_blocks` 在 `gpu_model_runner.py:2907/3045` 每次 capture 都执行 → 破坏的 manifest 会被覆盖 |
| 重启 B + 换 `KEY_PREFIX` 算不算合适的注入？ | **不算**：那是 lookup **miss**（正常重算），不是"命中后 import 失败"。1b 要验的是后者 |

### 7.10 A 组（86782EE9）：不崩了，但没收益（2026-10-28）

结果：20/20 成功、两轮稳定、B alive；但 p50 **743.8 ms**（关档 393.6），tput 2.59 rps
（B 组 8.22）。判据上"不崩"达成，"收益"未达成。

#### 根因一：双 SEP 造出 34 token 小段，被块对齐吃光

prompt 写成 `<SEP>prefix<SEP><SEP>side<SEP>` 时，`side` 只有 **34 token**：

```
段：prefix(4304) + side(34)
side: [0,34)  发布 span = [8,34) = 26 token
      但 KV 引用按 kernel block（32）下取整 → 复用窗口 [8,32) = 24 token
      span(8,34) ≠ 窗口(8,32) → 需要切片 → 压缩算子不可切 → 回退
      23 次/轮
```

**短段本身不值得复用**：34 token 里的 seam 8 + 对齐 2，剩下 24 token，收益可忽略，
却带来一次完整导入 + 一次回退。

**修法（C2）**：`min_segment_tokens`（建议 = 2×seam_sink 或 64）以下的段直接不参与 PIC
（不捕获、不发布、不 park、不导入），按普通段重算。默认值需谨慎：现有单测大量使用 2–4 token
的段，直接改默认会大面积破坏测试，应先加配置项、测试里显式关闭。

#### 根因二：A 组的 prompt 形态不是论文负载

论文的 compositional workload 是"**系统+文档+query**"，复用单元是大文档（>512 token），
query 每请求不同。而当前 A 组把 `side` 做成 34 token 的第二块，既不真实，也拿不到收益。

**修法（测试侧）**：A 组改成

```
prompt_i = <SEP>doc_i（约 4–5k token）<SEP> query_i（每请求不同）
           └── 可复用 ──┘        └── 不可复用（外部文本）──┘
```

同一 doc 被两个请求使用（第二个换 query），才是"重复内容 + 变化后缀"的真实形态。

#### 根因三：回退代价高（需要单独量化）

743.8 − 393.6 = **350 ms** 的额外开销。其中一部分是"导入白做"：park 阶段已经导入 KV、
materialize 之后才发现窗口对不上，于是重算。这条比小段问题更值得查——它决定"失败时有多贵"。

**测量方法**：同一 prompt 对比
- `--pic-seam-sink 0`（无部分窗口，应全部命中）
- 默认 seam 8（部分窗口，短段回退）

若 seam=0 时 p50 回到 400 ms 以下，说明 350 ms 主要来自"导入后回退"，而不是并发调度。

#### 决定（2026-10-28）：这两条推迟，不阻塞收益测量

| 问题 | 决定 | 理由 |
|---|---|---|
| 小段未跳过 | **用配置解决，不改代码** | --pic-min-segment-tokens（本版已加，默认 0）在 bench/部署里设 512 即可 |
| 回退前白导入 | **推迟到 C2** | 只有窗口与发布 span 不一致时才触发；≥4K 段 + 默认 seam 下窗口恒等于 span |

**重启条件**：若真实负载里出现
PIC transition range is outside the published span 或
	ransition slicing is unsupported **非零**，说明窗口又开始不一致，那时必须修导入前判定。

#### 根因四：segment=0 hash 撞

6/20 出现 `externally_owned_cache_entry`。`_segment_hash` 是 sha256(全部 token) 取前 16 字节，
**碰撞概率为零**，所以这 6 个请求的前缀 token 实际相同——是测试数据的 `unique_marker`
没有真正改变 token（例如 marker 在分词后与模板重复，或 publish 脚本复用同一前缀文件）。

**动作**：publish 20 组时打印每组的 `seg_hash[:8]`，确认 20 个都不同；若相同，改 marker 文本。

### 7.9 C1d：A 组（不同 prompt）暴露的两个 bug（2026-10-28）

B 组（同 prompt）已验收；A 组用 20 个不同 prompt 时出现以下两个问题，均已定位到具体分支。

#### W7（硬崩溃）：`PICTransitionOperator.slice` 对 affine/linear transition 的局部区间无解

代码事实（`vllm/v1/pic/state.py`）：

| 类 | 存储形态 | 有 `slice` 吗 |
|---|---|---|
| `PICGDNTransition` | **逐 token** 的 `key/value/log_decay/beta` + 可选的 chunk 参考 | ✅ 按 token 轴切 |
| `PICConvTransition` | 逐 token 的 conv 输入 | ✅ |
| `PICAffineTransition` | **整段压缩成一对** `(decay, zero_state)` | ❌ |
| `PICLinearTransition` | **整段压缩成一个** `(matrix, zero_state)` | ❌ |

```python
# state.py:804
for transition in self.transitions:
    if isinstance(transition, PICGDNTransition):
        sliced.append(transition.slice(local_start, local_end))
    elif local_start == 0 and local_end == length:   # 只有"整段"才放行
        sliced.append(transition)
    else:
        raise ValueError("PIC transition slicing is unsupported for this state kind")
```

因此：

- **B 组**（定长 prompt）：seam 子区间恰好等于整段 → 走 `elif`，通过；
- **A 组**（长度 4967–5015 浮动）：子区间 ≠ 整段 → 落到 `raise`；
- 而且这里是裸 `ValueError`，不在 `_run_pic_single_request_ranges` 的
  `except PICWorkerUnsupported` 覆盖内 → **直接打死 EngineCore**。

**为什么不能"补个 slice 方法"了事**：affine/linear 只存了整段的压缩结果，
`out = decay*state + zero`（或 `state @ matrix + zero`），从这一对里**无法反推**任意前缀的
转换——它需要逐 token 的中间系数。逐 token 存 `zero_state` 是 state 大小的 4352 倍，不可行。

#### 修法（支持任意长度，而非回退）

复用 GDN 已有的 **chunk 机制**（`chunk_size=64`，`_apply_chunked_reference`）：

1. **capture 侧**：affine/linear 也按 chunk 存"每 64 token 组合后的转换 + 每个 chunk 边界的
   checkpoint state"。4352 token 只有约 68 个 chunk，内存可忽略。
2. **`slice(local_start, local_end)`**：
   - 两个端点都落在 chunk 边界 → 直接组合 chunk 区间的转换；
   - 端点不齐 → **把 `local_start` 向下取整到 chunk 边界**，并让调用方从该边界开始重算
     （最多多算 63 个 token，代价有界）。
3. **调用方**：`apply_matched_mamba_transition` 的 `local_start` 由 runtime plan 的 seam 区间给出，
   需要允许"向下对齐"并把重算区间同步扩展。
4. **安全网**：任何仍然无法切分的 kind 一律抛 `PICWorkerUnsupported`（受控失败），由
   `_run_pic_single_request_ranges` 捕获后做**单请求**回退，绝不冒泡成 EngineCore 退出。

#### 为什么必须真支持（而不是回退）

复用段的长度是**随机的**（A 组 4967–5015 已证明），seam 子区间几乎永远不等于整段。
若只做"不支持就回退全量重算"，PIC 在真实负载下会退化成普通请求——收益归零。
所以这条是**功能必需**，不是健壮性优化。

#### 工作量修正

| 子项 | 天数 |
|---|---:|
| capture 侧 chunk 化（affine/linear）+ 序列化兼容 | 0.75 |
| `slice` 实现 + 端点对齐与重算区间联动 | 0.5 |
| 安全网（改抛 `PICWorkerUnsupported` + 单请求回退） | 0.25 |
| 单测（chunk 对齐 / 不齐 / 无法切分三种）+ A 组回归 | 0.5 |
| **合计** | **1.5–2 天** |

#### W7 实施计划（C1d，2026-10-28）

##### 结论先行：按 chunk（64 token）存，不按 token 存

| 表示形式 | 体积/层/段 | 能切任意位置 |
|---|---:|---|
| 整段压缩（当前 publish 回退形式） | ~32 KB | ❌ |
| 逐 token 原始张量 | ~18 MB | ✅ 任意 |
| **每 64 token 压缩一次（采用）** | **~2.2 MB** | ✅ 到 64 边界 |

"按 segment"说的是**作用域**（每段一个 operator），"按 chunk"说的是**表示形式**。整段压缩后
`A_seg = A_b · A_[a,b) · A_a`，只知乘积不知因子，因此任意前缀不可反推——这是必须 chunk 化的原因。

对 24 层、4352 token 的段：整段 0.8 MB、chunk 化 52 MB、逐 token 430 MB。选 chunk。

##### 落在我们的 segment 存储方案上

我们的 segment 就是 `<SEP> … <SEP>` 之间的 token 区间 `[start, end)`，长度任意。按 chunk 化后：

```
segment [start, end)                  len = end - start
├── 完整 chunk 数  full = len // 64
├── 已存区间       [start, start + full*64)
└── 尾巴           [start + full*64, end)     长度 = len % 64，< 64 → 不存，重算
```

规则与"之前的方式"一致，只是把"整段一个算子"换成"每 64 token 一个算子"：

| 情形 | 行为 |
|---|---|
| `len % 64 == 0` | 全部 chunk 都存，与现在等价 |
| `len % 64 != 0` | 存前 `full` 个 chunk，尾巴 `<64` token 回到重算区间 |
| `len < 64` | 一个 chunk 都不完整 → 整段重算（极短段，正常不发生） |
| KV 导入 | **不受影响**，仍覆盖整个 segment；尾巴只是重算它的 state/KV |

**修正（2026-10-28）**：chunk 网格**不能**简单锚定在 segment 起点——复用窗口的起点本来就不
一定在段首。代码里已经有证据：

```python
# vllm/v1/pic/runtime.py:185
candidate_start = max(local_start, seam_tokens)      # 段首被 seam 吃掉

# vllm/v1/pic/runtime.py:230-232
aligned_start = candidate_start + (target_phase - candidate_start) % alignment
reusable_local_start = max(reusable_starts)          # 再按 kernel block 对齐向上取整

# vllm/v1/pic/runtime.py:258-267
if reusable_start > item.start:
    runtime_ranges.append(PICRuntimeRange(..., action="recompute"))   # 段首那截已经要重算
```

所以复用窗口是 `[reusable_start, reusable_end)`，两端都可能不在段首/段尾上。

#### chunk 大小做成可配置，默认 64

不把 64 写死在代码里，而是走 PIC 的既有配置模式（`PICConfig` + `arg_utils`）：

| 项 | 值 |
|---|---|
| 配置字段 | `pic_config.transition_chunk_size: int = 64` |
| 命令行 | `--pic-transition-chunk-size` |
| 默认 | **64** |
| 校验 | 必须 2 的幂；必须整除/被整除于 kernel alignment（`chunk % alignment == 0`）；范围 `[alignment, 1024]` |
| 生效点 | ①capture 切 chunk ②`slice` 端点量化 ③runtime plan 的复用窗口收紧 ④KV 引用同步收紧 |
| 启动日志 | `PIC transition chunk=%d kernel_alignment=%d manager_block=%d` |

**校验必须 fail-closed**：配置不合法时启动即报错，而不是运行时才读错 KV。

```python
# 建议放在 PICConfig 校验里
if self.transition_chunk_size % alignment != 0:
    raise ValueError(
        f"--pic-transition-chunk-size={self.transition_chunk_size} must be a "
        f"multiple of the attention kernel block size ({alignment})"
    )
```

**代价随 chunk 变化的定量关系**（4352 token 的段、24 层）：

| chunk | 每段 chunk 数 | 过渡状态体积/段 | 每段最多多算 |
|---:|---:|---:|---:|
| 16 | 272 | ~209 MB | 30 token |
| 32 | 136 | ~105 MB | 62 token |
| **64（默认）** | **68** | **~52 MB** | **126 token** |
| 128 | 34 | ~26 MB | 254 token |
| 256 | 17 | ~13 MB | 510 token |

默认取 64 的理由：与 GDN 的 `chunk_size=64` 同粒度（不引入新常量）、多算占比 <3%、
体积在 2 GB 预算内可控。若缓存段数变多导致内存紧张，**调大 chunk 是纯配置操作，不改代码**。

#### 最终方案：每段一个矩阵，按 reuse 区间发布（2026-10-28 定稿）

##### 结论

**每个 segment 只存一个压缩矩阵**，不存逐 token、不做 chunk。
唯一要改的是：这个矩阵必须覆盖**将来被复用的那个区间**，而不是"整段"。

##### 为什么可以只存一个矩阵

| 事实 | 依据 |
|---|---|
| 复用窗口只有一种形状 | `a = seam`（`seam_sink` 默认 8，仅 `index>0`）、`b = kv_token_count` |
| KV 落位逐 token，不需要 block 对齐 | `native_kv.py:641` `arange`；`runtime.py:220` 对 `CANONICAL_PUBLIC` 跳过对齐 |
| 矩阵大小与 token 数无关（`[H,D,D]`） | `state.py:388 compact()`；本模型 H=16、D=128、fp32 |
| 逐 token 形式只是发布时的**中间态** | capture 时张量已在显存里（模型刚算完），切完即可丢弃 |

A 组崩溃的真正原因：每个 prompt 有 2 个段，第 2 段 `index=1` → `seam=8` → `a=8≠0`，
而发布的矩阵只覆盖 `[0,len)` → 需要 slice → 压缩矩阵不可切 → 崩。

##### 改法（一处）

```python
# _publish_pic_transition_operator：发布前先切到复用区间
span_start = seam_sink                  # 统一 8（含 index == 0）
span_end   = kv_published_token_count   # 与该段 KV 覆盖的 token 数一致
trimmed = operator.slice(span_start, span_end)   # 逐 token 形式，可切
publish(compact(trimmed), span=(span_start, span_end))
```

读取侧把 `(span_start, span_end)` 存进 manifest；runtime plan 的复用窗口直接取它，
**不再做任何对齐/截断**，也就不需要 slice。

##### 论文依据：seam 作用于每个 interior segment（2026-10-28 核实）

对照 `2607.01299v2.pdf` §4.3（Hypic，Full attention alignment with seam windows）原文：

> "for each **interior segment**, Hypic designates its first F tokens as a
> recomputation region, which we call the **seam window**."
>
> "At cache time, Hypic runs full attention over every token in each segment but
> caches only the **reusable suffix KV for interior segments** — the first F tokens
> are left uncached, since they are guaranteed to be recomputed at reuse time."
>
> "The **leading segment** is typically the system prompt, which has no left
> neighbor and always anchors at position 0, so no cross-segment deviation needs
> repair and Hypic caches it **in full without seam exclusion**."
>
> "In practice F is small — we use **F = 8** as the default."
>
> "When the model also has a causal convolution, **F ≥ k−1** guarantees the seam
> recompute itself warms up the boundary conv state."

结论（推翻此前的"段连续所以不需要 seam"）：

| 段 | seam | 缓存范围 |
|---|---|---|
| 段 0（leading，位置 0） | **无** | **整段** `[0, len)` |
| 段 i>0（interior） | **8** | **后缀** `[8, len)` |
| 尾部段（trailing，用户 query） | 论文端到端算，不缓存 | — |

原因不是"状态断层"，而是**注意力汇聚（intra-segment attention sink）**：每个 interior
段的首几个 token 在缓存时是在**不同的左上下文**下算出来的，复用时会偏离；这几个 token 的
KV 因此需要重算，且 `F ≥ k−1 = 3`（`linear_conv_kernel_dim = 4`）顺便把卷积边界状态也焐热了。

**粒度澄清（2026-10-28）**：seam 以 **segment**（<SEP> 之间的内容块）为单位，
与 **KV block 的数量无关**。一个 interior 段内部无论被划成多少个 block（16/32/544 token 一块），
都只有该段开头的 F 个 token 需要重算；block 只是 KV 的存储/对齐单位，不产生额外 seam。

| 粒度 | 单位 | 与 seam 的关系 |
|---|---|---|
| segment | <SEP> 之间的内容 | **seam 的单位**：每个 interior 段一个 |
| block | KV 存储块（= lock_size token） | 只影响 KV 落位与对齐，**不产生 seam** |

**所以实现应当是"按段发布后缀"**：段 0 发布 `[0,len)`，段 i>0 发布 `[8,len)`，
复用窗口与发布的 span 逐段一致，不需要任何 slice。这正是当前 C1d 实现（`span_start`）做的事。

##### seam 从哪取

不写死 8，走已有配置链：

```
--pic-seam-sink  →  PICConfig.seam_sink (默认 8)  →  request.pic_seam_sink（可选覆盖）
                 →  build_plan(seam_sink=...)     →  cache_plan.seam_tokens[i]
```

发布侧读同一个配置：

```python
seam = self.vllm_config.pic_config.seam_sink
span = (seam, kv_published_token_count)
```

**注意 per-request 覆盖**：`request.pic_seam_sink` 可以逐请求改值。若某请求把 seam 调大
（比如 16），而发布 span 从 8 开始，则该请求的复用窗口起点 > span 起点 → 需要 slice →
按 fail-closed 回退。这不是 bug，而是"该请求要求更保守的 seam，span 不匹配就不复用"。

要完全避免这种不匹配，可以让计划侧**直接采用 manifest 的 span 起点**，而 `seam_sink` 只作为
"至少重算多少 token"的下界：

```python
window_start = max(request_seam, manifest.span_start)
if window_start > manifest.span_start:
    raise PICWorkerUnsupported("seam exceeds published transition span")
```

##### 配套要求

| # | 要求 | 理由 |
|---|---|---|
| 1 | `seam_sink` 取**配置值**（`pic_config.seam_sink`，默认 8，`--pic-seam-sink`），且对所有段统一（`index==0` 也算） | 否则发布端无法预知复用端的 seam |
| 2 | manifest 记录 `span_start / span_end` | 读取端据此设置复用窗口 |
| 3 | 复用窗口必须等于发布 span，否则明确回退（`PICWorkerUnsupported`） | fail-closed |
| 4 | 删除已作废的 chunk 计划与 `--pic-transition-chunk-size` | 矩阵按 chunk 复制会 × 段长/64 |

##### 体积（本模型实测口径）

| 项 | 值 |
|---|---:|
| 单层状态 `[16,128,128]` fp32 | 1.0 MB |
| 单层矩阵 + zero_state | 2.0 MB |
| **18 层 GDN 合计** | **36 MB**（实测 38.4 MB，差额是 conv tail） |
| 改动后 | **不变** |
| 逐 token 形式（不存） | 1.29 GB |

##### 验收

| # | 检查 | 期望 |
|---|---|---|
| 1 | A 组 20 个不同 prompt，c=2 | 20/20，无 EngineDead |
| 2 | `action="reuse"` 区间 = `[seam, span_end)`，且 reuse token ≥ `len - 8 - 32` | 复用真实发生 |
| 3 | `fallback_reasons` 无 slice 相关项 | 走的是新路径 |
| 4 | 传输字节 | 与今天持平（≈38.4 MB transition） |
| 5 | 多算 token | ≤ 8 + 尾部未对齐部分（<40） |
| 6 | B 组回归 | 仍 1.72× / 1.60× |

##### 工作量

| 子项 | 天数 |
|---|---:|
| publish 侧切片 + manifest 记录 span | 0.25 |
| 读取侧用 span 设置窗口 + fail-closed | 0.15 |
| `seam_sink` 统一 + 计划构建调整 | 0.1 |
| 单测（span 命中 / 不匹配回退） | 0.15 |
| **合计** | **0.65 天** |

##### 风险

| 风险 | 缓解 |
|---|---|
| 已有缓存条目没有 span 字段 | 视为"整段 `[0,len)`"，只支持整段复用；部分复用回退 |
| seam 统一后所有段多算 8 token | 占比 0.18%，可忽略 |
| 尾部未对齐时 `b < span_end` | 该情形按 `PICWorkerUnsupported` 回退（已知有界） |

---

#### W8（软错误）：publish 的 block 数与 token 区间不自洽

```python
# vllm/v1/pic/mooncake_kv.py:1100
expected = (canonical_start + token_count + block_size - 1) // block_size
expected -= canonical_start // block_size
if len(ids) != expected:
    raise ValueError(f"native KV block span mismatch: expected {expected}, got {len(ids)}")
```

`expected` 由 `(canonical_start, token_count)` 推出，而 `ids` 是调用方
`_materialize_pic_public_kv` 选出的 block 列表。B 组固定长度时两者一致；A 组长度浮动时，
最后一个 allocator block 覆盖的 token 数大于 manifest span，于是 `len(ids)` 比 `expected`
多（报错里 135 vs 119，差 16 个 block = 512 token，正好是"多算了一个尾块"的量级）。

**修法**：让 `_materialize_pic_public_kv` 用与 `expected` 相同的公式反推需要多少 block
（或用 `get_full_local_block_span` 的结果裁剪 `ids`），保证"传入的 block 数"与
"(canonical_start, token_count, block_size)"严格自洽；并补一个"长度不整除"的单测。

#### 完成判据

| # | 检查 | 期望 |
|---|---|---|
| 1 | A 组 20 个不同 prompt，c=2 | 20/20 成功，无 EngineDead |
| 2 | `requires a transition handle` / `transition slicing is unsupported` | 0 |
| 3 | `native KV block span mismatch` | 0 |
| 4 | 回退路径 | 若某请求 slice 不支持，`fallback_reasons` 应出现对应项且输出仍正确 |
| 5 | B 组回归 | 仍为 1.72× / 1.60× |

### 7.8 W6 已取消：PIC prefill 不组 batch（2026-10-28 决定）

**决定**：PIC 的 prefill 保持"每步只跑一个请求"的隔离执行（`--pic-single-request`），
**不**与普通请求、也**不**与其他 PIC 请求组 batch。`--pic-batch` / `--pic-packed-batch`
保持默认关闭。

理由：

| # | 理由 |
|---|---|
| 1 | `exclusive_batch` 默认 `True`，PIC 与普通请求本来就被 scheduler 分到不同 step；所谓"混合 batch"在不关这个开关时根本不会出现，为它做融合没有对象 |
| 2 | 序列较长时，一次 forward 的固定开销被摊薄，组批带来的边际收益很小（Stage 10-A 实测：4 并发纯 prefill 只比 9-B 快 **7.8%**，decode 持平，单请求还略慢） |
| 3 | 组批会把"每个请求一条独立的 range/recurrent/transition 状态"变成"多请求交错"，是正确性风险最大的一块；当前优先保正确性 |
| 4 | 收益的主要来源已经定位在别处：重复 inline 导入（≈26 ms/请求，adopt 修复）与通信/计算重叠（W1），不是组批 |

**保留的开关**：`--pic-batch`(9-B)、`--pic-packed-batch`(10-A) 代码与验收都在，需要时可以显式打开，
不需要重新实现。

**将来若重启这件事，先回答三个问题**：

| # | 问题 | 怎么答 |
|---|---|---|
| 1 | 我们的 PIC 形态（640 recompute token）组批到底能省多少 | 一次测量：同一步里 1 个 vs 2 个 PIC 请求的 `pic_range.forward_dispatch` 差多少 |
| 2 | 长序列 vs 短序列的分界 | 用 §21.20 的 fixed/marginal 模型：`fixed = 39 ms` 摊到 `N` 个 token 上，`N` 越大收益越小；分界约在 `N ≈ 2000` |
| 3 | 组批是否破坏 range/transition 隔离 | Stage 10-A 的验收用例（三类请求逐 token 一致）必须重跑 |

#### 存档：原本的待决策内容


#### 现状：代码里有三条执行路径，默认全关

`vllm/config/pic.py` 里三个开关默认都是 `False`：

| 开关 | 默认 | 走哪条执行路径 | 混合 batch 行为 |
|---|---|---|---|
| `single_request` | False | `_run_pic_single_request_ranges`（9-A） | 只对**单请求** batch 生效；混合 batch 不走 |
| `batch` | False | `_run_pic_batch_ranges`（9-B） | **request-isolated microbatch**：每次模型调用只给一个请求非零 token，其余行填零；普通请求保持原生路径，但**不与 PIC 请求融合进同一次 forward** |
| `packed_batch` | False | `_run_pic_packed_batch_ranges`（10-A） | 打包执行，PIC 子集进 `batch_runtime_plans`，普通请求作为 `ordinary_requests` 一起传下去 |

三者全关时（bench 的默认配置），走 `elif num_reqs == 1` 之后的**原生 batched forward**：
PIC 请求的 block table 已被 `_attach_pic_native_slots` 挂好私有 KV，但它自己的
range/transition 逻辑**不执行**，`pic_runtime_result is None` 还会触发一次
`_copyback_pic_kv()`。

#### 关键判断（待确认）

混合 batch 若落到"原生 batched forward"，对 **full-attention** 模型输出正确（KV 在 block
table 里，只算 recompute 的 token）；但对 **hybrid（GDN/mamba）** 模型，transition 状态
不会被 apply，存在正确性风险。当前 bench 之所以安全，是因为它用
`--pic-single-request` 把 PIC 请求隔离成单请求微批。

#### 需要决定的事

| # | 问题 | 选项 |
|---|---|---|
| 1 | 混合 batch 里 PIC 请求走哪条：隔离微批、打包融合、还是原生 forward | 建议：默认强制隔离微批（安全），打包融合作为显式开关 |
| 2 | hybrid 模型在原生 forward 下缺 transition 是否正确 | 需要一次 A/B 输出对比（同一 prompt，PIC 单请求 vs 混合批） |
| 3 | scheduler 是否允许 PIC/non-PIC 同批调度 | 当前 `--pic-exclusive-batch` 控制；默认不混合 |
| 4 | 打包融合的性能收益 | 见 §21.20 的 fixed-cost 推算：每请求可能省 19 ms，但需实测 |

#### 估算

| 子项 | 天数 |
|---|---:|
| 混合 batch 的 hybrid 正确性验证（A/B 输出） | 0.5 |
| 默认策略实现（无 PIC 执行开关时，PIC 请求强制隔离微批或安全回退） | 0.5 |
| 打包融合的混合路径调试与验收 | 1–2 |
| **合计** | **2–3 天** |

### 7.7 W5 已取消：GPU 段接受与自己的 forward 串行（2026-10-28 决定）

**决定**：不把私有 KV 物化搬进 park 窗口。GPU 侧工作（H2D materialize + 私有物化）
接受与**本请求自己的 forward** 串行，不再试图与别的请求重叠。

理由：

| # | 理由 |
|---|---|
| 1 | 这段是 GPU kernel（SM + 显存带宽），与 forward 同 stream，本来就会串行；搬位置只改延迟分布，不改总量 |
| 2 | `pic_slot_attach = 7.8 ms` 是**分析档**数字（7 处 `cuda.synchronize`），生产档真实成本未知且很可能远小于此 |
| 3 | 带宽算术表明 107 MB 的设备流量只有 0.04 ms，7.8 ms 若真实存在也是 Python/kernel 启动开销，搬迁无益 |
| 4 | 搬迁会让"导入线程"产生 GPU 依赖，需要跨线程 event（`native_kv_transfer.py:145` 已在做），复杂度上升而收益不确定 |

**保留当前分工**（这是"并行"的唯一来源）：

| 线程 | 做什么 | 与谁并行 |
|---|---|---|
| import 线程 | TCP 收包 + 反序列化 + H2D（其中 CPU 部分占 73.9 ms） | 与**别的请求**的 forward 并行 |
| 主线程 | 私有物化 + forward | 自己的链，串行 |

**对上限的影响**（用 record §21.20 的口径）：

| 口径 | 每请求下限 | 上限 |
|---|---:|---:|
| CPU/网络 与 forward 重叠，GPU 段串行 | 34.4 + max(73.9, 94.9) = 129.3 ms | **1.63×** |
| 若把 H2D 也挪到主线程（不推荐） | 34.4 + (7.8 + 94.9) = 137.1 ms | 1.54× |

结论：**H2D 留在 import 线程是有价值的（值 0.09×）**，私有物化留在主线程串行即可。
W5 从 W 清单移除，原评估保留在下文仅作记录。

**唯一遗留的验证项**：生产档（`HYPIC_PIC_PROFILE_IMPORT=0`）下
`native-materialization kind=private` 的真实成本。若它确实 <1 ms，则"计算 94.9 ms"里
PIC 独有部分只剩 3.9 ms 左右的 range 准备，1.63× 的上限不需要任何调整。

---

#### （存档，已不执行）原 W5 评估：私有 KV 物化前移


#### 问题

PIC 的 KV 落在**共享 public blocks** 上，attention 读的是**请求自己的 block table 行**，因此
_attach_pic_native_slots → _materialize_pic_private_kv 每次请求要做一次
gather → rerotate → copy，把 public KV 复制进私有行。

| 项 | 实测 | 说明 |
|---|---:|---|
| execute_model.pic_slot_attach | **7.8 ms** | PIC 独有（Full 约 0.05 ms），其中主要是私有复制 |
| pic_range.* 五项合计 | 3.9 ms | PIC 独有的输入/元数据准备 |
| 与 Full 共有（prepare_inputs/attention_metadata/sample/update_states） | ≈12 ms | 动不了 |
| PIC 计算总量 | 94.9 ms | forward 70.9 + PIC 独有 11.7 + 共有 12 |

关键结论：这 11.7 ms 是 **PIC 自己带来的计算**，而我先前把计算整体当作不可优化是错的。
它落在关键路径上，占 c=1 p50 的 5.5%；消掉一半，上限可从 **1.63× 抬到约 1.72×**。

#### 前置：7.8 ms 是分析档的数字，必须先测生产档

`_materialize_pic_private_kv` 内部有 **7 处 `_pic_profile_sync`**，每处都是
`torch.cuda.synchronize(device)`（`gpu_model_runner.py:419-427`），只在
`HYPIC_PIC_PROFILE_IMPORT > 0` 时生效。而 record §9.5 的 `pic_slot_attach = 7.8 ms`
正是在分析档测的——这个数字包含同步停顿，不是生产档成本。

带宽算术也不支持 7.8 ms 是"拷贝本身"：

| 项 | 值 |
|---|---|
| 私有复制流量 | 53.5 MB 读 + 53.5 MB 写 ≈ 107 MB |
| 4090 显存带宽 | ~2.5 TB/s（保守按 500 GB/s 算） |
| 纯拷贝耗时 | 0.04 ms（按 500 GB/s 也只有 0.21 ms） |

如果 7.8 ms 在生产档真实存在，它**不是带宽，而是开销**：每层 Python 循环、
`build_kv_slot_mapping`、gather 的索引构造、多次 kernel launch。

因此 W5 的第一件事不是改代码，而是测一次生产档：

```bash
HYPIC_PIC_PROFILE_IMPORT=0   # 生产档，看 c=1 p50
HYPIC_PIC_PROFILE_IMPORT=2   # 分析档，grep 'native-materialization kind=private'
```

判据：生产档 c=1 p50 与分析档之差若只有 1–2 ms，说明 7.8 ms 主要是同步假象，
**W5 可以取消**；若差异仍是 7 ms 级别，才值得做。

#### 为什么可以前移

park 窗口里通信（81.7 ms）大部分是空的——计算只有 94.9 ms 且其中 70.9 ms 是 forward，
而 park 时该请求**不参与本步 forward**。把私有复制放进这个窗口，就能从关键路径上消失，
且不增加任何网络带宽。

#### 依赖与改造点（这是估算依据）

| # | 依赖 | 现状 | 改造 |
|---|---|---|---|
| 1 | 私有 block ids | 取自 lock_table.get_numpy_array()[row_idx]（gpu_model_runner.py:3527），park 时请求不在 batch、无 row | 改为直接取 NewRequestData.block_ids[group]（park 时已分配，spec 已携带） |
| 2 | pic_native_kv_mapping（slot plans） | 由 _build_pic_runtime_plan 在请求进入 worker 时构建 | park 时也要构建一次：用 _pic_park_state 的 stand-in（_prepare_pic_mooncake_destinations 已证明 stand-in 够用） |
| 3 | slot_plan.slot_mapping / source_start | 同上 | 同上 |
| 4 | 顺序 | 复制必须在导入的 H2D materialize **之后** | 在 _pic_park_poll 完成回调里、manager.complete 之前执行复制 |
| 5 | 幂等 | pic_private_kv_materialized 已按 key 去重（:3539） | 复用同一套 key，重复调用自动跳过 |
| 6 | 失败退化 | 复制失败当前直接抛 | 改为记录 fallback，走 inline 复制（保留现路径作为兜底） |

#### 澄清：它不占张量核，但占 GPU

"不占 GPU 计算带宽"只对了一半：

| 资源 | 私有复制是否占用 |
|---|---|
| 张量核 / FLOPs | 不占 |
| SM 占用（kernel 执行） | 占 |
| 显存带宽（DRAM） | 占（约 107 MB/请求） |
| 与 forward 的 stream 顺序 | 同 stream 串行 |

所以移进 park 窗口的收益是**不落在本请求的关键路径上**（延迟收益），不是"免费"。
GPU 总工作量不变，因此 GPU 饱和的 c=4 场景吞吐不会改善，甚至可能因与其他 forward
抢 SM 略差。W5 的定位应写成"低并发降延迟"，不能写成"提高上限"。

#### 估算

| 子项 | 工作 | 天数 |
|---|---|---:|
| A. 让 _materialize_pic_private_kv 接受 block_ids 参数（不再依赖 batch 行） | 小改，需覆盖 row 缺失的情况 | 0.25 |
| B. park 时构建 stand-in 的 runtime plan / mapping | 复用 _pic_park_state，但 plan 构建依赖 pic_worker_plan，需确认 stand-in 可满足 | 0.5 |
| C. 在导入完成回调里执行复制（含 stream 顺序、失败回退） | 与 §5.2 的 H2D 顺序一致 | 0.5 |
| D. 验证（p50、pic_slot_attach 归零、正确性五层） | 一轮 c=1/c=2 + A/B 输出一致 | 0.25 |
| **合计** | | **1–1.5 天** |

#### 判据

| 检查 | 期望 |
|---|---|
| execute_model.pic_slot_attach | 从 7.8 ms 降到 <1 ms（复制在 park 窗口完成，此处只做 attach 记账） |
| c=1 p50 | 从 219 ms 降到约 211 ms |
| c=2 上限 | 1.63× → 约 1.72× |
| 正确性 | 
eused_tokens/
ecompute_tokens 不变，A/B 输出逐 token 一致 |
| 失败路径 | 强制复制失败时能回退且请求仍正确（fallback 计数 +1） |

#### 风险

| 风险 | 说明 |
|---|---|
| stand-in 构建 plan 不完整 | runtime plan 依赖 pic_worker_plan（由 _build_pic_worker_plan(new_req_data) 产出），park 时只有 spec 里的 NewRequestData；若字段不全会退化为不前移，无害 |
| 复制与 forward 抢 stream | 复制在 park 窗口执行，与下一步 forward 同 stream 串行；需在复制后记录事件，避免与导入的 H2D 乱序 |
| 私有 block 生命周期 | park 时 block 已分配且请求持有，释放路径不变 |

