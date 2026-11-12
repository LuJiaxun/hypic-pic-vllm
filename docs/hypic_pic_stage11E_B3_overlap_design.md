# Stage 11E-B3 草案：PIC 请求级异步与通信/计算重叠

状态：**草案，未经人工确认，不属于冻结计划**。`hypic_pic_vllm_0221_stage_plan.md`
仍是唯一冻结依据；本文件只记录 11E-B3 的落地方案、风险和验收口径。

> 2026-10-21：人工决定先只做 deserialize 消除（11E-D1），本 B3 草案暂不实施。
> 当前代码中没有引入任何通信/计算重叠改动。

## 1. 为什么现在不能重叠

当前远端导入全部发生在 `GPUModelRunner._update_states()` 内部
（`vllm/v1/worker/gpu_model_runner.py:1632` → `_set_pic_runtime_plan()` →
`_build_pic_runtime_plan()`），也就是在 `execute_model()` 的同一个 step 里阻塞。
后果：

- 该 step 的 forward 必须等传输完成，同 batch 的其他 request 也被一起拦住；
- 即使 B 侧同时有多个 PIC 请求，也无法“A 传输时 B 计算”；
- scheduler 完全不知道 worker 正在等远端数据，因此没有请求级等待队列。

## 2. 目标语义

```text
scheduler 发现 PIC external hit
    -> 分配目标 native block（拿得到 destination）
    -> request.status = WAITING_FOR_REMOTE_KVS
    -> 下一次 SchedulerOutput 携带 prefetch descriptor（不调度该 request 的 forward）
worker 收到 prefetch descriptor
    -> 启动 native KV + transition 传输（后台线程）
    -> 本 step 继续计算其他 request
每个 step
    -> 轮询 pending import；完成的 request id 放进 ModelRunnerOutput
scheduler 收到 ready
    -> 提升为 WAITING -> 下一个 step 正常调度
    -> 该 request 的 prefill 只在此刻 invoke
```

关键不变量（与 Stage 9-A/10 一致）：

- ready 之前 attention 不得读取目标 KV，transition 不得 apply；
- 任一子传输失败只回退该 request；
- 取消/抢占/超时必须释放 transfer handle、lease 和已分配 block；
- 默认开关关闭时，行为与当前同步路径完全一致。

## 3. 需要改的文件

```text
vllm/v1/core/sched/output.py        SchedulerOutput 增加 pic_prefetch_reqs
vllm/v1/core/sched/scheduler.py     add_request/schedule 的等待与提升
vllm/v1/worker/gpu_model_runner.py  prefetch 启动、轮询、ready 上报、adopt
vllm/v1/outputs.py                  ModelRunnerOutput 增加 pic_ready_req_ids
vllm/v1/pic/transfer_engine.py      PICRequestTransfer 增加 ready/error 查询
vllm/config/pic.py / arg_utils.py   开关（默认关闭）
```

## 4. 实施顺序

1. **B3-0 只读轮询**：`PICRequestTransfer` 增加 `try_wait()`（非阻塞查询
   ready/error），不改任何调度行为。单测覆盖 pending/ready/error 三态。
2. **B3-1 prefetch 通道**：`SchedulerOutput.pic_prefetch_reqs` 携带
   `req_id`、每个 KV group 的 allocator block id、`pic_segments`、
   `pic_cache_plan` 的 external reference。worker 用现有
   `_prepare_pic_mooncake_destinations()` 推导 kernel block id（不需要把 request
   加进 `InputBatch`），随后 `begin_import_references()` + transition submit。
3. **B3-2 scheduler 等待**：`--pic-scheduler-wait` 打开时，external hit 的请求
   在 `schedule()` 里先 `allocate_slots()`，然后置
   `WAITING_FOR_REMOTE_KVS` 并 `continue`（结构参照 `load_kv_async` 分支，
   `scheduler.py:884`）。复用现有 `_try_promote_blocked_waiting_request()`
   （`scheduler.py:2579`）。
4. **B3-3 ready 上报与 adopt**：worker 在 `ModelRunnerOutput` 里回报
   `pic_ready_req_ids`；被提升的请求再次进入 `_update_states()` 时，直接 adopt
   已完成的 import（`self._pic_prefetched[req_id]`）而不是重新传输。
5. **B3-4 失败与清理**：导入失败 -> 该 request fallback 到普通 forward；
   请求取消/抢占 -> `discard()` 并释放 block；leak 测试 1000 次。

## 5. 风险

- **死锁**：prefetch 没启动或 ready 丢失会让请求永远停在
  `WAITING_FOR_REMOTE_KVS`。必须加超时兜底：超时后按普通路径回退，并记录
  `pic_prefetch_timeout`，禁止无限等待。
- **block 生命周期**：`allocate_slots()` 在请求未运行时持有 block，
  fallback/取消必须走 `kv_cache_manager.free()`。
- **单请求无收益**：严格串行单请求没有可重叠的 compute，B3 的收益只在并发或
  有其他 request 在跑时出现；性能验收必须用并发场景，不能只看单请求 p50。
- **与 KV connector 冲突**：若同一部署启用 PD 分离 connector，PIC 不能复用
  `kv_connector_metadata`，必须走独立字段。

## 6. 验收口径

功能：

- prefetch 启动后，同一 step 内其他 request 的 forward 正常完成；
- ready 之前 head 请求不在 `scheduled_new_reqs` 中；
- ready 之后 forward token 数等于 miss+seam token；
- 输出与完整重算逐 token 一致（seam 0/4 两种）。

性能（相同 prompt、warmup、eager、`max_tokens=1`）：

- 单请求：预期与当前同步路径持平（无重叠对象），不得退化；
- 2/4 并发 PIC：报告端到端 p50/p95、`WAITING_FOR_REMOTE_KVS` 平均等待、
  transfer 与 forward 的实际重叠比例；
- 1000 次并发请求：无 pool/lease/block 泄漏、无超时回退。

## 7. 当前不建议做的事

- 不改 `BlockPool` 分配算法；
- 不为了重叠伪造 `num_computed_tokens`；
- 不在这一步引入新的 Triton/CUDA kernel。
