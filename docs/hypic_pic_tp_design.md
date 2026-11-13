# PIC 在 TP > 1 下的适配：参考设计

状态：**设计稿，未实现**。当前代码只允许 PP=1，**对 TP 没有任何判断**——所以 TP>1 会静默出错，
不是拒绝执行。本文先给事实、再给设计。

---

## 1. 今天的代码事实

| 事实 | 位置 |
|---|---|
| 段的 KV 键里**没有** `tp_rank`：`segment/<hash>/group-<gid>` | `vllm/v1/pic/mooncake_kv.py:2931` |
| 递推状态的键里也**没有**：`segment/<hash>/hybrid-state` | `vllm/v1/pic/mooncake_kv.py:3006` |
| 能力检查只拒绝 PP，**不拒绝 TP** | `vllm/v1/worker/gpu_model_runner.py:11468` |
| `vllm/v1/pic/` 里没有任何 `tp_size` / `tp_rank` / `get_tp_group()` | 全目录 grep 为空 |

**后果（重要）**：TP=2 时两个 rank 都会往同一个键写**自己那一片** KV 和状态，后写覆盖先写；
复用侧读到的必然是**某一个 rank 的分片**，另一个 rank 拿到错的——**静默数值错误，不会报错**。
这比崩溃危险得多，所以第一件要做的事是把"TP>1 且未启用新键格式"变成显式失败。

---

## 2. 原生 vLLM Mooncake connector 怎么处理 TP

结论先给：**它不做聚合，它让每个 rank 独立传输自己那片。**

| 机制 | 位置 | 说明 |
|---|---|---|
| 每个 rank 注册自己的 KV 张量 | `register_kv_caches` `:1388` | base addr + block_len 是**本 rank** 的地址 |
| 同 TP 大小 → rank i ↔ rank i | `_get_tp_ratio` `:88` + `_compute_sender_transfer_plan` `:140` | ratio == 1 时直接 1:1，偏移为 0 |
| 异构 TP 按比例切片 | `_compute_sender_transfer_plan` `:155-174` | `tp_ratio > 0`：一个生产者分片写进消费者的一块；`< 0`：一个消费者从多个生产者拼 |
| 缓存可能是"复制"而非切分 | `producer_cache_replicated` `:156,165` | 复制型缓存只需偶数 rank 发一次 |
| 元数据带 TP 身份 | `MooncakeXferMetadata.remote_tp_size/remote_tp_rank` `:260-261` | 消费者声明自己是 TP 里的第几个 |
| 握手配对 | `handshake_target_ranks` `:1007`、`resolve_need_send` `:1174` | 每个消费者 rank 与一组生产者 rank 配对 |

**可以借用的**：`_expand_transfer_regions` 对 `blocks_first` 布局的 K/V 拆分（`:109`）。
**不需要借用的**：它的 P/D 握手与 `transfer_id` 配对——PIC 是按**段键**发布/读取，不需要
"这个请求配哪个 rank"的会话概念。

---

## 3. PIC 比原生多一个维度

原生只搬 **KV**（每层、按 head 切）。PIC 还要搬 **递推状态**：

- GDN 的 recurrent state 在 TP 下**也按 head 切**，所以每个 rank 的状态形状不同 → 它的
  **transition matrix 也不同**（`state @ M + c` 里 M 的维度由本地状态决定）；
- 因此矩阵**天然是 per-rank 的**，不可以跨 rank 共享；
- 好在：**切段、seam、span、窗口、block id 全是集群级一致的**（scheduler 单实例、seam 是固定值、
  `PICWindow` 由 token 位置算出），所以各 rank 对"复用哪一段、复用多少 token"的判断**无需通信**。

一句话：**KV 和状态都按 rank 切，决策不切。**

---

## 4. 设计

### 4.1 键 schema v2：加 TP 段

```
segment/<hash>/group-<gid>/tp-<rank>-of-<size>
segment/<hash>/hybrid-state/tp-<rank>-of-<size>
```

- `size` 写进键里，使 TP=1 的旧键与新键**不会误命中**（旧键没有 tp 段 → 直接不匹配，自然失效）；
- 读侧必须用**自己的 rank** 去查；查不到就是 miss（不做"取任意一个 rank 的"这种容错——那是错数据）。

### 4.2 发布：每 rank 各自发布自己那片

- 每 rank 在**同一步**里发布自己的 KV 分片与自己的矩阵，无 gather、无 all-to-all；
- 传输量 = 原来 × 1（每 rank 只发自己那份），**总带宽不变**，只多了 N 个小对象；
- 发布仍是"先写 payload、最后写 manifest"的顺序。

### 4.3 完成性：manifest 里放 TP 位图

manifest 增加：

```json
{ "tp_size": 2, "tp_published": [true, true], "format": 2 }
```

- **写**：每个 rank 写自己的 payload 后置位；**最后一个置位的 rank 写 manifest**（或由 rank 0 轮询位图后写）；
- **读**：要求 `all(tp_published)`；任何一位没置位 → **视为未命中**，该段走全量重算；
- 这条规则挡住了"一半 rank 发布成功、另一半 OOM/超时"的坏情况——**不然就会有人复用半份 KV**。

### 4.4 复用：各 rank 读自己的键，apply 自己的矩阵

- KV：按本地 slot mapping 落位（block id 全 rank 一致，映射函数相同）；
- 状态：加载本 rank 的矩阵并 `apply`，仍然是 matmul；
- **不新增跨 rank 通信**。这是本设计最重要的一点：TP 的正确性靠"各管各的"，不靠同步。

### 4.5 调度器不变

block id、segment 划分、seam、窗口都在 scheduler 侧算一次，各 rank 拿到的是**同一份计划**。
所以 `scheduler.py` 的改动只有：把 `tp_size` 透传给 worker（或者干脆让 worker 自己从
`parallel_config` 读，零改动）。

### 4.6 失败与回退

| 情况 | 行为 |
|---|---|
| 部分 rank 未发布 | 不算发布；复用侧 miss → 全量重算 |
| 某 rank 读失败 | 该请求回退全量（沿用现有 `fallback_reasons` 语义），**不 publish** |
| 某 rank apply 后非有限 | 该请求回退全量并记 metric |
| TP 大小与键里的 `size` 不一致 | 直接 miss（相当于缓存不跨 TP 大小共享） |

### 4.7 复制型部署（要确认的边界）

若某种模型在 TP 下**复制**而非切分 KV（例如部分 MLA 实现），则每个 rank 的 KV 相同，
按 rank 分键会浪费 N 倍空间。处理方式二选一：

- 读 `producer_cache_replicated` 类似的判定，复制型只让 rank 0 发布、全体读同一个键；
- 或者先只在"确定按 head 切"的模型（Qwen3.5 GDN 属于此类）上开启，其他一律要求 TP=1。

---

## 5. 需要改动的代码点

| 文件:位置 | 改动 | 规模 |
|---|---|---|
| `mooncake_kv.py:2931` | KV 键加 `/tp-<r>-of-<n>` | 3 行 |
| `mooncake_kv.py:3006` | hybrid-state 键同上 | 3 行 |
| `mooncake_kv.py` manifest 结构 | 加 `tp_size` / `tp_published` | ~30 行 |
| `mooncake_kv.py` 发布入口 | 每 rank 置位 + manifest 落盘时机 | ~30 行 |
| `mooncake_kv.py` 查找入口 | 校验 `tp_published` 全 true | ~15 行 |
| `gpu_model_runner.py` 能力检查 | 允许 TP；未启用 v2 键时**显式拒绝** TP>1 | ~10 行 |
| `config/pic.py` | 可选：`tp_replicated` 开关 | ~5 行 |

**总计约 100 行**，集中在 `mooncake_kv.py`，不触碰状态算子与调度器。

---

## 6. 分阶段落地

| 阶段 | 内容 | 验收 |
|---|---|---|
| **0（先做）** | TP>1 时**显式拒绝**（fail closed），而不是静默错 | 启动即报错 |
| **1** | v2 键 + per-rank 发布/读取，TP=2 | 三形状 9 条判据 + 语义答案；数值与 TP=1 一致（同一 prompt 的输出 token 相同） |
| **2** | manifest 位图与回退 | 人为杀掉一个 rank 的发布 → 必须 miss 而不是错 |
| **3** | 异构 TP（P/D 不同大小）| 复用原生 `_get_tp_ratio` 的切片逻辑；PIC 需要额外把"段在哪个 rank 分片"映射过去 |

---

## 7. 验收标准（阶段 1）

1. `test_all.sh` 三形状全 PASS，语义答案正确（TP=2 与 TP=1 相同）；
2. TP 两个 rank 的 `transition applied` 日志都出现，且 `apply` 的 range 一致；
3. `fallback_reasons={}`；
4. `perf` c=4 吞吐不低于 TP=1 的 0.8×（TP 下每 rank 都要付 import，但前向本身变快）；
5. 杀掉 rank 1 的发布后，TP=2 必须**不复用**（`lookup_hits=0`）而不是出错数值。

---

## 8. Qwen3.5 的切分维度（已确认）

**GDN 的时序状态按 head 切；卷积状态按 channel 切；`head_v_dim` / `head_k_dim` 不切。**

```python
# vllm/model_executor/layers/mamba/mamba_utils.py:229-233
temporal_state_shape = (
    divide(num_v_heads, tp_world_size),   # heads 切成每 rank
    head_v_dim,                            # v 维整块
    head_k_dim,                            # k 维整块
)
```

```python
# vllm/model_executor/layers/mamba/mamba_utils.py:223-227
conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
conv_state_shape = _orient_conv_shape(divide(conv_dim, tp_world_size), ...)
```

图层侧处处是 `num_v_heads // self.tp_size`（`qwen_gdn_linear_attn.py:1100-1101`、`:1418`、
`:1577-1578`），与上面的状态形状一致。

**对本设计的影响**：

| 状态 | 每 rank 形状 | 迁移算子 | 是否精确 |
|---|---|---|---|
| 时序状态 `h` | `[heads/tp, v, k]` | `h @ T_r + C_r`，`T_r=[heads/tp,k,k]` | **是**（按 head 块对角） |
| 卷积状态 | `[tokens, channels/tp]` | 滚动窗口，逐 channel，无通道间耦合 | **是**（depthwise） |

所以第 3 节的公式**直接成立**：`T` 无需广播、无需 gather，`C` 也无需按 v 分段。

### 仍需确认的

1. **全注意力层的 KV**：GQA 下按 `num_kv_heads` 切；若目标是 MLA 类模型需另行确认。
2. **block id 全 rank 一致**：block table 由 scheduler 分配、各 rank 复制（预期一致，实现时加断言）。
3. **D2H 带宽**：TP=2 每 rank 各发一半，**总时间应不变**；若变慢，说明 staging 池被两倍数量的
   对象打散（可参考现有 `slots=4` 预热逻辑）。
