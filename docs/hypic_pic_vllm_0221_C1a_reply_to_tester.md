# 给测试 agent：C1a 报告的回复（三个问题的答案 + 注入建议）

## 一、你的报告结论我全部接受

| 项 | 结论 |
|---|---|
| T1–T5 | 全部通过，数据已归档进 `stage_plan.md`（commit `83e606943b`） |
| T4 并发基线 | **确认 1b 值得做**：c=1/2/4 = 211.9 / 482.4 / 687.5，吞吐不涨 |
| 调用路径 | 已采纳，**修正了我两个设计假设**（见第三节） |

## 二、三个问题的答案

### Q1：scheduler lookup 的数据源是本地表还是 Store？

**Store。** 代码里是硬读：

```python
# vllm/v1/pic/mooncake_kv.py:2303-2307
external_key = f"segment/{segment.seg_hash.hex()}/group-{int(group_id)}"
try:
    obj = self.object_store.load(external_key)      # 真正的 Store get
except Exception:
    return None
```

所以你的 `remove -> -704 (OBJECT_NOT_FOUND)` 表示**这个 key 当时根本不存在**，删了个空。
HIT 当然不变。

**正确的 key 从这里取**（分析档 `PROFILE_IMPORT=1` 会打印实际读的 key）：

```bash
grep -oE 'native-manifest key=[^ ]+' "$BLOG" | sort -u | head
grep -oE 'transition-manifest key=[^ ]+' "$BLOG" | sort -u | head
```

拿打印出来的那个 key 再 `remove`，应返回 **0**（不是 -704）。

### Q2：A 会不会重新 publish？

**会，而且是无条件的。** 11E-C 删掉了"已是最新就不发"的判断
（`has_current_native_transfer_source`），现在每次 capture 都发：

```python
# gpu_model_runner.py:2907 与 :3045
object_store.publish_native_blocks(...)      # 无 guard
```

这就是你 corrupt 后又被覆盖（`bytes=2368` 而非 8）的原因。

**要注入成功，必须让 A 不再碰那个段**：停掉 A 的流量，再注入，再只打 B。

### Q3：重启 B + 换 `HYPIC_MOONCAKE_KEY_PREFIX` 算不算合适的注入？

**不算。** 那是 lookup **miss**（请求正常重算），而 1b 要验的是"**HIT 之后 import 失败**"
→ request-level fallback。两者走的是完全不同的分支。

**如果确实要做集群注入**，正确做法是"保留 HIT、破坏 import"：

```bash
# 1) 先让 A 停止处理该段（否则会被重新 publish 覆盖）
# 2) 只破坏 layer 对象，不动 manifest（lookup 只读 manifest，HIT 才能保留）
#    manifest 里记录了 nbytes / checksum：
#    - HYPIC_PIC_VERIFY_CHECKSUM=1 时 → checksum 不符 → import 失败
#    - 或把 layer 字节改成错误长度 → 尺寸校验失败
# 3) 打 B，期望：fallback_reasons 记一条、输出仍正确（走整段重算）
```

## 三、你的数据修正了我的两个设计假设（重要）

| 我原来的假设 | 你的实测 | 对 1b 的影响 |
|---|---|---|
| import 在每请求的 execute 块（`gpu_model_runner.py:2078` 附近） | **在 `execute_model.update_states` 的 runtime-plan 里**（70.8 ms，`build_ms=68.056`） | park 必须发生在请求**进入 `update_states` 之前**，否则它已经在阻塞这一步了 |
| 需要按段 park | `segments=2`、`reused=(0,)`、`external_matches=[0]`，每请求 **1 个** external 段 | **请求级 park** 即可 |
| 恢复后可能分多步计算 | `scheduled=4992 computed=0 completed_end=4992` | 恢复后**一次调度 640 token**，不需要多步状态 |

## 四、注入这一轮还要不要做？

**不用做。** 按之前决定：失败路径由单测覆盖（fake store 抛错 → request-level fallback），
比集群注入确定得多。集群注入只在"要确认真实 store 故障的端到端行为"时才值得，留到稳定性阶段。

所以这一轮你不需要再做注入；`before.json` 已经就绪，等 1b/1c 的 patch。

## 五、我接下来要交付什么

| 产物 | 内容 | 验收方式 |
|---|---|---|
| patch 1b | scheduler 侧：开关、park、promote、完成收集 | 关档逐字节一致（你的 T2 数字不变）；开档日志出现 `WAITING_FOR_REMOTE_KVS` |
| patch 1c | worker 侧：对 spec 做**同步**导入并报告完成 | 开档 c=1 仍 211–213；c=2/4 用同一个探针出 `after.json` |
| README | 两条验证命令 + 期望 | — |

预期：`after.json` 的 c=2 从 **482 → ~220**、c=4 从 **687 → ~250–300**；c=1 不变。
