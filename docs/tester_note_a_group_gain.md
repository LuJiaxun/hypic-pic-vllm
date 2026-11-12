# 给测试 agent：A 组收益测量（修正测试方法后）

**目标**：在**真实负载形态**下测 W1 的收益。前一轮 A 组（86782EE9）p50 743.8 ms 的失败，主因是
测试构造，不是代码——这一轮按下面的规范重造 prompt 和测量方式。

---

## 零、重要更新（2026-10-28）：不是小段问题，是对齐

上一轮把 A 组回退归因为"34 token 小段"是**误判**。实际原因是：

```
doc_i = 4839 token
KV 按 allocator block 发布（该模型 544 token/块）→ 只覆盖 floor(4839/544)*544 = 4352
复用窗口末端 = KV 覆盖末端        → 4352
发布 span 末端 = 段长（旧实现）    → 4839
窗口(0,4352) ≠ span(0,4839) → 差 487 → 需要切片 → 压缩算子不可切 → 回退
（每请求一次，40 次 = 2×20）
```

注意：不是 kernel block（32），而是 **allocator block（544）**——混合模型的 attention page
被撑到覆盖 mamba page，所以发布粒度是 544。

**几乎每个真实段都会中招**（长度不是 544 的倍数），与段大小无关。
`--pic-min-segment-tokens 512` 解决不了它。

已修（patch **`0630FB876640CD08698AF9586287C378`**）：发布 span 与 KV 覆盖统一按 allocator block（544）对齐，两者都落在 4352。

## 一、用哪个 patch

| 项 | 值 |
|---|---|
| **推荐** | **`0630FB876640CD08698AF9586287C378`**（含 `--pic-min-segment-tokens` 与 allocator 对齐修复） |
| 备选 | `86782EE9` / `D0FA3015`（能跑，但 A 组仍会因 487 token 差异回退） |
| pytest | 新 patch 期望 **164 passed** |

如果打新 patch，先核对指纹（README 的"打完后的指纹"表已随重建更新）。

---

## 二、上一轮为什么失败（务必避免）

| 问题 | 上一轮做法 | 后果 |
|---|---|---|
| **复用段太小** | `<SEP>prefix(4304)<SEP><SEP>side(34)<SEP>` | 34 token 段经 seam(8)+块对齐(32) 后只剩 24 token，窗口≠span → 23 次回退 |
| **publish 请求计入了 p50** | publish（冷启动、无复用）与 reuse 混在一起统计 | p50 被冷请求拉高，测不出复用收益 |
| **前缀不唯一** | `unique_marker` 未真正改变 token | 6/20 撞 hash |

---

## 三、prompt 规范（关键）

### 复用段必须 ≥ 4096 token

```text
publish 请求： <SEP> doc_i <SEP>
reuse   请求： <SEP> doc_i <SEP> query_i
               └─ 复用段 ─┘      └ 外部文本，不可复用 ┘
```

| 要求 | 说明 |
|---|---|
| `doc_i` 长度 | **4096–5000 token**（用 tokenizer 精确控制，打印实际值） |
| `doc_i` 在 publish 与 reuse 中**逐字符相同** | 否则 hash 不同，不会命中 |
| `query_i` | 每请求不同，100–300 token，**写在 SEP 之外** |
| 20 组 `doc_i` 互不相同 | 开头放 ≥64 token 的随机文本（不要用短 marker，可能被模板吞掉） |

### 自检（必做）

```python
# 打印 20 个 doc 的段 hash，必须 20 个互不相同
print(seg_hash[:8] for each doc)
```

---

## 四、测量方法（关键：分两阶段）

```text
阶段 1（预热，不计时）
  依次发送 20 个 publish 请求（<SEP>doc_i<SEP>），等全部完成
  → 此时 20 个 doc 已发布到 Mooncake

阶段 2（计时，只测这一阶段）
  并发 c=2 发送 20 个 reuse 请求（<SEP>doc_i<SEP> query_i，query 各不相同）
  → p50 / tput 只统计这一阶段的 20 个请求
```

**绝对不要把 publish 请求算进 p50**——它们是冷启动，没有任何复用。

关档基线用**同一阶段、同一批 reuse 请求**，只是不带 `--pic-async-import`（或关掉 PIC）。

---

## 五、启动参数

```bash
export PIC_ASYNC_FLAG="--pic-async-import"
# 若已打 D0FA3015，额外加：
#   --pic-min-segment-tokens 512
# 保持默认：--pic-seam-sink 8
```

另需**清一次 Mooncake 旧数据**（旧条目没有 `span_start` 字段，读回会按整段处理 → 可能回退）。

---

## 六、要交的数据

| 项 | 怎么取 |
|---|---|
| 阶段 2 的 success/failed | 探针输出 |
| 阶段 2 的 p50 / p95 / tput | 探针输出（**只 reuse 阶段**） |
| 关档同阶段 p50 / tput | 同上，关 PIC |
| `reuse body skipped` | `grep -c` |
| `slicing is unsupported` | `grep -c`，期望 **0** |
| `outside the published span` | `grep -c`，期望 **0** |
| `PIC transition range` 相关 | `grep -c`，期望 **0** |
| `publish trims blocks` | `grep -c`，记录次数（Bug1 线索） |
| `EngineDead` / `Traceback` | 期望 **0** |

---

## 七、判读

| 现象 | 结论 |
|---|---|
| 阶段 2 p50 **显著低于**关档（如 <300 ms） | ✅ W1 在真实负载下成立 |
| 阶段 2 p50 ≈ 关档 | ⚠️ 复用没生效：查 `reuse body skipped` 是否 ≈ 20 |
| `slicing is unsupported > 0` | ❌ 还有小段或窗口不匹配，把该行上下文贴回来 |
| `reuse body skipped` 明显 < 20 | 查 prompt 是否逐字符一致、doc 是否 ≥4096 |
| tput 低于关档 | 说明并发下互相干扰，需单独分析（不要和 p50 混谈） |

---

## 八、回报模板

```text
patch: D0FA3015（或 86782EE9）
pytest: __ passed / __ failed

doc 长度: min=__ max=__ token（20 组）
20 个 seg_hash 互不相同: 是/否

阶段 1 publish: 20/20 完成
阶段 2 reuse (c=2): success=__/20
  开档 p50=__ ms  p95=__ ms  tput=__ rps
  关档 p50=__ ms            tput=__ rps
  加速: __×

reuse body skipped: __
slicing is unsupported: __
outside the published span: __
publish trims blocks: __
EngineDead / Traceback: __ / __
```

---

## 九、两个不要误判的点

1. `no_result` 日志里有 `skip park reason=last_runnable` 的请求，属设计内，**不要当 bug 报**。
2. `slicing is unsupported` 出现只表示"该请求受控回退"，不崩；但**本轮应为 0**，
   非零就说明还有小段/不匹配，需要修。
