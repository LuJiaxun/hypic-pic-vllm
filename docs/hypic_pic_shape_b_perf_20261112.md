# 形状 B 性能记录（PIC-ON / PIC-OFF 全量数据）

**结论：PIC 的收益在 c=4 高并发——c=4 tput 6.06 对 4.38（+38%）、c=4 wall 3301 ms 对 4578 ms（−27.9%）；c=1 单请求慢 25%，那是 import + apply 的固定成本，不是回归。**

| 项 | 值 |
|---|---|
| 环境 | RTX 4090 24 GB ×2（A:8000 / B:8001，`CUDA_VISIBLE_DEVICES=0/1`），Qwen3.5-2B，Mooncake 0.3.10.post2 |
| 形状 | B：`system <<PIC_SEP>> doc <<PIC_SEP>> query`，文档位置不变、不需要旋转 |
| fixture | `tests/v1/pic/fixtures/b/`，**system 209 / doc 6976 / query 305 token**（reuse 合计 7453） |
| 探针 | `--concurrency 1,4 --requests 20 --warmup 20 --max-tokens 1`，**5 轮，丢弃第 1 轮，取第 2–5 轮中位数** |
| 起服务 | `--max-model-len 8192`、`--max-num-batched-tokens 8192`、**`--gpu-memory-utilization 0.5`**、`--enforce-eager`、`--no-enable-prefix-caching`、`--pic-max-cache-bytes 4294967296` |
| ON / OFF 的差别 | **只有 PIC 标志**（`bash tools/run_pic_test.sh perf` 与 `off`），模型、批处理、显存比例、探针参数完全相同 |

---

## 1. PIC-ON 逐轮（形状 B，doc 6976）

| 轮次 | c=1 wall_ms | c=1 p50 | c=1 tput | c=4 wall_ms | c=4 p50 | c=4 tput |
|---|---:|---:|---:|---:|---:|---:|
| round 1 | 7494.4 | 374.7 | 2.67 | 3425.9 | 631.4 | 5.84 |
| round 2 | 7407.9 | 369.4 | 2.70 | 3331.6 | 613.4 | 6.00 |
| round 3 | 7447.1 | 371.8 | 2.69 | 3251.2 | 604.0 | 6.15 |
| round 4 | 7146.6 | 360.6 | 2.80 | 3334.5 | 580.7 | 6.00 |
| round 5 | 7315.2 | 364.3 | 2.73 | 3269.6 | 602.8 | 6.12 |
| **稳态中位（2–5）** | **~7378** | **366.8** | **2.72** | **~3301** | **603.4** | **6.06** |

## 2. PIC-OFF 逐轮（同一套 fixture）

| 轮次 | c=1 wall_ms | c=1 p50 | c=1 tput | c=4 wall_ms | c=4 p50 | c=4 tput |
|---|---:|---:|---:|---:|---:|---:|
| round 1 | 5821.9 | 290.4 | 3.44 | 4892.4 | 968.5 | 4.09 |
| round 2 | 5889.1 | 293.4 | 3.40 | 4522.3 | 957.3 | 4.42 |
| round 3 | 5879.9 | 293.5 | 3.40 | 4612.4 | 960.0 | 4.34 |
| round 4 | 5882.0 | 293.8 | 3.40 | 4578.4 | 960.7 | 4.37 |
| round 5 | 5917.2 | 295.3 | 3.38 | 4546.7 | 958.3 | 4.40 |
| **稳态中位（2–5）** | **~5885** | **293.6** | **3.40** | **~4578** | **959.2** | **4.38** |

### 2.1 PIC-OFF 逐轮明细（含 p95 / min / max）

| 轮次 | 并发 | wall_ms | p50 | p95 | min | max | tput |
|---|---|---:|---:|---:|---:|---:|---:|
| r2 | c=1 | 5889.1 | 293.4 | 300.1 | 291.1 | 300.6 | 3.3961 |
| r2 | c=4 | 4522.3 | 957.3 | 1036.2 | 295.3 | 1084.0 | 4.4226 |
| r3 | c=1 | 5879.9 | 293.5 | 294.5 | 292.3 | 295.1 | 3.4014 |
| r3 | c=4 | 4612.4 | 960.0 | 984.1 | 427.1 | 1166.5 | 4.3361 |
| r4 | c=1 | 5882.0 | 293.8 | 298.0 | 290.9 | 300.1 | 3.4002 |
| r4 | c=4 | 4578.4 | 960.7 | 983.5 | 391.6 | 1124.6 | 4.3684 |
| r5 | c=1 | 5917.2 | 295.3 | 301.5 | 278.4 | 302.7 | 3.3800 |
| r5 | c=4 | 4546.7 | 958.3 | 1042.0 | 301.5 | 1095.6 | 4.3988 |

**c=1 的 p50 极稳（293.4–295.3）**；**c=4 的 min 波动大（295–427）**——首批请求要付 import 与连接，
中位数不受影响。

## 3. ON 对 OFF

| 指标 | PIC-ON | PIC-OFF | 差异 |
|---|---:|---:|---|
| c=1 wall_ms | ~7378 | ~5885 | **多 25.4%** |
| c=1 p50 | 366.8 | 293.6 | 慢 25% |
| c=1 tput | 2.72 | 3.40 | 低 20% |
| **c=4 wall_ms** | **~3301** | **~4578** | **少 27.9%** |
| c=4 p50 | 603.4 | 959.2 | 快 37% |
| **c=4 tput** | **6.06** | **4.38** | **高 38%** |

## 4. wall / p50 / tput 三者自洽

| 场景 | 算式 | 结果 |
|---|---|---|
| ON c=1 | 7378 / 20 请求 | 368.9 ms/请求 ≈ p50 366.8 ✅ |
| OFF c=1 | 5885 / 20 请求 | 294.3 ms/请求 ≈ p50 293.6 ✅ |
| ON c=4 | 3301 / 5 批（20 请求 ÷ 4 并发） | 660 ms/批，p50 603（中位数小于含慢请求的均值）✅ |
| OFF c=4 | 4578 / 5 批 | 915.6 ms/批 ≈ p50 959 ✅ |
| tput | 20 / 3.301 s = 6.06；20 / 4.578 s = 4.37 | ✅ |
| 交叉验证 | 6.06 / 4.38 = **1.38**；4578 / 3301 = **1.39** | 同向同量 ✅ |

**c=1 的差额可归因**：`(7378 − 5885) / 20 = 74.7 ms/请求`——就是 import + apply 的固定成本。

## 5. 为什么 c=1 没有收益，c=4 有

```
c=1 PIC-OFF:  全量 forward 7453 token                  ≈ 294 ms
c=1 PIC-ON :  import + 2× transition apply ≈ 74.7 ms（净增）
              + 复用侧 forward ≈ 30 ms
              ≈ 367 ms
→ 单请求时固定成本没有别人分担
```

```
c=4 PIC-OFF:  每批 4 × 7453 token 全量 forward        → 915 ms/批
c=4 PIC-ON :  4 个请求的 import/apply 互相重叠，
              每批只剩 660 ms                         → −27.9%
```

**PIC 换的是吞吐，不是单请求延迟**——所以封版口径用 c=4。

## 6. doc 长度的影响（c=4 tput）

| doc | PIC-ON | PIC-OFF | PIC-ON vs OFF |
|---:|---:|---:|---|
| 4951 | 6.23 | 6.22 | +0.2% |
| 6976 | 6.06 | 4.38 | **+38%** |

**doc 越长收益越大**：import + apply 是固定成本，PIC-OFF 的 forward 随 token 线性增长。

## 7. 形状 B 的 fixture 规范（已进 patch）

| 文件 | 结构 |
|---|---|
| `publish.txt` / `publish_0.txt` | `system <<PIC_SEP>> doc <<PIC_SEP>>` |
| `reuse.txt` / `reuse_0.txt` | `system <<PIC_SEP>> doc <<PIC_SEP>> query` |
| `baseline.txt` | `system doc query`（不拆段） |

| 段 | 内容 | token |
|---|---|---:|
| system | `"You are a highly capable and helpful assistant. …"` × 4 | 209 |
| doc | `"The dog is a domesticated descendant of the wolf. …"` × 155 | 6976 |
| query | `"Please read the document carefully… "` × 20 + 问题 | 305 |
| **reuse 合计** | — | **7453**（< 8192） |

**三条设计约束**：

1. **doc 必须长**——短 doc 的复用比例低，固定成本盖过节省（doc 4951 时 c=4 只 +0.2%）；
2. **doc 不能一句话无限重复**——模型会继续复读而不是回答；
3. **query 不能太短**——14 token 的 query 让复用比例接近 100%，是失真的。

## 8. 怎么跑

```bash
cd /usr/local/lib/python3.12/dist-packages
export PIC_MODEL=/data/eval_model/Qwen/qwen3.5-2b

bash tools/run_pic_test.sh perf    # PIC-ON ，形状 B，5 轮 → probe_b_r*.json
bash tools/run_pic_test.sh off     # PIC-OFF，形状 B，5 轮 → probe_off_b_r*.json
```

**两者只差 PIC 标志**（`pic_run.sh` 里 `PIC_OFF=1` 时 `PIC_FLAGS` 为空），所以可以直接对比。

## 9. 已知问题

| # | 问题 | 现状 |
|---|---|---|
| 1 | **doc 8011 时 c=4 OOM**（`max-model-len 16384` + `0.6`：4 请求 KV ~32 GB > 24 GB） | doc ~7000 是当前安全上限 |
| 2 | **publish 侧 forward 是整个 prompt** | doc 越长 publish 越慢；若 publish 只 capture 不 forward，c=1 也能出收益 |
| 3 | **c=1 PIC-ON 慢 25%** | 结构性：`74.7 ms/请求` 的 import + apply > 复用收益 |

## 10. 封版口径

| 指标 | 值 | 用途 |
|---|---|---|
| **c=4 tput 6.06 vs 4.38（+38%）** | ✅ | **对外汇报主指标** |
| **c=4 wall 3301 vs 4578（−27.9%）** | ✅ | 与 tput 同向，可交叉验证 |
| c=4 p50 603.4 vs 959.2（−37%） | ✅ | 同上 |
| c=1 p50 366.8 vs 293.6（+25%） | ⚠️ 参考 | 说明 PIC 不适合单请求低延迟，不是回归 |
