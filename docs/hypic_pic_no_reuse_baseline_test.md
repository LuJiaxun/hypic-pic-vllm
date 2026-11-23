# PIC 不复用基线（PIC-OFF）测试手册 —— 昇腾 NPU

这份文档回答一个问题：**在同一个模型、同一批请求、同一份 fixture 下，"不复用"（PIC-OFF）到底要花多少时间**，
从而把"复用带来的收益"算成一个可以引用的比值，而不是一句"感觉快了"。

不复用基线本身不是收益，它是对照组。收益 = `PIC-OFF 指标 ÷ PIC-ON 指标`。

---

## 1. 两个被测配置只差一件事

| 配置 | 启动参数差异 | 语义 |
|---|---|---|
| **PIC-OFF（不复用，基线）** | **不带任何 `--pic-*` 标志** | 每个请求都完整重算 prompt，不发布段、不查 Store |
| **PIC-ON（复用）** | `--pic-enable --pic-max-cache-bytes 4294967296 --pic-min-segment-tokens 512` | 段可发布/命中，命中段不重算 |

其余必须完全一致，否则两组不可比：

- 模型路径、`--max-model-len`、`--max-num-batched-tokens`
- `--gpu-memory-utilization`（NPU 上为显存/NPU 内存比例）
- `--no-enable-prefix-caching`（两组都关；本地前缀缓存会污染"复用"的归属）
- 探针参数：并发、请求数、warmup、`max_tokens`
- 机器状态：同一台机器、同一轮实验内连续跑完

实现上这两组已经由现有脚本固定：

```text
tools/pic_run.sh  第 127-140 行：PIC_OFF=1 时 PIC_FLAGS 为空，否则加 PIC 三个标志
tools/run_pic_test.sh           ：perf 模式 = PIC-ON，off 模式 = PIC-OFF
```

---

## 2. 前置条件

1. 已打好 PIC patch，且 `python3 -c "import vllm"` 在本机可用；
2. Mooncake master 可启动（PIC-ON 需要；PIC-OFF 不需要，但脚本会一起管）；
3. 模型与 fixture 存在：

```bash
ls "$PIC_MODEL"                                  # 模型目录
ls tests/v1/pic/fixtures/b/                      # publish.txt / reuse.txt / baseline.txt
ls tests/v1/pic/fixtures/c/                      # publish_a/b/c.txt / reuse.txt
```

4. 记录环境（写进报告，否则数字无法复现）：

```bash
npu-smi info | head -20
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
free -g | head -2
```

---

## 3. 第一步：跑 PIC-OFF 基线

```bash
cd /usr/local/lib/python3.12/dist-packages     # 含 vllm/ 的那一层

export PIC_MODEL=/path/to/Qwen3.5-2B
export PIC_LOG_DIR=$PWD/tests/v1/pic/logs
mkdir -p "$PIC_LOG_DIR"

# 5 轮，脚本自己丢弃第 1 轮、对第 2-5 轮取中位数
bash tools/run_pic_test.sh off 2>&1 | tee /tmp/pic_off.txt
```

脚本实际执行的探针（形状 B）：

```text
tests/v1/pic/run_pic_concurrency_probe.py \
  --base-url http://127.0.0.1:8001 --model qwen3.5-2b \
  --publish-prompt-file tests/v1/pic/fixtures/b/publish.txt \
  --reuse-prompt-file   tests/v1/pic/fixtures/b/reuse.txt \
  --concurrency 1,4 --requests 20 --warmup 20 --max-tokens 1
```

产物：

```bash
ls "$PIC_LOG_DIR"/probe_off_b_r*.json    # 5 个 JSON，逐轮明细
grep -A3 'PIC-OFF 形状 B 稳态' /tmp/pic_off.txt
```

期望输出形如：

```text
===== PIC-OFF 形状 B 稳态（丢弃第 1 轮，取第 2–5 轮中位数）=====
  c=1: p50 median=... ms  tput median=...
      p50 rounds 2-5: [...]
  c=4: p50 median=... ms  tput median=...
      p50 rounds 2-5: [...]
```

---

## 4. 第二步：确认"真的没复用"

不做这一步，基线可能在偷偷复用（例如本地前缀缓存被打开），收益会凭空缩小。

```bash
OFF_LOG="$PIC_LOG_DIR/B.log"      # pic_run.sh 固定写 $LOG_DIR/{A,B}.log
# 注意：PIC 关闭时指标快照可能仍打印 'lookup_hits': 0，所以这里只匹配非零值
grep -E "'lookup_hits': [1-9]|'matched_segments': [1-9]|'reused_tokens': [1-9]" "$OFF_LOG" | tail -3
grep -cE 'publish-matrix|native KV published|hybrid transition published' "$OFF_LOG"
```

两条命令都应输出空 / `0`。

PIC-OFF 的正确形态是：

| 检查 | PIC-OFF 期望 |
|---|---|
| 段发布日志（`publish-matrix` / `native KV published`） | **0 条** |
| `lookup_hits` / `matched_segments` / `reused_tokens` | **没有非零值** |
| 每请求实际 prefill token 数 | ≈ 完整 prompt 长度（无省略） |

如果 PIC-OFF 里出现发布或命中，先停：说明 `PIC_OFF=1` 没生效或被环境变量覆盖，基线不可用。

---

## 5. 第三步：跑 PIC-ON（同一轮实验内紧接着跑）

```bash
bash tools/run_pic_test.sh perf 2>&1 | tee /tmp/pic_on.txt
ls "$PIC_LOG_DIR"/probe_b_r*.json
grep -A3 '形状 B 稳态' /tmp/pic_on.txt
```

确认复用真的发生：

```bash
ON_LOG="$PIC_LOG_DIR/B.log"
grep -E "'lookup_hits': [1-9]" "$ON_LOG" | tail -2
grep -E "'fallback_reasons': \{\}" "$ON_LOG" | tail -2
grep -cE 'publish-matrix|native KV published' "$PIC_LOG_DIR/A.log"
```

第一、三条应有输出（命中与发布都发生），第二条确认没有走到回退路径。

---

## 6. 第四步：算收益

### 6.1 主指标

| 指标 | 取值来源 | 收益公式 |
|---|---|---|
| 单请求延迟 p50 | 稳态中位数 `c=1.p50` | **加速比 = OFF_p50 ÷ ON_p50** |
| 吞吐 | 稳态中位数 `c=4.tput` | **提升比 = ON_tput ÷ OFF_tput** |
| 尾延迟 p95 | 逐轮 JSON 的 `p95_ms` | 看是否退化（复用不应显著抬高 p95） |

### 6.2 填这张表

| 并发 | PIC-OFF p50 | PIC-ON p50 | 加速比 | PIC-OFF tput | PIC-ON tput | 吞吐提升 |
|---|---|---|---|---|---|---|
| c=1 |  |  |  |  |  |  |
| c=4 |  |  |  |  |  |  |

### 6.3 判定标准

| 结论 | 条件 |
|---|---|
| 复用有效 | `c=4` 吞吐提升 **≥ 10%**，且 `c=1` 加速比 > 1 |
| 复用基本无效 | 提升在 ±5% 内 |
| 复用有负收益 | 吞吐下降 > 5%，或 p95 明显变差（通常是导入串行化） |

> 参考：vLLM 0.22.1 / CUDA 上形状 B 的历史数据是 `c=4` 吞吐 **1.34–1.38×**、`c=1` 加速比约 **1.36**。
> 那组数字只用于判断量级是否合理，**不能直接当作昇腾的预期值**。

---

## 7. 昇腾环境专项注意

| 事项 | 做法 |
|---|---|
| 设备指定 | 用 `ASCEND_RT_VISIBLE_DEVICES`，不要沿用 `CUDA_VISIBLE_DEVICES` |
| 图模式 / 编译 | 两组必须一致；建议都加 `--enforce-eager` 排除算子编译差异 |
| 显存比例 | 两组一致，且给 PIC 池留出空间（PIC-ON 额外有 4 GiB 池预算） |
| 传输 | PIC 依赖 Mooncake Store，测试机通常只能 `"protocol": "tcp"`；**直连/零拷贝收益不要在此环境下评估** |
| 首轮 | 一定丢弃：NPU 首次编译、显存分配、Store 连接都会让第 1 轮偏高 |
| 温度/频率 | 长实验前后各看一次 `npu-smi info`，降频会让后跑的组吃亏 |

如果两组之间必须重启服务（本流程就是如此），请**交替顺序各跑一次**（OFF → ON → OFF → ON），
取两次同配置的中位数，抵消"后跑更热/更慢"的系统性偏差。

---

## 8. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| PIC-OFF 里也出现 `lookup_hits` | `PIC_OFF=1` 未生效 | 确认 `bash tools/pic_run.sh` 打印了 `PIC_OFF=1: servers start without any PIC flag` |
| 两组 p50 几乎一样 | 请求没有复用机会（每次 fixture 不同 / 段太短被 `pic-min-segment-tokens 512` 过滤） | 用 `fixtures/b` 的 publish→reuse 组合；必要时下调 min-segment 并记录 |
| PIC-ON 反而更慢 | 导入串行、Store 传输成为瓶颈 | 看 p50/p95 与 `KV_IMPORT_WORKERS`；先确认 Store 命中确实发生 |
| 数字波动大 | 单轮噪声、机器有其他负载 | 至少 5 轮丢首轮；两次交替实验取中位数 |

---

## 9. 报告模板

```text
环境
  机器 / NPU 型号：
  驱动 / CANN / torch_npu 版本：
  模型：Qwen3.5-2B
  Mooncake：0.3.10.post2，protocol=tcp
  命令：bash tools/run_pic_test.sh off ；bash tools/run_pic_test.sh perf

不复用基线（PIC-OFF，稳态中位数）
  c=1  p50 = ___ ms   tput = ___
  c=4  p50 = ___ ms   tput = ___
  复用确认：lookup_hits=0、无发布日志  ✅

复用（PIC-ON，稳态中位数）
  c=1  p50 = ___ ms   tput = ___
  c=4  p50 = ___ ms   tput = ___
  命中确认：lookup_hits>0、fallback_reasons={}  ✅

收益
  c=1 加速比 = ___    c=4 吞吐提升 = ___ %
  p95 变化 = ___

结论：复用有效 / 基本无效 / 有负收益
```

---

## 10. 一段话

> **"不复用基线"就是 `PIC_OFF=1` 跑 `tools/run_pic_test.sh off`**：服务端不带任何 PIC 标志，每个请求
> 完整重算 prompt。跑完先确认日志里**没有**发布与命中（否则基线被污染），再在同一轮实验里跑
> `tools/run_pic_test.sh perf` 得到 PIC-ON，最后用 `OFF p50 ÷ ON p50` 和 `ON tput ÷ OFF tput` 两个比值
> 表达收益。**两组只允许差 PIC 标志**；都关本地前缀缓存、都 discard 第 1 轮、取 2–5 轮中位数，
> 最好交替跑两遍消除机器状态偏差。
