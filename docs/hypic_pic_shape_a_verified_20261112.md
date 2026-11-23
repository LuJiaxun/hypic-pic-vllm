# 形状 A 验收记录（2026-11-12，封板）

**Patch**：`98FC6660102979B4AE6C3EA898D36414`（143 文件 / `+38085 −126`，base vLLM v0.22.1 / `0decac0d`）
**Tag**：`pic-shapeA-verified-2026-11-12`
**环境**：RTX 4090 ×2（A:8000 发布 / B:8001 复用），Qwen3.5-2B，Mooncake 0.3.10.post2

---

## 1. 结论

**通过并封板。** 执行层 100% 命中、语义层正确（复用三段 doc 的 KV 后答出 `dog`）、性能符合预期：

| 场景 | 数值 | 对比基线（224 / 596） |
|---|---|---|
| c=1 稳态 p50 | **247–251 ms** | 慢 ~10%（query 271 token 重算是形状 A 的固有成本） |
| c=4 快簇 p50 | **460 ms**（tput 7.77） | **快 23%** |
| c=4 慢簇 p50 | 601–625 ms | 持平（import 排队，已知） |

## 2. 功能判据（全部通过）

| 判据 | 值 |
|---|---|
| `lookup=hit` | 1/1（单段）、1/1（三段） |
| `reuse_tokens` | 4830 |
| `compute_tokens` | 280（头 279 + 尾 1） |
| `transition applied` | `range=[279,5109)`、`state_position=5108` |
| `segment_spans`（三段） | `[(0,271), (271,5110)]` |
| `fallback_reasons` | `{}` |
| `non-finite` | 0 |

## 3. 语义判据（**唯一正确性判据**，照 sglang）

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.5-2b",
  "messages": [{"role": "user", "content": "<prompt>"}],
  "temperature": 0, "max_tokens": 32,
  "chat_template_kwargs": {"enable_thinking": false}
}'
#   期望：finish=stop，输出含 "dog"
```

**必须用 `/v1/chat/completions` + `enable_thinking: false`**：

- `/v1/completions` **关不掉 `<think>`**，`max_tokens=1` 只会吐出 `<think>`；
- `enable_thinking: false` 是 **chat template 层面的硬开关**（预埋 `<think>\n\n</think>\n\n`），
  不是 system prompt 的弱指令——它让模型走训练时见过的“非 Thinking”路径。

## 4. 性能判据

**必须用 perf 模式（`DEBUG=0`）+ warmup ≥ 20**：

```bash
bash pic_run.sh perf
```

- diag 模式下 c=1 是 266 ms、perf 下 247–251 ms，**diag 慢 ~8%**（日志开销）；
- 冷启动第一轮 c=1 ~403 ms、c=4 ~689 ms 是 **arena 首次 pin + 连接/JIT**，不是稳态。

## 5. fixture 结构（本次验证用）

| 形状 | 结构 |
|---|---|
| 单段 | `query<<PIC_SEP>>doc<<PIC_SEP>>`（**query 在前**，doc 从 271 开始） |
| 三段 | `system<<PIC_SEP>>doc_a<<PIC_SEP>>doc_b<<PIC_SEP>>doc_c<<PIC_SEP>>query` |
| **不要用** | `<<PIC_SEP>>doc<<PIC_SEP>>`（只有 doc，见遗留 ①） |

约束：**每段 doc ≥ 512 token**（`--pic-min-segment-tokens 512`），system ≥ 197 token。

## 6. 遗留问题（不阻塞封板）

### ① `<<SEP>>doc<<SEP>>`（只有 doc、无 query）时 capture 失败

- 现象：`segments=[(0,4839)]`（doc 从 0 开始），`GDN transition validation failed` layer 0–22，
  `capture skipped reason=no_transition_or_exact_boundary`，`native KV published` / `publish-matrix` 不出现。
- 对照：`query<<SEP>>doc<<SEP>>` 时 `segments=[(0,271),(271,5110)]`，验证通过。
- 已确认**不是 fixture 问题**（`publish_0` 未变）也**不是 `PICWindow` 重构**（那次提交没有触碰分段）。
- 待查：`segmenter.split_text_and_tokenize` 的两条路径（预分词快路径 `_split_pretokenized_token_ids`
  与逐段 encode 的慢路径）对“分隔符是否留在 token stream 里”的处理是否一致——
  慢路径 `text.split(separator)` 后分隔符被**完全删除**，于是自身起点为 0；
  快路径保留分隔符 token，于是起点 = 分隔符长度。两条路径给出不同的坐标基准。
- 影响：形状 B（`doc<<SEP>>query`）与“只有 doc”的场景会踩到。

### ② c=4 慢簇（601–625 ms）

import 排队（`HYPIC_PIC_KV_IMPORT_WORKERS=1`；TCP 下一拥而上更慢，已有实测）。
若将来需要，可做“import 与 forward 重叠”（头 range 先算、复用 range 前再 wait），预计 ~620 → ~500 ms；
本项目当前决定保持“传输完成再调度”的简单结构，不做。

### ③ `transition_operator_bytes` 在四个 stage 都是 0

矩阵在 `cleanup_begin` 之前就已释放，`PIC-MEMORY-OWNERS` 覆盖不到峰值。
间接验证：`transition-transfer total_ms=49.3`（38 MB 量级）、`allocated_bytes` 无 1.9 GB 波动、
`compact shape=(16,128,128)`。若要直接观测，需在 publish 之前打点。

## 7. 本轮已实现的性能优化（都在这个 patch 里）

| 优化 | 收益 |
|---|---|
| staging-pool 预热（首次分配就填满 4 个 slot） | `transition-transfer` 111.9 → **49.3 ms**（−62.6） |
| 闭包就地释放（compact 后替换为矩阵） | **1.9 GB → 38 MB/请求** |
| 删除 `capture window` 观察 | 去掉每层 2 次 D2H 同步（18 层 = 36 次） |
| `PICWindow` 收敛块数计算 | 10 处 → 1 处；比较全为“覆盖” |

---

## 8. 封版材料：fixture 与判据（可复现）

以下内容取代“按报告手工重造 fixture”的做法。

### 8.1 生成 fixture

```bash
python3 tools/pic_make_fixtures.py --out /tmp/pic_fixtures \
  --model /data/eval_model/Qwen/qwen3.5-2b
#   会打印每段的 token 数；任何一段低于 --pic-min-segment-tokens(512) 会给出 WARNING
```

生成的文件：

| 文件 | 结构 | 用途 |
|---|---|---|
| `publish_0` / `reuse_0` | `query<<SEP>>doc_b<<SEP>>` | 单段形状 A（已验证） |
| `baseline_0` | `query doc_b`（无分隔符） | off 基线（同 token、不拆段） |
| `pub_a` / `pub_b` / `pub_c` | `system<<SEP>>doc_x<<SEP>>` | 分别发布三段 |
| `reuse_abc` | `system<<SEP>>a<<SEP>>b<<SEP>>c<<SEP>>query` | 三段复用（覆盖重排） |
| `baseline_abc` | 同内容不拆段 | 三段 off 基线 |

**文档内容用自然语言散文（×16 次）**，不是同一句话无限重复——重复的 doc 会让模型
**继续复读**而不是回答问题，把一次正确的复用变成一次失败的答案。

### 8.2 功能判据

```bash
export HYPIC_PIC_DEBUG=1; unset HYPIC_PIC_PROFILE_DEEP VLLM_CUSTOM_SCOPES_FOR_PROFILING
bash pic_run.sh diag

grep -E 'publish-matrix'          /tmp/A.log | tail -1   # span=[8,4838) tokens=4830
grep -E 'native KV published'     /tmp/A.log | tail -1   # blocks=153
grep -c 'GDN transition validation failed' /tmp/A.log    # 0
grep -c 'lookup=hit'              /tmp/B.log             # ≥ 复用次数
grep -E 'transition applied'      /tmp/B.log | tail -1   # range=[279,5109) state_position=5108
grep -E 'fallback_reasons'        /tmp/B.log | tail -1   # {}
grep -cE 'non-finite'             /tmp/B.log             # 0
```

### 8.3 语义判据（唯一正确性判据）

必须 `/v1/chat/completions` + `chat_template_kwargs: {"enable_thinking": false}`：

- `/v1/completions` **关不掉 `<think>`**，`max_tokens=1` 只会吐出 `<think>`；
- 在 system 里写「Do NOT use `<think>`」**无效**（模型不响应弱指令）；
- 原理：chat template 在 assistant 开头预埋 `<think>\n\n</think>\n\n`，让模型走训练时见过的
  “非 Thinking”路径，直接输出答案。

期望：`finish_reason=stop`、`content` 含 `dog`、无 `<think>`。

### 8.4 性能判据

```bash
bash pic_run.sh perf        # DEBUG=0
unset HYPIC_PIC_DEBUG HYPIC_PIC_PROFILE_DEEP VLLM_CUSTOM_SCOPES_FOR_PROFILING
python3 tests/v1/pic/run_pic_concurrency_probe.py \
  --base-url http://127.0.0.1:8001 --model qwen3.5-2b \
  --publish-prompt-file /tmp/pic_fixtures/publish_0.txt \
  --reuse-prompt-file   /tmp/pic_fixtures/reuse_0.txt \
  --concurrency 1,4 --requests 20 --warmup 20 --max-tokens 1 \
  --json-out /tmp/probe_perf_shapeA.json
```

| 指标 | 参考 |
|---|---|
| c=1 p50 / tput | **247–251 ms** / 4.0 |
| c=4 快簇 | **460 ms**（基线 596，快 23%） |
| c=4 慢簇 | 601–625 ms（import 排队，已知） |

**diag 模式的性能数字不可信**（13.9 万条日志，c=1 慢 ~8%）；**冷启动第一轮 403 ms 不是稳态**。

### 8.5 封版清单

| # | 项 | 状态 |
|---|---|---|
| 1 | patch 打上，五个检查器全绿 | ✅ |
| 2 | 功能：`lookup=hit`、`transition applied`、`fallback_reasons={}`、`non-finite=0` | ✅ |
| 3 | 语义：`enable_thinking:false` 下输出 `dog`、`finish=stop` | ✅ |
| 4 | 性能：perf 下 c=1 247–251 ms、c=4 快簇 460 ms | ✅ |
| 5 | 遗留：只有 doc 的 prompt、c=4 慢簇、`transition_operator_bytes` | 不阻塞 |
| 6 | 回退点：`pic-fixed-tput59-2026-11-10` | ✅ |
