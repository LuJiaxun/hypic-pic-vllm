# 给测试 agent：C1a 验收 + 并发基线（一份请求）

## 0. 这次要交付什么

| 序号 | 任务 | 产出 |
|---|---|---|
| T1 | 打 patch + 静态门 + pytest | 通过/失败 + 失败原因 |
| T2 | 生产档 bench（`PROFILE_IMPORT=0`） | `PERF:` 五行 + gate 计数 |
| T3 | 正确性五层 | 五个字段值 + A/B 输出 |
| T4 | **并发基线探针（本次新增，最重要）** | `/tmp/before.json` 内容 |
| T5 | 单请求完整日志 | 日志文件路径（含 scheduler 的 `[PIC-DEBUG]` 行） |
| T6 | 抢占交叉（`max_num_seqs=1`，同时 2 请求） | 第二个请求是否正常完成 |

T4 是这次的关键：**W1（异步）的收益只有在并发下才量得出来**，而现在 bench 是串行的。
所以需要一次并发基线，作为 W1 的对照。

## 1. Patch

| 项 | 值 |
|---|---|
| 文件 | `hypic-pic-vllm-0221-cu129-stage0-11E-E-C1a-full.patch` |
| MD5 | `1aa593daee5b165151a3883ff79632b5` |
| 大小 | 1,176,447 字节 / 81 文件 |
| 基线 | 上游 `0decac0d96c42b49572498019f0a0e3600f50398`（干净镜像直接打，**不需要先打 C**） |

```bash
cd /usr/local/lib/python3.12/dist-packages
md5sum hypic-pic-vllm-0221-cu129-stage0-11E-E-C1a-full.patch      # 必须是 1aa593da...
patch -p1 --batch --forward --dry-run < hypic-pic-vllm-0221-cu129-stage0-11E-E-C1a-full.patch && echo "DRYRUN OK"
patch -p1 --batch --forward < hypic-pic-vllm-0221-cu129-stage0-11E-E-C1a-full.patch && echo "APPLIED"
find . -name '*.rej' -o -name '*.orig' | head      # 必须为空
```

## 2. T1 静态门 + pytest

```bash
python3 -m compileall -q vllm/v1 tests/v1/pic && echo "compileall OK"
python3 tools/pic_check_unbound.py vllm/v1 tests/v1/pic     # 期望 clean: 307 files
python3 tools/pic_check_removed.py .                        # 期望 clean: 49 removed symbols
test ! -e vllm/v1/pic/transfer_engine.py && echo "TE module gone"
test -f vllm/v1/pic/async_import.py && echo "1a module present"

# 容器里可能残留旧测试缓存，先清
find . -name '*.pyc' -path '*tests/v1/pic*' -delete

python3 -m pytest tests/v1/pic -q
```

**期望：`123 passed`**。逐文件应为：

| 文件 | 用例数 |
|---|---:|
| `test_segmenter.py` | 98 |
| `test_async_import.py` | **9**（本次新增） |
| `test_native_kv_transfer.py` | 7 |
| `test_mooncake_kv.py` | 6 |
| `test_validation.py` | 3 |

**不是 123 就停下来**，把 `pytest --collect-only -q | tail -5` 的结果贴回来。

## 3. T2 生产档 bench

```bash
export MC_STORE_MEMCPY=0
grep -n 'export HYPIC_PIC_' mooncake_bench.sh      # 必须没有写死的 HYPIC_PIC_*
HYPIC_PIC_PROFILE_IMPORT=0 ROUNDS=5 ./mooncake_bench.sh 2>&1 | tee /tmp/c1a_p0.log
grep 'PERF:' /tmp/c1a_p0.log

BLOG=$(grep -o '/tmp/pic_mooncake_[0-9]*/B.log' /tmp/c1a_p0.log | head -1)
echo "PIC-PROFILE=$(grep -c 'PIC-PROFILE' "$BLOG")"     # 必须 0
echo "PIC-MEMORY=$(grep -c 'PIC-MEMORY' "$BLOG")"       # 必须 0
echo "snapshot=$(grep -c 'metrics snapshot' "$BLOG")"   # 必须 >0
grep 'metrics snapshot=' "$BLOG" | tail -1
```

**期望**：

| 指标 | 期望 |
|---|---|
| PIC p50 | 209–213 ms |
| Full p50 | 193–195 ms |
| ratio | 0.91–0.93 |
| `PIC-PROFILE` | 0 |
| `PIC-MEMORY` | 0 |

**必须不出现的记录**（这些通路已删）：

```bash
for k in native-transfer-total native-transfer-submit native-compact-materialize \
         "mode=layer" compat-payload store-gpu-direct; do
  echo "$k -> $(grep -c "$k" "$BLOG" || true)"      # 必须全 0
done
```

## 4. T3 正确性五层

直接用 `metrics snapshot` 的字段核对（N = 本轮请求数；65 请求的 bench 就是 N=65）：

| 字段 | 期望 |
|---|---|
| `external_kv_imported` | = N |
| `lookup_hits` / `lookup_total` | N / N |
| `reused_tokens` | 4352 × N（65 → 282880） |
| `recompute_tokens` | 640 × N（65 → 41600） |
| `forward_tokens` | 5640 × N（65 → 366600） |
| `fallback_reasons` | `{}` |
| `pool.free_bytes` | 首尾相同 |

A/B 输出：`max_tokens=10` 时 PIC 与 Full **逐 token 一致**；A/B 后再看一次 snapshot，
`external_kv_imported` **+1**。

## 5. T4 并发基线（本次新增，最重要）

**目的**：现在 bench 是串行的，W1（通信与 prefill 异步）的收益在串行下恒为 0。
需要一组并发基线，W1 做完后用同一脚本对比。

```bash
export HYPIC_PIC_PROFILE_IMPORT=1        # 探针需要打点来关联
export MC_STORE_MEMCPY=0

python3 tests/v1/pic/run_pic_concurrency_probe.py \
    --base-url http://127.0.0.1:8001 \
    --model <你的模型名> \
    --publish-prompt-file /tmp/publish.txt \
    --reuse-prompt-file /tmp/reuse.txt \
    --concurrency 1,2,4 --requests 20 --warmup 3 --max-tokens 1 \
    --json-out /tmp/before.json

cat /tmp/before.json
```

说明：

- `--publish-prompt-file` / `--reuse-prompt-file` 用你们现有 bench 里那两个 prompt 文件即可
  （第一次写入、后续复用的那对）；
- 探针只发 `/v1/completions`、只量延迟与成功数，**不做正确性判断**；
- 端口按实际服务地址改（PIC 实例是 8001，Full 是 8000）；
- 如果 `max_num_seqs` < 4，`--concurrency 4` 会被截断，请同时把 `max_num_seqs` 值告诉我。

**我需要的就是 `before.json` 的三行 p50**：`concurrency=1 / 2 / 4`。

判读：

| 现象 | 含义 | 对 W1 的意义 |
|---|---|---|
| p50 随并发**基本不变** | 导入已经重叠 | W1 收益小，要先查原因 |
| p50 随并发**线性增长** | 导入完全串行 | W1 收益明确，这就是它的验收基线 |

## 6. T5 单请求完整日志

```bash
HYPIC_PIC_PROFILE_IMPORT=1 ROUNDS=1 ./mooncake_bench.sh 2>&1 | tee /tmp/c1a_p1.log
```

我需要这份日志里的这几类行：

| 行 | 用途 |
|---|---|
| `[PIC-DEBUG] scheduler lookup HIT/MISS ... reused=/recompute=/matches=/external_matches=` | 确认每个请求有几个 external segment |
| `[PIC-PROFILE] runtime-plan` / `native-slot-attach` | 计划构建发生在哪一侧 |
| `external native KV import begin` / `ready` | 导入当前占用的时间窗 |
| `[PIC-PROFILE] forward` | forward 起止 |

**请特别确认**：`[PIC-DEBUG] scheduler ...` 这些行在哪个文件里（B.log 里有没有？没有的话
在哪个日志里）。1b 的改动全在 scheduler 侧，没有这些行我无法验证状态机。

## 7. 回传清单（照这个格式填）

```text
T1 静态门: compileall=OK/FAIL  unbound=OK/FAIL  removed=OK/FAIL  TE-gone=OK/FAIL
T1 pytest : 123 passed / <实际>
T2 PERF   : Round1 PIC=xx Full=xx ratio=xx | ... (五行)
T2 gate   : PIC-PROFILE=0  PIC-MEMORY=0  snapshot=xx
T2 不存在 : native-transfer-*=0  mode=layer=0  compat=0  store-gpu-direct=0
T3 五层   : external_kv_imported=xx reused=xx recompute=xx forward=xx fallback={} pool 首尾=xx/xx
T3 A/B    : 一致/不一致（贴两行输出）
T4 before.json: concurrency=1 p50=xx | 2 p50=xx | 4 p50=xx   (max_num_seqs=xx)
T5 日志   : /tmp/c1a_p1.log（scheduler 的 [PIC-DEBUG] 在 <文件> 里）
T6 抢占   : max_num_seqs=1 下同时发 2 个请求，第二个是否正常完成（不注入故障）
```

## 8. 故障注入是可选的（不必做）

最初列了"删 key 注入"，重新评估后**不是必需项**，原因：三条失败路径里有两条不需要集群级注入。

| 失败路径 | 怎么验 | 需要注入吗 |
|---|---|---|
| 导入失败（store 报错 / 对象缺失 / dtype 不符） | **单测**：用 fake store 在 `pytest` 里抛错，走 request-level fallback | ❌ 不需要 |
| 抢占与等待态交叉 | **`max_num_seqs=1` + 同时发 2 个请求**，让第二个把第一个挤掉 | ❌ 不需要 |
| 请求中途 abort | 客户端超时/断连（`--timeout` 调小即可） | ❌ 不需要 |

集群级注入只在"想确认真实 store 故障下的端到端行为"时才值得做，而单测已经覆盖了同一段代码。
**所以这一轮请跳过注入**；如果后面想补，最省事的做法是给复用实例换一个
`HYPIC_MOONCAKE_KEY_PREFIX`（全 miss，但注意那是"未命中"而不是"导入失败"，两者语义不同）。

## 9. 失败时先看这些

| 现象 | 先查 |
|---|---|
| pytest 不是 123 | 旧 `.pyc` 残留；`--collect-only` 看缺了哪个文件 |
| `env count != 5` | 容器里有旧变量残留（先 `unset` 掉 9 个 TE 变量与 5 个旧 `PROFILE_*`） |
| `PIC-PROFILE` 不为 0 | bench 脚本写死了 `HYPIC_PIC_PROFILE_IMPORT`（第 47 行历史问题） |
| `fallback_reasons` 非空 | 有请求整段重算；贴出 fallback 原因与对应 key |
| 探针跑不起来 | 贴完整 traceback；确认端口与模型名 |
