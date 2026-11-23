# PIC 验收 fixture（固定文本，已入库）

**17 个文本文件，共 354 KB。** 这些是**生成好的文本**，新人不需要生成、不需要 copy、不需要改路径。

## 三种形状

| 目录 | 结构 | 用途 | 规模（token） |
|---|---|---|---|
| `a/` | `query <<PIC_SEP>> doc <<PIC_SEP>>` | 文档在 query **之后**，复用要**重旋转**；段末 1 token 重算 | doc ~4950 |
| **`b/`** | `system <<PIC_SEP>> doc <<PIC_SEP>> query` | **性能测试用这个**：文档位置不变、不需要旋转 | **system ~200 / doc ~6970 / query ~305** |
| `c/` | `system <<PIC_SEP>> a <<PIC_SEP>> b <<PIC_SEP>> c <<PIC_SEP>> query` | 三段分别发布后一次请求同时命中（sglang 测法，验语义 `dog`） | 每段 ~700 |

**b 为什么这么长**：复用比例决定测不测得出收益——doc 太短时 `import + apply` 的固定成本盖过省下的
forward（doc 4951 时 c=4 只 +0.2%），doc ~7000 时 c=4 tput **+29.7%**。query 也不能太短，
14 token 的 query 让复用比例接近 100%，是失真的。


每个形状都有一份 `baseline.txt`：**同样的内容但不拆段**，用来看“同样的 token 不走 PIC 要多久”。

## 为什么有 `publish_0.txt` / `reuse_0.txt`

| 文件 | 用途 |
|---|---|
| `publish.txt` / `reuse.txt` | 给人读的名字 |
| **`publish_0.txt` / `reuse_0.txt`** | **与上面逐字节相同**，是 `tools/pic_run.sh` 与 `tools/run_pic_test.sh` 读的名字 |

**两套名字都入库，是为了让一次运行不依赖任何“准备步骤”**——`git apply` 之后文件就在，脚本直接读。

## 一键跑

```bash
# 脚本所在目录的上一级就是仓库根，即包含 vllm/ 包的那一层（安装后是 dist-packages）
cd /usr/local/lib/python3.12/dist-packages
export PIC_MODEL=/path/to/Qwen3.5-2B

bash tools/run_pic_test.sh all      # 一条命令跑完（diag + perf），结束时自动停服务

# 分步
bash tools/run_pic_test.sh diag    # a/b/c 执行层 9 条判据 + 语义答案
bash tools/run_pic_test.sh perf    # 形状 b，PIC 开，5 轮稳态（丢弃第 1 轮，取 2–5 轮中位数）
bash tools/run_pic_test.sh off     # 形状 b，PIC 关，同 5 轮 —— 与 perf 同一套 fixture，直接对比
bash tools/run_pic_test.sh trace   # 形状 b 的 c=1 / c=4 trace
```

**日志、trace 与 mooncake 配置都落在仓库内**（`tests/v1/pic/{logs,trace}`，均已 `.gitignore`）：

- 日志与 trace 相对仓库根，不再指向任何机器的 home 目录；
- **`mooncake_config.json` 由 `pic_run.sh` 自动生成**——它的 `local_hostname` 与
  `master_server_address` 是本机地址，入库换台机器就是错的，所以脚本用 `hostname -I` 现取现写；
  要自定义路径或内容时用 `PIC_MOONCAKE` 覆盖。

只需要一个必填环境变量：

```bash
export PIC_MODEL=/path/to/Qwen3.5-2B                # 必填，没有默认值
export PIC_MOONCAKE=/path/to/mooncake_config.json   # 可选（机器配置，不入库）
```

**`PIC_MODEL` 故意没有默认值**：写死一个别人机器上的路径，跑挂时错误会出现在 vLLM 的加载日志里，
而不是“你忘了设变量”这一行。

起服务用 `--gpu-memory-utilization 0.5`（doc ~7000 的 KV 约 10 GB，0.35 的 8.4 GB 不够；
0.6 在 c=4 可能 OOM）。这个值写死在 `tools/pic_run.sh` 里。

其它可选：`PIC_DEVICES`（默认 `0,1`）、`PIC_A_PORT`/`PIC_B_PORT`（8000/8001）、
`PIC_LOG_DIR`（`/tmp/pic_test_logs`）、`PIC_TRACE_DIR`（`/tmp/pic_trace`）、
`PIC_MAX_MODEL_LEN`（8192）、`PIC_MIN_SEGMENT_TOKENS`（512）。


## 三个必须知道的点

1. **段必须 ≥ 512 token**（`--pic-min-segment-tokens`）。低于阈值的段会被调度器**静默跳过**，
   表现是“lookup miss”，看起来像 PIC 的 bug。这些 fixture 的每一段都 ≥700 token。
2. **语义判据必须用 `/v1/chat/completions` + `chat_template_kwargs: {"enable_thinking": false}`**。
   `/v1/completions` 关不掉 `<think>`，`max_tokens=1` 只会吐出 `<think>`；
   在 system 里写“不要用 think”也无效（模型不响应弱指令）。
3. **性能只认 perf 模式 + warmup ≥ 20**：diag 慢 ~8%（日志开销），冷启动第一轮还要付
   pinned arena 分配与连接建立。

## 改动这些文件时

`tests/v1/pic/test_fixture_shapes.py` 会检查磁盘上的文件与 `tools/pic_fixture_shapes.py`
里定义的形状**逐字节一致**——所以改文本要同时改生成器，或者重新生成：

```bash
python3 tools/pic_make_fixtures.py --out tests/v1/pic/fixtures
```

这个检查存在的理由：文本与生成器一旦分叉，验收跑的就是没人维护的那一份。
