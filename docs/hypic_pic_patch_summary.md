# PIC patch 说明（vLLM 0.22.1 / cu129）

**Patch**：`hypic-pic-vllm-0221-cu129-stage0-11E-E-C1c-full.patch`
**Base**：`0decac0d96c42b49572498019f0a0e3600f50398`（= vLLM `v0.22.1`，干净镜像直接打）
**规模**：171 文件 / `+39757 −126`；代码侧 152 文件 / `+31419 −126`（`vllm` + `tools` + `tests`）

---

## 1. 这个 patch 做什么

在 vLLM 0.22.1 上实现 **PIC（位置无关缓存）**：把 prompt 按 `<<PIC_SEP>>` 切成段，段的 **KV** 与
**GDN 递推状态** 可在**不同请求**间复用，即使该段在新请求里**不在原来的位置**——办法是把缓存的 KV
**重旋转**到新位置，并把递推状态压成**一个线性算子**（`state @ M + c`）搬过去。

### 形状

| 形状 | 结构 | 复用时 |
|---|---|---|
| **A** | `query <<SEP>> doc <<SEP>>` | doc 位置变化 → 需要重旋转；头 query 与段末 1 token 重算 |
| **B**（计时口径） | `system <<SEP>> doc <<SEP>> query` | doc 位置不变 → 不旋转；只有 query 重算 |
| **C** | `system <<SEP>> a <<SEP>> b <<SEP>> c <<SEP>> query` | 三段分别发布，一次请求命中已见过的段 |

---

## 2. 核心模块（新增）

| 文件 | 行数 | 职责 |
|---|---:|---|
| `vllm/v1/worker/gpu_model_runner.py` | **+5201** | 执行侧：range 循环、公共 KV 物化、发布、导入、apply |
| `vllm/v1/pic/mooncake_kv.py` | **3049** | Mooncake 对象存储：KV/矩阵发布与查找、pinned staging 池、H2D |
| `vllm/v1/pic/live_capture.py` | **1470** | 捕获 GDN 的 recurrent/conv 输入与状态 |
| `vllm/v1/pic/native_kv.py` | **1177** | 原生 KV 的块/窗口语义（`PICWindow`、span、allocation、lease） |
| `vllm/v1/pic/state.py` | **1137** | transition：`retimed` / `slice` / `compact`（token 级 → 矩阵） |
| `vllm/v1/core/sched/scheduler.py` | +987 | 计划构建、公共块预留、park/promote |
| `vllm/v1/pic/runtime.py` | 453 | 单请求运行计划：哪些 range 复用、哪些重算 |
| `vllm/v1/pic/external_kv.py` | 389 | provider 契约（覆盖而非相等） |
| `vllm/v1/pic/mooncake_provider.py` | 372 | Mooncake 适配 |
| `vllm/v1/pic/snapshot.py` | 355 | 状态快照 |
| `vllm/config/pic.py` | 135 | 配置 + 单一诊断开关 |

---

## 3. 固化的规则（改动必须保持）

1. **发布 span = 复用窗口**：段内 `[seam, length-1)`，段末 1 个 token 重算（要出 logits）；
2. **矩阵永不切**；闭包先切到窗口、再 `compact`；
3. **`slice` 的坐标相对算子**，不是相对段；
4. **跨单位比较一律用“覆盖”**，不用相等（`PICWindow.fits()` / `covers()`）；
5. **正在复用的请求不捕获**；
6. **一个诊断开关**：`HYPIC_PIC_DEBUG` → `pic_config.debug`；
7. **`seam` = `seam_sink` = 8**（固定），不是分隔符 token 数；
8. **矩阵只由 Triton 生成**（`chunk_gated_delta_rule_fwd_h`），手写的 Torch 递推已删除。

---

## 4. 命令行（现在只需 1 个开关）

```bash
vllm serve ... --pic-enable --pic-max-cache-bytes 4294967296 \
                   --pic-min-segment-tokens 512 --no-enable-prefix-caching
```

| 类 | 参数 |
|---|---|
| **"不开就不工作"→ 跟随 `--pic-enable`** | `auto_enable`、`capture_live`、`restore_live`、`zero_copy`、`single_request`、`mooncake`、`async_import`（默认全 `True`） |
| **部署选择 → 写在启动脚本里** | `--pic-max-cache-bytes 4294967296`（4 GiB 池）、`--pic-min-segment-tokens 512` |
| **真·可选（仍默认 False）** | `--pic-copyback-kv`、`--pic-batch`、`--pic-packed-batch`、`--pic-mooncake-import-only`、`--pic-debug` |

**共 20 个配置字段**：9 个默认 True、5 个有默认值、6 个默认 False。

---

## 5. 测试与工具（全部入库，`git apply` 后一条命令）

| 文件 | 作用 |
|---|---|
| `tools/run_pic_test.sh`（353 行，755） | **一键验收**：`diag` / `perf` / `off` / `trace` / `all`；负责 publish、reuse、判据 |
| `tools/pic_run.sh`（273 行，755） | 起/停 A/B 与 mooncake master；`serve` 只起服务，`diag`/`perf`/`trace` 自带编排 |
| `tools/pic_fixture_shapes.py` | fixture 的**唯一定义**（`build(shape)` / `reusable_parts()`） |
| `tools/pic_make_fixtures.py` | 生成/校验 fixture（打印每段 token 数） |
| `tools/pic_compact_selftest.py` | 离线自测：矩阵 vs kernel 在**未见状态**上的一致性 |
| `tests/v1/pic/fixtures/{a,b,c}/` | **17 个固定文本，354 KB**，含 `publish_0` / `reuse_0` 别名 |
| `tests/v1/pic/test_window_alignment.py` | 39 个非对齐长度用例（抓了 12 类历史故障中的 8 类） |
| `tests/v1/pic/test_fixture_shapes.py` | 20 例：字节一致、段够长、文档不复读 |

### 用法

```bash
cd /usr/local/lib/python3.12/dist-packages      # 含 vllm/ 包的那一层
export PIC_MODEL=/path/to/Qwen3.5-2B
bash tools/run_pic_test.sh diag    # a/b/c × 10 条判据 + 语义答案
bash tools/run_pic_test.sh perf    # 形状 b，5 轮（丢第 1 轮取中位）
bash tools/run_pic_test.sh off     # 同一套 fixture，不带任何 PIC 开关
bash tools/run_pic_test.sh trace
```

**日志/配置落 `tests/v1/pic/{logs,trace}`；`mooncake_config.json` 由脚本用 `hostname -I` 生成**（机器相关，不入库）。

---

## 6. 性能（形状 B，doc 6976，5 轮稳态）

| 指标 | PIC-ON | PIC-OFF | 差异 |
|---|---:|---:|---|
| c=1 wall / p50 / tput | 7378 ms / 366.8 / 2.72 | 5885 ms / 293.6 / 3.40 | 慢 25% |
| **c=4 wall / p50 / tput** | **3301 ms / 603.4 / 6.06** | 4578 ms / 959.2 / 4.38 | **wall −27.9%、p50 −37%、tput +38%** |

**交叉验证**：`6.06/4.38 = 1.38`、`4578/3301 = 1.39`——两个指标一致。
**定位**：PIC 换的是**吞吐**，不是单请求延迟（c=1 的差额 ≈ `74.7 ms/请求` 的 import + apply）。

---

## 7. 本会话的主要改动（按主题）

| 主题 | 做了什么 |
|---|---|
| **窗口语义** | 引入 `PICWindow(token_start, token_end, block_size)`：`blocks()` 用 ceil、`fits()`/`covers()` 用覆盖，10 处重复计算收敛到 1 处 |
| **内存** | 闭包就地释放（1.9 GB → **38 MB/请求**）；staging 池首次分配即填满 4 个 slot（`transition-transfer` 111.9 → **49.3 ms**） |
| **诊断** | 删除 `capture window`（每层 2 次 D2H 同步）；9 处日志全部 debug-gated；`PIC_DEBUG` 环境变量可覆盖模式 |
| **删除手写递推** | `_apply_chunked_reference`、`:579` 分支、FP32 兜底、`validate_gdn_layer` 全部删除（共 −202/+10）；`apply()` 只剩「矩阵 / Triton / `ValueError`」三条出口 |
| **测试树** | fixture 入库（含 `_0` 别名）、`serve` 模式、`lookup_hits` 判据、`off` 模式、39+20 个离线用例 |
| **CLI 收敛** | 10 个开关 → 1 个；必须开的跟 `--pic-enable`，数值显式写在启动脚本 |
| **形状 c 语义** | doc 加 `segment X is cat/dog/bird` 标记（否则模型答 "which one" 而不是 `dog`） |

---

## 8. 已知限制

| # | 项 | 状态 |
|---|---|---|
| 1 | `<<SEP>>doc<<SEP>>`（只有 doc、无 query）以前 capture 失败 | **已解除**（拦它的 `validate_gdn_layer` 已删）；**待复测** |
| 2 | c=4 存在慢簇 ~600–625 ms | **接受**：import 排队（`KV_IMPORT_WORKERS=1`，并行更慢已有实测） |
| 3 | `transition_operator_bytes` 各阶段读 0 | 观测缺口；有间接证据（transfer 49.3 ms、无 1.9 GB 波动） |
| 4 | CPU / 无 CUDA 环境无法应用 transition | **故意**：`apply()` 抛 `ValueError`（fail closed） |
| 5 | doc ~8011 + c=4 OOM | doc ~7000 是当前安全上限 |
| 6 | `use_fla_reference` 字段、`pic_transition_validation` 字段已无消费者 | 死代码，留待下一版清理 |
