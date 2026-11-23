# HYPIC PIC — 交接（2026-10-23）

会话若中断、或换 API key 后只能新开会话，从这一页开始即可接上。仓库里的一切都已提交。

## 当前状态

| 项 | 值 |
|---|---|
| 仓库 | `D:\WorkingDocs\CodexDirs\HYPIC\vllm` |
| 分支 | `backup/11E-C-with-breaker-20261022` |
| HEAD | 见 `git log --oneline -1`（写本文时为 `7b42aa47be`） |
| 工作区 | 干净 |
| 基线 patch | `patches/11E-C-3A09990D-baseline.patch`，MD5 `3A09990D9540E52B0ED5EE15E6012CA7` |
| 真机 | 三卡 HOST_A，Mooncake 0.3.10.post2，TCP，容器 `hypic_vllm-openai-v0.22.1-cu129` |

## 已验证

| 版本 | 结果 |
|---|---|
| 3A09990D 基线 | R1–R6 全过：pytest 148、PIC **210.9 / 212.7 / 211.5 ms**、0.914–0.919×、`native-receive-total=39`、`bytes=53477376`、`store-host-staging=39`、`compat-payload=0` |
| A1 `KEEP_PAYLOAD` | PASS：pytest 145、PIC 211.3–212.3 ms、0.911–0.919×、跨进程 39 次 |
| A2 `COMPACT_TRANSITION` | PASS：pytest 145、PIC 210.6–210.8 ms、0.914–0.920×、`external_kv_imported=39`、`reused_tokens=169728`、`recompute_tokens=24960`、`fallback_reasons` 空 |

## 待验证

| 项 | 内容 |
|---|---|
| **A3+A5** | patch `patches/hypic-pic-vllm-0221-cu129-stage0-11E-E-A3A5-overlap-batch/`，MD5 **`97D67D4484A3B7B32042944C4A93EC16`**，commit `d3ea6c440d`。删 `NATIVE_KV_OVERLAP` + `NATIVE_KV_BATCH`（含逐层回退），并修了 `PROFILE_IMPORT=2` 把 level-1 打点全关的门控 bug。期望 pytest 145，`mode=layer` 为 0 |

## 当前基线（2026-10-28 封版：11E-E-C1c，B 组验收）

| 项 | 值 |
|---|---|
| patch | `patches/hypic-pic-vllm-0221-cu129-stage0-11E-E-C1c-full/hypic-pic-vllm-0221-cu129-stage0-11E-E-C1c-full.patch` |
| MD5 | **`A10B00A58CF3B39C99224453B72A3DE6`**（1,290,526 字节，89 文件，commit `b2fb26ca00`） |
| 内容 | A1–A6 + B + C + 1a + 1b + 1c（**单份累计，从上游 base 打**） |
| 收益（B 组，c=2） | p50 **229.3 ms**（关档 393.6，**1.72×**）、tput **8.22 rps**（关档 5.13，**1.60×**） |
| 单请求 | Full 193.5 / PIC 208.7（0.927×，c=1 不 park，属预期） |
| 判据 | `ADOPT=PARK=38`、`FAILED=0`、`TIMEOUT=0`、时序检查 clean |
| pytest | **139 passed** |
| 开关 | `--pic-async-import`（`async_import` 默认 False）；`exclusive_batch` 默认 True |
| 执行模式 | PIC prefill 不组 batch（`--pic-single-request`），W6 已取消 |
| 未通过 | **A 组（不同 prompt）触发两个 bug**：block span mismatch（软）、transition slicing unsupported（硬崩），见 §7.9 与 record §21.21 |

### 上一条基线（A4+A6，仅供对账）

`patches/hypic-pic-vllm-0221-cu129-stage0-11E-E-A4A6-final/`，MD5 `9BF2AC19818A36481DBCAD806F098B7A`，
PIC 212.5 / 212.9 ms、ratio 0.913 / 0.921、pytest 142、开关 18。

### 硬规则：bench 前先查脚本

`mooncake_bench.sh` 曾写死 `export HYPIC_PIC_PROFILE_IMPORT=2`，覆盖父 shell 的 `=0`，
导致"关闭档"实测仍在采集档（PIC 230–244 ms、数千条打点）。脚本已改为
`${HYPIC_PIC_PROFILE_IMPORT:-0}`，但**每次 bench 前仍要复查**：

```bash
grep -n 'export HYPIC_PIC_' mooncake_bench.sh
```

## 待验证 / 未交付

| 项 | 状态 |
|---|---|
| B 组（5 个 `PROFILE_*` 合并） | ✅ 已验并入 `B2F6EFD6`，开关 18 → 14 |
| 11E-G（通信与 prefill 异步） | 设计已写入 `stage_plan.md`，未实现 |
| `git stash` 里的 `wip: 11E-E A3-A6` | 历史半成品，可丢弃 |

## 删除计划（A→B，已全部完成）

| 组 | 开关 | 状态 |
|---|---|---|
| A1 | `KEEP_PAYLOAD` | ✅ 已验（`3093D298`） |
| A2 | `COMPACT_TRANSITION` | ✅ 已验（`0E935982`） |
| A3 | `NATIVE_KV_OVERLAP` | ✅ 已验（`5697A1DC`） |
| A4 | `TRANSITION_RECEIVE` | ✅ 已验（`24E95791`） |
| A5 | `NATIVE_KV_BATCH` | ✅ 已验（`5697A1DC`） |
| A6 | `NATIVE_KV_DIRECT` | ✅ 已验（并入 `9BF2AC19`） |
| B | 5 个 `PROFILE_*` 细分开关 | ✅ 已验（`B2F6EFD6`） |

保留不动的 14 个：`PROFILE_IMPORT`、`KEY_PREFIX`、`ARENA_CACHE_BYTES`、`VERIFY_CHECKSUM`，
以及原生 TransferEngine 的入口与配置（`TRANSFER_ENGINE`、`ENGINE_ID`、`BOOTSTRAP_ADDR`、
`TP_RANK`、`PP_RANK`、`STATE_WORKERS`、`TRANSFER_WORKERS`、`NATIVE_KV_BATCH_DESCRIPTORS`、
`NATIVE_KV_TRANSFER_MODE`、`TRANSITION_TRANSPORT`）。

## 规律（必须遵守）

1. 代码改动只用编辑工具，逐处改——`docs/..._stage_plan.md` 的「交付纪律」；
2. 每次改动立刻提交，工作区不留未提交代码——「git 备份」；
3. 每份 patch README 必须有六节，其中「打点说明」强制——「打点说明」；
4. 交付前必跑：`tools/pic_check_unbound.py vllm/v1 tests/v1/pic`（clean）
   与 `python3 -m compileall -q vllm/v1 tests/v1/pic`（0）。

## 关键文件

| 文件 | 用途 |
|---|---|
| `docs/hypic_pic_vllm_0221_stage_plan.md` | 阶段计划、三条纪律、11E-E/G 设计 |
| `docs/hypic_pic_vllm_0221_acceleration_record.md` | 实测数据，§21.16 为 3A09990D 复核 |
| `vllm/v1/pic/mooncake_kv.py` | 数据通路（native/transition 收发） |
| `vllm/v1/worker/gpu_model_runner.py` | 计划构建、导入提交/等待 |
| `tools/pic_check_unbound.py` | 按作用域的未绑定名检查 |
| `tools/pic_profile_breakdown.py` | 把 `[PIC-PROFILE]` 日志拆成耗时表 |
| `patches/past_patches/` | 历史可回退 patch（含 3A09990D、B90EFAD5） |

## 恢复步骤

```bash
cd /d/WorkingDocs/CodexDirs/HYPIC/vllm
git log --oneline -5            # 看最新提交
git status                      # 应为干净
# 待验证项：打对应 patches/<stage>/<name>.patch，按该目录 README 的验证命令走
```
