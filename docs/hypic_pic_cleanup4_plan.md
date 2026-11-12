# 第 ④ 步清理：四个原型开关 + `mode`（一次做完的执行单）

**目标**：删掉 `copyback_kv`、`batch`、`packed_batch`、`mooncake_import_only`、`mode` 五条早期原型路径，
约 2,500–3,500 行。**做完只跑一次测试**：`bash tools/test_all.sh`（三形状 9 条判据）+ `perf` / `off`。

**回退点**：`git checkout pic-pre-cleanup4-2026-11-12`

---

## 0. 先补一次 grep（唯一未完成的信息）

```bash
grep -rn 'pic_kv_copyback' vllm/v1/worker/gpu_model_runner.py
grep -rn 'PICAttentionKVCacheCopyBack\|PICKVLayerTarget' vllm/ tests/
```

**为什么**：`copyback_kv` 的 11 个删除点已定位（见下），但 `self.pic_kv_copyback` 还有**消费方法**
（trace 里的 `copyback_pic_kv` 阶段），方法体要一起删，否则留下死属性。

---

## 4a. `copyback_kv`（13 处 / 3 文件 + 1 模块）

| 文件:行 | 删除内容 |
|---|---|
| `vllm/config/pic.py:62-63` | 字段 `copyback_kv: bool = False` + docstring |
| `vllm/engine/arg_utils.py:532` | `pic_copyback_kv: bool = PICConfig.copyback_kv` |
| `vllm/engine/arg_utils.py:1205` | `pic_group.add_argument("--pic-copyback-kv", ...)` |
| `vllm/engine/arg_utils.py:2290` | `copyback_kv=self.pic_copyback_kv,` |
| `vllm/v1/pic/__init__.py:22-25` | `from vllm.v1.pic.kv_copyback import (...)` |
| **`vllm/v1/pic/kv_copyback.py`** | **整文件删除** |
| `vllm/v1/worker/gpu_model_runner.py:187-190` | 同上的 import 块 |
| `:755` | `self.pic_kv_copyback = ...` |
| `:766` | `... or pic_config.copyback_kv` |
| `:784-787` | `if pic_config.copyback_kv: self.pic_kv_copyback = ...` |
| `:792` + 实参 | 日志里的 `copyback_kv=%s` |
| **其余** | `pic_kv_copyback` 的消费方法与 `copyback_pic_kv` 阶段（第 0 步查出来的） |

**判据**：`grep -c copyback vllm/ -r` 结果为 0。

---

## 4b. `pic_config.batch`（6 处 / 2 文件）

| 文件 | 删除 |
|---|---|
| `vllm/config/pic.py:71-77` | `batch: bool = False` + docstring |
| `vllm/engine/arg_utils.py` | `pic_batch` 字段、`--pic-batch`、构造参数 |
| `vllm/v1/worker/gpu_model_runner.py`（5 处） | `pic_config.batch` 的所有分支 |
| `vllm/v1/core/sched/scheduler.py`（1 处） | 同上 |

**注意**：`batch` 与 `single_request` 是互斥的两种执行方式；`single_request` 是现在唯一在用的，
删 `batch` 时确认没有 `if not single_request: ... batch` 这种二选一结构需要改写。

---

## 4c. `mooncake_import_only`（9 处 / 4 文件）

| 文件 | 删除 |
|---|---|
| `vllm/config/pic.py:94-100` | 字段 + docstring |
| `vllm/engine/arg_utils.py`（3 处） | 字段、CLI、构造参数 |
| `vllm/v1/core/sched/scheduler.py`（3 处） | **park/promote 分支**——这是唯一有逻辑风险的地方 |
| `vllm/v1/worker/gpu_model_runner.py`（2 处） | |

**风险**：它改变“只导全注意力 KV、不做 hybrid 复用”的行为；删掉后该场景会走正常 PIC。
**若本项目不再需要这个场景**（当前部署不做），直接删。

---

## 4d. `packed_batch`（21 处 / 6 文件）

| 文件 | 处数 | 先确认 |
|---|---:|---|
| `vllm/v1/worker/gpu_model_runner.py` | 10 | 打包混批的执行分支 |
| **`vllm/model_executor/models/radio.py`** | **5** | **是“读 PIC 配置”还是“被 PIC 侧调用”？** 读配置 → 只删 PIC 侧；被调用 → 涉及跨模块接口 |
| `vllm/engine/arg_utils.py` | 3 | 字段/CLI/构造 |
| `vllm/config/pic.py` | 1 | 字段 |
| 其余 | 2 | |

**这是唯一渗到模型层的一条**，删之前必须先回答 `radio.py` 那 5 处的性质。

---

## 4e. `mode`（单一取值）

`PICConfig.mode: PICMode = "transition_rope_recompute"`——只有这一个值在用。
先 `grep -rn 'pic_config.mode\|PICMode' vllm/` 确认没有别处枚举分支，再删字段与提及。

---

## 验证（只做一次）

```bash
cd /usr/local/lib/python3.12/dist-packages

# 1. 静态
python3 -m compileall -q vllm
python3 tools/pic_check_unbound.py          vllm/v1 tests/v1/pic
python3 tools/pic_check_missing_imports.py  vllm/v1 tests/v1/pic
python3 tools/pic_check_forward_refs.py     vllm/v1 tests/v1/pic
python3 tools/pic_check_dataclass_order.py  vllm/v1 tests/v1/pic
python3 -m pytest tests/v1/pic/test_fixture_shapes.py tests/v1/pic/test_window_alignment.py -q

# 2. 功能（三形状 9 条判据 + 语义 dog）
bash tools/test_all.sh 2>&1 | tee /tmp/test_all.txt
grep -E 'ALL PASS|SOME FAIL' /tmp/test_all.txt

# 3. 性能（应与清理前一致：6.06 / 4.38）
bash tools/run_pic_test.sh perf
bash tools/run_pic_test.sh off
```

**通过标准**：① 四个检查器全 clean；② `ALL PASS`；③ perf c=4 tput ≈ 6.06、off ≈ 4.38（±5%）。

**不通过**：`git checkout pic-pre-cleanup4-2026-11-12`，按 4a→4e 逐条二分。

---

## 为什么建议一次删完再测（而不是逐条测）

这五条都在**同一条执行路径的相邻分支**上（`gpu_model_runner.execute_model` 里的 gate 群）。
逐条测要跑五次三形状 + 五次 perf（每次约 15–20 分钟），而它们的失败模式是**同一个**：
删漏一处 → `compileall` 或 `unbound` 检查立刻报，根本走不到上机。
**静态检查能拦住绝大多数问题**，所以一次删完、一次上机，是这里更省时间的做法。
