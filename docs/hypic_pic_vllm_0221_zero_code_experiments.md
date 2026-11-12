# Stage 11E 验证与零代码实验清单（2026-10-21）

本轮不做代码改动。目标是先把 Stage 11E-C 的基线验证做完，再用四个零代码实验回答
"下一步做 B3（异步重叠）还是传输/字节优化"。

输入数据：`docs/hypic_pic_vllm_0221_acceleration_record.md` 第 20 节
（跨机基线见 `acceleration_record` 第 21 节：COMPACT_TRANSITION 组 Full 193.0 ms、
PIC 271.1 ms、0.718×；11E-B 的 Store-pull 默认组是 274.3 ms / 0.708×，波动更大。）

固定结论（用于判断优先级，不要在实验前推翻）：

```text
跨进程新增成本 = native-receive 86.5 + native-materialize 23.5 + transition 35.5 ≈ 145.5 ms
本地 PIC 省下的计算 = 69.6 ms
完美重叠最多隐藏 forward 65 ms  →  PIC 约 209 ms，仍 0.92×
等价条件：跨进程新增成本 < 64 ms 才可能 >1×；92 MB 需要 ≥1.4 GB/s
```

## E0：静态门与测试基线（阻塞项）

```bash
cd /workspace/vllm
python3 -m compileall -q vllm/v1/pic
python3 -m pytest tests/v1/pic/test_segmenter.py tests/v1/pic/test_mooncake_kv.py \
  tests/v1/pic/test_native_kv_transfer.py tests/v1/pic/test_transfer_engine.py -q
ruff check --select F821 vllm/v1/pic/mooncake_kv.py tests/v1/pic/test_segmenter.py
```

已知有 **6 个 legacy 用例失败**（未随 Store-pull 契约更新）。请把失败用例名和
断言差异贴回来，我按契约改断言；在测试转绿之前不进入下一阶段。

## E1：11E-C 的 A/B 验证（约 30 分钟）

目的：确认 `store-host-staging` 生效、`deserialize_ms=0.000`、且不回退基线
（第 21 节：PIC p50 271.1 ms，窗口 265–281 ms）。

```bash
export HYPIC_PIC_COMPACT_TRANSITION=1
export HYPIC_PIC_PROFILE_IMPORT=1
unset HYPIC_PIC_VERIFY_CHECKSUM
unset HYPIC_PIC_TRANSITION_RECEIVE            # 默认 host
# A、B 使用同一条命令、同一份 patch
vllm serve /data/eval_model/Qwen/qwen3.5-2b --port 8000 \
  --enforce-eager --pic-enable --pic-auto-enable --pic-zero-copy \
  --pic-single-request --pic-capture-live --pic-restore-live \
  --pic-mooncake --pic-max-cache-bytes 2147483648 --pic-debug \
  --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.35
```

跑一次 publish + 10 次 reuse（沿用 `mooncake_bench.sh` 的
请求形状），然后：

```bash
grep -E 'transition-(store-total|transfer|total|deserialize)' /tmp/pic_b.log | tail -20
grep -E 'native-(direct|receive|materialize)-total' /tmp/pic_b.log | tail -20
grep -E 'fallback|native_slot_attach_failed' /tmp/pic_b.log | tail -20
```

必填记录：

| 项目 | 11E-B | 11E-C | 判定 |
|---|---:|---:|---|
| transition `representation=` | ？（待填） | 期望 `store-host-staging` | 与 11E-B 不同即需解释 |
| transition `deserialize_ms` | ？（待填） | 期望 `0.000` | 非 0 即未命中快路径 |
| `transition-store-total.get_ms` | ？ | ？ | 与 27.9 / 35.5 对比 |
| `native-direct-total` 是否出现 | 未出现 | 未出现（本阶段不改 native） | 只记录；已决定不追（见下） |
| PIC p50（4k, mt=1） | 271.1 | ？ | 窗口 265–281 ms |
| full p50 | 193.0 | ？ | 对照组 |
| fallbacks | 0 | 期望 0 | 非 0 先查 cause |

注意：E1 只验证"无 deserialize + 无回退"，**不要期待大幅加速**，主导成本是
native-receive 86.5 ms。

## E2：Store 调参矩阵（约 1 小时，零代码）

目的：0.82 GB/s 是配置天花板还是协议天花板。若带宽能上去，全部结论都要重算。

矩阵（改 Mooncake master 的 `global_segment_size` / `local_buffer_size` 后重启
master，再重启 A/B）：

| 组合 | global_segment_size | local_buffer_size | 备注 |
|---|---:|---:|---|
| S0 | 8 GiB | 2 GiB | 当前基线，复用已有数据 |
| S1 | 16 GiB | 2 GiB | 只加段大小 |
| S2 | 8 GiB | 4 GiB | 只加本地 buffer |
| S3 | 16 GiB | 4 GiB | 全加 |

每组跑：

```bash
python3 store_bw_test.py      # 16/32/64 MB
# 选 get 最好的组合，再跑一次整链路
bash mooncake_bench.sh
```

记录表（get / put GB/s）——**已测完，结论：参数无效，停止调**：

| 组合 | get 16 MB | get 32 MB | get 64 MB | put 64 MB | PIC p50 |
|---|---:|---:|---:|---:|---:|
| S0 | 0.671 | 0.683 | 0.796 | 5.288 | 274.3 |
| S1 | 0.682 | 0.686 | 0.786 | 5.298 | — |
| S2 | 0.657 | 0.784 | 0.832 | 5.205 | — |
| S3 | 0.654 | 0.727 | 0.768 | 5.258 | — |

干净启动（master 全新、Mem Storage 0 B）复测 S0：0.739 / 0.658 / 0.768，无差异。
master 启动日志没有 `global_segment_size` 字段（只有 `global_file_segment_size`），
这两个参数是 client `setup()` 传的，不参与传输路径。

**E2 结论：`global_segment_size` / `local_buffer_size` 对带宽无影响，停止调参。**

⚠️ 但 E2 带出一个更重要的事实：**put 5.2–5.4 GB/s 与 get 0.65–0.83 GB/s 相差 6–8 倍**。
put 是**本机 segment 写**（master 与 client 同机、`put` 立即返回），不能当作 TCP 读能力；
而 transition 路径用一次批量读拿到 1.38 GB/s，说明**单对象 get 的 0.8 只是单次调用上限**，
不是链路上限。因此 20.5 节“批量无用、P0 撤回”的推论需要重测。最简脚本：

```bash
python3 tests/v1/pic/run_store_read_path_probe.py \
  --master HOST_A:50088 --host HOST_A --reps 5
```

它只比两件事：`N 次 store.get()` vs `1 次 batch_get_into_multi_buffers`，默认跑
`6:53.5`（native 形状）和 `90:38.4`（transition 形状）。**batch 明显快 → 改 native；
两者接近 → 不改。**

每 token 成本表仍有效（用于判断某个带宽下是否回本）：

| get 带宽 | 传输 µs/token | +H2D | 每 token 净 | 摊平 39.5 ms 固定开销需要 |
|---:|---:|---:|---:|---:|
| 0.62（当前） | 19.8 | 24.3 | **−8.3** | 永远回不了本 |
| 0.82（裸测峰值） | 15.0 | 19.4 | −3.4 | 永远回不了本 |
| 1.06 | 11.6 | 16.0 | 0 | 临界，固定开销仍亏 |
| 1.40 | 8.8 | 13.2 | +2.8 | 约 1.41 万 reused token |
| 4.90 | 2.5 | 6.9 | +9.1 | 约 4352 token（当前形状即可） |

即：**只有 ≥1.06 GB/s 符号才翻正**；到 1.4 GB/s 也需配长复用段才真正回本。

## E3：并发探针（约 30 分钟，零代码）

目的：回答"现在并发 PIC 的 import 是否串行"，这决定 B3 有没有场景。
关键事实：`_update_states()` 里每个新请求的 import 都是同步阻塞的
（`vllm/v1/worker/gpu_model_runner.py:1632` 在 `for new_req_data in
scheduled_new_reqs` 循环内），所以并发 N 个 PIC 很可能是 N×。

```bash
export HYPIC_PIC_PROFILE_IMPORT=1
python3 tests/v1/pic/run_pic_concurrency_probe.py \
  --base-url http://127.0.0.1:8000 --model qwen3.5-2b \
  --publish-prompt-file /tmp/publish.txt \
  --reuse-prompt-file /tmp/reuse.txt \
  --concurrency 1,2,4 --requests 20 --warmup 3 --max-tokens 1 \
  --json-out /tmp/pic_concurrency.json
```

（没有现成 prompt 文件时省略这两个参数，用脚本内置的 filler 构造
`X <<PIC_SEP>> B <<PIC_SEP>> Y`；token 数取决于 tokenizer，只用于相对比较。）

配套日志：

```bash
python3 tools/pic_import_profile.py /tmp/pic_b.log --json > /tmp/pic_b.json
```

记录表：

| 并发 | p50 (ms) | p95 (ms) | tput (rps) | native-import 次数 | transition-transfer 条数 |
|---:|---:|---:|---:|---:|---:|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |

判据：

- p50 随并发**近似线性**增长（≈ N × 单请求 import 时间）→ import 串行，
  **B3-a（先全部 submit 再统一 wait）立刻有收益**，优先级最高；
- p50 基本持平 → 已有重叠，B3 只对"与其他请求的 forward 重叠"有意义，
  优先级降到 RDMA/带宽之后。

## E4：RDMA 复核（10 分钟）

```bash
ibv_devinfo -v | head -40
ls /sys/class/infiniband 2>/dev/null
ibstat 2>/dev/null | head -20
ethtool -i $(ip route get HOST_A | awk '{print $5; exit}') 2>/dev/null
```

三种结果对应三种走法：

1. 有设备但容器不可见 → 加 `--device /dev/infiniband`、`--cap-add IPC_LOCK`、
   `--ulimit memlock=-1` 后重跑 E2/E3；
2. 宿主机也没有 → 确认无 RDMA，PIC 在 TCP 上只能做到"打平 Full"，B3 只剩摊薄意义；
3. 有设备且可用 → 把 Mooncake `protocol` 改为 `rdma`，重跑 E2 + 整链路，
   预期 PIC 到 ~170 ms（报告 P0 估算），这时 B3 与 kernel 优化都不必先做。

**E4 结果（2026-10-21，已确认）：无 RDMA 网卡。** 决策矩阵里的 RDMA 分支全部作废，
按下面的"无 RDMA 矩阵"走。盈亏分析见
`docs/hypic_pic_vllm_0221_acceleration_record.md` 20.6.1：TCP + fp16 KV 下每复用
1 个 token 要付 24.3 µs 传输+H2D、只省 16.0 µs 计算，**符号是反的**，不是调优能补的。

## 决策矩阵（无 RDMA 版，E2/E4 已出结果）

| E1 | E3 | 下一步 |
|---|---|---|
| 正常 | 任意 | **先做受控读路径实验**（`run_store_read_path_probe.py`）判定批量是否有用 |
| 正常 | import 串行 | 批量方案 + **B3-a**（并发摊薄）→ 按段准入 |
| 正常 | 已有重叠 | 批量方案 → 按段准入 |
| 回退 / fallback | — | 停，先修 11E-C |

## 下一步代码候选（按收益/成本排序，E2 之后更新）

| 候选 | 收益依据 | 成本 | 依赖 |
|---|---|---|---|
| **native KV 改 host-staging 批量接收**（与 11E-C 同模式，不用 GPU 注册） | 若读路径受单次调用限制：86.5 → 约 41 ms，省约 45 ms | 中（复用 `_store_state_host_plan` 的思路） | 受控实验证明 batch ≫ singles |
| **按段准入**：预计传输时间 > 预计计算节省时，按 miss 本地重算 | 直接消除 0.708× 这类负收益请求 | 小（配置 + 一段准入判断） | 无 |

已排除（2026-10-22 决定不做）：KV 改 fp8、transition 降 bf16。这两项各自能把字节
砍半，但会牵动精度与 wire 格式，不在本阶段范围内。
| B3-a / B3-b | 只在并发场景摊薄；单请求为 0 | 中 | E3 证明 serialized |
| P1 forward batch 合并 | 65 → 约 17 ms（理论） | 大（架构改动） | — |
| ~~Store 调参~~ | **已测：四组无差异，停止** | — | E2 已否决 |

**已决定不做：native direct（`store-gpu-direct` 的本机直写契约）。** 无 RDMA 的 TCP
部署下，vLLM/Mooncake 的 `register_buffer` 实际会失败（vLLM 自己只记 error 后继续），
Mooncake 内部仍按 host staging 搬运；PIC 再单独造一条"直写 GPU"的路径既与参考实现
不一致，也拿不到真实收益。因此 20.3 中 `native-materialize` 的 cpu 3 + h2d 19 ms
**保持不变**，盈亏分析里的 4.4 µs/token H2D 成本照旧计入。
`HYPIC_PIC_TRANSITION_RECEIVE=gpu` 作为对照实验开关保留，但不进入路线图。

**建议顺序**：E2（免费）→ 按段准入（先止损）→ E3（决定 B3 是否有场景）→
D3 native 批量接收（已实现，待真机验收）。

## 贴回来的最小数据集

```text
E0: 失败用例名 + 断言差异
E1: transition representation / deserialize_ms / store-total.get_ms；PIC & Full p50；fallbacks
E2: 四组 get GB/s + 最优组合的 PIC p50
E3: p50/p95/tput 三行 + native-import 次数
E4: 已完成——无 RDMA 网卡
```

有了 E1–E3，我就能直接给出"按段准入阈值 / B3-a"的取舍。

