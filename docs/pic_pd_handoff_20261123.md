# HYPIC PIC / P-D 分离交接记录（2026-11-23）

本文给后续代码 agent 使用，目标是避免重复踩坑。**只把已验证事实与未解决问题分开，不把推测写成结论。**

## 1. 当前已验证通过的部分

| 项目 | 状态 | 证据/命令 |
|---|---|---|
| 混部 PIC（A/B/C） | **PASS** | `bash tools/test_all.sh` |
| 纯 P/D TransferEngine | **PASS** | `bash tools/test_pd_transfer_engine.sh` |
| shape C 语义（chat + `enable_thinking:false`） | **PASS，出 dog** | 同一 P/D 探针 |
| 原混部 `pic_run.sh` | **保持未修改** | 不要向其中注入 P/D connector |

纯 P/D 已验证：attention KV + GDN/conv state 通过 TransferEngine 交接，P 首 token relay 到 D，D 不重算原 prompt，并发 4/4，0 failed transfers。

**重要：P/D 通过不代表 PIC Store 复用通过。** 当前未闭环的是 **P/D + PIC segment reuse**。

## 2. 目前真正的阻塞点

两 P（producer）隔离实验中，publish 仍会在 native Mooncake batch put 中 segfault：

```text
public KV materialized ...
PIC native KV publish trims blocks to the declared span: declared=48 blocks got=51
!!!!!!! Segfault encountered !!!!!!!
```

随后 `EngineDeadError` / HTTP 500。master 侧通常看到：

```text
PutStart:(Req=1/0/1, Item=6/6)
PutEnd:(Req=0/0/0, Item=0/0)
```

`PutStart Req=1/0/1, Item=6/6` **不是失败**，而是 1/1 个 batch 请求成功、6/6 个对象成功；但 PutEnd 没有完成，客户端随后崩溃。

当前证据指向 **native Store batch put 的传输/收尾路径**，不是 PIC segment 算法，也不是 D/proxy。

## 3. 连接器/Store 一定要分清

| 名称 | 用途 |
|---|---|
| `MooncakeConnector` | **P/D 当次请求** KV + GDN/conv state handoff |
| `MooncakeStoreConnector` | **原生 block-hash 前缀缓存**，不是 PIC |
| PIC 自定义 `MooncakeNativeKVObjectStore` | **按 PIC segment 查询、发布、拉取 KV 和 GDN 状态** |
| `MooncakePICExternalKVProvider` | 将 PIC external refs 导入请求原生 block |

PIC 不使用 `MooncakeStoreConnector`。PIC 的 key 是：

```text
segment/<seg_hash>/group-N@tp_rank:R@pp_rank:P
```

原生前缀复用的 key 是：

```text
model@tp_rank...@group@<chunk_hash>
```

**不要因为原生 block prefix 与 PIC segment 都有 KV 就混用它们。**

## 4. HND/NHD 的坑

- NHD/HND 是 **HBM 中 KV cache 的物理布局**，不是 Mooncake Store 的对象格式。
- Mooncake Store 只保存原始字节，不负责统一布局。
- 原生 `MooncakeConnector` 默认强制 HND；混部 PIC 的路径是在 backend 默认布局上验证的。
- 不能简单认为“换成 HND 就都能工作”，也不能认为“保留 NHD 就与 P/D 无关”。

已落地的两个 HND 功能修复（在 `pic-hnd-clean` 分支）：

1. `cabc687ec7`：gather 不再把 token offset 当 head index；
2. `d7018f7019`：publish 的 `_native_cache_axes` 在无歧义时按实际 shape 选择 K/V/block 轴。

日志已确认 gather 正常：

```text
PIC gather layout=HND key_shape=(15572, 32, 2, 256)
block_indices=[125,172] offsets=[0,31]
```

**但 HND 修复并没有消除 native batch-put segfault。**

## 5. 已排除或尚未证实的假设

### 不是“旧 patch 叠加”

已用 Git 对象内容的 SHA256 对照过服务器源码，服务器与当前 patch 一致。不要再以“混合树”为默认解释。

### 不是简单“两个 Store client”导致

混部 PIC 的 scheduler/worker 本身就可能有两个 Store client，已能正常工作。`Mounting segment` 出现两次不等于 P/D 冲突。

### 不是 fixture/trim 单独导致

原混部成功日志也存在同 segment、`declared=48 got=51`，publish 成功。trim 不是充分根因。

### 不是 fresh master 单独导致

fresh master 的原混部也能通过。

### 不是 KV 注册单独导致

`PIC_DISABLE_PD_REGISTRATION=1` 后仍然 segfault。它只排除了 `batch_register_memory` 这一层，**没有证明 TransferEngine 初始化无冲突**。

### 尚未证实的两个方向

- TransferEngine 与 PIC Store 初始化共存时的 native 冲突；
- `batch_put_from_multi_buffers` 的 direct GPU-buffer 路径（flat-run/view/pointer/lifetime）。

不要把这些方向写成已确认根因。

## 6. 当前分支与实验脚本

### 分支

- `pic-hnd-clean`：当前 HND 路线，含上述两个功能修复与实验脚本。
- `pd-pic-clean`：NHD 实验分支，保留，不回写。
- `backup/nhd-experiment-20261120`：NHD 实验备份。
- 纯 P/D 验收基点：`1aa2c3e761`。

### 脚本

| 脚本 | 用途 | 注意 |
|---|---|---|
| `tools/test_all.sh` | 混部基线 | 已通过 |
| `tools/test_pd_transfer_engine.sh` | 纯 P/D | 已通过 |
| `tools/pic_two_p_reuse_e2e.sh` | 两 producer，无 D/proxy | 当前用于定位 native publish 崩溃 |
| `tools/pic_pd_pic_reuse_e2e.sh` | 完整 P/D + PIC Store | publish_a/b/c 后 reuse |
| `tools/pic_pure_reuse_e2e.sh` | 混部 publish/reuse 控制 | 使用 `pic_run.sh`，不是单 P |
| `tools/pic_store_reuse_e2e.sh` | **原生 MooncakeStoreConnector 前缀复用** | 与 PIC 不同 |
| `tools/native_official_pd_prefix_reuse.sh` | 独立官方 P/D + 原生前缀实验 | 尚未作为通过证据 |

## 7. 已加入的诊断开关（均不是生产配置）

### `PIC_DISABLE_PD_REGISTRATION=1`

跳过 `batch_register_memory` 与 sender listener，保留 TransferEngine 初始化。

**实测仍 segfault**，所以不是单纯注册问题。

### `PIC_DISABLE_TRANSFER_ENGINE=1`

进一步跳过 `TransferEngine.initialize()`，用于验证初始化共存。**尚未作为通过证据**，需要实际运行。

### `PIC_STORE_HOST_PUT=1`

绕过 `batch_put_from_multi_buffers`，把相同 block 序列化为 host bytes，调用 `store.put(bytes)`。

**这是当前最关键的 direct-buffer 对照实验**，不能据此宣布已修复。

## 8. 下一步建议顺序

### A. 先跑 host put

```bash
PIC_STORE_HOST_PUT=1 \
READY_TIMEOUT=300 PUBLISH_TIMEOUT=300 \
PIC_DEBUG=1 HYPIC_PIC_DEBUG=1 \
timeout 600 bash tools/pic_two_p_reuse_e2e.sh 2>&1 | tee /tmp/two_p_host_put.log
```

- 若通过：问题在 direct GPU-buffer put，继续处理 get/restore 的 host fallback。
- 若仍 segfault：继续做下一步。

### B. 若 host put 仍失败，跳过 TransferEngine 初始化

```bash
PIC_DISABLE_TRANSFER_ENGINE=1 \
READY_TIMEOUT=300 PUBLISH_TIMEOUT=300 \
PIC_DEBUG=1 HYPIC_PIC_DEBUG=1 \
timeout 600 bash tools/pic_two_p_reuse_e2e.sh 2>&1 | tee /tmp/two_p_no_engine.log
```

- 若通过：是 TransferEngine 初始化与 PIC Store 的 native 冲突。
- 若仍失败：检查 connector 后台线程、Store client 初始化顺序和 native 客户端 bug，不再改布局。

### C. 需要看的日志

```bash
grep -nE 'PIC gather|native KV publish|declared=|PIC put batch|Segfault|EngineDeadError' \
  /tmp/pic_two_p_publish.log | tail -60

grep -oE 'PutStart:\([^)]*\)|PutEnd:\([^)]*\)|Keys: [0-9]+' \
  /tmp/pic_two_p_master.log | tail -12

grep -nE 'lookup_hits|matched_segments|reused_tokens|external_kv_imported|fallback_reasons' \
  /tmp/pic_two_p_reuse.log | tail -30
```

## 9. 后续如果同时支持“有 SEP 的 PIC”和“无 SEP 的原生前缀”

分派原则应是：

```text
有 SEP（实际 PIC 分隔符）→ 只走 PIC
无 SEP → 只走原生 MooncakeStoreConnector 前缀复用
```

不要让 MultiConnector 的“先命中者胜”替你做决定。若要同一服务自动路由，需要在请求级加 `use_pic` 开关；仅做分开实验则不用改核心代码。

## 10. 最重要的交接原则

1. **先保混部和纯 P/D 不回退，再查 PIC Store publish。**
2. 不要一遇到 native segfault 就改布局；当前 gather/axis 已有日志证明正常。
3. 不要把 `PutStart Req=1/0/1` 误读成失败。
4. 不要把 `PIC_MINIMAL=1` 当成无 connector 的纯混部；它仍带 connector。无 connector 要用 `PIC_PURE=1`。
5. 不要把原生 `MooncakeStoreConnector` 当成 PIC Store。
6. 不要把诊断开关当生产配置。
7. 先用 host put / no-engine 两个对照确定 native 层，再谈修复。
