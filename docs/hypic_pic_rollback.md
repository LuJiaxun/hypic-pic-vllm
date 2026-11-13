# PIC 回退与封版（vLLM 0.22.1 / cu129）

本文件是 PIC 的**回退口径**与**封版清单**。任何时候要回到一个已知状态，按这里的命令走；
不要在记忆里保存 commit 号——封版点会变，本文件跟着变。

---

## 1. 封版点

| tag | 日期 | 状态 | 关键数字 |
|---|---|---|---|
| `pic-shapeA-2026-11-11` | 2026-11-11 | **复用链路首次跑通**（执行层已验；**输出一致性未验**） | 形状 A：c=4 tput **6.49**、窗口 4830（`279 + 4830 + 1 = 5110`）、`transition applied range=[279,5109)`、`state_position=5108`、`publish-matrix span=[8,4838) tokens=4830`、`native KV blocks=153` |
| `pic-fixed-tput59-2026-11-10` | 2026-11-10 | 上一稳定点（H2D 不阻塞、位置校验、attach 去重、`cudaMemGetInfo` 删除） | 形状 A：c=1 p50 **268.9 ms** / tput 3.77、c=4 tput **5.9**、1-token step 27.1 ms |

两个封版点都有 **annotated tag + 同名 backup 分支 + 离线副本**：

| 封版 | tag | 分支 | 离线副本 |
|---|---|---|---|
| 形状 A | `pic-shapeA-2026-11-11` | `backup/pic-shapeA-2026-11-11` | `backups/pic-shapeA-2026-11-11/` |
| 上一稳定点 | `pic-fixed-tput59-2026-11-10` | `backup/pic-fixed-tput59-2026-11-10` | `backups/pic-fixed-tput59-2026-11-10/` |

离线副本内容：`hypic-pic-*.patch`（交付 patch 本身）、`IDENTITY.txt`（md5 / commit / 关键数字）、
`TAG_MESSAGE.txt`（封版说明原文）。**这两个目录不依赖 git 历史**，仓库被改写也能用。

---

## 2. 回退方式（三选一）

```bash
# ① 回到封版（工作区干净时；会切到 detached HEAD）
git checkout pic-shapeA-2026-11-11

# ② 只回代码，保留后续提交历史（在现有分支上继续）
git checkout backup/pic-shapeA-2026-11-11 -- vllm tools tests

# ③ 离线重放（不依赖 git；从干净镜像重建）
git clone --branch v0.22.1 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
md5sum ../hypic-pic-vllm-0221-cu129-stage0-11E-E-C1c-full.patch
#   形状 A 封版应为 5467b668d9a6e04fef392012629c2278
git apply ../hypic-pic-vllm-0221-cu129-stage0-11E-E-C1c-full.patch
```

**注意 `③` 的路径**：`md5sum` 校验的是交付 patch（在 `vllm/` 目录之外），不要写成 `./`。

回退后必须确认 patch 能干净应用：

```bash
git apply --check ../hypic-pic-vllm-...patch   # 无输出 = 可应用
```

---

## 3. 回退后先跑（离线，5 + 1）

```bash
python3 tools/pic_check_unbound.py          vllm/v1 tests/v1/pic
python3 tools/pic_check_forward_refs.py     vllm/v1 tests/v1/pic
python3 tools/pic_check_dataclass_order.py  vllm/v1 tests/v1/pic
python3 tools/pic_check_missing_imports.py  vllm/v1 tests/v1/pic
python3 -c "import vllm.v1.pic.state, vllm.v1.pic.native_kv, vllm.v1.pic.mooncake_kv, vllm.v1.worker.gpu_model_runner; print('imports ok')"
```

这四条静态检查覆盖的正是“`compileall` 能过、`import` 才炸”的三类错误（前向引用、dataclass 默认值顺序、
缺失 import），它们在本项目里各自至少抓到过一次线上故障。

---

## 4. 升级/降级单点开关（不改代码的回退）

| 开关 | 关掉后行为 |
|---|---|
| `vllm/config/pic.py: async_import` | **默认已改为 True**（阻塞式导入是它替换掉的旧行为）；要回到 C1b 的阻塞路径，把它改成 False 并重启 |
| `--pic-enable`（去掉） | 整条 PIC 关闭，等同全量 prefill |
| `HYPIC_PIC_DEBUG`（unset） | 诊断日志全关（生产默认） |
| ~~`HYPIC_PIC_NATIVE_KV_DIRECT`~~ | **已删除**：直连不再有开关，store 有 API 就走直连 |
| ~~`HYPIC_PIC_NATIVE_ARENA_CACHE_BYTES`~~ / ~~`HYPIC_PIC_KV_ARENA_BYTES`~~ | **已删除**：host arena 整条移除 |
| `PIC_TP` / `PIC_CUDA_A` / `PIC_CUDA_B` | 测试脚本用：TP 大小与两侧设备 |

守卫已删除，**默认值就是安全阀**。

---

## 5. 封版需要记录什么

每次封版，本文件与离线副本的 `IDENTITY.txt` 都要写清五件事：

1. **patch MD5 与字节数**（唯一身份，不是 commit 号）；
2. **已验证的判据**（例如 `49 命中 = 49 执行 = 49 apply`、`fallback_reasons={}`）；
3. **未验证的部分**（例如“输出一致性未验”——执行对不等于结果对）；
4. **形态与窗口数字**（`279 + 4830 + 1 = 5110`、`range=[279,5109)`）；
5. **已知剩余成本**（tail 61 ms、物化 10–17 ms、每请求 91 MB 传输）。

**第 3 条最重要**：一个封版如果只写“跑通了”，下一个人会把“执行正确”读成“结果正确”。

## 5. 当前传输形态（2026-11-13 起）

| 方向 | 路径 |
|---|---|
| 发布 | `batch_put_from_multi_buffers`：连续块是一段物理连续区，**每层 1 段、无 gather** |
| 接收 | `batch_get_into_multi_buffers`：**直接写目标块**，`host_staging=None`，无 H2D |
| transition | **1 个对象 / 1 块 36.6 MB 设备 slab**，按偏移切 54 个视图 |
| 失败语义 | store 无 API 或返回负值 → **抛错**（不再静默走 host） |

**host staging 已删除**（`_PinnedStagingPool`、arena 租还/预热、`PICNativeKVHostStaging`），
见 `hypic_pic_tp_design.md` §9。
