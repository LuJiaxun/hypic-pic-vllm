# Stage 11-B 调试经验总结（给下一个 Agent）

## 六、Stage 11-B 补充踩坑（2026-10-09）

### 6.1 Mooncake publish `-800`：segment 空间不足

现象：A publish 到第 23 层 layer 时返回 `-800`，publish 整体回滚，master Keys
变为 0。

根因：`mooncake_config.json` 的 `global_segment_size` 只有 `1073741824`（1 GiB）。

修复：改为 `8589934592`（8 GiB），并将 `local_buffer_size` 改为
`2147483648`（2 GiB）。实测修改前出现 `PutStart=4/4`、`PutEnd=3/3`、
`PutRevoke=1/1`、`Del=3/3`；修改后为 `PutStart=7/7`、`PutEnd=7/7`。

Qwen3.5-2B 的 full-attention group 虽然 payload 约 12.75 MB，但需要连续完成
23 层 layer 数据和 manifest 的写入，不能只按最终 payload 大小估算 segment 容量。

### 6.2 Master 陈旧 segment 归属

现象：B 的 `store.get()` 连接历史上已经退出的 vLLM 进程 endpoint，例如 15131 或
16927，出现 `Connection refused` 和 `TRANSFER_FAIL`。

根因：master 中存在 `segment_already_exists`，key 到 segment 的映射仍指向旧进程。

修复：重启 master（`pkill -9 -f mooncake_master` 后重新启动），确认 master ready 后
再启动 A，等待 A 的 `/v1/models` 返回 200 后再启动 B。

注意：master 的 `client_ttl` 只维护 key 元数据，不会替代 segment 数据的生命周期
管理；A 退出后，B 不能依赖已经失效的旧 endpoint。

### 6.3 单卡 24 GB 的 A+B 参数约束

已验证参数：

```text
--gpu-memory-utilization 0.35
--max-model-len 4096
--max-num-batched-tokens 4096
--pic-max-cache-bytes 1073741824
```

必须先启动 A 并等待完全 ready，再启动 B，避免两个 profiling 峰值重叠导致 OOM。
A+B 同时运行时 GPU 占用约 18.5 GB，剩余约 5.5 GB。

### 6.4 判断 import 是否真正命中的日志

命中时应看到：

```text
external_kv_imported: 1
external_matches=[0]
```

并且不应出现 `TRANSFER_FAIL`。miss 时通常看到 `external_matches=[]`、
`SocketHandShakePlugin Connection refused` 或 `manifest is missing`。

scheduler 的 `lookup MISS` 在 import-only 模式下不一定表示失败，因为该模式不跳过
本地 hybrid segment；判断依据应是 `external_matches` 和 worker import 日志。

### 6.5 logits 一致性验证

让 B 执行 import-only 请求，让 A 执行普通完整 forward；两边都开启 `logprobs=5`，
逐 token 对比 text、token id 和 logprob。

验收标准：text 和 tokens 一致，最大 logprob 差异不超过 `1e-4`。本次实测最大差异为
`0.00000000`。

## 七、Stage 11-B 最终状态

- ✅ compileall 通过，86 个测试通过，3 个 warning；
- ✅ A publish 成功，full-attention manifest 包含 34 个 native blocks；
- ✅ B 跨进程 import 成功，`external_kv_imported: 1`；
- ✅ `external_matches=[0]`，无传输错误；
- ✅ 无 remote block ID 直接写入 block table；
- ✅ logits 一致，最大差异为 0；
- ✅ publish 幂等，lease 和 pool 不增长；
- ✅ 删除 manifest 后 request-level fallback 成功；
- ✅ 无显式 import-only flag 时 external discovery 关闭并安全回退。

Stage 11-B 已完成。下一阶段是 Stage 11-C：将 native KV import 与 transition/state
restore、seam forward、miss segment forward 和 decode 连接起来，不能把当前
import-only 结果宣称为完整 hybrid PIC 复用。
