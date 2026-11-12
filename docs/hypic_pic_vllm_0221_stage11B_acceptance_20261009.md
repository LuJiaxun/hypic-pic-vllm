# HYPIC PIC vLLM Stage 11-B 验收报告

日期：2026-10-09  
环境：vLLM 0.22.1-cu129 / Mooncake 0.3.10.post2 / Qwen3.5-2B hybrid / RTX 4090 24 GB 单卡  
Patch：`hypic-pic-vllm-0221-cu129-stage0-11B-block-size-fix.patch`  
Patch MD5：`B929604D924423D941B4D8FE71772F9E`  
结论：**全部验收项通过**

## 1. 验证环境

容器：`vllm/vllm-openai:v0.22.1-cu129`，使用 `--shm-size 32g`、`--ipc=host`、
`--network=host`、`--gpus all` 和 `/home:/home`。

Mooncake 配置文件：

```json
{
  "local_hostname": "HOST_B",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "tcp",
  "device_name": "",
  "master_server_address": "HOST_B:50088",
  "global_segment_size": 8589934592,
  "local_buffer_size": 2147483648
}
```

两个 vLLM 进程均设置：

```bash
export MC_STORE_MEMCPY=0
export MOONCAKE_CONFIG_PATH=mooncake_config.json
```

A 使用端口 8000，B 使用端口 8001；A 完全 ready 后才启动 B。

## 2. 验收结果

- `compileall`：通过。
- 单元测试：`86 passed, 3 warnings`。
- block-size 测试覆盖逻辑 block 544、native block 32、1088 tokens 和 34 个 native blocks。
- A publish 和 manifest 写入成功：`blocks=34`，23 层数据完整写入。
- B 跨进程 import 成功：`external_kv_imported: 1`，`external_matches=[0]`。
- 无 remote block ID 直接写入 block table 的日志。
- A/B 输出一致：text 和 tokens 相同，最大 logprob 差异 `0.00000000`。
- 5 次重复请求后 publish 次数、cache entry 和 pool free bytes 稳定。
- 删除 manifest 后仅当前请求 fallback，普通请求仍成功；随后可重新 publish。
- 不带 `--pic-mooncake-import-only` 时 external discovery 关闭，生产默认安全回退。

## 3. 关键运行约束

`global_segment_size` 必须至少为 8 GiB。1 GiB 配置会在写入 Qwen3.5-2B
full-attention group 的最后一层时返回 Mooncake `-800`，使 publish 整体回滚。

Mooncake master 若保留历史死进程的 segment 归属，B 可能连接到陈旧 endpoint 并出现
`Connection refused` 或 `TRANSFER_FAIL`。应先重启 master，再在 master ready 后依次启动
A、B。

24 GB 单卡 A+B 的已验证配置为：`--gpu-memory-utilization 0.35`、
`--max-model-len 4096`、`--max-num-batched-tokens 4096`、
`--pic-max-cache-bytes 1073741824`。

## 4. 阶段边界

本报告只证明 Stage 11-B 的 full-attention native KV import-only 链路正确：

```text
manifest discovery → request-owned native destination → worker import
```

不包含 hybrid recurrent/conv/transition state 远程复用、segment skip、seam forward
或端到端加速；这些属于 Stage 11-C。
