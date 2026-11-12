# HYPIC PIC vLLM 0.22.1-cu129 Stage 11-C 验收报告

日期：2026-10-10  
环境：vLLM 0.22.1-cu129 / Mooncake 0.3.10.post2 / Qwen3.5-2B hybrid / RTX 4090 24 GB 单卡  
累计 patch：`hypic-pic-vllm-0221-cu129-stage0-11D-fix.patch`  
基线：`0decac0d96c42b49572498019f0a0e3600f50398`

## 结论

Stage 11-C Phase 1 功能 gate 全部通过。验证使用 `--pic-mooncake`，不使用
`--pic-mooncake-import-only`。

## 环境与启动顺序

Mooncake 配置必须提供至少 8 GiB global segment space；已验证配置为 8 GiB
global space、2 GiB local buffer：

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

先启动并确认 `mooncake_master`，再启动 producer A；A 的 `/v1/models` 就绪后
再启动 consumer B。RTX 4090 双进程验证使用 `--gpu-memory-utilization 0.35`、
`--max-model-len 4096`、`--max-num-batched-tokens 4096`。

`--pic-mooncake-import-only` 是 Stage 11-B 的 KV-only 验证模式，不覆盖
Stage 11-C 的 hybrid transition reuse。

## 验收结果

- `compileall`：exit 0。
- `tests/v1/pic/test_segmenter.py`：89 passed。
- A 发布 native KV：group 3、34 blocks；发布 hybrid transition：segment 0/1。
- Mooncake master：Keys 11、PutStart=11/11、Mem Storage 约 905 MB。
- B lookup HIT：`reused=(0,)`、`recompute=(1,)`、`matches=[0]`。
- native KV checksum verified、`transition applied`、`reuse body skipped`。
- `forward_tokens=647`，full forward 为 1735，减少约 63%。
- `reused_tokens=1088`、`recompute_tokens=640`。
- 与 full-recompute baseline 对比：max logprob diff 为 0，logits 和 tokens 一致。
- 连续两次 `max_tokens=8` 请求成功，第二次无 500 或 transition handle KeyError。
- 删除 hybrid-state manifest 后按 request fallback，普通请求仍成功。

## 已修复的三个 gate bug

1. `resolved_entries` 填充为对象、消费按二元组解包，导致 `TypeError`。
2. `release_request` 后 transition handle 失效，导致 decode 阶段 `KeyError`。
3. worker 改写 scheduler 共享的 `PICSegmentEntry.native_kv_refs`，导致第二次请求
   看不到 external ref，跳过重新导入。

当前修复将 imported native refs 和 transition handles 都放入
`CachedRequestState`，runtime planner 使用 request-local refs，并在 request finish、
fallback 和 plan refresh 时释放资源。

## Stage 11-D 状态

Stage 11-D 当前不需要新增代码，进入稳定性与性能测试：重复请求、并发请求、取消、
eviction、manifest fallback、普通请求回归、资源/lease 稳定性和长时间运行。
