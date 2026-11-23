# Stage 10-C-6 Mooncake compatibility follow-up

最终目标是让 Mooncake 持久化 vLLM native KV cache，并支持后续请求对 PIC
非前缀 segment 的真实复用。本轮在已验证的 Stage 10-C-6 external native-KV
interface 上增加 Mooncake provider、worker fail-closed 注册和 TCP loopback 验证，
属于实现最终目标所需的前置层，不宣称远程 KV 复用已经完成。

## 边界

- 不调用 worker 侧不存在的 scheduler `BlockPool`。
- 不伪造 native block ID。
- provider 只有在调用方提供真实 native block resolver 和目标写入回调时才会导入。
- 没有这些回调时，`--pic-mooncake` 仍可启动，但 external reference 按 request
  回退。
- 不修改 attention、CUDA、C++ 或 Triton。
- 本轮暂不实现 Mooncake Store 的远程 KV object 格式、RDMA 或 hybrid state
  远程导入；这些是最终可用版本的后续工作。

## 静态和单元测试

```bash
python3 -m compileall -q vllm/v1/pic/mooncake_provider.py \
  vllm/v1/pic/__init__.py vllm/v1/worker/gpu_model_runner.py
python3 -m pytest tests/v1/pic/test_segmenter.py -q
```

测试应覆盖 provider 成功导入、lease 释放、block span 不匹配、传输失败、没有
destination resolver、未注册 provider 和重复注册。

## TCP loopback

需要安装与 `vllm/vllm-openai:v0.22.1-cu129` CUDA 版本匹配的 Mooncake Python
包。先启动 target：

```bash
python3 tests/v1/pic/mooncake_tcp_loopback.py target \
  --address 127.0.0.1:12345 \
  --metadata-server 127.0.0.1:2379
```

另一个终端启动 initiator：

```bash
python3 tests/v1/pic/mooncake_tcp_loopback.py initiator \
  --address 127.0.0.1:12347 \
  --target-address 127.0.0.1:12345 \
  --metadata-server 127.0.0.1:2379
```

预期输出包括 `tcp loopback checksum matched` 和 `provider lease released`。
脚本只验证 Mooncake TCP transfer 和 provider 生命周期，不代表远程 KV 已接入
vLLM native attention pool。

## vLLM smoke

```bash
vllm serve /path/to/model \
  --enforce-eager \
  --pic-enable --pic-auto-enable --pic-zero-copy --pic-single-request \
  --pic-capture-live --pic-restore-live \
  --pic-mooncake --pic-debug
```

在没有真实 native destination resolver 的环境中，预期是服务正常启动，external
reference 记录 request-local fallback；普通请求和 PIC 请求仍应 HTTP 200，不能有
traceback、HTTP 400/500 或 pool/lease 泄漏。

## 性能记录

本轮没有真实远程 KV import 时，Mooncake 端到端 speedup 记录为“不适用”。最终
版本必须在真实 Mooncake KV put/get 和非前缀 PIC 命中链路上记录普通基线、PIC 延迟、
throughput、token reuse 和跨进程传输开销；不能把 TCP loopback checksum 测试当作
性能收益。
