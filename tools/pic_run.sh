#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 用法: bash tools/pic_run.sh perf|diag|trace
# 环境变量：
#   PIC_BASE      fixture 目录（默认 <repo>/tests/v1/pic/fixtures/a）
#   PIC_MODEL     模型路径
#   PIC_TP        tensor parallel size（默认 1；>1 时两个服务端各自需要这么多张卡）
#   PIC_CUDA_A / PIC_CUDA_B  两个服务端各自用哪些卡（默认 0 / 1）
#   PIC_MOONCAKE  mooncake 配置文件
#   PIC_TRACE_DIR trace 输出目录
#   PIC_LOG_DIR   日志与 pid 目录
#
# 路径自适应：脚本所在目录的上一级就是仓库根。仓库根指 **包含 vllm/ 包的那一层**，
# 安装后通常是 /usr/local/lib/python3.12/dist-packages，所以用法是：
#   cd /usr/local/lib/python3.12/dist-packages && bash tools/<script>.sh <mode>
set -u

MODE=${1:-perf}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"        # <repo>/tools
VLLM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"           # <repo>

# Everything the run writes lands inside the tree, so nothing depends on where the machine keeps its
# files. The two inputs that cannot be relative - the model and, optionally, the mooncake config - have
# no default: an unset variable stops the run instead of silently pointing at someone else's disk.
BASE="${PIC_BASE:-$VLLM_DIR/tests/v1/pic/fixtures/a}"
TRACE_DIR="${PIC_TRACE_DIR:-$VLLM_DIR/tests/v1/pic/trace}"
LOG_DIR="${PIC_LOG_DIR:-$VLLM_DIR/tests/v1/pic/logs}"
MODEL="${PIC_MODEL:-/path/to/Qwen3.5-2B}"
# Tensor parallel size, and the devices each server gets.  With TP>1 one server needs several
# devices, so they are named per side rather than assumed to be 0 and 1.
TP="${PIC_TP:-1}"
CUDA_A="${PIC_CUDA_A:-0}"
CUDA_B="${PIC_CUDA_B:-1}"
TP_FLAG=""
if [ "$TP" -gt 1 ] 2>/dev/null; then
  TP_FLAG="--tensor-parallel-size $TP"
fi
MOONCAKE_CONFIG="${PIC_MOONCAKE:-$LOG_DIR/mooncake_config.json}"
mkdir -p "$TRACE_DIR" "$LOG_DIR"

if [ -z "$MODEL" ]; then
  echo "FATAL: PIC_MODEL 未设置（export PIC_MODEL=/path/to/Qwen3.5-2B）" >&2
  exit 1
fi

case "$MODE" in
  perf)  DEBUG=0; TRACE=0 ;;
  diag)  DEBUG=1; TRACE=0 ;;
  trace) DEBUG=0; TRACE=1 ;;
  # serve brings the servers up and stops there: the caller publishes, warms up and measures.
  # The other modes publish and warm up on their own, which is right for a hand run and wrong for a
  # driver that is about to publish the same documents again - whichever process publishes first is
  # where the log lands, and the checks look in a fixed place.
  serve) DEBUG=1; TRACE=0; SERVE_ONLY=1 ;;
  *) echo "MODE 必须是 perf|diag|trace|serve"; exit 1 ;;
esac
echo "===== MODE=$MODE  DEBUG=$DEBUG  TRACE=$TRACE  BASE=$BASE  VLLM_DIR=$VLLM_DIR ====="
echo "===== TP=$TP  A=[$CUDA_A]  B=[$CUDA_B]  FLAGS=${TP_FLAG:-none} ====="

# ---------- [1] 清场 ----------
for pat in "vllm serve" "VLLM::EngineCore" "EngineCore"; do
  for pid in $(pgrep -f "$pat" 2>/dev/null || true); do
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    kill -9 "$pid" 2>/dev/null || true
  done
done
pkill -9 mooncake_master 2>/dev/null || true
sleep 15
ps aux | grep -E "vllm|EngineCore|mooncake" | grep -v grep || echo "无残留 ✅"
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -3

# ---------- [2] Mooncake master ----------
rm -rf /tmp/mooncake_store* /var/lib/mooncake* ~/.mooncake* /tmp/mooncake_master.log
rm -f "$LOG_DIR/A.log" "$LOG_DIR/B.log"
nohup mooncake_master --rpc_port 50088 --metrics_port 9004 \
  --client_ttl 3600 --default_kv_lease_ttl 3600000 --default_kv_soft_pin_ttl 7200000 \
  > /tmp/mooncake_master.log 2>&1 &
for i in $(seq 1 30); do
  grep -qE "state=serving" /tmp/mooncake_master.log && { echo "master ready ✅"; break; }
  sleep 2
  [ "$i" -eq 30 ] && { echo "FATAL master"; tail -20 /tmp/mooncake_master.log; exit 1; }
done

# ---------- [2.5] mooncake config（自动生成，host 用本机 IP）----------
# 这个文件的 local_hostname 与 master_server_address 是本机地址，不能入库——换了机器就是错的。
# 所以由脚本生成一次：hostname -I 取本机 IP，失败退回 127.0.0.1。PIC_MOONCAKE 可覆盖路径与内容。
if [ ! -f "$MOONCAKE_CONFIG" ]; then
  LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  [ -z "$LOCAL_IP" ] && LOCAL_IP="127.0.0.1"
  cat > "$MOONCAKE_CONFIG" <<JSON
{
    "local_hostname": "$LOCAL_IP",
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "tcp",
    "device_name": "",
    "master_server_address": "$LOCAL_IP:50088",
    "global_segment_size": 8589934592,
    "local_buffer_size": 2147483648
}
JSON
  echo "generated $MOONCAKE_CONFIG (local_ip=$LOCAL_IP)"
else
  echo "using existing $MOONCAKE_CONFIG"
fi
export MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG"

# ---------- [3] 环境 ----------
export MC_STORE_MEMCPY=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MC_TCP_ENABLE_CONNECTION_POOL=1
unset HYPIC_PIC_PROFILE_IMPORT HYPIC_PIC_PROFILE_DEEP
unset HYPIC_PIC_KV_IMPORT_WORKERS HYPIC_PIC_STATE_WORKERS
# Diagnostics are an input, not a consequence of the mode: serve starts servers and does nothing else,
# so if it also decided whether a run logs anything, a caller that only wants servers could not get the
# logs the checks read. PIC_DEBUG overrides the mode; the modes keep their old meaning when it is unset.
if [ -n "${PIC_DEBUG:-}" ]; then
  DEBUG="$PIC_DEBUG"
fi
if [ "$DEBUG" = "1" ]; then export HYPIC_PIC_DEBUG=1; else unset HYPIC_PIC_DEBUG; fi
if [ "$DEBUG" = "1" ]; then PIC_DEBUG_FLAG="--pic-debug"; else PIC_DEBUG_FLAG=""; fi
if [ "$TRACE" = "1" ]; then export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1; else unset VLLM_CUSTOM_SCOPES_FOR_PROFILING; fi

# PIC_OFF=1 starts the same servers without the PIC flags. That is the only difference between the two
# halves of a comparison: same model, same batching, same fixture. Measuring the baseline with a
# different command line is how a configuration difference gets mistaken for a feature's gain.
if [ "${PIC_OFF:-0}" = "1" ]; then
  PIC_FLAGS=""
  echo "PIC_OFF=1: servers start without any PIC flag"
else
  # One switch: the bridge, the per-range loop, capture/restore, the external store, the async
  # import, the segment floor and the pool budget are the defaults now, because this deployment
  # never turned any of them off.
  # Only the switch and the two numbers: everything that has to be on for PIC to work at all
  # follows --pic-enable, while the pool budget and the segment floor are choices about this run
  # and are written here rather than buried in a default.
  PIC_FLAGS="--pic-enable --pic-max-cache-bytes 4294967296 --pic-min-segment-tokens 512 $PIC_DEBUG_FLAG"
fi

# ---------- [4] 起 A ----------
( export CUDA_VISIBLE_DEVICES="$CUDA_A"
  vllm serve "$MODEL" --served-model-name qwen3.5-2b --port 8000 \
    $TP_FLAG \
    --max-model-len 8192 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.5 --enforce-eager \
    --no-enable-prefix-caching \
    $PIC_FLAGS \
    > "$LOG_DIR/A.log" 2>&1 ) &

# ---------- [5] 起 B ----------
PROF=""
if [ "$TRACE" = "1" ]; then
  PROF="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false}"
fi
( export CUDA_VISIBLE_DEVICES="$CUDA_B"
  vllm serve "$MODEL" --served-model-name qwen3.5-2b --port 8001 \
    $TP_FLAG \
    --max-model-len 8192 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.5 --enforce-eager \
    --no-enable-prefix-caching \
    $PIC_FLAGS \
    $PROF > "$LOG_DIR/B.log" 2>&1 ) &

for i in $(seq 1 90); do
  A=$(curl -sf http://127.0.0.1:8000/v1/models >/dev/null && echo 1 || echo 0)
  B=$(curl -sf http://127.0.0.1:8001/v1/models >/dev/null && echo 1 || echo 0)
  [ "$A" = 1 ] && [ "$B" = 1 ] && { echo "A/B ready ✅"; break; }
  sleep 5
  [ "$i" -eq 90 ] && { echo "FATAL A/B"; tail -30 "$LOG_DIR/A.log"; tail -30 "$LOG_DIR/B.log"; exit 1; }
done

if [ "${SERVE_ONLY:-0}" = "1" ]; then
  echo "servers ready; no publish, no warmup (serve mode)"
  exit 0
fi

# ---------- [6] publish ----------
python3 - <<PY
import glob, json, urllib.request
candidates = sorted(glob.glob("$BASE/publish*.txt"))
if not candidates:
    raise SystemExit("no publish*.txt in $BASE")
pub = open(candidates[0]).read()
req = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
    data=json.dumps({"model":"qwen3.5-2b","prompt":pub,"max_tokens":1}).encode(),
    headers={"Content-Type":"application/json"})
print("A publish:", urllib.request.urlopen(req, timeout=600).read().decode()[:60])
PY

if [ "$DEBUG" = "1" ]; then
  echo "等待 A 侧 publish 完成..."
  for i in $(seq 1 90); do
    if grep -q 'Mooncake native KV published' "$LOG_DIR/A.log" 2>/dev/null \
       && grep -q 'Mooncake hybrid transition published' "$LOG_DIR/A.log" 2>/dev/null; then
      echo "A publish completed ✅"
      break
    fi
    sleep 1
    [ "$i" -eq 90 ] && { echo "FATAL: A publish 90s 未完成"; tail -20 "$LOG_DIR/A.log"; exit 1; }
  done
else
  echo "A publish completed（DEBUG=0，跳过日志轮询）✅"
  sleep 10
fi
sleep 2

# ---------- [7] warmup ----------
python3 - <<PY
import glob, json, urllib.request
candidates = sorted(glob.glob("$BASE/reuse*.txt"))
if not candidates:
    raise SystemExit("no reuse*.txt in $BASE")
reu = open(candidates[0]).read()
for _ in range(2):
    req = urllib.request.Request("http://127.0.0.1:8001/v1/completions",
        data=json.dumps({"model":"qwen3.5-2b","prompt":reu,"max_tokens":1}).encode(),
        headers={"Content-Type":"application/json"})
    try: urllib.request.urlopen(req, timeout=600)
    except Exception as e: print("warmup err", e)
print("warmup done")
PY
sleep 5

# ---------- [8] diag 单请求 ----------
if [ "$MODE" = "diag" ]; then
  python3 - <<PY
import glob, json, urllib.request
candidates = sorted(glob.glob("$BASE/reuse*.txt"))
if not candidates:
    raise SystemExit("no reuse*.txt in $BASE")
reu = open(candidates[0]).read()
req = urllib.request.Request("http://127.0.0.1:8001/v1/completions",
    data=json.dumps({"model":"qwen3.5-2b","prompt":reu,"max_tokens":1}).encode(),
    headers={"Content-Type":"application/json"})
print("reuse:", urllib.request.urlopen(req, timeout=600).read().decode()[:60])
PY
  sleep 3
  echo "--- ① single-request ---"
  grep -E 'single-request skip/recompute|single-request early runtime' "$LOG_DIR/B.log" | tail -3
  echo "--- ② plan ---"
  grep -E 'PIC-TRACE plan' "$LOG_DIR/B.log" | tail -2
  echo "--- ③ executed ---"
  grep -c 'single-request early runtime executed' "$LOG_DIR/B.log"
  echo "--- ④ skipped ---"
  grep -E 'pic path skipped|pic path' "$LOG_DIR/B.log" | tail -3
  echo "--- ⑤ refused ---"
  grep -E 'refused' "$LOG_DIR/B.log" | tail -5
  echo "--- ⑥ PIC-DEBUG ---"
  grep -E 'PIC-DEBUG' "$LOG_DIR/B.log" | tail -10
  echo "--- ⑦ 传输/阶段 ---"
  grep -E 'transition-phases|transition-transfer' "$LOG_DIR/B.log" | tail -2
  exit 0
fi

# ---------- [9] perf / trace ----------
if [ "$TRACE" = "1" ]; then
  curl -s -X POST http://127.0.0.1:8001/start_profile >/dev/null; sleep 5
fi
cd "$VLLM_DIR"
python3 tests/v1/pic/run_pic_concurrency_probe.py \
  --base-url http://127.0.0.1:8001 --model qwen3.5-2b \
  --publish-prompt-file "$(ls "$BASE"/publish*.txt | sort | head -1)" \
  --reuse-prompt-file   "$(ls "$BASE"/reuse*.txt | sort | head -1)" \
  --concurrency 1,4 --requests 20 --warmup 3 --max-tokens 1 \
  --json-out /tmp/probe_shapeA_${MODE}.json
if [ "$TRACE" = "1" ]; then
  curl -s -X POST http://127.0.0.1:8001/stop_profile >/dev/null; sleep 8
  cd "$TRACE_DIR"
  RANK0=($(ls -t rank0.*.pt.trace.json 2>/dev/null))
  if [ "${#RANK0[@]}" -ge 2 ]; then
    SMALL=$(ls -lS rank0.*.pt.trace.json | tail -1 | awk '{print $NF}')
    LARGE=$(ls -lS rank0.*.pt.trace.json | head -1 | awk '{print $NF}')
    mv "$SMALL" pic_shapeB_${MODE}_c1.pt.trace.json
    mv "$LARGE" pic_shapeB_${MODE}_c4.pt.trace.json
    echo "trace 重命名完成："
    ls -lh pic_shapeB_${MODE}_c*.pt.trace.json
  else
    echo "rank0 trace 不足 2 个"
    ls -lh "$TRACE_DIR"/*.pt.trace.json 2>/dev/null | tail -5
  fi
fi

echo "===== 结果 /tmp/probe_shapeA_${MODE}.json ====="
cat /tmp/probe_shapeA_${MODE}.json 2>/dev/null || echo "探针未生成"
