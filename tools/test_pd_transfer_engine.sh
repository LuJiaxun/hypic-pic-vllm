#!/usr/bin/env bash
# Validate a hybrid P/D handoff entirely through Mooncake TransferEngine.
# Required: PIC_MODEL. Optional: PIC_TP, PIC_CUDA_P, PIC_CUDA_D, MOONCAKE_PROTOCOL.
#
# Runs with python3 only: the container images here do not provide a `python`
# alias, so every interpreter call goes through $PYTHON (override with
# PIC_PYTHON if the image names it differently).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PIC_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "FATAL: neither python3 nor python is on PATH (set PIC_PYTHON)" >&2
  exit 1
fi

MODEL="${PIC_MODEL:?set PIC_MODEL to the model directory}"
TP="${PIC_TP:-1}"
CUDA_P="${PIC_CUDA_P:-0}"
CUDA_D="${PIC_CUDA_D:-1}"
PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
BASE_PORT="${PIC_PD_BASE_PORT:-8000}"
P_PORT=$((BASE_PORT + 10))
D_PORT=$((BASE_PORT + 20))
PROXY_PORT="$BASE_PORT"
LOG_DIR="${PIC_LOG_DIR:-/tmp}"
mkdir -p "$LOG_DIR"
TP_FLAG=()
if [ "$TP" -gt 1 ]; then
  TP_FLAG=(--tensor-parallel-size "$TP")
fi

cleanup() {
  for pattern in 'pic_pd_proxy.py' 'mooncake_connector_proxy.py' 'vllm serve' 'EngineCore'; do
    pkill -f "$pattern" 2>/dev/null || true
  done
}
# Clean the slate before starting, but do NOT clean up on exit by default: after
# a FAIL the operator needs the live P/D/proxy to reproduce the request and read
# their logs, and a trap that fires on exit made every failure a restart.  Set
# PIC_PD_CLEANUP_ON_EXIT=1 to restore the old teardown-on-exit behaviour.
cleanup
sleep 2
if [ "${PIC_PD_CLEANUP_ON_EXIT:-0}" = "1" ]; then
  trap cleanup EXIT
fi

common=(
  "$MODEL" --served-model-name qwen3.5-2b
  --max-model-len 8192 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.5 --enforce-eager
  --no-enable-prefix-caching
)

(
  export CUDA_VISIBLE_DEVICES="$CUDA_P"
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
  nohup vllm serve "${common[@]}" --port "$P_PORT" \
    "${TP_FLAG[@]}" \
    --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$PROTOCOL\"}}" \
    >"$LOG_DIR/pic_pd_prefill.log" 2>&1
) &
P_PID=$!
disown "$P_PID" 2>/dev/null || true

(
  export CUDA_VISIBLE_DEVICES="$CUDA_D"
  nohup vllm serve "${common[@]}" --port "$D_PORT" \
    "${TP_FLAG[@]}" \
    --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$PROTOCOL\"}}" \
    >"$LOG_DIR/pic_pd_decode.log" 2>&1
) &
D_PID=$!
disown "$D_PID" 2>/dev/null || true

ready() {
  # The P/D proxy answers /health; vLLM servers answer /v1/models.  Probe both
  # so a proxy 404 is not mistaken for "not up yet".
  curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
  curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 && return 0
  return 1
}

for entry in "prefill:$P_PORT" "decode:$D_PORT"; do
  name="${entry%%:*}"
  port="${entry##*:}"
  up=0
  for _ in $(seq 1 120); do
    if ready "$port"; then
      up=1
      break
    fi
    if ! kill -0 "$P_PID" 2>/dev/null && [ "$name" = "prefill" ]; then
      break
    fi
    if ! kill -0 "$D_PID" 2>/dev/null && [ "$name" = "decode" ]; then
      break
    fi
    sleep 2
  done
  if [ "$up" -ne 1 ]; then
    echo "FATAL: $name server did not become ready on port $port" >&2
    tail -n 40 "$LOG_DIR/pic_pd_$name.log" >&2 || true
    exit 1
  fi
done

PROXY="$SCRIPT_DIR/pic_pd_proxy.py"
if [ ! -f "$PROXY" ]; then
  echo "FATAL: proxy script missing: $PROXY" >&2
  echo "It ships with this patch; re-apply the patch or copy tools/pic_pd_proxy.py." >&2
  exit 1
fi

nohup "$PYTHON" "$PROXY" \
  --prefill "http://127.0.0.1:$P_PORT" 8998 \
  --decode "http://127.0.0.1:$D_PORT" \
  --relay-first-token --port "$PROXY_PORT" \
  >"$LOG_DIR/pic_pd_proxy.log" 2>&1 &
PROXY_PID=$!
disown "$PROXY_PID" 2>/dev/null || true

for _ in $(seq 1 30); do
  if ready "$PROXY_PORT"; then
    break
  fi
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "FATAL: proxy exited; see $LOG_DIR/pic_pd_proxy.log" >&2
    tail -n 40 "$LOG_DIR/pic_pd_proxy.log" >&2 || true
    exit 1
  fi
  sleep 1
done
if ! ready "$PROXY_PORT"; then
  echo "FATAL: proxy did not become ready on port $PROXY_PORT" >&2
  tail -n 40 "$LOG_DIR/pic_pd_proxy.log" >&2 || true
  exit 1
fi

echo "===== P/D relay up: proxy=$PROXY_PORT prefill=$P_PORT decode=$D_PORT TP=$TP protocol=$PROTOCOL ====="

RC=0

# Same probe and fixture style as the existing PIC runs: real publish/reuse
# traffic through the P/D proxy, including concurrency and latency numbers.
echo "===== publish/reuse traffic (fixtures/b) ====="
"$PYTHON" "$REPO_ROOT/tests/v1/pic/run_pic_concurrency_probe.py" \
  --base-url "http://127.0.0.1:$PROXY_PORT" --model qwen3.5-2b \
  --publish-prompt-file "$REPO_ROOT/tests/v1/pic/fixtures/b/publish.txt" \
  --reuse-prompt-file "$REPO_ROOT/tests/v1/pic/fixtures/b/reuse.txt" \
  --concurrency "${PIC_PD_CONCURRENCY:-1,4}" \
  --requests "${PIC_PD_REQUESTS:-4}" --warmup "${PIC_PD_WARMUP:-2}" \
  --max-tokens 8 --json-out "$LOG_DIR/pic_pd_concurrency.json" || RC=1

# Shape-C semantics plus P/D-specific evidence: TransferEngine moved GDN/conv
# pages, and D scheduled only the relayed token rather than the whole prompt.
echo "===== shape C semantics + transport evidence ====="
PD_BASELINE_ARGS=()
if [ -n "${PIC_PD_BASELINE_URL:-}" ]; then
  PD_BASELINE_ARGS=(--baseline-url "$PIC_PD_BASELINE_URL")
fi
"$PYTHON" "$REPO_ROOT/tests/v1/pic/run_pd_transfer_probe.py" \
  --base-url "http://127.0.0.1:$PROXY_PORT" --model qwen3.5-2b \
  --prompt-file "$REPO_ROOT/tests/v1/pic/fixtures/c/reuse.txt" \
  --decode-log "$LOG_DIR/pic_pd_decode.log" \
  --prefill-log "$LOG_DIR/pic_pd_prefill.log" \
  "${PD_BASELINE_ARGS[@]}" \
  --json-out "$LOG_DIR/pic_pd_semantics.json" || RC=1

# Optional independent experiment: restart only P with local prefix caching
# enabled and prove hits_delta > 0. It is deliberately not part of the P/D
# verdict, because P/D handoff does not require local cache reuse.
if [ "${PIC_PD_CHECK_PREFILL_REUSE:-0}" = "1" ]; then
  echo "===== independent Prefill local-reuse experiment ====="
  bash "$SCRIPT_DIR/pic_prefill_reuse_e2e.sh" || \
    echo "prefill local reuse experiment failed (P/D verdict unchanged)" >&2
fi

keep_hint() {
  echo "" >&2
  echo "=== P/D services left running for debugging ===" >&2
  echo "  P:     http://127.0.0.1:$P_PORT/v1/models" >&2
  echo "  D:     http://127.0.0.1:$D_PORT/v1/models" >&2
  echo "  proxy: http://127.0.0.1:$PROXY_PORT/health" >&2
  echo "  logs:  $LOG_DIR/pic_pd_{prefill,decode,proxy}.log" >&2
  echo "  clean: pkill -f 'vllm serve'; pkill -f pic_pd_proxy" >&2
}

if [ "$RC" -eq 0 ]; then
  echo "=== P/D TransferEngine: PASS ==="
  if [ "${PIC_PD_CHECK_PREFILL_REUSE:-0}" = "1" ]; then
    echo "reuse: P-local=see independent pic_prefill_reuse_e2e.sh result above"
  else
    echo "reuse: P-local=disabled(--no-enable-prefix-caching) pic=disabled"
  fi
  echo "note: D external 100% = P→D per-request handoff, NOT P cache reuse"
  echo "services are still running; stop them with: pkill -f 'vllm serve'; pkill -f pic_pd_proxy"
else
  echo "=== P/D TransferEngine: FAIL (see $LOG_DIR/pic_pd_*.log) ===" >&2
  keep_hint
fi
exit "$RC"
