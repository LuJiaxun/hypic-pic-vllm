#!/usr/bin/env bash
# End-to-end Prefill local-prefix-cache reuse check.
#
# Restarts only the Prefill server with --enable-prefix-caching, sends the same
# prompt twice through the existing probe, and checks prefix_cache_hits delta.
# P/D and PIC are not required. The server is left running for inspection unless
# PIC_REUSE_CLEANUP_ON_EXIT=1 is set.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${PIC_MODEL:-/path/to/Qwen3.5-2B}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
CUDA_P="${PIC_CUDA_P:-0}"
PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/tests/v1/pic/fixtures/c/reuse.txt}"
LOG_DIR="${PIC_LOG_DIR:-/tmp}"
PREFILL_LOG="${PREFILL_LOG:-$LOG_DIR/pic_prefill_reuse_e2e.log}"
JSON_OUT="${JSON_OUT:-$LOG_DIR/prefill_reuse_e2e.json}"
MAX_TOKENS="${MAX_TOKENS:-8}"
REPEAT="${REPEAT:-2}"
PYTHON="${PIC_PYTHON:-python3}"
mkdir -p "$LOG_DIR"

stop_prefill() {
  pgrep -f "vllm serve.*$PREFILL_PORT" | xargs -r kill 2>/dev/null || true
}

if [ "${PIC_REUSE_CLEANUP_ON_EXIT:-0}" = "1" ]; then
  trap stop_prefill EXIT
fi

# This experiment changes only P. D/proxy are deliberately left alone.
stop_prefill
sleep 3

common=(
  "$MODEL" --served-model-name qwen3.5-2b
  --port "$PREFILL_PORT"
  --max-model-len 8192 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.5 --enforce-eager
  --enable-prefix-caching
)

(
  export CUDA_VISIBLE_DEVICES="$CUDA_P"
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
  nohup vllm serve "${common[@]}" \
    --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$PROTOCOL\"}}" \
    >"$PREFILL_LOG" 2>&1
) &
P_PID=$!
disown "$P_PID" 2>/dev/null || true

echo "===== Prefill reuse experiment ====="
echo "model=$MODEL port=$PREFILL_PORT log=$PREFILL_LOG"
echo "prefix caching: enabled; PIC: not required"

up=0
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PREFILL_PORT/v1/models" >/dev/null 2>&1; then
    up=1
    break
  fi
  if ! kill -0 "$P_PID" 2>/dev/null; then
    break
  fi
  sleep 2
done
if [ "$up" -ne 1 ]; then
  echo "FATAL: Prefill did not become ready on port $PREFILL_PORT" >&2
  tail -n 60 "$PREFILL_LOG" >&2 || true
  exit 2
fi

set +e
PREFILL_URL="http://127.0.0.1:$PREFILL_PORT" \
PREFILL_LOG="$PREFILL_LOG" \
PROMPT_FILE="$PROMPT_FILE" \
MAX_TOKENS="$MAX_TOKENS" \
REPEAT="$REPEAT" \
JSON_OUT="$JSON_OUT" \
PIC_PYTHON="$PYTHON" \
  bash "$SCRIPT_DIR/pic_prefill_reuse_probe.sh"
RC=$?
set -e

echo ""
if [ "$RC" -eq 0 ]; then
  echo "=== Prefill local prefix reuse: PASS ==="
  echo "prefix_cache_hits increased after repeated identical prompts."
else
  echo "=== Prefill local prefix reuse: FAIL (exit=$RC) ===" >&2
  echo "Check that the running config has --enable-prefix-caching and that both requests used the same prompt." >&2
  grep -Ei 'enable_prefix_caching|prefix.caching|prefix cache' "$PREFILL_LOG" | head -30 >&2 || true
fi

echo ""
echo "metrics: curl -s http://127.0.0.1:$PREFILL_PORT/metrics | grep -E 'vllm:prefix_cache_(queries|hits)'"
echo "log:     $PREFILL_LOG"
echo "json:    $JSON_OUT"
echo "clean:   pgrep -f 'vllm serve.*$PREFILL_PORT' | xargs -r kill"
exit "$RC"
