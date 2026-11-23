#!/usr/bin/env bash
# Two prefill-role services, no decode/proxy: P1 publishes, P2 reuses.
# Both have PIC custom Store and MooncakeConnector(kv_producer).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FIX="$REPO_ROOT/tests/v1/pic/fixtures/c"
LOG_DIR="${PIC_LOG_DIR:-/tmp}"
P1_LOG="$LOG_DIR/pic_two_p_publish.log"
P2_LOG="$LOG_DIR/pic_two_p_reuse.log"
MASTER_LOG="$LOG_DIR/pic_two_p_master.log"
CONFIG="${MOONCAKE_CONFIG_PATH:-$LOG_DIR/mooncake_two_p.json}"
MODEL="${PIC_MODEL:-/path/to/Qwen3.5-2B}"
MASTER_PORT="${MOONCAKE_MASTER_PORT:-50088}"
P1_PORT="${PIC_P1_PORT:-8000}"
P2_PORT="${PIC_P2_PORT:-8001}"
PYTHON="${PIC_PYTHON:-python3}"
export PIC_DEBUG=1
export HYPIC_PIC_DEBUG=1
mkdir -p "$LOG_DIR"

cleanup() {
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -f 'mooncake_master' 2>/dev/null || true
}
if [ "${PIC_REUSE_CLEANUP_ON_EXIT:-0}" = "1" ]; then
  trap cleanup EXIT
fi

report_failure() {
  echo "=== two-P experiment failed; recent logs ===" >&2
  for file in "$P1_LOG" "$P2_LOG" "$MASTER_LOG"; do
    if [ -f "$file" ]; then
      echo "--- $file (tail 40) ---" >&2
      tail -n 40 "$file" >&2 || true
    fi
  done
}
trap report_failure ERR

pkill -f 'vllm serve' 2>/dev/null || true
pkill -f 'mooncake_master' 2>/dev/null || true
sleep 5
rm -rf /tmp/mooncake_store* /var/lib/mooncake* ~/.mooncake*

LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
LOCAL_IP="${LOCAL_IP:-127.0.0.1}"
cat > "$CONFIG" <<JSON
{
  "local_hostname": "$LOCAL_IP",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "tcp",
  "device_name": "",
  "master_server_address": "$LOCAL_IP:$MASTER_PORT",
  "global_segment_size": 8589934592,
  "local_buffer_size": 2147483648
}
JSON

nohup mooncake_master --rpc_port "$MASTER_PORT" --metrics_port 9004 \
  --client_ttl 3600 --default_kv_lease_ttl 3600000 \
  --default_kv_soft_pin_ttl 7200000 >"$MASTER_LOG" 2>&1 &
MASTER_PID=$!
disown "$MASTER_PID" 2>/dev/null || true
sleep 3
if ! kill -0 "$MASTER_PID" 2>/dev/null; then
  echo "FATAL: mooncake_master exited; see $MASTER_LOG" >&2
  tail -n 60 "$MASTER_LOG" >&2 || true
  exit 2
fi

P_CONFIG='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"tcp"}}'
if [ "${PIC_DISABLE_PD_REGISTRATION:-0}" = "1" ]; then
  # Diagnostic only: no batch_register_memory and no sender listener.
  P_CONFIG='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"tcp","disable_kv_cache_registration":true}}'
  echo "DIAG: batch_register_memory is disabled; this is not a real P/D run."
fi
if [ "${PIC_DISABLE_TRANSFER_ENGINE:-0}" = "1" ]; then
  # Diagnostic only: no TransferEngine.initialize and no registration.
  P_CONFIG='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"tcp","disable_transfer_engine":true,"disable_kv_cache_registration":true}}'
  echo "DIAG: TransferEngine.initialize is disabled; this is not a real P/D run."
fi
common=(
  "$MODEL" --served-model-name qwen3.5-2b
  --max-model-len 8192 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.5 --enforce-eager
  --no-enable-prefix-caching
)

(
  export CUDA_VISIBLE_DEVICES="${PIC_CUDA_P1:-0}"
  export MOONCAKE_CONFIG_PATH="$CONFIG"
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
  nohup vllm serve "${common[@]}" --port "$P1_PORT" \
    --pic-enable --pic-max-cache-bytes 4294967296 \
    --pic-min-segment-tokens 512 --pic-mooncake --pic-debug \
    --kv-transfer-config "$P_CONFIG" >"$P1_LOG" 2>&1
) &
P1_PID=$!
disown "$P1_PID" 2>/dev/null || true

(
  export CUDA_VISIBLE_DEVICES="${PIC_CUDA_P2:-1}"
  export MOONCAKE_CONFIG_PATH="$CONFIG"
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=8999
  nohup vllm serve "${common[@]}" --port "$P2_PORT" \
    --pic-enable --pic-max-cache-bytes 4294967296 \
    --pic-min-segment-tokens 512 --pic-mooncake --pic-debug \
    --kv-transfer-config "$P_CONFIG" >"$P2_LOG" 2>&1
) &
P2_PID=$!
disown "$P2_PID" 2>/dev/null || true

wait_ready() {
  local pid="$1" port="$2" name="$3" log="$4"
  local deadline=$((SECONDS + ${READY_TIMEOUT:-300}))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "FATAL: $name P exited before ready (pid=$pid)" >&2
      tail -n 60 "$log" >&2 || true
      return 1
    fi
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "$name P is ready on port $port"
      return 0
    fi
    sleep 2
  done
  echo "FATAL: $name P did not become ready within ${READY_TIMEOUT:-300}s" >&2
  tail -n 60 "$log" >&2 || true
  return 1
}

wait_ready "$P1_PID" "$P1_PORT" publish "$P1_LOG" || exit 1
wait_ready "$P2_PID" "$P2_PORT" reuse "$P2_LOG" || exit 1

wait_published() {
  local expected="$1" name="$2"
  local deadline=$((SECONDS + ${PUBLISH_TIMEOUT:-300}))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$P1_PID" 2>/dev/null; then
      echo "FATAL: publish P exited while waiting for $name (pid=$P1_PID)" >&2
      tail -n 80 "$P1_LOG" >&2 || true
      return 1
    fi
    local kv state
    kv=$(grep -c 'Mooncake native KV published' "$P1_LOG" 2>/dev/null || true)
    state=$(grep -c 'Mooncake hybrid transition published' "$P1_LOG" 2>/dev/null || true)
    if [ "${kv:-0}" -ge "$expected" ] && [ "${state:-0}" -ge "$expected" ]; then
      echo "  $name: Store publish confirmed (native_kv=$kv, hybrid_state=$state)"
      return 0
    fi
    sleep 1
  done
  echo "FATAL: $name publish was not confirmed within ${PUBLISH_TIMEOUT:-300}s" >&2
  tail -n 80 "$P1_LOG" >&2 || true
  return 1
}

for i in 1 2 3; do
  case "$i" in
    1) name=publish_a ;;
    2) name=publish_b ;;
    3) name=publish_c ;;
  esac
  echo "===== P1 publish $name ====="
  set +e
  "$PYTHON" - "$FIX/$name.txt" "$P1_PORT" <<'PY'
import json
import sys
import urllib.request

prompt = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({"model": "qwen3.5-2b", "prompt": prompt, "max_tokens": 1}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[2]}/v1/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=600) as response:
    print(response.read().decode()[:80])
PY
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "FATAL: $name request failed (exit=$rc)" >&2
    tail -n 80 "$P1_LOG" >&2 || true
    exit "$rc"
  fi
  wait_published "$i" "$name" || exit 1
  sleep 3
done

echo "===== P2 reuse after all three Store publications ====="
sleep 5
set +e
"$PYTHON" - "$FIX/reuse.txt" "$P2_PORT" <<'PY'
import json
import sys
import urllib.request

prompt = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({
    "model": "qwen3.5-2b",
    "prompt": prompt,
    "temperature": 0,
    "max_tokens": 32,
}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[2]}/v1/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=300) as response:
    result = json.loads(response.read().decode())
answer = result["choices"][0]["text"]
print(f"answer={answer!r}")
print(f"contains_dog={'dog' in answer.lower()}")
if "dog" not in answer.lower():
    raise SystemExit(1)
PY
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  echo "=== two-P reuse: FAIL (semantic/request exit=$RC) ===" >&2
  tail -n 80 "$P2_LOG" >&2 || true
  exit "$RC"
fi

set +e
"$PYTHON" - "$P2_LOG" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
values = {}
for name in ("lookup_hits", "matched_segments", "reused_tokens", "external_kv_imported"):
    found = re.findall(rf"['\"]{name}['\"]\s*:\s*(\d+)", text)
    values[name] = int(found[-1]) if found else 0
print("P2 metrics:", values)
checks = {
    "lookup_hits": values["lookup_hits"] > 0,
    "matched_segments": values["matched_segments"] > 0,
    "reused_tokens": values["reused_tokens"] > 0,
    "external_kv_imported": values["external_kv_imported"] > 0,
    "transition_applied": "transition applied" in text,
    "executed": "single-request early runtime executed" in text,
    "no_fallback": "'fallback_reasons': {}" in text,
}
for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'} {name}")
if not all(checks.values()):
    raise SystemExit(1)
PY
RC=$?
set -e
if [ "$RC" -eq 0 ]; then
  echo "=== two-P producer publish(a,b,c) -> reuse: PASS ==="
else
  echo "=== two-P producer reuse: FAIL (evidence exit=$RC) ===" >&2
fi
echo "P1 log: $P1_LOG"
echo "P2 log: $P2_LOG"
echo "master log: $MASTER_LOG"
exit "$RC"
