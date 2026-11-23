#!/usr/bin/env bash
# P/D + Mooncake Store reuse experiment.
#
# Request 1 goes through the P/D proxy and publishes KV to the shared Store.
# Request 2 repeats the same prompt: P must hit the Store during prefill, while
# D still receives the request through MooncakeConnector. Reuse is not assumed
# to be bit-exact: acceptance is Store evidence plus the expected semantic word.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${PIC_MODEL:-/path/to/Qwen3.5-2B}"
P_PORT="${PREFILL_PORT:-8010}"
D_PORT="${DECODE_PORT:-8020}"
PROXY_PORT="${PROXY_PORT:-8000}"
CUDA_P="${PIC_CUDA_P:-0}"
CUDA_D="${PIC_CUDA_D:-1}"
PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
MASTER_PORT="${MOONCAKE_MASTER_PORT:-50088}"
LOG_DIR="${PIC_LOG_DIR:-/tmp}"
P_LOG="$LOG_DIR/pic_pd_store_prefill.log"
D_LOG="$LOG_DIR/pic_pd_store_decode.log"
PROXY_LOG="$LOG_DIR/pic_pd_store_proxy.log"
MASTER_LOG="$LOG_DIR/pic_pd_store_master.log"
CONFIG="${MOONCAKE_CONFIG_PATH:-$LOG_DIR/mooncake_pd_store.json}"
JSON_OUT="${JSON_OUT:-$LOG_DIR/pic_pd_store_reuse.json}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/tests/v1/pic/fixtures/c/reuse.txt}"
MAX_TOKENS="${MAX_TOKENS:-32}"
EXPECT="${EXPECT:-dog}"
STORE_SETTLE_SECS="${STORE_SETTLE_SECS:-5}"
PYTHON="${PIC_PYTHON:-python3}"
mkdir -p "$LOG_DIR"

if ! command -v mooncake_master >/dev/null 2>&1; then
  echo "FATAL: mooncake_master is not on PATH"
  exit 2
fi
PROXY="$SCRIPT_DIR/pic_pd_proxy.py"
if [ ! -f "$PROXY" ]; then
  echo "FATAL: missing proxy: $PROXY"
  exit 2
fi

cleanup() {
  for pattern in pic_pd_proxy.py 'vllm serve' mooncake_master; do
    pkill -f "$pattern" 2>/dev/null || true
  done
}
if [ "${PIC_REUSE_CLEANUP_ON_EXIT:-0}" = "1" ]; then
  trap cleanup EXIT
fi

# A fresh master makes request 1 a Store write and request 2 a Store read.
pgrep -f "mooncake_master.*$MASTER_PORT" | xargs -r kill 2>/dev/null || true
pgrep -f 'vllm serve' | xargs -r kill 2>/dev/null || true
pkill -f pic_pd_proxy.py 2>/dev/null || true
sleep 3

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

nohup mooncake_master --port "$MASTER_PORT" >"$MASTER_LOG" 2>&1 &
MASTER_PID=$!
disown "$MASTER_PID" 2>/dev/null || true
sleep 3
if ! kill -0 "$MASTER_PID" 2>/dev/null; then
  echo "FATAL: mooncake_master exited; see $MASTER_LOG" >&2
  tail -n 40 "$MASTER_LOG" >&2 || true
  exit 2
fi

common=(
  "$MODEL" --served-model-name qwen3.5-2b
  --max-model-len 8192 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.5 --enforce-eager
  --enable-prefix-caching
  --mamba-cache-mode align
)

P_CONFIG="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$PROTOCOL\"}},{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"mamba_cache_mode\":\"align\"}}]}}"
D_CONFIG="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$PROTOCOL\"}},{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"mamba_cache_mode\":\"align\"}}]}}"

(
  export CUDA_VISIBLE_DEVICES="$CUDA_P"
  export MOONCAKE_CONFIG_PATH="$CONFIG"
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
  nohup vllm serve "${common[@]}" --port "$P_PORT" \
    --kv-transfer-config "$P_CONFIG" >"$P_LOG" 2>&1
) &
P_PID=$!
disown "$P_PID" 2>/dev/null || true

(
  export CUDA_VISIBLE_DEVICES="$CUDA_D"
  export MOONCAKE_CONFIG_PATH="$CONFIG"
  nohup vllm serve "${common[@]}" --port "$D_PORT" \
    --kv-transfer-config "$D_CONFIG" >"$D_LOG" 2>&1
) &
D_PID=$!
disown "$D_PID" 2>/dev/null || true

for entry in "prefill:$P_PORT" "decode:$D_PORT"; do
  name="${entry%%:*}"
  port="${entry##*:}"
  up=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      up=1
      break
    fi
    sleep 2
  done
  if [ "$up" -ne 1 ]; then
    echo "FATAL: $name did not become ready" >&2
    echo "--- last 100 log lines ---" >&2
    tail -n 100 "$LOG_DIR/pic_pd_store_$name.log" >&2 || true
    echo "--- connector/layout errors ---" >&2
    grep -Ei 'ValueError|MambaSpec|MooncakeStore|mamba_cache_mode|EngineCore' \
      "$LOG_DIR/pic_pd_store_$name.log" | tail -40 >&2 || true
    exit 2
  fi
done

nohup "$PYTHON" "$PROXY" \
  --prefill "http://127.0.0.1:$P_PORT" 8998 \
  --decode "http://127.0.0.1:$D_PORT" \
  --relay-first-token --port "$PROXY_PORT" >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
disown "$PROXY_PID" 2>/dev/null || true

for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "FATAL: proxy exited; see $PROXY_LOG" >&2
    tail -n 40 "$PROXY_LOG" >&2 || true
    exit 2
  fi
  sleep 1
done

export P_PORT PROXY_PORT PROMPT_FILE MAX_TOKENS EXPECT STORE_SETTLE_SECS JSON_OUT P_LOG D_LOG
"$PYTHON" - <<'PY'
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
proxy = f"http://127.0.0.1:{os.environ['PROXY_PORT']}"
prefill = f"http://127.0.0.1:{os.environ['P_PORT']}"


def metrics():
    request = urllib.request.Request(prefill + "/metrics")
    with opener.open(request, timeout=30.0) as response:
        text = response.read().decode("utf-8", "replace")
    values = {"queries": 0.0, "hits": 0.0}
    for name in values:
        pattern = re.compile(
            rf"^vllm:external_prefix_cache_{name}(?:_total)?(?:\{{[^}}]*\}})?\s+"
            rf"([0-9.eE+-]+)\s*$",
            re.MULTILINE,
        )
        values[name] = sum(float(match.group(1)) for match in pattern.finditer(text))
    return values


def post(prompt):
    body = json.dumps(
        {
            "model": "qwen3.5-2b",
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": int(os.environ["MAX_TOKENS"]),
            "temperature": 0.0,
            "stream": False,
            "ignore_eos": True,
            "return_token_ids": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        proxy + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=600.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    text = message.get("content") or message.get("reasoning") or ""
    return text, list(choice.get("token_ids") or [])


prompt = Path(os.environ["PROMPT_FILE"]).read_text(encoding="utf-8").strip()
before = metrics()
first_text, first_ids = post(prompt)
print(f"request 1 (publish): tokens={len(first_ids)} text={first_text!r}")
time.sleep(float(os.environ["STORE_SETTLE_SECS"]))
second_text, second_ids = post(prompt)
print(f"request 2 (reuse):  tokens={len(second_ids)} text={second_text!r}")
after = metrics()
p_log = Path(os.environ["P_LOG"]).read_text(encoding="utf-8", errors="replace")
d_log = Path(os.environ["D_LOG"]).read_text(encoding="utf-8", errors="replace")
store_hits = [int(v) for v in re.findall(r"kvpool hit tokens:\s*(\d+)", p_log)]
result = {
    "queries_before": before["queries"],
    "queries_after": after["queries"],
    "queries_delta": after["queries"] - before["queries"],
    "hits_before": before["hits"],
    "hits_after": after["hits"],
    "hits_delta": after["hits"] - before["hits"],
    "store_log_hits": store_hits[-4:],
    "store_reuse": after["hits"] > before["hits"] or any(v > 0 for v in store_hits),
    "expected": os.environ["EXPECT"],
    "expected_in_second": os.environ["EXPECT"].lower() in second_text.lower(),
    "first_answer": first_text,
    "second_answer": second_text,
    "pd_relay_events": len(re.findall(r"PIC-PD relay request=", d_log)),
    "pd_state_events": len(re.findall(r"PIC-PD state request=", p_log)),
}
result["pass"] = result["store_reuse"] and result["expected_in_second"]
print(json.dumps(result, ensure_ascii=False, indent=2))
Path(os.environ["JSON_OUT"]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
sys.exit(0 if result["pass"] else 1)
PY
RC=$?

echo ""
if [ "$RC" -eq 0 ]; then
  echo "=== P/D + Mooncake Store reuse: PASS ==="
  echo "request 1 published KV; request 2 hit the Store on P; semantic check '${EXPECT}' passed."
else
  echo "=== P/D + Mooncake Store reuse: FAIL (exit=$RC) ===" >&2
  echo "P external hits and request logs:" >&2
  grep -Ei 'external_prefix_cache|kvpool hit tokens|PIC-PD state|batch_put|lookup' "$P_LOG" | tail -40 >&2 || true
fi
echo "P log:     $P_LOG"
echo "D log:     $D_LOG"
echo "proxy log: $PROXY_LOG"
echo "json:      $JSON_OUT"
exit "$RC"
