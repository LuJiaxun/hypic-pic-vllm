#!/usr/bin/env bash
# Stock vLLM + Mooncake P/D + native prefix-cache experiment.
# No PIC flags, no custom vLLM source edits, no custom proxy file.
set -uo pipefail

ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MODEL="${PIC_MODEL:-/path/to/Qwen3.5-2B}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/tests/v1/pic/fixtures/c/reuse.txt}"
LOG_DIR="${PIC_LOG_DIR:-/tmp}"
P_LOG="$LOG_DIR/native_pd_prefill.log"
D_LOG="$LOG_DIR/native_pd_decode.log"
PROXY_LOG="$LOG_DIR/native_pd_proxy.log"
MASTER_LOG="$LOG_DIR/native_pd_master.log"
CONFIG="${MOONCAKE_CONFIG_PATH:-$LOG_DIR/native_pd_mooncake.json}"
PROXY_PY="$LOG_DIR/native_pd_proxy.py"
MASTER_PORT="${MOONCAKE_MASTER_PORT:-50088}"
PYTHON="${PIC_PYTHON:-python3}"
mkdir -p "$LOG_DIR"

pkill -f 'vllm serve' 2>/dev/null || true
pkill -f 'mooncake_master' 2>/dev/null || true
pkill -f 'native_pd_proxy.py' 2>/dev/null || true
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

nohup mooncake_master --rpc_port "$MASTER_PORT" --metrics_port 9004 \
  --client_ttl 3600 --default_kv_lease_ttl 3600000 \
  --default_kv_soft_pin_ttl 7200000 >"$MASTER_LOG" 2>&1 &
MASTER_PID=$!
disown "$MASTER_PID" 2>/dev/null || true
sleep 3
if ! kill -0 "$MASTER_PID" 2>/dev/null; then
  echo "FATAL: mooncake_master exited; see $MASTER_LOG" >&2
  tail -n 50 "$MASTER_LOG" >&2 || true
  exit 2
fi

common=(
  "$MODEL" --served-model-name qwen3.5-2b
  --max-model-len 8192 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.5 --enforce-eager
  --enable-prefix-caching
  --mamba-cache-mode align
)
P_CONFIG='{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"tcp"}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"mamba_cache_mode":"align"}}]}}'
D_CONFIG='{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"tcp"}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"mamba_cache_mode":"align"}}]}}'

(
  export CUDA_VISIBLE_DEVICES="${PIC_CUDA_P:-0}"
  export MOONCAKE_CONFIG_PATH="$CONFIG"
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
  nohup vllm serve "${common[@]}" --port 8010 --engine-id p-engine \
    --kv-transfer-config "$P_CONFIG" >"$P_LOG" 2>&1
) &
P_PID=$!
disown "$P_PID" 2>/dev/null || true

(
  export CUDA_VISIBLE_DEVICES="${PIC_CUDA_D:-1}"
  export MOONCAKE_CONFIG_PATH="$CONFIG"
  nohup vllm serve "${common[@]}" --port 8020 --engine-id d-engine \
    --kv-transfer-config "$D_CONFIG" >"$D_LOG" 2>&1
) &
D_PID=$!
disown "$D_PID" 2>/dev/null || true

for port in 8010 8020; do
  ready=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "FATAL: vLLM on $port did not become ready" >&2
    [ "$port" = 8010 ] && tail -n 80 "$P_LOG" >&2 || tail -n 80 "$D_LOG" >&2
    exit 2
  fi
done

# A standalone proxy, created only for this experiment. It uses the stock
# Mooncake P/D handoff parameters and does not modify the vLLM installation.
cat > "$PROXY_PY" <<'PY'
import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

P_URL = os.environ.get("PD_P_URL", "http://127.0.0.1:8010")
D_URL = os.environ.get("PD_D_URL", "http://127.0.0.1:8020")
P_ENGINE = os.environ.get("PD_ENGINE_ID", "p-engine")
P_HOST = os.environ.get("PD_HOST", "127.0.0.1")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, body):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(req, timeout=600) as response:
        return json.loads(response.read().decode())


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        request_id = request.get("id", "native-pd-request")
        p_body = dict(request)
        p_body["max_tokens"] = 1
        p_body["temperature"] = 0.0
        p_body["stream"] = False
        p_body["return_token_ids"] = True
        p_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "transfer_id": request_id,
        }
        p_response = post(P_URL + "/v1/completions", p_body)
        prompt_ids = list(p_response.get("prompt_token_ids") or [])
        first = p_response["choices"][0]
        first_ids = list(first.get("token_ids") or [])
        if not prompt_ids or not first_ids:
            self.send_error(502, "P response has no prompt/first token ids")
            return
        d_body = dict(request)
        d_body["prompt_token_ids"] = prompt_ids + [first_ids[0]]
        d_body["stream"] = False
        d_body["return_token_ids"] = True
        d_body["kv_transfer_params"] = {
            "do_remote_prefill": True,
            "remote_engine_id": P_ENGINE,
            "remote_bootstrap_addr": P_HOST,
            "transfer_id": request_id,
        }
        d_response = post(D_URL + "/v1/completions", d_body)
        raw = json.dumps(d_response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
PY

PD_HOST="$LOCAL_IP" nohup "$PYTHON" "$PROXY_PY" >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
disown "$PROXY_PID" 2>/dev/null || true
sleep 2
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
  echo "FATAL: proxy exited; see $PROXY_LOG" >&2
  tail -n 50 "$PROXY_LOG" >&2 || true
  exit 2
fi

export P_PORT=8010 PROXY_PORT=8000 PROMPT_FILE P_LOG
"$PYTHON" - <<'PY'
import json
import os
import re
import urllib.request
from pathlib import Path

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post_prompt():
    body = json.dumps({
        "model": "qwen3.5-2b",
        "prompt": Path(os.environ["PROMPT_FILE"]).read_text(encoding="utf-8").strip(),
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
        "return_token_ids": True,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=600) as response:
        return json.loads(response.read().decode())


def hits():
    req = urllib.request.Request("http://127.0.0.1:8010/metrics")
    with opener.open(req, timeout=30) as response:
        text = response.read().decode("utf-8", "replace")
    values = []
    for line in text.splitlines():
        if line.startswith("vllm:external_prefix_cache_hits"):
            values.append(float(line.rsplit(" ", 1)[1]))
    return sum(values)


before = hits()
first = post_prompt()
print("first answer:", first["choices"][0].get("text", ""))
second = post_prompt()
after = hits()
print("second answer:", second["choices"][0].get("text", ""))
p_log = Path(os.environ["P_LOG"]).read_text(encoding="utf-8", errors="replace")
store_hits = [int(v) for v in re.findall(r"kvpool hit tokens:\s*(\d+)", p_log)]
result = {
    "hits_before": before,
    "hits_after": after,
    "hits_delta": after - before,
    "store_log_hits": store_hits[-4:],
    "pass": after > before or any(v > 0 for v in store_hits),
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["pass"] else 1)
PY
RC=$?
if [ "$RC" -eq 0 ]; then
  echo "=== native vLLM P/D + Mooncake Store prefix reuse: PASS ==="
else
  echo "=== native prefix reuse: FAIL (exit=$RC) ===" >&2
  grep -Ei 'external_prefix_cache|kvpool hit tokens|error|Traceback' "$P_LOG" | tail -40 >&2 || true
fi
echo "P log: $P_LOG"
echo "D log: $D_LOG"
echo "proxy log: $PROXY_LOG"
exit "$RC"
