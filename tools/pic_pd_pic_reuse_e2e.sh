#!/usr/bin/env bash
# P/D + PIC segment-reuse experiment.
#
# MooncakeConnector carries the current request from P to D. PIC's own Store
# serves cross-request segment reuse on P. Request 1 publishes PIC segments;
# request 2 must hit them on P while D still receives the P/D state handoff.
# Acceptance is PIC lookup evidence plus the expected semantic word, not exact
# output tokens (reuse is not assumed to be bit-exact).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${PIC_MODEL:-/path/to/Qwen3.5-2B}"
P_PORT="${PREFILL_PORT:-8010}"
D_PORT="${DECODE_PORT:-8020}"
PROXY_PORT="${PROXY_PORT:-8000}"
PURE_B_PORT="${PIC_PURE_B_PORT:-8001}"
CUDA_P="${PIC_CUDA_P:-0}"
CUDA_D="${PIC_CUDA_D:-1}"
PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
MASTER_PORT="${MOONCAKE_MASTER_PORT:-50088}"
LOG_DIR="${PIC_LOG_DIR:-/tmp}"
P_LOG="$LOG_DIR/pic_pd_pic_prefill.log"
D_LOG="$LOG_DIR/pic_pd_pic_decode.log"
PURE_B_LOG="$LOG_DIR/pic_pd_pic_pure_b.log"
REUSE_LOG="$P_LOG"
if [ "${PIC_PURE:-0}" = "1" ]; then
  REUSE_LOG="$PURE_B_LOG"
fi
PROXY_LOG="$LOG_DIR/pic_pd_pic_proxy.log"
MASTER_LOG="$LOG_DIR/pic_pd_pic_master.log"
CONFIG="${MOONCAKE_CONFIG_PATH:-$LOG_DIR/mooncake_pd_pic.json}"
JSON_OUT="${JSON_OUT:-$LOG_DIR/pic_pd_pic_reuse.json}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/tests/v1/pic/fixtures/c/reuse.txt}"
# Optional single-file override. By default publish a/b/c in order, matching shape C.
PUBLISH_FILE="${PUBLISH_FILE:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
EXPECT="${EXPECT:-dog}"
STORE_SETTLE_SECS="${STORE_SETTLE_SECS:-5}"
PYTHON="${PIC_PYTHON:-python3}"
mkdir -p "$LOG_DIR"

# Export these explicitly so CUDA assertion failures point at the launch site.
# PIC_MINIMAL=1 starts only P and uses raw completions, without D or the proxy.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-0}"
export PIC_DEBUG="${PIC_DEBUG:-0}"
export HYPIC_PIC_DEBUG="${HYPIC_PIC_DEBUG:-0}"
PIC_MINIMAL="${PIC_MINIMAL:-0}"
# PIC_PURE=1 starts P without MooncakeConnector, for a true colocated-PIC
# control run. Use together with PIC_MINIMAL=1; D/proxy stay disabled.
PIC_PURE="${PIC_PURE:-0}"
if [ "$PIC_PURE" = "1" ] && [ "$PIC_MINIMAL" != "1" ]; then
  echo "FATAL: PIC_PURE=1 requires PIC_MINIMAL=1 (no P/D handoff in the control run)" >&2
  exit 2
fi
if [ -z "$MAX_TOKENS" ]; then
  if [ "$PIC_MINIMAL" = "1" ]; then
    MAX_TOKENS=1
  else
    MAX_TOKENS=32
  fi
fi

report_cuda_errors() {
  for file in "$P_LOG" "$D_LOG" "$PROXY_LOG"; do
    if [ -f "$file" ] && grep -Eq 'device-side assert|CUDA error|Assertion|index out of|illegal memory' "$file"; then
      echo "===== CUDA/assert diagnostics: $file =====" >&2
      grep -n -B20 -A40 -E 'device-side assert|CUDA error|Assertion|index out of|illegal memory' "$file" | tail -200 >&2
    fi
  done
}
trap report_cuda_errors EXIT

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

nohup mooncake_master --rpc_port "$MASTER_PORT" --metrics_port 9004 \
  --client_ttl 3600 --default_kv_lease_ttl 3600000 \
  --default_kv_soft_pin_ttl 7200000 >"$MASTER_LOG" 2>&1 &
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
  --no-enable-prefix-caching
)

P_CONFIG="{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$PROTOCOL\"}}"
D_CONFIG="{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$PROTOCOL\"}}"
P_CONNECTOR_ARGS=(--kv-transfer-config "$P_CONFIG")
if [ "$PIC_PURE" = "1" ]; then
  P_CONNECTOR_ARGS=()
fi

(
  export CUDA_VISIBLE_DEVICES="$CUDA_P"
  export MOONCAKE_CONFIG_PATH="$CONFIG"
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
  nohup vllm serve "${common[@]}" --port "$P_PORT" \
    --pic-enable --pic-max-cache-bytes 4294967296 \
    --pic-min-segment-tokens 512 --pic-mooncake \
    "${P_CONNECTOR_ARGS[@]}" >"$P_LOG" 2>&1
) &
P_PID=$!
disown "$P_PID" 2>/dev/null || true

if [ "$PIC_PURE" = "1" ]; then
  (
    export CUDA_VISIBLE_DEVICES="$CUDA_D"
    export MOONCAKE_CONFIG_PATH="$CONFIG"
    nohup vllm serve "${common[@]}" --port "$PURE_B_PORT" \
      --pic-enable --pic-max-cache-bytes 4294967296 \
      --pic-min-segment-tokens 512 --pic-mooncake >"$PURE_B_LOG" 2>&1
  ) &
  PURE_B_PID=$!
  disown "$PURE_B_PID" 2>/dev/null || true
fi

if [ "$PIC_MINIMAL" != "1" ]; then
  (
    export CUDA_VISIBLE_DEVICES="$CUDA_D"
    export MOONCAKE_CONFIG_PATH="$CONFIG"
    nohup vllm serve "${common[@]}" --port "$D_PORT" \
      --kv-transfer-config "$D_CONFIG" >"$D_LOG" 2>&1
  ) &
  D_PID=$!
  disown "$D_PID" 2>/dev/null || true
fi

entries="prefill:$P_PORT"
if [ "$PIC_PURE" = "1" ]; then
  entries="$entries pure_b:$PURE_B_PORT"
fi
if [ "$PIC_MINIMAL" != "1" ]; then
  entries="$entries decode:$D_PORT"
fi
for entry in $entries; do
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
    tail -n 100 "$LOG_DIR/pic_pd_pic_$name.log" >&2 || true
    echo "--- connector/layout errors ---" >&2
    grep -Ei 'ValueError|MambaSpec|MooncakeStore|mamba_cache_mode|EngineCore' \
      "$LOG_DIR/pic_pd_pic_$name.log" | tail -40 >&2 || true
    exit 2
  fi
done

if [ "$PIC_MINIMAL" != "1" ]; then
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
fi

export P_PORT PROXY_PORT PROMPT_FILE PUBLISH_FILE MAX_TOKENS EXPECT STORE_SETTLE_SECS JSON_OUT P_LOG D_LOG PIC_MINIMAL REUSE_LOG PURE_B_PORT PIC_PURE
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
    text = Path(os.environ["REUSE_LOG"]).read_text(encoding="utf-8", errors="replace")
    values = {}
    for name in ("lookup_hits", "matched_segments", "reused_tokens", "external_kv_imported"):
        matches = re.findall(rf"['\"]{name}['\"]\s*:\s*(\d+)", text)
        values[name] = int(matches[-1]) if matches else 0
    return values


def post(prompt, reuse=False):
    if os.environ["PIC_MINIMAL"] == "1":
        body = json.dumps(
            {
                "model": "qwen3.5-2b",
                "prompt": prompt,
                "max_tokens": int(os.environ["MAX_TOKENS"]),
                "temperature": 0.0,
                "stream": False,
                "ignore_eos": True,
                "return_token_ids": True,
            }
        ).encode("utf-8")
        if reuse and os.environ.get("PIC_PURE") == "1":
            endpoint = f"http://127.0.0.1:{os.environ['PURE_B_PORT']}/v1/completions"
        else:
            endpoint = prefill + "/v1/completions"
    else:
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
        endpoint = proxy + "/v1/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=600.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choice = payload["choices"][0]
    if "message" in choice:
        message = choice.get("message") or {}
        text = message.get("content") or message.get("reasoning") or ""
    else:
        text = choice.get("text", "")
    return text, list(choice.get("token_ids") or [])


def post_publish(prompt):
    body = json.dumps(
        {
            "model": "qwen3.5-2b",
            "prompt": prompt,
            "max_tokens": 1,
        }
    ).encode("utf-8")
    if os.environ["PIC_MINIMAL"] == "1":
        endpoint = prefill + "/v1/completions"
    else:
        endpoint = proxy + "/v1/completions"
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=600.0) as response:
        return response.read().decode("utf-8", "replace")


def wait_published(expected):
    for _ in range(120):
        text = Path(os.environ["P_LOG"]).read_text(encoding="utf-8", errors="replace")
        kv = text.count("Mooncake native KV published")
        state = text.count("Mooncake hybrid transition published")
        if kv >= expected and state >= expected:
            print(f"  Store publish confirmed {expected}: native_kv={kv}, hybrid_state={state}")
            return
        time.sleep(1.0)
    raise RuntimeError(f"publish {expected} was not confirmed in the Store within 120s")


if os.environ.get("PUBLISH_FILE"):
    publish_files = [os.environ["PUBLISH_FILE"]]
else:
    publish_files = [
        str(Path(os.environ["PROMPT_FILE"]).parent / f"publish_{name}.txt")
        for name in ("a", "b", "c")
    ]
reuse_prompt = Path(os.environ["PROMPT_FILE"]).read_text(encoding="utf-8").strip()
before = metrics()
print(f"publish_files={publish_files} reuse_file={os.environ['PROMPT_FILE']}")
publish_answers = []
for index, file_name in enumerate(publish_files, 1):
    publish_prompt = Path(file_name).read_text(encoding="utf-8").strip()
    answer = post_publish(publish_prompt)
    publish_answers.append(answer)
    print(f"request {index} (publish {Path(file_name).name}): {answer[:80]}")
    wait_published(index)
    time.sleep(3.0)
time.sleep(float(os.environ["STORE_SETTLE_SECS"]))
second_text, second_ids = post(reuse_prompt, reuse=True)
print(f"request 2 (reuse):  tokens={len(second_ids)} text={second_text!r}")
after = metrics()
p_log = Path(os.environ["P_LOG"]).read_text(encoding="utf-8", errors="replace")
d_path = Path(os.environ["D_LOG"])
d_log = d_path.read_text(encoding="utf-8", errors="replace") if d_path.exists() else ""
delta = {name: after[name] - before[name] for name in before}
pic_reuse = (
    delta["lookup_hits"] > 0
    and delta["matched_segments"] > 0
    and delta["reused_tokens"] > 0
)
result = {
    "pic_before": before,
    "pic_after": after,
    "pic_delta": delta,
    "pic_reuse": pic_reuse,
    "expected": os.environ["EXPECT"],
    "expected_in_second": (
        True
        if os.environ["PIC_MINIMAL"] == "1"
        else os.environ["EXPECT"].lower() in second_text.lower()
    ),
    "publish_answers": publish_answers,
    "second_answer": second_text,
    "pd_relay_events": len(re.findall(r"PIC-PD relay request=", d_log)),
    "pd_state_events": len(re.findall(r"PIC-PD state request=", p_log)),
}
result["pass"] = pic_reuse and result["expected_in_second"]
print(json.dumps(result, ensure_ascii=False, indent=2))
Path(os.environ["JSON_OUT"]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
sys.exit(0 if result["pass"] else 1)
PY
RC=$?

echo ""
if [ "$RC" -eq 0 ]; then
  if [ "$PIC_MINIMAL" = "1" ]; then
    echo "=== PIC-only P segment reuse: PASS ==="
  else
    echo "=== P/D + PIC segment reuse: PASS ==="
    echo "request 1 published PIC segments; request 2 hit them on P; semantic check '${EXPECT}' passed."
  fi
else
  echo "=== P/D + PIC segment reuse: FAIL (exit=$RC) ===" >&2
  echo "P PIC lookup evidence and request logs:" >&2
  grep -Ei 'lookup_hits|matched_segments|reused_tokens|external_kv_imported|PIC-PD state|publish' "$P_LOG" | tail -40 >&2 || true
fi
echo "P log:     $P_LOG"
if [ "$PIC_PURE" = "1" ]; then
  echo "pure B log: $PURE_B_LOG"
fi
echo "D log:     $D_LOG"
echo "proxy log: $PROXY_LOG"
echo "json:      $JSON_OUT"
exit "$RC"
