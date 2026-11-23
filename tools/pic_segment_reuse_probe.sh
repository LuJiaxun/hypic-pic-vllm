#!/usr/bin/env bash
# Verify PIC segment reuse from Prefill-side logs (requires PIC and its Store).
# Exit 0: reuse observed. Exit 1: no reuse. Exit 2: PIC/Store not configured.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PREFILL_URL="${PREFILL_URL:-http://127.0.0.1:8010}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/tests/v1/pic/fixtures/c/reuse.txt}"
PREFILL_LOG="${PREFILL_LOG:-/tmp/pic_pd_prefill.log}"
MAX_TOKENS="${MAX_TOKENS:-8}"
REPEAT="${REPEAT:-2}"
JSON_OUT="${JSON_OUT:-}"
PYTHON="${PIC_PYTHON:-python3}"

# PIC is deliberately opt-in here: a P/D-only run must not manufacture a Store
# dependency just to measure reuse.  Set PIC_ENABLED=1 and provide its config.
if [ "${PIC_ENABLED:-0}" != "1" ]; then
  echo "PIC not enabled, skip (set PIC_ENABLED=1 and MOONCAKE_CONFIG_PATH or PIC_MOONCAKE)"
  exit 2
fi
CONFIG="${MOONCAKE_CONFIG_PATH:-${PIC_MOONCAKE:-}}"
if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
  echo "PIC Store config missing, skip (set MOONCAKE_CONFIG_PATH or PIC_MOONCAKE)"
  exit 2
fi
if [ ! -f "$PREFILL_LOG" ]; then
  echo "PREFILL_LOG not found: $PREFILL_LOG"
  exit 2
fi

export PREFILL_URL PROMPT_FILE PREFILL_LOG MAX_TOKENS REPEAT JSON_OUT
"$PYTHON" - <<'PY'
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = os.environ["PREFILL_URL"].rstrip("/")
log_path = Path(os.environ["PREFILL_LOG"])
NAMES = ("lookup_hits", "matched_segments", "reused_tokens", "external_kv_imported")


def snapshot():
    if not log_path.exists():
        return {name: 0 for name in NAMES}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for name in NAMES:
        matches = re.findall(rf"['\"]{name}['\"]\s*:\s*(\d+)", text)
        result[name] = int(matches[-1]) if matches else 0
    return result


before = snapshot()
prompt = Path(os.environ["PROMPT_FILE"]).read_text(encoding="utf-8").strip()
for index in range(int(os.environ["REPEAT"])):
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
    request = urllib.request.Request(
        base + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=600.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    print(f"request {index + 1}: {payload['choices'][0].get('text')!r}")

after = snapshot()
delta = {name: after[name] - before[name] for name in NAMES}
reuse = delta["lookup_hits"] > 0 and delta["matched_segments"] > 0 and delta["reused_tokens"] > 0
result = {
    "before": before,
    "after": after,
    "delta": delta,
    "pic_segment_reuse": reuse,
    "prompt_file": os.environ["PROMPT_FILE"],
    "prefill_log": os.environ["PREFILL_LOG"],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
if os.environ.get("JSON_OUT"):
    Path(os.environ["JSON_OUT"]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
sys.exit(0 if reuse else 1)
PY
