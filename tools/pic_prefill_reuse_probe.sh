#!/usr/bin/env bash
# Verify local HBM prefix-cache reuse on the Prefill server, independently of P/D.
# Exit 0: reuse observed. Exit 1: no reuse. Exit 2: request/setup error.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PREFILL_URL="${PREFILL_URL:-http://127.0.0.1:8010}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/tests/v1/pic/fixtures/c/reuse.txt}"
MAX_TOKENS="${MAX_TOKENS:-8}"
REPEAT="${REPEAT:-2}"
PREFILL_LOG="${PREFILL_LOG:-/tmp/pic_pd_prefill.log}"
JSON_OUT="${JSON_OUT:-}"
PYTHON="${PIC_PYTHON:-python3}"

export PREFILL_URL PROMPT_FILE MAX_TOKENS REPEAT PREFILL_LOG JSON_OUT
"$PYTHON" - <<'PY'
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = os.environ["PREFILL_URL"].rstrip("/")


def metrics():
    request = urllib.request.Request(base + "/metrics")
    with opener.open(request, timeout=30.0) as response:
        text = response.read().decode("utf-8", "replace")
    values = {"queries": 0.0, "hits": 0.0}
    for name in values:
        pattern = re.compile(
            rf"^vllm:prefix_cache_{name}(?:_total)?(?:\{{[^}}]*\}})?\s+"
            rf"([0-9.eE+-]+)\s*$",
            re.MULTILINE,
        )
        values[name] = sum(float(match.group(1)) for match in pattern.finditer(text))
    return values


prompt = Path(os.environ["PROMPT_FILE"]).read_text(encoding="utf-8").strip()
before = metrics()
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
    choice = payload["choices"][0]
    print(f"request {index + 1}: finish={choice.get('finish_reason')} text={choice.get('text')!r}")

after = metrics()
result = {
    "queries_before": before["queries"],
    "queries_after": after["queries"],
    "queries_delta": after["queries"] - before["queries"],
    "hits_before": before["hits"],
    "hits_after": after["hits"],
    "hits_delta": after["hits"] - before["hits"],
    "repeat": int(os.environ["REPEAT"]),
    "prompt_file": os.environ["PROMPT_FILE"],
    "reuse": after["hits"] > before["hits"],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
if os.environ.get("JSON_OUT"):
    Path(os.environ["JSON_OUT"]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
print(
    "Next: grep -Ei 'num_local_cached_tokens|Prefix cache hit rate' "
    + os.environ["PREFILL_LOG"]
    + " | tail"
)
sys.exit(0 if result["reuse"] else 1)
PY
