#!/bin/bash
# 形状 C：system <<SEP>> a <<SEP>> b <<SEP>> c <<SEP>> query
# 用法：bash tools/test_shape_a.sh
#
# 这个脚本自己起服务、自己 publish、自己 reuse、自己判据——不再经过 run_pic_test.sh。
# 端点统一 /v1/completions：publish 与 reuse 同构，doc 的 token 化一致，A/B 的 key 才能对上。
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FIX="$REPO_ROOT/tests/v1/pic/fixtures/c"
LOG_DIR="${PIC_LOG_DIR:-$REPO_ROOT/tests/v1/pic/logs}"
mkdir -p "$LOG_DIR"

export PIC_MODEL="${PIC_MODEL:?set PIC_MODEL to the directory holding the model}"
export PIC_BASE="$FIX"
export PIC_DEBUG=1
export PIC_LOG_DIR="$LOG_DIR"

echo "===== 清场 ====="
pkill -9 -f 'vllm serve' 2>/dev/null || true
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f 'EngineCore' 2>/dev/null || true
pkill -9 -f 'mooncake_master' 2>/dev/null || true
# 回收 mooncake_master 的僵尸进程，否则长期反复跑会累积 <defunct>
wait 2>/dev/null || true
sleep 15

echo "===== 起服务（PIC-ON，形状 C）====="
bash "$SCRIPT_DIR/pic_run.sh" serve

echo "===== publish_a/b/c（A:8000，/v1/completions）====="
for name in publish_a publish_b publish_c; do
  python3 - "$FIX/$name.txt" <<'PY'
import json, sys, urllib.request
pub = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({"model": "qwen3.5-2b", "prompt": pub, "max_tokens": 1})
req = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
    data=body.encode(), headers={"Content-Type": "application/json"})
print(f"  publish: {urllib.request.urlopen(req, timeout=600).read().decode()[:60]}")
PY
  sleep 3
done

sleep 5

echo "===== reuse（B:8001，/v1/completions，与 publish 同构）====="
python3 - "$FIX/reuse.txt" <<'PY'
import json, sys, urllib.request
reu = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({
    "model": "qwen3.5-2b",
    "prompt": reu,
    "temperature": 0, "max_tokens": 32,
})
req = urllib.request.Request("http://127.0.0.1:8001/v1/completions",
    data=body.encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
answer = r["choices"][0]["text"]
print(f"finish={r['choices'][0]['finish_reason']}")
print(f"answer={answer!r}")
print(f"contains dog: {'dog' in answer.lower()}")
PY

sleep 3

echo "===== key 比对 ====="
echo "A publish key:"; grep -oE 'key=segment/[a-f0-9]+' "$LOG_DIR/A.log" | tail -1
echo "B lookup key:";  grep -oE "mooncake:hypic/native-kv/v1/segment/[a-f0-9]+" "$LOG_DIR/B.log" | tail -1

echo "===== 形状 C 的 9 条判据 ====="
A_LOG="$LOG_DIR/A.log"
B_LOG="$LOG_DIR/B.log"
FAIL=0
check() {
  local side=$1 name=$2 pattern=$3 forbidden=${4:-0}
  local log=$([ "$side" = "A" ] && echo "$A_LOG" || echo "$B_LOG")
  local hits=$(grep -cE "$pattern" "$log" 2>/dev/null) || hits=0
  local passed
  if [ "$forbidden" = "1" ]; then
    [ "$hits" -eq 0 ] && passed=1 || passed=0
  else
    [ "$hits" -gt 0 ] && passed=1 || passed=0
  fi
  [ "$passed" = "0" ] && FAIL=1
  local mark=$([ "$passed" = "1" ] && echo "PASS" || echo "FAIL")
  printf "  %-4s %-4s %-20s found %s\n" "$mark" "$side" "$name" "$hits"
}
check_fixed() {
  local side=$1 name=$2 pattern=$3
  local log=$([ "$side" = "A" ] && echo "$A_LOG" || echo "$B_LOG")
  local hits=$(grep -cF "$pattern" "$log" 2>/dev/null) || hits=0
  local passed=0
  [ "$hits" -gt 0 ] && passed=1
  [ "$passed" = "0" ] && FAIL=1
  local mark=$([ "$passed" = "1" ] && echo "PASS" || echo "FAIL")
  printf "  %-4s %-4s %-20s found %s\n" "$mark" "$side" "$name" "$hits"
}
check A "publish-matrix" "publish-matrix"
check A "native-kv" "native KV published"
check A "no-gdn-failure" "GDN transition validation failed" 1
check B "lookup-hit" "'lookup_hits': [1-9]"
check B "transition-applied" "transition applied"
check B "executed" "single-request early runtime executed"
check_fixed B "no-fallback" "'fallback_reasons': {}"
check B "no-nonfinite" "non-finite" 1
check B "no-slice-oob" "slice is outside the range" 1

echo ""
if [ "$FAIL" = "0" ]; then
  echo "=== shape c: PASS ==="
else
  echo "=== shape c: FAIL ==="
fi

echo ""
echo "日志：$A_LOG / $B_LOG"
exit $FAIL
