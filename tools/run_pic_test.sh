#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# PIC 验收脚本：diag + perf + trace
# 用法：
#   bash tools/run_pic_test.sh diag    # A/B/C 执行层判据（C 加语义答 dog）
#   bash tools/run_pic_test.sh perf    # 形状 A 性能，5 轮取稳态
#   bash tools/run_pic_test.sh trace   # 采 trace
#   bash tools/run_pic_test.sh all     # 先 diag 再 perf
#
# 路径自适应：脚本所在目录的上一级就是仓库根。仓库根指 **包含 vllm/ 包的那一层**，
# 安装后通常是 /usr/local/lib/python3.12/dist-packages，所以用法是：
#   cd /usr/local/lib/python3.12/dist-packages && bash tools/<script>.sh <mode>
set -u

MODE=${1:-all}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"        # <repo>/tools
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"           # <repo>
FIX="$REPO_ROOT/tests/v1/pic/fixtures"              # <repo>/tests/v1/pic/fixtures
LOG_DIR="${PIC_LOG_DIR:-$REPO_ROOT/tests/v1/pic/logs}"
TRACE_DIR="${PIC_TRACE_DIR:-$REPO_ROOT/tests/v1/pic/trace}"
export PIC_LOG_DIR="$LOG_DIR"
mkdir -p "$LOG_DIR"

cleanup_servers() {
  pkill -9 -f 'vllm serve' 2>/dev/null || true
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
  pkill -9 -f 'EngineCore' 2>/dev/null || true
  pkill -9 -f 'mooncake_master' 2>/dev/null || true
  sleep 10
}

prepare_fixture_zero() {
  local shape=$1
  local d="$FIX/$shape"
  # The aliases ship with the patch, so this is only a safety net for a hand-edited tree.
  if [ ! -f "$d/publish_0.txt" ] && [ -f "$d/publish.txt" ]; then
    cp "$d/publish.txt" "$d/publish_0.txt"
  fi
  # c 只有 publish_a/b/c，没有 publish.txt
  if [ ! -f "$d/publish_0.txt" ] && [ -f "$d/publish_a.txt" ]; then
    cp "$d/publish_a.txt" "$d/publish_0.txt"
  fi
  if [ ! -f "$d/reuse_0.txt" ] && [ -f "$d/reuse.txt" ]; then
    cp "$d/reuse.txt" "$d/reuse_0.txt"
  fi
  if [ ! -f "$d/publish_0.txt" ] || [ ! -f "$d/reuse_0.txt" ]; then
    echo "FATAL: $d 缺少 publish_0.txt / reuse_0.txt"
    return 1
  fi
  return 0
}

start_servers() {
  local debug=$1
  local shape=$2
  echo "===== 起服务（DEBUG=$debug, SHAPE=$shape）====="
  prepare_fixture_zero "$shape" || return 1
  cd "$REPO_ROOT" || return 1
  export PIC_BASE="$FIX/$shape"
  # The launcher only brings the servers up; publishing, reuse and the checks are this script's job.
  # The checks read per-request diagnostics, so the driver is the one that asks for them.
  export PIC_DEBUG="$debug"
  bash "$SCRIPT_DIR/pic_run.sh" serve
}

# The off half of a comparison: same shape, same fixture, no PIC flags. Everything else - model,
# batching, memory fraction, probe arguments - is identical, so the difference is the feature.
start_servers_off() {
  local shape=$1
  echo "===== 起服务（PIC-OFF, SHAPE=$shape）====="
  prepare_fixture_zero "$shape" || return 1
  cd "$REPO_ROOT" || return 1
  export PIC_BASE="$FIX/$shape"
  export PIC_OFF=1
  bash "$SCRIPT_DIR/pic_run.sh" perf
}





run_perf() {
  cleanup_servers
  start_servers 0 b || { echo "FATAL: 起服务失败"; return 1; }
  unset HYPIC_PIC_DEBUG HYPIC_PIC_PROFILE_DEEP VLLM_CUSTOM_SCOPES_FOR_PROFILING

  cd "$REPO_ROOT" || return 1
  echo ""
  echo "===== perf：连跑 5 轮探针（形状 B，system 197 / doc 7000 / query 274）====="
  for i in 1 2 3 4 5; do
    echo "---------- round $i ----------"
    python3 tests/v1/pic/run_pic_concurrency_probe.py \
      --base-url http://127.0.0.1:8001 --model qwen3.5-2b \
      --publish-prompt-file "$FIX/b/publish.txt" \
      --reuse-prompt-file   "$FIX/b/reuse.txt" \
      --concurrency 1,4 --requests 20 --warmup 20 --max-tokens 1 \
      --json-out "$LOG_DIR/probe_b_r${i}.json" 2>/dev/null
    python3 - "$LOG_DIR/probe_b_r${i}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d.get("levels", []):
    print(f"  c={r['concurrency']}: p50={r['p50_ms']:.1f}  tput={r['throughput_rps']:.4f}")
PY
  done

  echo ""
  echo "===== 形状 B 稳态（丢弃第 1 轮，取第 2–5 轮中位数）====="
  python3 - "$LOG_DIR" <<'PY'
import json, sys, glob, statistics
log_dir = sys.argv[1]
rounds = {}
for f in sorted(glob.glob(f"{log_dir}/probe_b_r*.json")):
    for r in json.load(open(f)).get("levels", []):
        rounds.setdefault(r["concurrency"], []).append({
            "p50": r["p50_ms"], "tput": r["throughput_rps"]
        })
for c, rs in sorted(rounds.items()):
    steady = rs[1:]
    p50s = [x["p50"] for x in steady]
    tputs = [x["tput"] for x in steady]
    print(f"  c={c}: p50 median={statistics.median(p50s):.1f}ms  tput median={statistics.median(tputs):.4f}")
    print(f"      p50 rounds 2-5: {[f'{x:.1f}' for x in p50s]}")
PY

  cleanup_servers
  echo ""
  echo "=== perf: DONE ==="
}

run_off() {
  cleanup_servers
  start_servers_off b || { echo "FATAL: 起服务失败 (PIC-OFF)"; return 1; }
  unset HYPIC_PIC_DEBUG HYPIC_PIC_PROFILE_DEEP VLLM_CUSTOM_SCOPES_FOR_PROFILING

  cd "$REPO_ROOT" || return 1
  echo ""
  echo "===== PIC-OFF：连跑 5 轮探针（形状 B）====="
  for i in 1 2 3 4 5; do
    echo "---------- round $i ----------"
    python3 tests/v1/pic/run_pic_concurrency_probe.py \
      --base-url http://127.0.0.1:8001 --model qwen3.5-2b \
      --publish-prompt-file "$FIX/b/publish.txt" \
      --reuse-prompt-file   "$FIX/b/reuse.txt" \
      --concurrency 1,4 --requests 20 --warmup 20 --max-tokens 1 \
      --json-out "$LOG_DIR/probe_off_b_r${i}.json" 2>/dev/null
    python3 - "$LOG_DIR/probe_off_b_r${i}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d.get("levels", []):
    print(f"  c={r['concurrency']}: p50={r['p50_ms']:.1f}  tput={r['throughput_rps']:.4f}")
PY
  done

  echo ""
  echo "===== PIC-OFF 形状 B 稳态（丢弃第 1 轮，取第 2–5 轮中位数）====="
  python3 - "$LOG_DIR" <<'PY'
import json, sys, glob, statistics
log_dir = sys.argv[1]
rounds = {}
for f in sorted(glob.glob(f"{log_dir}/probe_off_b_r*.json")):
    for r in json.load(open(f)).get("levels", []):
        rounds.setdefault(r["concurrency"], []).append({
            "p50": r["p50_ms"], "tput": r["throughput_rps"]
        })
for c, rs in sorted(rounds.items()):
    steady = rs[1:]
    p50s = [x["p50"] for x in steady]
    tputs = [x["tput"] for x in steady]
    print(f"  c={c}: p50 median={statistics.median(p50s):.1f}ms  tput median={statistics.median(tputs):.4f}")
    print(f"      p50 rounds 2-5: {[f'{x:.1f}' for x in p50s]}")
PY

  cleanup_servers
  echo ""
  echo "=== PIC-OFF: DONE ==="
}

run_trace() {
  cleanup_servers
  cd "$REPO_ROOT" || return 1
  export PIC_BASE="$FIX/b"
  bash "$SCRIPT_DIR/pic_run.sh" trace
  echo ""
  echo "===== trace 产物 ====="
  ls -lh "$TRACE_DIR"/pic_shapeB_trace_c*.pt.trace.json 2>/dev/null \
    || echo "trace 文件未生成"
  echo "trace 目录: $TRACE_DIR"
  cleanup_servers
}

case "$MODE" in
  perf) run_perf; exit $? ;;
  off)  run_off;  exit $? ;;
  trace) run_trace; exit $? ;;
  all)
    # diag lives in tools/test_shape_{a,b,c}.sh now; this script measures.
    run_perf; PERF_RC=$?
    run_off;  OFF_RC=$?
    echo ""
    echo "========== 总结 =========="
    echo "  perf: DONE"
    echo "  off : DONE"
    exit $((PERF_RC + OFF_RC))
    ;;
  *) echo "用法: bash tools/run_pic_test.sh perf|off|trace|all"; exit 1 ;;
esac
