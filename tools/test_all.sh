#!/bin/bash
# 串跑 A/B/C。每个形状自己起停服务、自己发布、自己判据，这里只按顺序调用并汇总。
# 用法：bash tools/test_all.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FAIL=0

for shape in a b c; do
  echo ""
  echo "########## shape $shape ##########"
  bash "$SCRIPT_DIR/test_shape_$shape.sh"
  RC=$?
  [ "$RC" -ne 0 ] && FAIL=1
done

echo ""
if [ "$FAIL" = "0" ]; then
  echo "########## ALL PASS ##########"
else
  echo "########## SOME FAIL ##########"
fi
exit $FAIL
