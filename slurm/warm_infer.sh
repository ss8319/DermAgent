#!/bin/bash
# Run one inference iteration on the persistent warm worker (no model reload).
# Usage: warm_infer.sh <task> <benchmark.jsonl> <limit> <output.jsonl> [model] [max_tokens]
set -euo pipefail
REQ_DIR=/fs04/scratch2/ub62/ssim0070/DermAgent/results/warm/requests
mkdir -p "$REQ_DIR"
TASK=$1; BENCH=$2; LIMIT=$3; OUT=$4
MODEL=${5:-qwen/qwen3.5-27b}; MAXTOK=${6:-8000}
rm -f "$OUT" "$OUT.done"
REQ="$REQ_DIR/req_$(date +%s%N 2>/dev/null || echo $$).json"
cat > "$REQ" <<EOF
{"task":"$TASK","benchmark":"$BENCH","limit":$LIMIT,"output":"$OUT",
 "model":"$MODEL","max_tokens":$MAXTOK,"max_tool_images":3}
EOF
echo "[warm_infer] queued $TASK -> $OUT ; waiting for worker ..."
while [ ! -f "$OUT.done" ]; do sleep 3; done
echo "[warm_infer] done: $OUT"
