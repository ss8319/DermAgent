#!/bin/bash
# Resumable weight prefetch with a stall watchdog onto the H200 node's /tmp. Usage (inside the job, via srun --overlap):
#   bash prefetch_model.sh <hf_id>     e.g. zai-org/GLM-4.6V
ID=$1; L=/tmp/bench_weights/$ID
source /fs04/scratch2/ub62/ssim0070/sglang/slurm/env.sh >/dev/null 2>&1
source /fs04/scratch2/ub62/ssim0070/benchmark_runs/local_weights.sh
for i in 1 2 3 4 5 6; do
  ( local_weights "$ID" ) & P=$!
  last=0; same=0
  while kill -0 $P 2>/dev/null; do
    sleep 30; now=$(du -s --block-size=1M "$L" 2>/dev/null | cut -f1)
    if [ "$now" = "$last" ]; then same=$((same+1)); else same=0; fi; last=$now
    if [ $same -ge 8 ]; then echo "stalled at ${now} MB, restarting (try $i) $(date)"; pkill -P $P; kill $P 2>/dev/null; pkill -f snapshot_download; break; fi
  done
  wait $P 2>/dev/null; [ -f "$L/.complete" ] && { echo "PREFETCH DONE $ID $(date)"; exit 0; }
  sleep 5
done
echo "PREFETCH FAILED $ID $(date)"; exit 1
