#!/bin/bash
# Resumable Flash-Next weight prefetch onto the H200 node's /tmp; retries on stall/failure. Run via:
#   srun --jobid=61562245 --overlap -n1 -c4 --mem=8G bash prefetch_flashnext.sh
source /fs04/scratch2/ub62/ssim0070/sglang/slurm/env.sh >/dev/null 2>&1
source /fs04/scratch2/ub62/ssim0070/benchmark_runs/local_weights.sh
for i in 1 2 3 4 5 6; do
  # kill a download that has not grown for 4 minutes (stall seen 22:55 11-05), then resume
  ( local_weights Qwen/Qwen3.8-Flash-Next ) & P=$!
  last=0; same=0
  while kill -0 $P 2>/dev/null; do
    sleep 30; now=$(du -s --block-size=1M /tmp/bench_weights/Qwen/Qwen3.8-Flash-Next 2>/dev/null | cut -f1)
    if [ "$now" = "$last" ]; then same=$((same+1)); else same=0; fi; last=$now
    if [ $same -ge 8 ]; then echo "stalled at ${now} MB, restarting (try $i) $(date)"; pkill -P $P; kill $P 2>/dev/null; pkill -f 'snapshot_download' ; break; fi
  done
  wait $P 2>/dev/null; [ -f /tmp/bench_weights/Qwen/Qwen3.8-Flash-Next/.complete ] && { echo "PREFETCH DONE $(date)"; exit 0; }
  sleep 5
done
echo "PREFETCH FAILED $(date)"; exit 1
