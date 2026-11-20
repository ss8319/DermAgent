#!/bin/bash
# Detached: release the Flash-Next canary job once the Flash-Next server answers on the H200 holder.
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; J=$B/dx/dermagent-qwen3.8-flash-next/jobs
until grep -q 'Qwen3.8-Flash-Next' $B/holders/h200x4/ENDPOINT 2>/dev/null && curl -sf --max-time 5 http://m3u010:30745/health >/dev/null; do sleep 20; done
sleep 30; mv $J/01_canary.json.hold $J/01_canary.json; echo "canary released $(date)" >> $B/logs/flashnext_release.log
