#!/bin/bash
# mp_shard_lock.sh <lockdir> <jobid>: make Slurm job <jobid> own <lockdir>. A lock whose owner job is no longer RUNNING is stale and is
# taken over. Default: block until owned. NOWAIT=1: exit 1 at once if a different RUNNING job owns it (11-09: surplus holders then do
# ablation work instead of waiting idle).
L=$1; J=$2
while true; do
  if mkdir $L 2>/dev/null; then echo $J > $L/owner; echo "[lock] $L taken by job $J $(date)"; exit 0; fi
  o=$(cat $L/owner 2>/dev/null)
  [ "$o" = "$J" ] && exit 0
  st=$(squeue -h -j "$o" -o %T 2>/dev/null)
  if [ -n "$o" ] && [ "$st" != "RUNNING" ]; then echo "[lock] $L owner $o is '$st': stale, taking over"; rm -rf $L; continue; fi
  [ -n "${NOWAIT:-}" ] && { echo "[lock] $L owned by running job $o"; exit 1; }
  sleep 300
done
