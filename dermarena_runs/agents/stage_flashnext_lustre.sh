#!/bin/bash
# Stage Qwen3.8-Flash-Next (338 GB) into the shared HF cache (owner OK 11-06), so a new H200 copies it from Lustre instead of downloading it.
# Resumable: a stalled attempt (no growth for 5 min) is killed and retried. local_weights ignores the cache until no *.incomplete remains.
source /fs04/scratch2/ub62/ssim0070/sglang/slurm/env.sh >/dev/null 2>&1
export HF_TOKEN="$(grep -m1 '^HUGGINGFACE_TOKEN=' /home/ssim0070/ub62_scratch/ssim0070/.env | cut -d= -f2- | tr -d '"')" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
C=$HUGGINGFACE_HUB_CACHE; M=$C/models--Qwen--Qwen3.8-Flash-Next; PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
for i in $(seq 1 40); do
  $PY -I -c "from huggingface_hub import snapshot_download as s; import sys; print('SNAPSHOT', s('Qwen/Qwen3.8-Flash-Next', cache_dir=sys.argv[1], max_workers=8))" "$C" & P=$!
  last=-1; same=0
  while kill -0 $P 2>/dev/null; do
    sleep 30; now=$(du -s --block-size=1M $M/blobs 2>/dev/null | cut -f1)
    if [ "$now" = "$last" ]; then same=$((same+1)); else same=0; fi; last=$now
    [ $same -ge 10 ] && { echo "stalled at ${now} MB, try $i $(date)"; kill $P; sleep 3; kill -9 $P 2>/dev/null; break; }
  done
  wait $P; rc=$?
  [ $rc -eq 0 ] && [ -z "$(ls $M/blobs/*.incomplete 2>/dev/null)" ] && { du -shL $M/snapshots/* ; echo "STAGE_DONE $(date)"; exit 0; }
  echo "attempt $i ended rc=$rc at $(du -sh $M/blobs 2>/dev/null | cut -f1) $(date)"; sleep 10
done
echo "STAGE_FAILED $(date)"; exit 1
