#!/bin/bash
# stop the prefetch watchdog + its downloader (pids only), then fetch the last shard with a resumable curl
for p in $(pgrep -f "agents/prefetch_flashnext.sh"); do [ "$p" != "$$" ] && kill $p 2>/dev/null; done
for p in $(pgrep -f "huggingface_hub import snapshot_download"); do kill $p 2>/dev/null; done
sleep 3
cd /tmp/bench_weights/Qwen/Qwen3.8-Flash-Next || exit 1
F=${1:-model-00127-of-00131.safetensors}
tok=$(grep -m1 '^HUGGINGFACE_TOKEN=' /home/ssim0070/ub62_scratch/ssim0070/.env | cut -d= -f2- | tr -d '"' | awk '{print $1}')
U="https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/$F"
echo "size:"; curl -sIL -H "Authorization: Bearer $tok" "$U" | grep -iE '^HTTP|content-length' | tail -3
for i in 1 2 3 4 5 6 7 8; do
  curl -L -sS -C - --retry 3 --speed-limit 200000 --speed-time 60 -H "Authorization: Bearer $tok" -o "$F.part" "$U" && { mv "$F.part" "$F"; echo "got $F $(stat -c %s $F) $(date)"; exit 0; }
  echo "curl try $i rc=$? at $(stat -c %s $F.part 2>/dev/null) bytes $(date)"; sleep 5
done
exit 1
