#!/bin/bash
# gpu_opt_check.sh (2026-11-16, owner: "make this yr main check for every 2hr GPU optimisation; be SMART and efficient").
# One read-only command for the 2-hourly GPU plan. It answers: is any GPU shape our pending work fits on available sooner than
# the jobs we already have queued? Method (the 11-16 "do we need H200 / smaller GPU" analysis):
#   1. Our jobs: state, estimated start, end.
#   2. Free GPUs per node right now (80 GB and 48 GB types), and whether our account may use that partition.
#   3. Earliest start per GPU SHAPE (1/2/4 x 80 GB, 1/2 x 48 GB) on every partition we can use. Each shape is tested with our
#      normal QOS AND with irq: fitq/m3h estimates for a fresh job are biased by our own pending jobs (per-user GPU caps); irq has
#      no per-user cap, so it shows whether capacity is really free.
#   4. LEVER lines: a shape that starts >= LEVER_H hours before our earliest queued job of the same GPU class.
# Shape <-> workload (bf16, no quantisation): 1x80 GB = a 27B model (HuatuoGPT-3-27B, Qwen3.6-27B judge, Qwen3.5-27B);
#   4x80 GB = Flash-Next, GLM-4.6V (~200 GB); 1x48 GB = Qwen3.5-9B; 2x48 GB = a 27B model at tp2.
# Usage: bash gpu_opt_check.sh   (~30 s; never submits or cancels anything)
LEVER_H=${LEVER_H:-2}
S=/fs04/scratch2/ub62/ssim0070/sglang/slurm/bench_holder.sbatch
now=$(date +%s); ep() { [[ "$1" =~ ^20[0-9-]+T ]] && date -d "$1" +%s 2>/dev/null; }
hrs() { [ -n "$1" ] && echo $(( ($1 - now) / 3600 ))h || echo "?"; }
echo "== GPU opt check $(date '+%m-%d %H:%M')"

echo "-- 1. our jobs"
squeue -u $USER -h -o "%i|%j|%T|%S|%e|%b|%P" 2>/dev/null | while IFS='|' read -r j n st s e g p; do
  printf "  %-9s %-22s %-8s %-7s start %-19s (in %s) end %s\n" $j "$n" $st "$p" "$s" "$(hrs $(ep $s))" "$e"; done
OUR80=$(squeue -u $USER -h -t PD -o "%S|%b|%P" 2>/dev/null | grep -E "A100|H100|H200|gpu:4|fit|m3h|bdi" | grep -v desktop | cut -d'|' -f1 | sort | head -1)
echo "  earliest queued 80 GB job: ${OUR80:-none} (in $(hrs $(ep "$OUR80")))"

echo "-- 2. free GPUs now (node: partition type free; * = our account cannot use the partition)"
OKP=" fit m3h gpu bdi bdi-imaging desktop comp "
scontrol show nodes -o 2>/dev/null | python3 -c '
import sys,re
ok=sys.argv[1].split()
for l in sys.stdin:
    d=dict(re.findall(r"(\S+?)=(\S+)",l)); m=re.match(r"gpu:([A-Za-z0-9_-]+):(\d+)",d.get("Gres",""))
    if not m or m.group(1) not in ("A100","H100","H200","A40","L40S"): continue
    st=d.get("State","")
    if any(x in st for x in ("DOWN","DRAIN","MAINT","FAIL","NOT_RESP")): continue
    u=re.search(r"gres/gpu(?::[^=]*)?=(\d+)",d.get("AllocTRES","") or ""); free=int(m.group(2))-(int(u.group(1)) if u else 0)
    if free<=0: continue
    parts=d.get("Partitions","?"); use=any(p in ok for p in parts.split(","))
    node=d["NodeName"]; cpu=int(d.get("CPUTot",0))-int(d.get("CPUAlloc",0)); mem=(int(d.get("RealMemory",0))-int(d.get("AllocMem",0)))//1024
    print("  %-8s %-14s %-5s free %d  cpu_free %3d  mem_free %dG%s" % (node, parts, m.group(1), free, cpu, mem, "" if use else "  *"))
' "$OKP"

echo "-- 3. earliest start per shape (normal QOS | irq = real capacity, no per-user cap)"
t() { sbatch --test-only -A ub62 "$@" $S x 2>&1 | grep -oE 'start at [^ ]+' | sed 's/start at //'; }
shape() {  # label, gres, cpus, mem, partition, qos, extra
  local a b; a=$(t -p $5 -q $6 --gres=$2 -c $3 --mem=$4 -t 1-00:00:00 $7); b=$(t -p $5 -q irq --gres=$2 -c $3 --mem=$4 -t 1-00:00:00 --requeue $7)
  printf "  %-26s %-9s %-19s (%4s) | irq %-19s (%4s)\n" "$1" "$5" "${a:-n/a}" "$(hrs $(ep $a))" "${b:-n/a}" "$(hrs $(ep $b))"
  echo "$1|$a|$b" >> $TMPF; }
TMPF=$(mktemp); trap 'rm -f $TMPF' EXIT
shape "1x80GB A100 (27B model)"  gpu:A100:1 8  100G fit  fitq "--exclude=m3u009"
shape "1x80GB H100 (27B model)"  gpu:H100:1 8  100G m3h  m3h  ""
shape "1x80GB H200 (27B model)"  gpu:H200:1 8  100G fit  fitq "--exclude=m3u011"
shape "4x80GB A100 (FN / GLM)"   gpu:A100:4 32 400G fit  fitq "--exclude=m3u009"
shape "4x80GB H100 (FN / GLM)"   gpu:H100:4 32 400G m3h  m3h  ""
shape "4x80GB A100 bdi"          gpu:A100:4 32 400G bdi  irq  ""
d=$(t -p desktop -q desktopq --gres=gpu:A40:1 -c 16 --mem=150G -t 1-00:00:00); printf "  %-26s %-9s %s\n" "1x48GB A40 desktop (9B)" desktop "${d:-n/a: desktop allows one job per user and ours is queued or running}"
shape "1x48GB A40 gpu (9B)"      gpu:A40:1  8  100G gpu  normal ""
shape "2x48GB A40 gpu (27B tp2)"  gpu:A40:2  16 150G gpu  normal ""
printf "  %-26s %-9s %s\n" "2x48GB L40S (27B tp2)" gpu "$(t -p gpu -q normal --gres=gpu:L40S:2 -c 16 --mem=150G -t 1-00:00:00)"

echo "-- 4. levers (a shape >= ${LEVER_H} h earlier than our earliest queued 80 GB job)"
o=$(ep "$OUR80"); n=0
while IFS='|' read -r lab a b; do
  for x in "$a" "$b"; do e=$(ep "$x"); [ -n "$e" ] && [ -n "$o" ] && [ $(( o - e )) -ge $(( LEVER_H * 3600 )) ] && { echo "  LEVER: $lab could start $x, $(( (o - e) / 3600 )) h before our queued job"; n=1; break; }; done
done < $TMPF
[ $n = 0 ] && echo "  none: our queued jobs are the earliest 80 GB capacity; idle GPUs (if any) are on partitions marked *"
