#!/bin/bash
# fn_t11_first.sh (11-13 09:4x) — make Flash-Next MedAgent-Pro T1.1 finish before the H200 job ends (19:00): server capacity is the
# bottleneck (~70 cases/h, 160 running cap), so T1.2 cases started now cannot finish today and only slow the T1.1 tail.
# Every 5 min: for each 8-way part k (0-6) whose T1.1 backlog is 0, stop that part's runner ONCE by its exact process group
# (it is then only running T1.2 cases, which are re-run later; finished cases are kept). The supervisor is stopped first
# (FN_SUPERVISOR_STOP) so it does not restart those parts. Part 7 runs as 7/24, 15/24, 23/24 on gpu2a100b: never touched.
# Ends when T1.1 backlog is 0 everywhere (prints T11_DONE) or on agents/FN_T11_FIRST_STOP.
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; SLUG=medagentpro-qwen3.8-flash-next; D=$B/dx/$SLUG; L=$D/logs/fn_t11_first.log
touch $D/FN_SUPERVISOR_STOP; declare -A KILLED
log() { echo "$(date '+%m-%d %H:%M') [t11-first] $*" | tee -a $L; }
left() { python3 - <<'PY'
import json,glob
D="/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/medagentpro-qwen3.8-flash-next"; R="/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks/T1.1_disease_screening/test.jsonl"
done=set()
for f in glob.glob(f"{D}/T1.1/predictions.p*.jsonl"):
    for l in open(f):
        try: done.add(json.loads(l)["_id"])
        except Exception: pass
n=[0]*8
for i,l in enumerate(x for x in open(R) if x.strip()):
    if json.loads(l)["_id"] not in done: n[i%8]+=1
print(" ".join(map(str,n)))
PY
}
log "start; supervisor stop file set"
while [ ! -f $B/agents/FN_T11_FIRST_STOP ]; do
  n=($(left)); tot=0; for x in "${n[@]}"; do tot=$((tot+x)); done
  log "T1.1 left per part: ${n[*]} (total $tot)"
  [ $tot -eq 0 ] && { log "T11_DONE"; break; }
  for k in 0 1 2 3 4 5 6; do
    [ "${n[$k]}" -eq 0 ] && [ -z "${KILLED[$k]:-}" ] || continue
    for j in 61981433 62450847; do
      g=$(timeout 60 srun --jobid=$j --overlap -n1 -c1 --mem=256M bash -c 'for p in $(pgrep -f "[r]un_dermarena_v1_medagentpro.py"); do a=$(tr "\0" " " < /proc/$p/cmdline); echo "$a" | grep -qF "/'$SLUG'/" && echo "$a" | grep -qE -- "--shard '$k'/8( |$)" && ps -o pgid= -p $p; done' 2>/dev/null | tr -d ' ' | sort -u)
      for pg in $g; do log "part $k: T1.1 done -> stopping its runner (host $j pgid $pg; it only holds T1.2 cases now)"; timeout 60 srun --jobid=$j --overlap -n1 -c1 --mem=256M bash -c "kill -TERM -- -$pg" 2>/dev/null; KILLED[$k]=1; done
    done
  done
  sleep 300
done
