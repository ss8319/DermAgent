#!/bin/bash
# a40_ondemand.sh (11-16, replaces a40_successor.sh): keep ONE desktop A40 tools host for MedAgent-Pro x Flash-Next, but only
# while a Flash-Next server is running or due within LEAD_S (2 h). 11-14..16 the A40 was held "ready" and idled ~36 h while every
# server job sat in the queue (M3 cancels idle GPU jobs). Each pass (2 min):
#   - no bench-a40tools job in the queue, Flash-Next cases left, and a server job is RUNNING or its estimated start is within
#     LEAD_S -> submit one (7 d) and add its id to agents/mp_tool_hosts.txt (mp_fn_supervisor3 places runners' tools there).
#   - exits when no Flash-Next case is left. Stop: touch agents/A40_ONDEMAND_STOP. It never cancels a job.
# "Server job" = a job of ours whose holder queue holds a 10*fn_server*.req.
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; L=$B/logs/a40_ondemand.log; A=$B/logs/gpu_watchdog/alerts.log
FN=$B/dx/medagentpro-qwen3.8-flash-next; LEAD_S=${LEAD_S:-7200}
exec 8>$B/logs/a40_ondemand.flock; flock -n 8 || exit 0
log() { echo "$(date '+%m-%d %H:%M') [a40] $*" | tee -a $L; }
left() { python3 - "$FN" <<'PY'
import json,glob,sys
D=sys.argv[1]; R="/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks"; n=0
for t,d in (("T1.1","T1.1_disease_screening"),("T1.2","T1.2_disease_confirmation")):
    done=set()
    for f in glob.glob(f"{D}/{t}/predictions.p*.jsonl"):
        for l in open(f):
            try: done.add(json.loads(l)["_id"])
            except Exception: pass
    n+=sum(1 for l in open(f"{R}/{d}/test.jsonl") if l.strip() and json.loads(l)["_id"] not in done)
print(n)
PY
}
need() {  # 0 = a server job runs or is due within LEAD_S; prints the reason
  local q j name st s h now=$(date +%s) t
  q=$(squeue -u $USER -h -o "%i|%j|%T|%S" 2>/dev/null) || return 1
  while IFS='|' read -r j name st s; do
    h=${name#bench-}; [ -d $B/holders/$h ] || h=${h//-/_}
    ls $B/holders/$h/queue/10*fn_server*.req >/dev/null 2>&1 || continue
    [ "$st" = RUNNING ] && { echo "$name running"; return 0; }
    [[ "$s" =~ ^20[0-9-]+T ]] || continue
    t=$(date -d "$s" +%s 2>/dev/null) || continue
    [ $((t - now)) -le $LEAD_S ] && { echo "$name due $s"; return 0; }
  done <<< "$q"
  return 1
}
log "armed (pid $$, lead ${LEAD_S}s)"
while [ ! -f $B/agents/A40_ONDEMAND_STOP ]; do
  q=$(squeue -u $USER -h -n bench-a40tools -o %i 2>/dev/null); rc=$?
  if [ $rc -eq 0 ] && [ -z "$q" ] && why=$(need); then
    l=$(left); [ "$l" = 0 ] && { log "no Flash-Next case left: exit"; exit 0; }
    if [ -n "$l" ]; then
      out=$(cd /fs04/scratch2/ub62/ssim0070/sglang/slurm && env $(env | grep -oE '^SLURM_[A-Z_]+' | sed 's/^/-u /') sbatch -J bench-a40tools -p desktop --qos=desktopq --gres=gpu:A40:1 -c 16 --mem=150G -t 7-00:00:00 bench_holder.sbatch a40tools 2>&1)
      J=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$')
      if [ -n "$J" ]; then
        echo "$J  # bench-a40tools on demand ($why; a40_ondemand.sh $(date '+%m-%d %H:%M'))" >> $B/agents/mp_tool_hosts.txt
        log "submitted $J ($why)"; echo "$(date '+%m-%d %H:%M') A40 tools host submitted on demand: $J ($why)" >> $A; SUB=$(date +%s); WARN=""
      else log "submit failed: $out"; [ -z "${FAILAL:-}" ] && { FAILAL=1; echo "$(date '+%m-%d %H:%M') A40 ON-DEMAND submit failing (retrying every 2 min): $out" >> $A; }; fi
    fi
  fi
  # postcondition: a submitted A40 must be RUNNING within 30 min
  if [ -n "${SUB:-}" ] && [ -z "${WARN:-}" ] && [ $(( $(date +%s) - SUB )) -gt 1800 ] && [ -n "$q" ] && ! squeue -h -j $q -t R 2>/dev/null | grep -q .; then
    WARN=1; echo "$(date '+%m-%d %H:%M') A40 tools host $q NOT running 30 min after submit: $(squeue -h -j $q -o '%T %R %S')" >> $A; fi
  sleep 120 8>&-
done
log STOP
