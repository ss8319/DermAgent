#!/bin/bash
# Self-healing fallback for the Qwen3.5 MedAgent-Pro full run (owner 11-09: "design for losing GPUs"; see memory feedback-design-for-losing-gpus).
# Light bash loop, detached on the login node; inspects and launches only through srun into the L40S tools job (every Qwen runner lives there).
# Every 5 min: a shard (0-3 of 4) that is not finished and has had no runner for 3 checks in a row gets one, on a live A100 server
# (preferred: shards 0,2 -> gpu2a100; 1,3 -> gpu2a100b; else the other). Shards 0-1 are left alone while a live replacement holder owns
# locks/TAKEOVER_01 (its Slurm job id inside); a flag whose owner job is not RUNNING is removed. Stop: touch <D>/SUPERVISOR_STOP. DRY=1: print only.
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; D=$B/dx/medagentpro-qwen3.5-27b; L=61714874; LOG=$D/logs/supervisor.log
R=/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks
SA=$(python3 -c "import json;print(json.load(open('$D/T1.1/RUN_META.json'))['server_args'])")
declare -A PREF=([0]="gpu2a100 gpu2a100b" [1]="gpu2a100b gpu2a100" [2]="gpu2a100 gpu2a100b" [3]="gpu2a100b gpu2a100")
declare -A GPU=([0]=2 [1]=3 [2]=0 [3]=1)
declare -A MISS=([0]=0 [1]=0 [2]=0 [3]=0)
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
remaining() {  # $1 shard -> number of unfinished cases (both tasks)
  python3 - "$1" "$D" "$R" <<'PY'
import json,glob,sys
s=int(sys.argv[1]); D=sys.argv[2]; R=sys.argv[3]; n=0
for t,d in (("T1.1","T1.1_disease_screening"),("T1.2","T1.2_disease_confirmation")):
    done=set()
    for f in glob.glob(f"{D}/{t}/predictions.p*.jsonl"):
        for l in open(f):
            try: done.add(json.loads(l)["_id"])
            except Exception: pass
    ids=[json.loads(l)["_id"] for l in open(f"{R}/{d}/test.jsonl") if l.strip()]
    n+=sum(1 for i,x in enumerate(ids) if i%4==s and x not in done)
print(n)
PY
}
log "supervisor start (pid $$, DRY=${DRY:-0}) server_args: $SA"
while [ ! -f $D/SUPERVISOR_STOP ]; do
  out=$(srun --jobid=$L --overlap -n1 -c1 --mem=1G bash -c 'echo OK; ps -eo args | grep "[r]un_dermarena_v1_medagentpro.py" | grep medagentpro-qwen3.5-27b | grep -oE -- "--shard [0-3]/4"' 2>/dev/null)
  if ! echo "$out" | grep -q '^OK'; then log "cannot inspect the L40S job (srun failed): skip"; sleep 300; continue; fi
  if [ -f $D/locks/TAKEOVER_01 ]; then
    o=$(cat $D/locks/TAKEOVER_01 2>/dev/null); st=$(squeue -h -j "${o:-0}" -o %T 2>/dev/null)
    if [ "$st" != RUNNING ]; then log "takeover flag owner '${o}' is '${st}': removing the flag"; [ -z "$DRY" ] && rm -f $D/locks/TAKEOVER_01; fi
  fi
  all_done=1
  for s in 0 1 2 3; do
    if echo "$out" | grep -q -- "--shard $s/4"; then MISS[$s]=0; all_done=0; continue; fi
    rem=$(remaining $s); [ "$rem" -gt 0 ] && all_done=0
    [ "$rem" -eq 0 ] && { MISS[$s]=0; continue; }
    if [ $s -le 1 ] && [ -f $D/locks/TAKEOVER_01 ]; then MISS[$s]=0; continue; fi
    MISS[$s]=$((MISS[$s]+1)); log "shard $s: no runner (check ${MISS[$s]}/3), $rem cases left"
    [ ${MISS[$s]} -lt 3 ] && continue
    url=""; for h in ${PREF[$s]}; do e=$(awk '{print $1}' $B/holders/$h/ENDPOINT 2>/dev/null); [ -n "$e" ] && curl -sf -m 20 "${e%/v1}/health" >/dev/null && { url=$e; break; }; done
    [ -z "$url" ] && { log "shard $s: no live A100 server"; continue; }
    log "shard $s: starting a runner on $url, tools on L40S GPU ${GPU[$s]}"
    if [ -z "$DRY" ]; then
      setsid nohup env MP_REMOTE_CPUS=4 MP_REMOTE_MEM=16G bash $B/tools/guarded.sh "${url%/v1}/health" bash $B/agents/run_medagentpro_remote_v2.sh $L "$url" full \
        MP_SLUG=medagentpro-qwen3.5-27b MP_BACKBONE=Qwen/Qwen3.5-27B "MP_TOOL_GPUS='${GPU[$s]}'" MP_NPROC=1 MP_THREADS=32 MP_SHARD_BASE=$s MP_SHARD_TOTAL=4 \
        MP_ALLOC_CONF=expandable_segments:True "MP_SERVER_ARGS='$SA'" >> $D/logs/supervisor_s$s.out 2>&1 < /dev/null &
      disown
    fi
    MISS[$s]=0
  done
  [ $all_done = 1 ] && { log "all 4 shards finished: supervisor exits"; break; }
  [ -n "$DRY" ] && { log "DRY run: one pass only"; break; }
  sleep 300
done
