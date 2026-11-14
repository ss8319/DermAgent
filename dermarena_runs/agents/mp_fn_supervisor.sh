#!/bin/bash
# mp_fn_supervisor.sh (11-12) — MedAgent-Pro x Qwen3.8-Flash-Next on EVERY live Flash-Next server (optimisation #2, owner-agreed 11-11).
# Why: the run had one server (h200x4d, ends 12 Nov 19:00) with its 4 runners' tools on bdiimg GPU 3; bdiimg could not load a second
# Flash-Next server while those tools (and Qwen's on GPU 2) sat on it. This supervisor:
#  1. SWITCH: while the old run (shards x/4, one 4-process runner) is alive, wait until no Qwen runner remains on bdiimg (Qwen done), then
#     stop the old runner by its exact process group. (Its in-flight cases are re-run; finished cases are kept: resume skips them.)
#     h200x4d's holder then moves to 70_fn_server_hold.req (same server, no restart) and bdiimg's 70_fn_server_hold.req loads Flash-Next.
#  2. Every 5 min: shards 0-7 of 8 (case i -> i % 8; run_sig does not include the shard count). A shard with work and no runner gets
#     one (1 process, $TH threads) on the live server with the fewest runners (at most $CAP per server), tools on a free GPU of a tools
#     host that is not a Flash-Next server and lives > 3 h. Per GPU at most floor(total/23500) runners of ANY MedAgent-Pro run.
#  3. Rebalance: a server with >= 6 runners while another live server has <= 2 -> stop one runner there (restarted on the emptier server).
# Servers register in dx/<slug>/servers/<job>.url (70_fn_server_hold.req). Runner clients live in THIS job (persistent host job).
# Stop: touch dx/<slug>/FN_SUPERVISOR_STOP. DRY=1: one pass, print only.
SLUG=medagentpro-qwen3.8-flash-next; source /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/mp_hosts.sh
B=$HB; D=$B/dx/$SLUG; LOG=$D/logs/fn_supervisor.log; mkdir -p $D/servers $D/logs
R=/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks
TOT=8; TH=${FN_THREADS:-54}; CAP=${FN_CAP:-8}; BDI=61718805
NOTOOLS=" 61718805 62288438 62399604 61714874 "      # Flash-Next server jobs (+ L40S, ends 12 Nov 11:52): never a tools slot
SA=$(python3 -c "import json;print(json.load(open('$D/T1.1/RUN_META.json'))['server_args'])")
# 11-13 (owner OK): next servers run --max-running-requests 220 (see META_OVERRIDES.txt): runners must record the new args and
# resume past the run_sig change. Set FN_NEW_CAP=0 to keep the old args.
if [ "${FN_NEW_CAP:-1}" = 1 ]; then SA=${SA/--max-running-requests 160 --cuda-graph-max-bs-decode 160/--max-running-requests 220 --cuda-graph-max-bs-decode 224}; export DX_ALLOW_META_CHANGE=1; fi
exec 7>$D/logs/fn_supervisor.flock; flock -n 7 || { echo "another fn supervisor holds the lock"; exit 0; }
declare -A MISS; for s in $(seq 0 $((TOT-1))); do MISS[$s]=0; done
log() { echo "$(date '+%m-%d %H:%M') [fn] $*" | tee -a $LOG; }
remaining() { python3 - "$D" "$R" $TOT <<'PY'
import json,glob,sys
D,R,T=sys.argv[1],sys.argv[2],int(sys.argv[3]); n=[0]*T
for t,d in (("T1.1","T1.1_disease_screening"),("T1.2","T1.2_disease_confirmation")):
    done=set()
    for f in glob.glob(f"{D}/{t}/predictions.p*.jsonl"):
        for l in open(f):
            try: done.add(json.loads(l)["_id"])
            except Exception: pass
    ids=[json.loads(l)["_id"] for l in open(f"{R}/{d}/test.jsonl") if l.strip()]
    for i,x in enumerate(ids):
        if x not in done: n[i%T]+=1
print(" ".join(map(str,n)))
PY
}
fn_servers() { local f u; for f in $D/servers/*.url; do [ -f "$f" ] || continue; u=$(cat $f)
  curl -sf -m 15 "$u/models" 2>/dev/null | grep -q '"Qwen/Qwen3.8-Flash-Next"' && echo $u; done; }
old_runners() {  # "pgid" of the old 4-shard run (shards x/4) on bdiimg
  srun --jobid=$BDI --overlap -n1 -c1 --mem=1G bash -c 'for p in $(pgrep -f "[r]un_dermarena_v1_medagentpro.py"); do a=$(tr "\0" " " < /proc/$p/cmdline)
    echo "$a" | grep -qF "/medagentpro-qwen3.8-flash-next/" && echo "$a" | grep -qE -- "--shard [0-9]+/4( |$)" && ps -o pgid= -p $p; done' 2>/dev/null | tr -d ' ' | sort -u; }
slots() {  # "host gpu free" for GPUs that can take one more runner (any-slug count, memory check), longest-lived host first
  local j end; for j in $(live_hosts); do
    case "$NOTOOLS" in *" $j "*) continue;; esac
    end=$(date -d "$(squeue -h -j $j -o %e)" +%s 2>/dev/null || echo 0); [ $((end - $(date +%s))) -lt 10800 ] && continue
    srun --jobid=$j --overlap -n1 -c1 --mem=1G bash -c 'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits | tr -d " "
      echo "#"; for p in $(pgrep -f "[r]un_dermarena_v1_medagentpro.py"); do tr "\0" "\n" < /proc/$p/environ | grep "^CUDA_VISIBLE_DEVICES=" | cut -d= -f2; done' 2>/dev/null \
    | awk -v j=$j -F, '/^#/{m=1;next} !m{g[$1]=$2; t[$1]=$3; next} m{n[$1]++} END{for(i in g) if(g[i]>=23000 && n[i]+0 < int(t[i]/23500)) print j, i, g[i]}'
  done; }
launch() {  # $1 shard $2 server url $3 host $4 gpu
  log "shard $1: start on $2 ($TH threads), tools on host $3 GPU $4"
  [ -n "$DRY" ] && return 0
  setsid nohup env MP_REMOTE_CPUS=4 MP_REMOTE_MEM=16G bash $B/tools/guarded.sh "${2%/v1}/health" bash $B/agents/run_medagentpro_remote_v2.sh $3 "$2" full \
    MP_SLUG=$SLUG MP_BACKBONE=Qwen/Qwen3.8-Flash-Next "MP_TOOL_GPUS='$4'" MP_NPROC=1 MP_THREADS=$TH MP_SHARD_BASE=$1 MP_SHARD_TOTAL=$TOT \
    MP_ALLOC_CONF=expandable_segments:True "MP_SERVER_ARGS='$SA'" >> $D/logs/fnsup_s$1.out 2>&1 < /dev/null &
  disown
}
log "fn supervisor start (pid $$, DRY=${DRY:-0}, $TOT shards, $TH threads, cap $CAP/server) server_args: $SA"
while [ ! -f $D/FN_SUPERVISOR_STOP ]; do
  old=$(old_runners)
  if [ -n "$old" ]; then
    q=$(SLUG=medagentpro-qwen3.5-27b runners_on $BDI); echo "$q" | grep -q '^OK$' || { log "cannot inspect bdiimg"; sleep 300; continue; }
    nq=$(echo "$q" | grep -v '^OK$' | grep -c .)
    if [ "$nq" -gt 0 ]; then log "old 4-shard run alive; $nq Qwen runner(s) still use bdiimg tools: wait"; [ -n "$DRY" ] && break; sleep 300; continue; fi
    log "SWITCH: Qwen is off bdiimg; stopping the old 4-shard run (pgid $(echo $old))"
    [ -z "$DRY" ] && for g in $old; do srun --jobid=$BDI --overlap -n1 -c1 --mem=1G bash -c "kill -TERM -- -$g" 2>/dev/null; done
    [ -n "$DRY" ] && break; sleep 90; continue
  fi
  run=$(all_runners) || { log "cannot inspect a tools host: skip pass"; sleep 300; continue; }
  servers=$(fn_servers); rem=($(remaining)); left=0; for x in "${rem[@]}"; do left=$((left+x)); done
  nrun=$(echo "$run" | grep -c .)
  [ $left -eq 0 ] && [ $nrun -eq 0 ] && { log "all $TOT shards finished: supervisor exits"; break; }
  if [ -z "$servers" ]; then log "no live Flash-Next server ($left cases left, $nrun runners)"; [ -n "$DRY" ] && break; sleep 300; continue; fi
  cnt() { echo "$run" | awk -v u=$1 '$5==u' | grep -c .; }
  # rebalance: one runner per pass from a crowded server to an empty-ish one
  hi=""; hn=0; lo=""; ln=99; for u in $servers; do n=$(cnt $u); [ $n -gt $hn ] && { hn=$n; hi=$u; }; [ $n -lt $ln ] && { ln=$n; lo=$u; }; done
  if [ $hn -ge 6 ] && [ $ln -le 2 ] && [ "$hi" != "$lo" ]; then
    set -- $(echo "$run" | awk -v u=$hi '$5==u' | tail -1)
    log "rebalance: $hi has $hn runners, $lo has $ln: stopping shard $3 (host $1 pgid $2) to restart it on $lo"
    [ -z "$DRY" ] && srun --jobid=$1 --overlap -n1 -c1 --mem=1G bash -c "kill -TERM -- -$2" 2>/dev/null
    MISS[$3]=1
  fi
  sl=""
  for s in $(seq 0 $((TOT-1))); do
    echo "$run" | awk -v s=$s '$3==s' | grep -q . && { MISS[$s]=0; continue; }
    [ ${rem[$s]} -eq 0 ] && { MISS[$s]=0; continue; }
    MISS[$s]=$((MISS[$s]+1)); [ ${MISS[$s]} -lt 2 ] && [ $nrun -gt 0 ] && continue   # first start (no runners at all): no grace period
    best=""; bn=99; for u in $servers; do n=$(cnt $u); [ $n -lt $bn ] && { bn=$n; best=$u; }; done
    [ $bn -ge $CAP ] && { log "shard $s: every server has $CAP runners"; continue; }
    [ -z "$sl" ] && sl=$(slots)
    slot=$(echo "$sl" | head -1); [ -z "$slot" ] && { log "shard $s: no free tools GPU (${rem[$s]} cases left)"; continue; }
    set -- $slot; launch $s "$best" $1 $2
    sl=$(echo "$sl" | tail -n +2)    # one runner per GPU per pass (its tools are still loading)
    run="$run"$'\n'"$1 0 $s $2 $best"; MISS[$s]=0
  done
  [ -n "$DRY" ] && { log "DRY run: one pass only"; break; }
  sleep 300
done
