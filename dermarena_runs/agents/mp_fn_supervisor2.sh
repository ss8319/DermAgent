#!/bin/bash
# mp_fn_supervisor2.sh (11-13, owner GO "run with best plan") — MedAgent-Pro x Qwen3.8-Flash-Next after the H200 (h200x4e) ends.
# Differences from mp_fn_supervisor.sh (which stays stopped: FN_SUPERVISOR_STOP):
#  - shards = $TOT (default 2) of the REMAINING cases (case i -> i % TOT; resume skips every _id done by any earlier shard layout),
#    $TH threads per runner (default 96 = one A100 server's running-request cap), so 2 runners on one A40 can feed 2 A100 servers.
#  - each server registers its own args (servers/<job>.args, written by 10_fn_server_v2.req); a runner gets ITS server's args as
#    MP_SERVER_ARGS, so run_sig (in every prediction row) records which server config produced each case. DX_ALLOW_META_CHANGE=1.
#  - tools hosts = agents/mp_tool_hosts.txt minus any job that serves Flash-Next; per GPU at most floor(total/21000) runners
#    (A40 46 GB -> 2; measured 19.6 GB per runner on 11-13), free >= 21000 MiB, host lives > 3 h; longest-lived host first.
#  - rebalance: a server with >= 2 runners while another live server has 0 -> stop one runner there (restarted on the empty one).
#    Costs that runner's in-flight cases (resume keeps finished ones); worth it when a second server arrives.
# T1.1 first: each runner finishes its T1.1 shard before T1.2 (runner order), and all runners start together.
# Stop: touch dx/<slug>/FN_SUP2_STOP. DRY=1: one pass, print only.
SLUG=medagentpro-qwen3.8-flash-next; source /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/mp_hosts.sh
B=$HB; D=$B/dx/$SLUG; LOG=$D/logs/fn_supervisor2.log; mkdir -p $D/servers $D/logs
R=/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks
TOT=${FN_TOT:-2}; TH=${FN_THREADS:-96}; CAP=${FN_CAP:-4}
export DX_ALLOW_META_CHANGE=1
exec 7>$D/logs/fn_supervisor2.flock; flock -n 7 || { echo "another fn supervisor2 holds the lock"; exit 0; }
declare -A MISS; for s in $(seq 0 $((TOT-1))); do MISS[$s]=0; done
log() { echo "$(date '+%m-%d %H:%M') [fn2] $*" | tee -a $LOG; }
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
  [ -f "${f%.url}.args" ] || continue
  curl -sf -m 15 "$u/models" 2>/dev/null | grep -q '"Qwen/Qwen3.8-Flash-Next"' && echo $u; done; }
server_jobs() { local f; for f in $D/servers/*.url; do [ -f "$f" ] && basename $f .url; done; }
args_of() { local f; for f in $D/servers/*.url; do [ "$(cat $f 2>/dev/null)" = "$1" ] && cat "${f%.url}.args" && return; done; }
slots() {  # "host gpu free" for GPUs that can take one more runner, longest-lived host first
  local j end sj=" $(server_jobs | tr '\n' ' ') "; for j in $(live_hosts); do
    case "$sj" in *" $j "*) continue;; esac
    end=$(date -d "$(squeue -h -j $j -o %e)" +%s 2>/dev/null || echo 0); [ $((end - $(date +%s))) -lt 10800 ] && continue
    srun --jobid=$j --overlap -n1 -c1 --mem=1G bash -c 'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits | tr -d " "
      echo "#"; for p in $(pgrep -f "[r]un_dermarena_v1_medagentpro.py"); do tr "\0" "\n" < /proc/$p/environ | grep "^CUDA_VISIBLE_DEVICES=" | cut -d= -f2; done' 2>/dev/null \
    | awk -v j=$j -F, '/^#/{m=1;next} !m{g[$1]=$2; t[$1]=$3; next} m{n[$1]++} END{for(i in g) if(g[i]>=21000 && n[i]+0 < int(t[i]/21000)) print j, i, g[i]}'
  done; }
launch() {  # $1 shard $2 server url $3 host $4 gpu
  local SA; SA=$(args_of "$2")
  log "shard $1/$TOT: start on $2 ($TH threads), tools on host $3 GPU $4, server args: $SA"
  [ -n "$DRY" ] && return 0
  setsid nohup env MP_REMOTE_CPUS=8 MP_REMOTE_MEM=24G bash $B/tools/guarded.sh "${2%/v1}/health" bash $B/agents/run_medagentpro_remote_v2.sh $3 "$2" full \
    MP_SLUG=$SLUG MP_BACKBONE=Qwen/Qwen3.8-Flash-Next "MP_TOOL_GPUS='$4'" MP_NPROC=1 MP_THREADS=$TH MP_SHARD_BASE=$1 MP_SHARD_TOTAL=$TOT \
    MP_ALLOC_CONF=expandable_segments:True DX_ALLOW_META_CHANGE=1 "MP_SERVER_ARGS='$SA'" >> $D/logs/fnsup2_s${1}of$TOT.out 2>&1 < /dev/null &
  disown
}
log "fn supervisor2 start (pid $$, DRY=${DRY:-0}, $TOT shards, $TH threads, cap $CAP/server)"
while [ ! -f $D/FN_SUP2_STOP ]; do
  run=$(all_runners) || { log "cannot inspect a tools host: skip pass"; sleep 300; continue; }
  servers=$(fn_servers); rem=($(remaining)); left=0; for x in "${rem[@]}"; do left=$((left+x)); done
  nrun=$(echo "$run" | grep -c .)
  [ $left -eq 0 ] && [ $nrun -eq 0 ] && { log "all cases finished: supervisor2 exits"; break; }
  if [ -z "$servers" ]; then log "no live Flash-Next server ($left cases left [${rem[*]}], $nrun runners)"; [ -n "$DRY" ] && break; sleep 300; continue; fi
  cnt() { echo "$run" | awk -v u=$1 '$5==u' | grep -c .; }
  hi=""; hn=0; lo=""; ln=99; for u in $servers; do n=$(cnt $u); [ $n -gt $hn ] && { hn=$n; hi=$u; }; [ $n -lt $ln ] && { ln=$n; lo=$u; }; done
  # 11-13 19:00: rebalance OFF by default (FN_REBALANCE=1 to enable): moving a runner kills its in-flight cases, which near the end of
  # T1.1 are 8+ h of work each. A new server gets new work instead (see fn_t11_finisher.sh / T1.2 runners).
  if [ "${FN_REBALANCE:-0}" = 1 ] && [ $hn -ge 2 ] && [ $ln -eq 0 ] && [ "$hi" != "$lo" ]; then
    set -- $(echo "$run" | awk -v u=$hi '$5==u' | tail -1)
    log "rebalance: $hi has $hn runners, $lo has none: stopping shard $3 (host $1 pgid $2) to restart it on $lo"
    [ -z "$DRY" ] && srun --jobid=$1 --overlap -n1 -c1 --mem=1G bash -c "kill -TERM -- -$2" 2>/dev/null
    MISS[$3]=1
  fi
  sl=""
  for s in $(seq 0 $((TOT-1))); do
    echo "$run" | awk -v s=$s '$3==s' | grep -q . && { MISS[$s]=0; continue; }
    [ ${rem[$s]} -eq 0 ] && { MISS[$s]=0; continue; }
    MISS[$s]=$((MISS[$s]+1)); [ ${MISS[$s]} -lt 2 ] && [ $nrun -gt 0 ] && continue
    best=""; bn=99; for u in $servers; do n=$(cnt $u); [ $n -lt $bn ] && { bn=$n; best=$u; }; done
    [ $bn -ge $CAP ] && { log "shard $s: every server has $CAP runners"; continue; }
    [ -z "$sl" ] && sl=$(slots)
    slot=$(echo "$sl" | head -1); [ -z "$slot" ] && { log "shard $s: no free tools GPU (${rem[$s]} cases left)"; continue; }
    set -- $slot; launch $s "$best" $1 $2
    sl=$(echo "$sl" | tail -n +2)
    run="$run"$'\n'"$1 0 $s $2 $best"; MISS[$s]=0
  done
  [ -n "$DRY" ] && { log "DRY run: one pass only"; break; }
  sleep 300
done
log "fn supervisor2 stop"
