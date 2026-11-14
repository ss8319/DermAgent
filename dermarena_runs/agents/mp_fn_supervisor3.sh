#!/bin/bash
# mp_fn_supervisor3.sh (11-13 19:1x, first-principles review; replaces mp_fn_supervisor2.sh) — MedAgent-Pro x Qwen3.8-Flash-Next, T1.1 FIRST.
# Roles (each = one runner process, tools on a tools-host GPU; the A40 fits 2 runners, measured 19.6 GB each):
#   t11      MP_TASKS=T1.1, shard 0/1, $TH11 threads (>= all remaining T1.1 cases, so every case is in flight at once),
#            placed on the FIRST live server (the one with the most Flash-Next capacity is not known; first registered wins).
#   t12_s0/1 MP_TASKS=T1.2, shards 0/2 and 1/2, $TH12 threads. While T1.1 has cases left they go ONLY to servers without the t11
#            runner, so T1.2 never takes capacity from T1.1. Once T1.1 is done they may use any server.
# No rebalancing: moving a runner kills its in-flight cases (8+ h of work each near the end).
# Each server's own args (servers/<job>.args) go to its runners as MP_SERVER_ARGS -> run_sig (per-case server provenance).
# Stop: touch dx/<slug>/FN_SUP3_STOP. DRY=1: one pass, print only.
SLUG=medagentpro-qwen3.8-flash-next; source /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/mp_hosts.sh
B=$HB; D=$B/dx/$SLUG; LOG=$D/logs/fn_supervisor3.log; mkdir -p $D/servers $D/logs
R=/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks
TH11=${FN_TH11:-140}; TH12=${FN_TH12:-112}   # 11-14: 112 > cap 96 so one runner keeps its server full (cases spend time in tools)
export DX_ALLOW_META_CHANGE=1
exec 7>$D/logs/fn_supervisor3.flock; flock -n 7 || { echo "$(date '+%m-%d %H:%M') [fn3] another process holds the supervisor lock (a runner started by an old supervisor?): exit" | tee -a $LOG; exit 0; }
log() { echo "$(date '+%m-%d %H:%M') [fn3] $*" | tee -a $LOG; }
left() { python3 - "$D" "$R" <<'PY'
import json,glob,sys
D,R=sys.argv[1],sys.argv[2]; out=[]
for t,d in (("T1.1","T1.1_disease_screening"),("T1.2","T1.2_disease_confirmation")):
    done=set()
    for f in glob.glob(f"{D}/{t}/predictions.p*.jsonl"):
        for l in open(f):
            try: done.add(json.loads(l)["_id"])
            except Exception: pass
    ids=[json.loads(l)["_id"] for l in open(f"{R}/{d}/test.jsonl") if l.strip()]
    out.append(sum(1 for x in ids if x not in done))
    if t == "T1.2": out += [sum(1 for i, x in enumerate(ids) if i % 2 == k and x not in done) for k in (0, 1)]
print(*out)
PY
}
fn_servers() { local f u; for f in $D/servers/*.url; do [ -f "$f" ] || continue; [ -f "${f%.url}.args" ] || continue; u=$(cat $f)
  curl -sf -m 15 "$u/models" 2>/dev/null | grep -q '"Qwen/Qwen3.8-Flash-Next"' && echo $u; done; }
server_jobs() { local f; for f in $D/servers/*.url; do [ -f "$f" ] && basename $f .url; done; }
args_of() { local f; for f in $D/servers/*.url; do [ "$(cat $f 2>/dev/null)" = "$1" ] && cat "${f%.url}.args" && return; done; }
runners() {  # "host pgid role url" for every Flash-Next runner on a live tools host; role = t11 | t12_s<k> | old (any other layout)
  local j; for j in $(live_hosts); do
    timeout 120 srun --jobid=$j --overlap -n1 -c1 --mem=1G bash -c 'echo OK; for p in $(pgrep -f "[r]un_dermarena_v1_medagentpro.py"); do
        a=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null); echo "$a" | grep -qF "/medagentpro-qwen3.8-flash-next/" || continue
        t=$(echo "$a" | grep -oE -- "--tasks [^ ]+" | cut -d" " -f2); s=$(echo "$a" | grep -oE -- "--shard [0-9]+/[0-9]+" | cut -d" " -f2)
        u=$(tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep "^OPENROUTER_BASE_URL=" | cut -d= -f2-)
        r=old; [ "$t" = T1.1 ] && [ "$s" = 0/1 ] && r=t11; [ "$t" = T1.2 ] && case $s in 0/2) r=t12_s0;; 1/2) r=t12_s1;; esac
        echo "$(ps -o pgid= -p $p | tr -d " ") $r $u"; done' 2>/dev/null | { read ok; [ "$ok" = OK ] || exit 1; sed "s/^/$j /"; } || return 1
  done; }
slot() {  # first "host gpu" that can take one more runner (free >= 21000 MiB and fewer runners than total/21000), longest-lived host first
  local j end sj=" $(server_jobs | tr '\n' ' ') "; for j in $(live_hosts); do
    case "$sj" in *" $j "*) continue;; esac
    end=$(date -d "$(squeue -h -j $j -o %e)" +%s 2>/dev/null || echo 0); [ $((end - $(date +%s))) -lt 10800 ] && continue
    timeout 120 srun --jobid=$j --overlap -n1 -c1 --mem=1G bash -c 'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits | tr -d " "
      echo "#"; for p in $(pgrep -f "[r]un_dermarena_v1_medagentpro.py"); do tr "\0" "\n" < /proc/$p/environ | grep "^CUDA_VISIBLE_DEVICES=" | cut -d= -f2; done' 2>/dev/null \
    | awk -v j=$j -F, '/^#/{m=1;next} !m{g[$1]=$2; t[$1]=$3; next} m{n[$1]++} END{for(i in g) if(g[i]>=21000 && n[i]+0 < int(t[i]/21000)) print j, i}' | head -1
  done | head -1; }
cap_of() { local c; c=$(args_of "$1" | grep -oE -- '--max-running-requests [0-9]+' | grep -oE '[0-9]+$'); echo ${c:-0}; }
launch() {  # $1 role  $2 server url  $3 host  $4 gpu
  local SA tasks sh th; SA=$(args_of "$2")
  case $1 in t11) tasks=T1.1; sh="0 1"; th=$TH11;; t12_s0) tasks=T1.2; sh="0 2"; th=$TH12;; t12_s1) tasks=T1.2; sh="1 2"; th=$TH12;; esac
  local cap; cap=$(echo "$SA" | grep -oE -- '--max-running-requests [0-9]+' | grep -oE '[0-9]+$')   # 11-14 review: 112 threads used half an H200
  [ -n "$cap" ] && [ $1 != t11 ] && th=$((cap + 16))
  set -- $1 $2 $3 $4 $sh
  log "start $1 on $2 ($th threads), tools on host $3 GPU $4, server args: $SA"
  [ -n "$DRY" ] && return 0
  setsid nohup env MP_REMOTE_CPUS=8 MP_REMOTE_MEM=24G bash $B/tools/guarded.sh "${2%/v1}/health" bash $B/agents/run_medagentpro_remote_v2.sh $3 "$2" full \
    MP_SLUG=$SLUG MP_BACKBONE=Qwen/Qwen3.8-Flash-Next "MP_TOOL_GPUS='$4'" MP_NPROC=1 MP_THREADS=$th MP_SHARD_BASE=$5 MP_SHARD_TOTAL=$6 \
    MP_TASKS=$tasks MP_LOG_TAG=_$1 MP_ALLOC_CONF=expandable_segments:True DX_ALLOW_META_CHANGE=1 "MP_SERVER_ARGS='$SA'" >> $D/logs/fnsup3_$1.out 2>&1 < /dev/null 7>&- &
  disown
}
log "fn supervisor3 start (pid $$, DRY=${DRY:-0}, t11 threads $TH11, t12 threads $TH12)"
while [ ! -f $D/FN_SUP3_STOP ]; do
  run=$(runners) || { log "cannot inspect a tools host: skip pass"; sleep 300 7>&-; continue; }
  touch $D/logs/fn_supervisor3.heartbeat   # 11-14 review: the watchdog relaunches a supervisor whose heartbeat is > 30 min old (hung srun)
  set -- $(left); l11=$1; l12=$2; l12s0=$3; l12s1=$4
  [ -z "$l12s1" ] && { log "case count failed: skip pass"; sleep 300 7>&-; continue; }
  servers=$(fn_servers)
  if [ "$l12" != "${LAST12:-}" ]; then LAST12=$l12; LASTCH=$(date +%s); fi
  if [ -n "$(echo "$run" | grep .)" ] && [ $(( $(date +%s) - ${LASTCH:-$(date +%s)} )) -gt 43200 ] && [ $(( $(date +%s) - ${LASTAL:-0} )) -gt 21600 ]; then
    LASTAL=$(date +%s); echo "$(date '+%m-%d %H:%M') FN-T1.2 NO PROGRESS for 12 h ($l12 left, runners alive): check full_t12_s*_p0.log" >> $B/logs/gpu_watchdog/alerts.log; fi
  [ $l11 -eq 0 ] && [ $l12 -eq 0 ] && [ -z "$(echo "$run" | grep .)" ] && { log "all cases finished: supervisor3 exits"; break; }
  if [ -z "$servers" ]; then log "no live Flash-Next server (T1.1 left $l11, T1.2 left $l12)"; [ -n "$DRY" ] && break; sleep 300 7>&-; continue; fi
  has() { echo "$run" | awk -v r=$1 '$3==r' | grep -q .; }
  srv_of() { echo "$run" | awk -v r=$1 '$3==r{print $4}' | head -1; }
  first=$(echo "$servers" | head -1)
  want=""
  [ $l11 -gt 0 ] && ! has t11 && want="t11"
  for r in t12_s0 t12_s1; do k=${r#t12_s}; lk=$([ $k = 0 ] && echo ${l12s0:-$l12} || echo ${l12s1:-$l12}); [ "$lk" -gt 0 ] && ! has $r && want="$want $r"; done
  for r in $want; do
    if [ $r = t11 ]; then tgt=$first
    else  # T1.2: while T1.1 runs, only on a server without the t11 runner
      # 11-14: choose the eligible server with the FEWEST runners (was: the first one), so a replacement runner never doubles up on
      # one server while another (e.g. a new H200) has none.
      t11srv=$(srv_of t11); tgt=""; bn=99; bc=0
      for u in $servers; do
        if [ $l11 -gt 0 ] && { [ "$u" = "$t11srv" ] || { [ -z "$t11srv" ] && [ "$u" = "$first" ]; }; }; then continue; fi
        n=$(echo "$run" | awk -v u="$u" '$4==u' | grep -c .); c=$(cap_of $u)   # 11-14: ties -> larger cap (H200 over A100)
        if [ $n -lt $bn ] || { [ $n -eq $bn ] && [ $c -gt ${bc:-0} ]; }; then bn=$n; bc=$c; tgt=$u; fi
      done
      # 11-14: at most ONE T1.2 runner per server (tool slots are scarce: the A40 holds 2). Two runners on one cap-96 server only
      # queue, and would leave the next server (fit 4xA100, H200) without a runner until a slot frees.
      [ -n "$tgt" ] && [ $bn -gt 0 ] && { log "$r waits: every live server already has a runner (one per server)"; continue; }
      [ -z "$tgt" ] && { log "$r waits: T1.1 still running ($l11 left) and no second server"; continue; }
    fi
    sl=$(slot); [ -z "$sl" ] && { log "$r: no free tools GPU"; break; }
    launch $r $tgt $sl
    run="$run"$'\n'"${sl% *} 0 $r $tgt"; sleep 30 7>&-
  done
  # 11-14 auto-fallback UPGRADE: every role runs, and a live server WITHOUT a runner has >= 2x the cap of the smallest-cap server
  # that has a T1.2 runner (e.g. an H200, cap 220, arriving while the runners sit on 4xA100s, cap 96) and no tools slot is free:
  # stop that runner; the next pass relaunches its role on the bigger server (ties go to the larger cap). Finished cases are kept;
  # in-flight ones restart. Never ping-pongs (the freed small server is < 2x of the big one).
  if [ -z "$want" ] && [ -n "$servers" ]; then
    big=""; bigc=0
    for u in $servers; do echo "$run" | awk -v u="$u" '$4==u' | grep -q . && continue; c=$(cap_of $u); [ $c -gt $bigc ] && { bigc=$c; big=$u; }; done
    if [ -n "$big" ]; then
      vic=$(echo "$run" | awk '$3 ~ /^t12_s/ && $2 > 1' | while read h pg r u; do c=$(cap_of $u); [ $c -gt 0 ] && echo "$c $h $pg $r $u"; done | sort -n | head -1)
      if [ -n "$vic" ]; then set -- $vic
        # 11-14 review: only while the victim's shard has >= 3x its server cap left (a tail runner is not capacity-bound, and
        # killing it would throw away up to ~100 in-flight multi-hour cases for nothing)
        vl=$([ "$4" = t12_s0 ] && echo $l12s0 || echo $l12s1)
        if [ "$1" -gt 0 ] && [ $bigc -ge $(( 2 * $1 )) ] && [ $vl -ge $(( 3 * $1 )) ] && [ -z "$(slot)" ]; then
          log "UPGRADE: stop $4 on $5 (cap $1, host $2 pgid $3) so the next pass moves it to $big (cap $bigc, no runner)"
          echo "$(date '+%m-%d %H:%M') FN-UPGRADE moving $4 from $5 (cap $1) to $big (cap $bigc)" >> $B/logs/gpu_watchdog/alerts.log
          [ -z "$DRY" ] && timeout 120 srun --jobid=$2 --overlap -n1 -c1 --mem=1G kill -TERM -- -$3 2>/dev/null
        fi
      fi
    fi
  fi
  # 11-14 review: a live server with NO runner for >= 60 min while no role is waiting (3rd server, the one an UPGRADE left, or a
  # finished shard's server in the tail) -> RELEASE_<job>; its 10_ req stops holding and the watchdog frees the job (idle-GPU policy).
  declare -A NR 2>/dev/null; now=$(date +%s)
  for u in $servers; do
    if echo "$run" | awk -v u="$u" '$4==u' | grep -q . || [ -n "$want" ]; then unset "NR[$u]"; continue; fi
    [ -z "${NR[$u]:-}" ] && NR[$u]=$now
    if [ $(( now - ${NR[$u]} )) -ge 3600 ]; then
      for f in $D/servers/*.url; do [ "$(cat $f 2>/dev/null)" = "$u" ] || continue; jb=$(basename $f .url)
        [ -f $D/servers/RELEASE_$jb ] && continue
        log "RELEASE: server $u (job $jb) has had no runner for $(( (now - ${NR[$u]}) / 60 )) min and no role waits"
        [ -z "$DRY" ] && touch $D/servers/RELEASE_$jb; done
    fi
  done
  [ -n "$DRY" ] && { log "DRY: one pass only"; break; }
  sleep 300 7>&-
done
log "fn supervisor3 stop"
