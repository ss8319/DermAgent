#!/bin/bash
# v4 (11-11): v3 + DRAIN rule (below). v3 (11-10): like v2, but runners may live on ANY tools host (agents/mp_tool_hosts.txt via agents/mp_hosts.sh), not only the L40S job
# 61714874, which ends 12 Nov 11:52. Every 5 min: a shard (0-3 of 4) that is unfinished and has had no runner on any live host for 3 checks
# gets one: tools on the free GPU slot of the longest-lived host, backbone = the live Qwen3.5-27B server with the fewest runners
# (64 threads if it has none, else 32). Stop: touch <D>/SUPERVISOR4_STOP. DRY=1: one pass, print only.
source /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/mp_hosts.sh
B=$HB; SLUG=medagentpro-qwen3.5-27b; D=$B/dx/$SLUG; LOG=$D/logs/supervisor.log
R=/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks
SA=$(python3 -c "import json;print(json.load(open('$D/T1.1/RUN_META.json'))['server_args'])")
declare -A MISS=([0]=0 [1]=0 [2]=0 [3]=0)
log() { echo "$(date '+%m-%d %H:%M') [v4${DRAIN_ONLY:+-drain}] $*" | tee -a $LOG; }
remaining() {  # $1 shard -> unfinished cases (both tasks)
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
log "supervisor v4 start (pid $$, DRY=${DRY:-0}) hosts: $(live_hosts | tr '\n' ' ') server_args: $SA"
while [ ! -f $D/SUPERVISOR4_STOP ]; do
  hosts=$(live_hosts)
  [ -z "$hosts" ] && { log "no live tools host"; sleep 300; continue; }
  run=$(all_runners) || { log "cannot inspect a live tools host (srun failed): skip this pass"; sleep 300; continue; }
  all_done=1; slots=""; servers=""
  for s in 0 1 2 3; do
    if echo "$run" | awk -v s=$s '$3==s' | grep -q .; then MISS[$s]=0; all_done=0; continue; fi
    rem=$(remaining $s); [ "$rem" -gt 0 ] && all_done=0
    [ "$rem" -eq 0 ] && { MISS[$s]=0; continue; }
    [ -n "${DRAIN_ONLY:-}" ] && continue   # 11-11: v3 (job 62451127) keeps doing restarts; this instance only drains
    MISS[$s]=$((MISS[$s]+1)); log "shard $s: no runner on any host (check ${MISS[$s]}/3), $rem cases left"
    [ ${MISS[$s]} -lt 3 ] && continue
    slots=$(free_slots "$run")
    slot=$(echo "$slots" | head -1); [ -z "$slot" ] && { log "shard $s: no free tools GPU on any live host"; continue; }
    [ -z "$servers" ] && servers=$(qwen_servers)
    [ -z "$servers" ] && { log "shard $s: no live Qwen3.5-27B server"; continue; }
    best=""; bn=99; for u in $servers; do n=$(echo "$run" | awk -v u=$u '$5==u' | wc -l); [ $n -lt $bn ] && { bn=$n; best=$u; }; done
    [ $bn -ge 2 ] && { log "shard $s: every live server already has 2 runners"; continue; }
    th=$([ $bn -eq 0 ] && echo 64 || echo 32); set -- $slot; hj=$1; hg=$2
    log "shard $s: starting a runner on $best ($th threads), tools on host $hj GPU $hg"
    if [ -z "$DRY" ]; then
      setsid nohup env MP_REMOTE_CPUS=4 MP_REMOTE_MEM=16G bash $B/tools/guarded.sh "${best%/v1}/health" bash $B/agents/run_medagentpro_remote_v2.sh $hj "$best" full \
        MP_SLUG=$SLUG MP_BACKBONE=Qwen/Qwen3.5-27B "MP_TOOL_GPUS='$hg'" MP_NPROC=1 MP_THREADS=$th MP_SHARD_BASE=$s MP_SHARD_TOTAL=4 \
        MP_ALLOC_CONF=expandable_segments:True "MP_SERVER_ARGS='$SA'" >> $D/logs/supervisor_s$s.out 2>&1 < /dev/null &
      disown
    fi
    run="$run"$'\n'"$hj 0 $s $hg $best"; MISS[$s]=0
  done
  # DRAIN (11-11, owner: no idle GPUs): one runner per pass leaves a host that ends within DRAIN_H hours for a free slot on a host that
  # lives >= 12 h longer (so a freed standby GPU gets work at once, and nothing is left on a host when it ends). Shards with < 40
  # unfinished cases stay (they will likely finish first). The moved runner keeps its server; resume re-runs only its in-flight cases.
  if [ $all_done = 0 ] && [ -z "$DRY" ]; then
    now=$(date +%s); run=$(all_runners) || run=""
    while read -u 9 hj pg s hg u; do
      [ -z "$hj" ] && continue
      he=$(date -d "$(squeue -h -j $hj -o %e)" +%s 2>/dev/null) || continue
      [ $((he - now)) -gt $(( ${DRAIN_H:-12} * 3600 )) ] && continue
      rem=$(remaining $s); [ "$rem" -lt 40 ] && continue
      tgt=""; while read -u 8 tj tg tf; do te=$(date -d "$(squeue -h -j $tj -o %e)" +%s 2>/dev/null) || continue
        [ $((te - he)) -ge 43200 ] && { tgt="$tj $tg"; break; }; done 8<<< "$(free_slots "$run" < /dev/null)"
      [ -z "$tgt" ] && continue
      set -- $tgt; tj=$1; tg=$2
      n=$(echo "$run" | awk -v u=$u -v p=$pg '$5==u && $2!=p' | wc -l); th=$([ $n -eq 0 ] && echo 64 || echo 32)
      log "drain: shard $s ($rem left) leaves host $hj (ends $(squeue -h -j $hj -o %e)) -> host $tj GPU $tg, server $u, $th threads"
      srun --jobid=$hj --overlap -n1 -c1 --mem=1G bash -c "kill -TERM -- -$pg" 2>/dev/null
      for i in 1 2 3 4 5 6; do sleep 15; runners_on $hj | awk -v p=$pg '$1==p' | grep -q . || break; done
      setsid nohup env MP_REMOTE_CPUS=4 MP_REMOTE_MEM=16G bash $B/tools/guarded.sh "${u%/v1}/health" bash $B/agents/run_medagentpro_remote_v2.sh $tj "$u" full \
        MP_SLUG=$SLUG MP_BACKBONE=Qwen/Qwen3.5-27B "MP_TOOL_GPUS='$tg'" MP_NPROC=1 MP_THREADS=$th MP_SHARD_BASE=$s MP_SHARD_TOTAL=4 \
        MP_ALLOC_CONF=expandable_segments:True "MP_SERVER_ARGS='$SA'" >> $D/logs/supervisor_s$s.out 2>&1 < /dev/null &
      disown; MISS[$s]=0; break
    done 9<<< "$run"
  fi
  [ $all_done = 1 ] && { log "all 4 shards finished: supervisor exits"; break; }
  [ -n "$DRY" ] && { log "DRY run: one pass only"; break; }
  sleep 300
done
