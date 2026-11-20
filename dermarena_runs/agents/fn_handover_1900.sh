#!/bin/bash
# fn_handover_1900.sh (11-13, owner GO "run with best plan") — runs detached on the LOGIN node; one shot at the H200 (h200x4e
# 62399604) walltime end, 13 Nov 19:02. Light: a few sruns + file moves.
#  1. wait until 19:03 or until the H200 server fails 3 health checks in a row
#  2. stop the T1.1 governor; kill every leftover Flash-Next runner of the OLD layout (x/8, x/24) by exact pid, incl. the 3 paused
#     (SIGSTOPped) catch-up runners on gpu2a100b (their in-flight cases are re-run later; finished cases are kept)
#  3. unregister the H200 server; gpu2a100b stops being a tools host
#  4. gpu2a100b: judge every finished Flash-Next case once (Qwen3.6-27B, same judge as all runs), then stop its server
#     -> the watchdog releases it 2 h after it goes idle (owner exception). gpu2a100 + A40 stay as tools hosts (plan).
#  5. start agents/mp_fn_supervisor2.sh (waits for the next registered Flash-Next server: A100 / H200 via 10_fn_server_v2.req)
# Stop before it acts: touch agents/FN_HANDOVER_STOP
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; SLUG=medagentpro-qwen3.8-flash-next; D=$B/dx/$SLUG
L=$D/logs/fn_handover.log; A=$B/logs/gpu_watchdog/alerts.log; H200=62399604; U=http://m3u010:30504
log() { echo "$(date '+%m-%d %H:%M') [handover] $*" | tee -a $L; }
ovl() { local j=$1; shift; timeout 120 env $(env | grep -oE '^SLURM_[A-Z_]+' | grep -v '^SLURM_CONF' | sed 's/^/-u /') srun --jobid=$j --overlap -n1 -c1 --mem=512M "$@" 2>/dev/null; }
log "armed (pid $$): acts at 19:03 or when the H200 server dies"
f=0
while [ ! -f $B/agents/FN_HANDOVER_STOP ]; do
  [ $(date +%s) -ge $(date -d 'today 19:03' +%s) ] && break
  if curl -sf -m 30 $U/health >/dev/null; then f=0; else f=$((f+1)); fi
  [ $f -ge 3 ] && { log "H200 server failed 3 health checks: acting early"; break; }
  sleep 60
done
[ -f $B/agents/FN_HANDOVER_STOP ] && { log "FN_HANDOVER_STOP: exit without acting"; exit 0; }
log "acting"; echo "$(date '+%m-%d %H:%M') HANDOVER h200x4e ended: cleaning old Flash-Next runners, judging on gpu2a100b, starting supervisor2" >> $A
touch $B/agents/FN_T11_FIRST_STOP
sleep 240   # guarded.sh stops runners after 3 failed health checks; give it time, then clean up what is left
for j in 61981434 61981433 62450847; do
  pids=$(ovl $j bash -c 'for p in $(pgrep -f "[r]un_dermarena_v1_medagentpro.py"); do a=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null)
           echo "$a" | grep -qF "/medagentpro-qwen3.8-flash-next/" && echo "$a" | grep -qE -- "--shard [0-9]+/(8|24)( |$)" && echo $p; done')
  [ -z "$pids" ] && { log "job $j: no old-layout Flash-Next runner left"; continue; }
  log "job $j: killing old-layout Flash-Next runners (pids: $(echo $pids))"
  ovl $j bash -c "kill -CONT $(echo $pids) 2>/dev/null; kill -KILL $(echo $pids)"
done
rm -f $D/servers/$H200.url $D/servers/$H200.args
grep -v '^61981434' $B/agents/mp_tool_hosts.txt > $B/agents/mp_tool_hosts.txt.new && mv $B/agents/mp_tool_hosts.txt.new $B/agents/mp_tool_hosts.txt
log "unregistered the H200 server; gpu2a100b removed from tools hosts"
Q=$B/holders/gpu2a100b/queue
cp $B/agents/medagentpro_reqs/08c_judge_fn_once.req $Q/
cp $B/agents/medagentpro_reqs/090_tools_host_stop_server.req $Q/
log "queued on gpu2a100b: 08c_judge_fn_once.req, then 090_tools_host_stop_server.req"
setsid nohup bash $B/agents/mp_fn_supervisor2.sh >> $D/logs/fn_supervisor2.out 2>&1 < /dev/null &
disown; log "started mp_fn_supervisor2.sh (pid $!)"
echo "$(date '+%m-%d %H:%M') HANDOVER done: supervisor2 waiting for the next Flash-Next server" >> $A
