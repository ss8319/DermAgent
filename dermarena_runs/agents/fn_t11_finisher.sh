#!/bin/bash
# fn_t11_finisher.sh (11-13 19:0x): finish MedAgent-Pro x Flash-Next T1.1 INCLUDING judging, with no session in the loop.
# 1. wait until every T1.1 case has a prediction and no T1.1-only runner is alive
# 2. stop the 2-node A100 server (fn_pp2_server.sh) so gpu2a100b is free; its T1.2 runners lose only just-started cases
# 3. queue 12b_judge_fn_t11_dp2.req + 090_tools_host_stop_server.req on gpu2a100b -> one judge pass over every Flash-Next case
# 4. when the judge request is in done/, relaunch fn_pp2_server.sh (T1.2 continues; supervisor3 starts T1.2 runners on it)
# Detached on the login node. Stop: touch agents/FN_FINISHER_STOP.
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; D=$B/dx/medagentpro-qwen3.8-flash-next; L=$D/logs/fn_t11_finisher.log; A=$B/logs/gpu_watchdog/alerts.log
STOP=$B/agents/FN_FINISHER_STOP; Q=$B/holders/gpu2a100b/queue
log() { echo "$(date '+%m-%d %H:%M') [finisher] $*" | tee -a $L; }
ovl() { local j=$1; shift; env $(env | grep -oE '^SLURM_[A-Z_]+' | grep -v '^SLURM_CONF' | sed 's/^/-u /') timeout 120 srun --jobid=$j --overlap -n1 -c1 --mem=512M "$@" 2>/dev/null; }
t11_left() { python3 - <<'PY'
import json,glob
D="/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/medagentpro-qwen3.8-flash-next"; R="/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks"
done=set()
for f in glob.glob(f"{D}/T1.1/predictions.p*.jsonl"):
    for l in open(f):
        try: done.add(json.loads(l)["_id"])
        except Exception: pass
print(sum(1 for l in open(f"{R}/T1.1_disease_screening/test.jsonl") if l.strip() and json.loads(l)["_id"] not in done))
PY
}
t11_runner_alive() { ovl 62450847 bash -c 'pgrep -f "[r]un_dermarena_v1_medagentpro.py.*--tasks T1.1 " >/dev/null && echo y'; }
log "armed (pid $$)"
while [ ! -f $STOP ]; do n=$(t11_left); [ "${n:-1}" = 0 ] && [ -z "$(t11_runner_alive)" ] && break; sleep 600; done
[ -f $STOP ] && { log "STOP"; exit 0; }
log "T1.1 complete: stopping the 2-node server and judging on gpu2a100b"; echo "$(date '+%m-%d %H:%M') FN T1.1 COMPLETE (all 2,806 predicted): judging on gpu2a100b" >> $A
touch $B/agents/FN_PP2_STOP; sleep 20; rm -f $D/servers/61981433.url
for J in 61981433 61981434; do ovl $J bash -c 'pkill -u $USER -f "sglang.launch_server --model-path /tmp/bench_weights/Qwen/Qwen3.8-Flash-Next"; sleep 20; pkill -9 -u $USER -f "sglang" ; true'; done
cp $B/agents/medagentpro_reqs/12b_judge_fn_t11_dp2.req $B/agents/medagentpro_reqs/090_tools_host_stop_server.req $Q/
log "queued 12b judge + 090 on gpu2a100b"
while [ ! -f $STOP ]; do
  [ -f $Q/done/12b_judge_fn_t11_dp2.req ] && break
  [ -f $Q/failed/12b_judge_fn_t11_dp2.req ] && { log "JUDGE FAILED (see holders/gpu2a100b/out/runs.log)"; echo "$(date '+%m-%d %H:%M') FN T1.1 judge FAILED on gpu2a100b" >> $A; break; }
  sleep 120
done
grep -A3 "T1.1" $D/T1.1/SCORE.txt 2>/dev/null | head -4 | tee -a $L
echo "$(date '+%m-%d %H:%M') FN T1.1 JUDGED: see $D/T1.1/SCORE.txt; relaunching the 2-node server for T1.2" >> $A
rm -f $B/agents/FN_PP2_STOP; setsid nohup bash $B/agents/fn_pp2_server.sh >> $D/logs/fn_pp2.out 2>&1 < /dev/null & disown
log "relaunched fn_pp2_server.sh for T1.2; done"
