#!/bin/bash
# bash sim.sh <scenario>; see README.md
SC=${1:?scenario}; D=$(cd $(dirname $0) && pwd); B0=/fs04/scratch2/ub62/ssim0070/benchmark_runs
S=$D/run_$SC${SIM_TAG:-}; rm -rf $S; mkdir -p $S/holder/queue $S/reqs
# 1. the holder script exactly as stored in the pending job, with the sandbox edits listed below
scontrol write batch_script 62728289 - 2>/dev/null > $S/holder_orig.sh || cp /fs04/scratch2/ub62/ssim0070/sglang/slurm/bench_holder.sbatch $S/holder_orig.sh
python3 - $S <<'PY'
import sys,re
S=sys.argv[1]; s=open(S+'/holder_orig.sh').read()
s=s.replace('H=/fs04/scratch2/ub62/ssim0070/benchmark_runs/holders/$NAME;','H=$SIM_S/holder;')
# fake server start/stop (keep the names and the CUR/SP contract)
a=s.index('stop_server() {'); b=s.index('export OPENROUTER_BASE_URL')
s=s[:a]+'''stop_server() { echo "[sim] stop_server (was: ${CUR:-none})" >> $SIM_S/server_calls.log; SP=""; CUR=""; rm -f $SIM_S/health; }
start_server() {
  echo "[sim] start_server model='$1' ctx=$2 args='$3'" >> $SIM_S/server_calls.log
  case "$1" in /tmp/bench_weights/*) ;; *) echo "SERVER FAILED: $1 (sim: not a local weights folder)"; return 1;; esac
  if [ -f $SIM_S/fail_start_once ] && [[ "$1" == *"$(cat $SIM_S/fail_start_once)"* ]]; then rm -f $SIM_S/fail_start_once; echo "SERVER FAILED: $1 (sim: forced)"; return 1; fi
  SP=4194000; CUR="$1|$2|$3"; touch $SIM_S/health; echo "server ready: $1 ($3)"; return 0
}
'''+s[b:]
s=s.replace('if [ -z "$req" ]; then sleep 60; continue; fi','if [ -z "$req" ]; then echo "[sim] queue empty"; break; fi\n  case "$(basename $req)" in 1*) echo "[sim] next would be $(basename $req): Flash-Next, outside this dry-run"; break;; esac')
assert '[sim] queue empty' in s
open(S+'/holder_sim.sh','w').write(s)
PY
# 2. the queue: real requests, in the order the real queue will hold them
for r in 01_a1_qwen35_9b 02_a1_huatuo27b 02b_a1_huatuo27b_retry 03_a1_glm46v 03b_a1_glm46v_retry 04_a1_judge; do
  sed -e "s#/fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/a1_reqs/_a1_common.sh#$S/reqs/_a1_common.sh#g" \
      -e "s#/fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/a1_reqs/03_a1_glm46v.req#$S/holder/queue_src/03_a1_glm46v.req#g" \
      -e "s#source \$B0/lock.sh#source $S/reqs/lock.sh#g" -e "s#\$B0/dx/\$s/.lock#$S/lockroot/dx/\$s/.lock#g" \
      -e "s#\$B0/logs/gpu_watchdog/alerts.log#$S/alerts.log#g" \
      -e "s#\$B0/holders/\*/out/runs.log#\$B0/holders/*/out/runs.log $S/holder/out/runs.log#g" $B0/agents/a1_reqs/$r.req > $S/holder/queue/$r.req
done
mkdir -p $S/holder/queue_src; cp $S/holder/queue/03_a1_glm46v.req $S/holder/queue_src/
sed -e "s#\$B0/holders/\*/out/runs.log#\$B0/holders/*/out/runs.log $S/holder/out/runs.log#g" $B0/agents/a1_reqs/_a1_common.sh > $S/reqs/_a1_common.sh
cp $B0/agents/medagentpro_reqs/10_fn_server_v2.req $S/holder/queue/
# locks: a sandbox copy of the real lock state (the real dx/ and tx_mcqa/ folders are never written by the dry-run)
sed -e "s#local L=/fs04/scratch2/ub62/ssim0070/benchmark_runs #local L=$S/lockroot #" $B0/lock.sh > $S/reqs/lock.sh
grep -q "local L=$S/lockroot " $S/reqs/lock.sh || { echo "lock sandbox edit failed"; exit 1; }
for k in dx tx_mcqa; do for a in qwen3.5-9b huatuogpt-3-27b glm-4.6v; do f=$B0/$k/$a-findingskept-img/.lock; [ -f $f ] && { mkdir -p $S/lockroot/$k/$a-findingskept-img; cp $f $S/lockroot/$k/$a-findingskept-img/; }; done; done
grep -rl "benchmark_runs/dx/\$s/.lock\|source \$B0/lock.sh\|gpu_watchdog/alerts.log" $S/holder/queue/0*.req $S/holder/queue_src 2>/dev/null && { echo "sandbox edit missed a path"; exit 1; }
# 3. fakes
export SIM_S=$S SIM_SC=$SC PATH=$D/bin:$PATH SLURM_JOB_ID=99999917 SLURM_RESTART_COUNT=0
echo 1000000 > $S/clock
date() { if [ "${1:-}" = "+%s" ]; then cat $SIM_S/clock; else command date "$@"; fi; }
sleep() { echo $(( $(cat $SIM_S/clock) + ${1%.*} )) > $SIM_S/clock; echo "[sim] sleep $1 (fake clock +$1)"; sim_tick; }
apptainer() { echo "[sim] apptainer $1 ... (no network in the dry-run)" >&2; return 1; }
curl() { case "$*" in *get_server_info*) echo '{"max_running_requests": 30}';; *) [ -f $SIM_S/health ];; esac; }
kill() { :; }
sim_tick() { :; }
mark() { echo "$*"; }
bash() {   # everything a request runs through `bash <script>`: clean_gpu, guarded benchmark, judge
  case "$*" in
    *clean_gpu.sh*) echo "[sim] clean_gpu";;
    *run_dx_model_findingskept_img.sh*)
      local slug; slug=$(echo "$*" | grep -oE '[a-z0-9.-]+-findingskept-img'); echo "[sim] RUN: $*"; echo "[sim]   DX_EXAMINEE_CONTEXT=${DX_EXAMINEE_CONTEXT:-unset}"
      local n; n=$(grep -c "RUN:.*$slug" $SIM_S/holder/out/runs.log 2>/dev/null); local how; how=$(sim_outcome $slug $n)
      case "$how" in
        ok)     echo "DX_MODEL_DONE $slug $(command date)"; return 0;;
        dead)   echo "SERVER DEAD: (sim) health failed 3 times — stopping"; rm -f $SIM_S/health; return 9;;
        canary) echo "CANARY FAILED T1.1 parse_rate=0.80 scored_rate=0.80 — stopping $slug"; return 3;;
      esac;;
    *judge_dx.sh*) local slug=${@: -1}; echo "[sim] JUDGE: $*"; echo "JUDGE_DONE $slug $(command date)";;
    *) echo "[sim] UNEXPECTED bash call: $*"; return 1;;
  esac
}
export -f date sleep apptainer curl kill bash mark 2>/dev/null
# staged-weights markers on this node's /tmp so local_weights takes its "already staged" path (removed at the end)
W=/tmp/bench_weights; made=""
stage() { [ -e "$W/$1/.complete" ] || { mkdir -p "$W/$1" && touch "$W/$1/.complete" && made="$made $1"; }; }
stage FreedomIntelligence/HuatuoGPT-3-27B; stage Qwen/Qwen3.6-27B
case $SC in
  A) # expected night: Qwen still running on the A40 at first, finishes while the judge waits
     stage zai-org/GLM-4.6V
     sim_outcome() { echo ok; }
     sim_tick() { [ "$(cat $SIM_S/clock)" -ge 1001200 ] && ! grep -qs "^DX_MODEL_DONE qwen3.5-9b-findingskept-img " $SIM_S/holder/out/runs.log && echo "DX_MODEL_DONE qwen3.5-9b-findingskept-img (sim: A40 finished)"; return 0; };;
  B) # Huatuo server dies under 0.90; GLM weights cannot be fetched (twice); Qwen finishes during the judge wait
     sim_outcome() { case "$1:$2" in huatuogpt*:1) echo dead;; *) echo ok;; esac; }
     sim_tick() { [ "$(cat $SIM_S/clock)" -ge 1001200 ] && ! grep -qs "^DX_MODEL_DONE qwen3.5-9b-findingskept-img " $SIM_S/holder/out/runs.log && echo "DX_MODEL_DONE qwen3.5-9b-findingskept-img (sim: A40 finished)"; return 0; };;
  C) # Huatuo canary fails: 02b must NOT retry or reload; GLM first server start fails once (holder retries itself)
     stage zai-org/GLM-4.6V
     sim_outcome() { case "$1" in huatuogpt*) echo canary;; *) echo ok;; esac; };;
  D) # Qwen on the A40 never finishes: the judge must give up after 3 h, not wait forever
     stage zai-org/GLM-4.6V
     sim_outcome() { echo ok; };;
  E) # the A40 job is gone (lock owner not in squeue) and Qwen unfinished: 01 runs here; the judge must not wait for a dead job
     stage zai-org/GLM-4.6V; stage Qwen/Qwen3.5-9B
     squeue() { case "$*" in *62749880*) return 0;; *) command squeue "$@";; esac; }; export -f squeue
     sim_outcome() { case "$1" in qwen*) echo dead;; *) echo ok;; esac; };;
esac
export -f sim_outcome sim_tick
[ $SC = C ] && echo GLM > $S/fail_start_once
( source $S/holder_sim.sh fita100x4fn2_sim ) > $S/holder.out 2>&1; echo "holder exit code: $?" >> $S/holder.out
for m in $made; do rm -f $W/$m/.complete; rmdir -p $W/$m 2>/dev/null; done
echo "=== scenario $SC"; echo "--- holder log"; grep -vE '^GPU 0-3' $S/holder.out
echo "--- server starts/stops"; cat $S/server_calls.log 2>/dev/null
echo "--- runs.log"; cat $S/holder/out/runs.log 2>/dev/null
echo "--- queue after: done=[$(ls $S/holder/queue/done | tr '\n' ' ')] failed=[$(ls $S/holder/queue/failed | tr '\n' ' ')] left=[$(ls $S/holder/queue/*.req 2>/dev/null | xargs -n1 basename | tr '\n' ' ')]"
echo "--- fake clock advanced: $(( $(cat $S/clock) - 1000000 )) s"
