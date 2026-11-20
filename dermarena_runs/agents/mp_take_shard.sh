#!/bin/bash
# mp_take_shard.sh <shard 0-3> <server url .../v1> <unused, was L40S gpu> <threads>
# 11-09: move ONE Qwen MedAgent-Pro shard onto a new server at <threads> threads (one shard per new server ~doubles its rate).
# 11-10 v2: host-aware (agents/mp_hosts.sh, agents/mp_tool_hosts.txt). Stops whatever runner serves that shard on ANY live tools host,
# by its exact process group, then starts the shard with its tools on the free GPU slot of the longest-lived host (the L40S job ends
# 12 Nov 11:52). Foreground. Env PREFER_HOST=<job id> tries that host first (11-11: spread tool hosts). Resume skips finished _ids, so only cases in flight re-run.
S=$1; URL=$2; T=${4:-64}
SLUG=medagentpro-qwen3.5-27b; source /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/mp_hosts.sh
B=$HB; D=$B/dx/$SLUG
SA=$(python3 -c "import json;print(json.load(open('$D/T1.1/RUN_META.json'))['server_args'])")
for i in 1 2 3 4 5 6; do
  run=$(all_runners) || { echo "[take] cannot inspect a tools host, retry $i"; sleep 30; continue; }
  old=$(echo "$run" | awk -v s=$S '$3==s {print $1, $2}')
  [ -z "$old" ] && break
  echo "[take] shard $S: stopping the existing runner(s) (host pgid: $(echo $old)), try $i $(date)"
  echo "$old" | while read j g; do srun --jobid=$j --overlap -n1 -c1 --mem=1G bash -c "kill -TERM -- -$g" 2>/dev/null; done
  sleep 45
done
run=$(all_runners) || { echo "[take] cannot inspect the tools hosts: not starting"; exit 5; }
echo "$run" | awk -v s=$S '$3==s' | grep -q . && { echo "[take] shard $S: old runner still alive: not starting a second one"; exit 5; }
slot=""; for i in 1 2 3 4 5 6; do [ -n "${PREFER_HOST:-}" ] && slot=$(free_slots "$run" | awk -v h=$PREFER_HOST '$1==h' | head -1); [ -z "$slot" ] && slot=$(free_slots "$run" | head -1); [ -n "$slot" ] && break; echo "[take] no free tools GPU yet ($i/6)"; sleep 60; run=$(all_runners); done
[ -z "$slot" ] && { echo "[take] shard $S: no free tools GPU on any live host: leaving it to the supervisor"; exit 6; }
set -- $slot; HJ=$1; HG=$2
echo "[take] shard $S: starting on $URL, tools on host $HJ GPU $HG, $T threads $(date)"
MP_REMOTE_CPUS=4 MP_REMOTE_MEM=16G exec bash $B/tools/guarded.sh "${URL%/v1}/health" bash $B/agents/run_medagentpro_remote_v2.sh $HJ "$URL" full \
  MP_SLUG=$SLUG MP_BACKBONE=Qwen/Qwen3.5-27B "MP_TOOL_GPUS='$HG'" MP_NPROC=1 MP_THREADS=$T MP_SHARD_BASE=$S MP_SHARD_TOTAL=4 \
  MP_ALLOC_CONF=expandable_segments:True "MP_SERVER_ARGS='$SA'"
