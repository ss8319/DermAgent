#!/bin/bash
# Start the DermAgent service workers on ANOTHER node's GPUs (tools: one process per GPU), talking to a backbone server
# elsewhere (owner 2026-11-05: Flash-Next takes all 4 H200; tools go to the L40S). Usage:
#   run_dermagent_remote.sh <tools_jobid> <backbone_url> <backbone> <slug> [extra env as VAR=val ...]
# The srun step must STAY ALIVE (Slurm kills a step's processes when its srun exits: 11-06 00:25 first attempt died),
# so the srun client itself is detached on this node (setsid nohup) and writes the service output.
set -euo pipefail
JOB=$1; URL=$2; BB=$3; SLUG=$4; shift 4
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs
mkdir -p $B/dx/$SLUG/logs $B/dx/$SLUG/jobs
rm -f $B/dx/$SLUG/jobs/STOP
ENVS="OPENROUTER_BASE_URL=$URL DA_BACKBONE=$BB DA_SLUG=$SLUG DA_TOOL_GPUS='0 1 2 3' DA_NPROC=4 DA_THREADS=20 $*"
setsid nohup srun --jobid=$JOB --overlap --cpu-bind=none -n1 -c16 --mem=200G bash -c "export $ENVS; exec bash $B/agents/run_dermagent_v1.sh service" \
  > $B/dx/$SLUG/logs/remote_service.out 2>&1 < /dev/null &
disown
sleep 5; echo "launched srun client pid $! for job $JOB"
