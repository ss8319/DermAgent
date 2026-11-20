#!/bin/bash
# Run run_medagentpro_v1.sh <mode> on ANOTHER node's GPUs (tools: one process per GPU) against a backbone server elsewhere
# (Flash-Next takes all 4 H200, so tools go to the held L40S, as for DermAgent). Synchronous: the srun client stays alive until the mode ends.
#   run_medagentpro_remote.sh <tools_jobid> <backbone_url> <mode> [VAR=val ...]
set -uo pipefail
JOB=$1; URL=$2; MODE=$3; shift 3
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs
ENVS="OPENROUTER_BASE_URL=$URL MP_TOOL_GPUS='0 1 2 3' MP_NPROC=4 $*"
srun --jobid=$JOB --overlap --cpu-bind=none -n1 -c16 --mem=200G bash -c "export $ENVS; exec bash $B/agents/run_medagentpro_v1.sh $MODE"
