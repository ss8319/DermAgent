#!/bin/bash
#SBATCH -J mp-qwen-supervisor4
#SBATCH -p comp
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 7-00:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/benchmark_runs/logs/%x-%j.out
# Hosts agents/mp_qwen_supervisor_v4.sh in its own CPU job, so it outlives any editor session. The runners it starts keep their srun
# clients in this job: do not cancel this job while those runners work (check dx/medagentpro-qwen3.5-27b/logs/supervisor.log).
exec env DRAIN_ONLY=1 bash /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/mp_qwen_supervisor_v4.sh
