#!/bin/bash
#SBATCH -J mp-qwen-supervisor
#SBATCH -p comp
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 4-00:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/benchmark_runs/logs/%x-%j.out
# Hosts agents/mp_qwen_supervisor.sh in its own CPU job, so it outlives any editor session. The runners it starts keep their srun
# clients in this job: do not cancel this job while those runners work (check dx/medagentpro-qwen3.5-27b/logs/supervisor.log).
exec bash /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/mp_qwen_supervisor.sh
