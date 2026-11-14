#!/bin/bash
#SBATCH -J abl-l40s-judge9b
#SBATCH -p comp
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 1-00:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/benchmark_runs/holders/l40s_judge/%x-%j.out
# Holds the srun client for agents/ops/l40s_judge9b.sh inside the L40S tools job, so the judge outlives any editor session.
# Stop it safely: touch benchmark_runs/holders/l40s_judge/STOP (the script stops the judge and its server; judging resumes later).
exec srun --jobid=61714874 --overlap -n1 -c8 --mem=100G --gres=gpu:4 bash /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/ops/l40s_judge9b.sh
