#!/bin/bash
#SBATCH -J abl-l40s-judgeHT
#SBATCH -p comp
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 1-00:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/benchmark_runs/holders/l40s_judge/%x-%j.out
# After the 9B judge (62443299): judge the HuatuoGPT-3 27B masked-text predictions already on disk (2,082 of 6,964) on the L40S spare memory,
# so the H200 has less to judge. Same watchdog as l40s_judge9b.sh. Stop: touch holders/l40s_judge/STOP.
exec srun --jobid=61714874 --overlap -n1 -c8 --mem=100G --gres=gpu:4 env JUDGE_SLUGS=huatuogpt-3-27b-textonly bash /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/ops/l40s_judge.sh
