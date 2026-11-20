#!/bin/bash
#SBATCH -J mp-qwen-64thr
#SBATCH -p comp
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 3-00:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/benchmark_runs/logs/%x-%j.out
# 11-11 owner-approved optimisation #1: Qwen MedAgent-Pro servers are half-loaded (32 req). Restart shards 3, 0, 1 at 64 threads on their
# current servers (threads are not in run_sig). mp_take_shard.sh stops each shard's runner by exact process group and restarts it with its
# tools on the longest-lived free slot; this job keeps their srun clients alive (do not cancel while they run).
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs
PREFER_HOST=62450847 bash $B/agents/mp_take_shard.sh 3 http://m3n100:30734/v1 x 64 > $B/logs/qwen64_s3.out 2>&1 &
sleep 120
bash $B/agents/mp_take_shard.sh 0 http://m3u023:30505/v1 x 64 > $B/logs/qwen64_s0.out 2>&1 &
sleep 120
bash $B/agents/mp_take_shard.sh 1 http://m3u023:30505/v1 x 64 > $B/logs/qwen64_s1.out 2>&1 &
wait
