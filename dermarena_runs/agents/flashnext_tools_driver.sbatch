#!/bin/bash
#SBATCH -J fn-tools-driver
#SBATCH -p comp
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/benchmark_runs/logs/fn-tools-driver-%j.out
# Holds the Flash-Next tool workers on the L40S (job 61714874) from a CPU job, so they do not depend on a chat session: a step dies when its srun client
# exits. Replaces any workers started from a session node; resume by _id loses only cases in flight. Exits when the service ends.
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; L=61714874; S=dermagent-qwen3.8-flash-next-clean
srun --jobid=$L --overlap --cpu-bind=none -n1 -c1 --mem=1G bash -c "pkill -f '[r]un_dermagent_v1.sh service'; pkill -f '[r]un_dermarena_v1.py --service'; sleep 10"
rm -f $B/dx/$S/jobs/STOP
ENVS="OPENROUTER_BASE_URL=http://m3u013:30862/v1 DA_BACKBONE=Qwen/Qwen3.8-Flash-Next DA_SLUG=$S DA_TOOL_GPUS='0 1 2 3' DA_NPROC=4 DA_THREADS=20 DA_STRIP_NOTOOL=1 DA_ALLOC_CONF=expandable_segments:True"
exec srun --jobid=$L --overlap --cpu-bind=none -n1 -c16 --mem=200G bash -c "export $ENVS; exec bash $B/agents/run_dermagent_v1.sh service" >> $B/dx/$S/logs/remote_service.out 2>&1
