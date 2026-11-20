#!/bin/bash
#SBATCH -J abl-l40s-judgeHT2
#SBATCH -p comp
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 1-00:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/benchmark_runs/holders/l40s_judge/%x-%j.out
# 11-11: judge HuatuoGPT masked-text on the L40S spare memory. SGLang tp4 refuses unbalanced free memory across GPUs ("memory capacity is
# unbalanced"): GPU 1 is empty (shard 3 moved to the A40) while GPUs 0/2/3 carry ~19.9 GB of tools each. A placeholder step holds 19.5 GB
# on GPU 1 for exactly as long as the judge runs, then is released.
PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python3; RDY=/fs04/scratch2/ub62/ssim0070/benchmark_runs/holders/l40s_judge/placeholder_ready.$SLURM_JOB_ID; rm -f $RDY
srun --jobid=61714874 --overlap -n1 -c1 --mem=4G --gres=gpu:4 env CUDA_VISIBLE_DEVICES=1 $PY -c "import torch,time; x=torch.empty(int(19.5*2**30),dtype=torch.uint8,device='cuda:0'); open('$RDY','w').write('ok'); print('placeholder holds 19.5 GB on L40S GPU 1',flush=True); time.sleep(10**7)" &
BP=$!
for i in $(seq 1 60); do [ -f $RDY ] && break; kill -0 $BP 2>/dev/null || break; sleep 10; done
[ -f $RDY ] || { echo "placeholder never became ready: not starting the judge"; kill $BP; exit 4; }
srun --jobid=61714874 --overlap -n1 -c1 --mem=500M nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
srun --jobid=61714874 --overlap -n1 -c8 --mem=100G --gres=gpu:4 env JUDGE_SLUGS=huatuogpt-3-27b-textonly bash /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/ops/l40s_judge.sh; rc=$?
kill $BP; wait $BP 2>/dev/null; rm -f $RDY; echo "placeholder released, judge rc=$rc $(date)"; exit $rc
