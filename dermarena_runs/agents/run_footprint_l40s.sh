#!/bin/bash
# Detached footprint test on the L40S node (job 61714874), GPU 0.
srun --jobid=61714874 --overlap -n1 -c8 --mem=120G --gres=gpu:1 bash -c 'export CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1; /fs04/scratch2/ub62/ssim0070/dermagent/bin/python3 -u /fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/tool_footprint_test.py 12 60'
echo "FOOTPRINT EXIT $? $(date)"
