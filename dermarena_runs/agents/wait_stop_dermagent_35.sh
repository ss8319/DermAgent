#!/bin/bash
# Detached waiter: when the DermAgent x Qwen3.5-27B full job (07_full) is ALLDONE, touch jobs/STOP so the 066 service exits
# and the H200 holder moves to the next queued req (judge, then the Flash-Next server).
J=/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/dermagent-qwen3.5-27b/jobs
until [ -f $J/07_full.ALLDONE ]; do sleep 30; done
sleep 20; touch $J/STOP; echo "STOP touched $(date)" >> /fs04/scratch2/ub62/ssim0070/benchmark_runs/logs/wait_stop_dermagent_35.log
