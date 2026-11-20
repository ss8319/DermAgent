#!/bin/bash
# When the Flash-Next full job (02_full) is ALLDONE: stop the tool workers (jobs/STOP) and release the H200 server
# (jobs/STOP_SERVER) so the holder moves on to the next queued req (judge).
J=/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/dermagent-qwen3.8-flash-next/jobs
until [ -f $J/02_full.ALLDONE ]; do sleep 30; done
sleep 20; touch $J/STOP $J/STOP_SERVER; echo "STOP+STOP_SERVER touched $(date)" >> /fs04/scratch2/ub62/ssim0070/benchmark_runs/logs/wait_stop_flashnext.log
