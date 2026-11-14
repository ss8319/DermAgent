#!/bin/bash
# a40_successor.sh (11-14): the desktop A40 tools host (62450847, desktop QOS = 1 job per user) ends 2026-11-18 02:26 while
# MedAgent-Pro x Flash-Next T1.2 still needs a tools host for days. desktop rejects a 2nd job while one runs, so this waits for
# 62450847 to leave the queue, submits the SAME job again (same submit line), and adds the new id to agents/mp_tool_hosts.txt so
# mp_fn_supervisor3.sh places runners' tools there. Detached; stop: touch agents/A40_SUCCESSOR_STOP. Postcondition: alerts if the
# submit fails or the new job is not RUNNING within 30 min.
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; L=$B/logs/a40_successor.log; A=$B/logs/gpu_watchdog/alerts.log; OLD=62450847
log() { echo "$(date '+%m-%d %H:%M') [a40succ] $*" | tee -a $L; }
log "armed (pid $$): waiting for $OLD to end"
while squeue -h -j $OLD 2>/dev/null | grep -q . ; do [ -f $B/agents/A40_SUCCESSOR_STOP ] && { log STOP; exit 0; }; sleep 120 9>&-; done
[ -f $B/agents/A40_SUCCESSOR_STOP ] && { log STOP; exit 0; }
cd /fs04/scratch2/ub62/ssim0070/sglang/slurm || exit 1
out=$(env $(env | grep -oE '^SLURM_[A-Z_]+' | sed 's/^/-u /') sbatch -J bench-a40tools -p desktop --qos=desktopq --gres=gpu:A40:1 -c 16 --mem=150G -t 7-00:00:00 bench_holder.sbatch a40tools 2>&1)
J=$(echo "$out" | grep -oE '[0-9]{6,}' | tail -1)
[ -z "$J" ] && { log "SUBMIT FAILED: $out"; echo "$(date '+%m-%d %H:%M') A40 SUCCESSOR SUBMIT FAILED: $out" >> $A; exit 1; }
echo "$J  # bench-a40tools successor of $OLD (a40_successor.sh $(date '+%m-%d %H:%M'))" >> $B/agents/mp_tool_hosts.txt
log "submitted $J and added it to mp_tool_hosts.txt"; echo "$(date '+%m-%d %H:%M') A40 tools host successor submitted: $J" >> $A
for i in $(seq 1 15); do squeue -h -j $J -t R | grep -q . && { log "$J RUNNING"; exit 0; }; sleep 120; done
log "$J not running after 30 min: $(squeue -h -j $J -o '%T %R')"; echo "$(date '+%m-%d %H:%M') A40 successor $J NOT running after 30 min" >> $A
