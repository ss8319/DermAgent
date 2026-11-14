#!/bin/bash
# gpu_watchdog.sh (owner-approved 2026-11-12 21:3x) — detect idle GPUs in our running jobs, fix what the PLAYBOOK allows,
# alert the rest, and release a GPU job idle > 2 h with nothing queued (owner standing exception to "never scancel").
# M3's rule (email 2026-11-12, job 62151003): job ran > 5 h AND at least ONE allocated GPU averages < 5 % utilisation
# (job lifetime) -> flagged / cancelled. So we watch every GPU separately, not only the job as a whole.
# Runs detached on the login node (light: one srun + nvidia-smi per job every 5 min; survives session disconnects). Every $EVERY s, per running GPU job:
#   sample per-GPU utilisation -> logs/gpu_watchdog/samples/<job>.tsv ; status table -> status.txt ; events -> alerts.log
# Rules (U = utilisation %):
#   R1 JOB IDLE      all GPUs U < 5 for >= 20 min (tools hosts: 60 min)          -> ALERT + auto-fix (below)
#   R2 GPU AT RISK   one GPU's average since watching began < 5 over >= 3 h       -> ALERT (M3 would flag this GPU)
#   R3 RELEASE       all GPUs U < 5 for >= 120 min AND holder queue empty AND standby list empty AND no MedAgent-Pro
#                    runner/tools and no helper on the job AND not a Flash-Next server with work left -> scancel + ALERT
#   H  HELPERS       Flash-Next supervisor dead while it has work and no STOP file -> relaunch (detached, on the CPU job)
# Auto-fix for R1 on a holder with an empty queue: move the first file from standby_reqs/<holder-name or any>/ into its queue.
# Stop: touch logs/gpu_watchdog/STOP. DRY=1: no scancel / no relaunch / no queue moves (log what it would do).
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; W=$B/logs/gpu_watchdog; S=$W/samples; mkdir -p $S $B/standby_reqs/any
EVERY=${EVERY:-300}; FN=$B/dx/medagentpro-qwen3.8-flash-next
exec 9>$W/watchdog.flock; flock -n 9 || { echo "another watchdog holds the lock"; exit 0; }
alert() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $W/alerts.log; }
ovl() { local j=$1; shift; timeout 90 env $(env | grep -oE '^SLURM_[A-Z_]+' | grep -v '^SLURM_CONF' | sed 's/^/-u /') srun --jobid=$j --overlap -n1 -c1 --mem=512M "$@" 2>/dev/null; }
tools_hosts() { grep -oE '^[0-9]+' $B/agents/mp_tool_hosts.txt 2>/dev/null; }
fn_left() { python3 - <<'PY'
import json,glob
D="/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/medagentpro-qwen3.8-flash-next"; R="/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks"; n=0
for t,d in (("T1.1","T1.1_disease_screening"),("T1.2","T1.2_disease_confirmation")):
    done=set()
    for f in glob.glob(f"{D}/{t}/predictions.p*.jsonl"):
        for l in open(f):
            try: done.add(json.loads(l)["_id"])
            except Exception: pass
    n+=sum(1 for l in open(f"{R}/{d}/test.jsonl") if l.strip() and json.loads(l)["_id"] not in done)
print(n)
PY
}
alert "watchdog start (pid $$, every ${EVERY}s, DRY=${DRY:-0})"
while [ ! -f $W/STOP ]; do
  now=$(date +%s); : > $W/status.new
  printf "%-10s %-20s %-8s %-28s %-10s %s\n" job name elapsed "util% per GPU (now)" idle_min note >> $W/status.new
  for line in $(squeue -h -u $USER -t R -o "%i|%j|%b|%M" | grep -i "gpu"); do
    IFS='|' read -r j name gres el <<< "$line"
    out=$(ovl $j bash -c 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d " "; echo "#"; pgrep -fa "[r]un_dermarena_v1_medagentpro.py|[m]p_fn_supervisor.sh|[j]udge_loop|[j]udge_worker|[f]inal_retry" | wc -l')
    [ -z "$out" ] && { printf "%-10s %-20s %-8s %s\n" $j $name $el "cannot inspect" >> $W/status.new; continue; }
    utils=$(echo "$out" | sed '/^#/,$d' | awk -F, -v t=$now -v j=$j '{print t"\t"j"\t"$1"\t"$2"\t"$3}' | tee -a $S/$j.tsv | cut -f4 | tr '\n' ' ')
    nproc=$(echo "$out" | sed -n '/^#/{n;p}')
    maxu=$(echo $utils | tr ' ' '\n' | sort -n | tail -1)
    # minutes since any GPU of this job was >= 5 %
    last_busy=$(awk -F'\t' '$4>=5{t=$1} END{print t+0}' $S/$j.tsv); first=$(head -1 $S/$j.tsv | cut -f1)
    [ "$last_busy" -eq 0 ] && last_busy=$first; idle=$(( (now - last_busy) / 60 ))
    hold=${name#bench-}; q=$B/holders/$hold/queue; nq=$(ls $q/*.req 2>/dev/null | wc -l)
    th=20; tools_hosts | grep -qx $j && th=60
    note=""
    # R2: per-GPU average since watching began, once >= 3 h of samples
    span=$(( (now - first) / 60 ))
    if [ $span -ge 180 ]; then
      risk=$(awk -F'\t' '{s[$3]+=$4; n[$3]++} END{for(g in s) if(s[g]/n[g]<5) printf "GPU%s avg %.1f%% ", g, s[g]/n[g]}' $S/$j.tsv)
      [ -n "$risk" ] && { note="$note M3-RISK:$risk"; grep -q "R2 $j " $W/alerts.log 2>/dev/null && [ $(( now - $(stat -c %Y $W/alerts.log) )) -lt 3600 ] || alert "R2 $j $name: $risk over ${span} min watched (M3 flags < 5 % per GPU over > 5 h)"; }
    fi
    if [ $idle -ge $th ]; then
      note="$note IDLE ${idle}min"
      [ $(( idle % 30 )) -lt 5 ] && alert "R1 $j $name idle ${idle} min (all GPUs < 5 %), queue reqs: $nq, agent/helper procs: ${nproc:-?}"
      # auto-fix: empty holder queue -> pull one standby request
      if [ -d $q ] && [ $nq -eq 0 ]; then
        sb=$(ls $B/standby_reqs/$hold/*.req $B/standby_reqs/any/*.req 2>/dev/null | head -1)
        if [ -n "$sb" ]; then alert "FIX $j $name: queue empty -> moving standby $(basename $sb)"; [ -z "$DRY" ] && mv "$sb" $q/; fi
      fi
      # R3: release
      if [ $idle -ge 120 ] && [ ${nproc:-1} -eq 0 ] && [ $nq -eq 0 ] && [ -z "$(ls $B/standby_reqs/$hold/*.req $B/standby_reqs/any/*.req 2>/dev/null)" ]; then
        # 11-14: also hold every job named in servers/*.jobs (a multi-job server, e.g. the 2-node pp2 server registers only its
        # first job's .url; its second job 61981434 was released at 11:27 as "idle" while it was half of the server).
        fnsrv=0; { [ -f $FN/servers/$j.url ] || tools_hosts | grep -qx $j || cat $FN/servers/*.jobs 2>/dev/null | tr ' ' '\n' | grep -qx $j; } && [ "$(fn_left)" -gt 0 ] && fnsrv=1
        if [ $fnsrv -eq 1 ]; then alert "R3-HOLD $j $name idle ${idle} min but it is a Flash-Next server or tools host and Flash-Next has work left: NOT releasing (check supervisor / server)";
        else alert "R3 RELEASE $j $name: idle ${idle} min, empty queue, no standby work, no agent/helper processes (owner exception 2026-11-12)"; [ -z "$DRY" ] && scancel $j; fi
      fi
    fi
    printf "%-10s %-20s %-8s %-28s %-10s %s\n" $j $name $el "$utils" $idle "$note" >> $W/status.new
  done
  # H: Flash-Next supervisor (11-14: watches mp_fn_supervisor3.sh, the live one; the old v1 rule was disabled by FN_SUPERVISOR_STOP
  # and would have relaunched the retired v1). Stop file: FN_SUP3_STOP.
  if [ ! -f $FN/FN_SUP3_STOP ]; then
    if ! pgrep -f '[m]p_fn_supervisor3\.sh' >/dev/null 2>&1 && [ "$(fn_left)" -gt 0 ]; then
      alert "H fn supervisor3 not running (work left): relaunching detached on $(hostname)"
      [ -z "$DRY" ] && setsid nohup bash $B/agents/mp_fn_supervisor3.sh >> $FN/logs/fnsup3_supervisor.out 2>&1 < /dev/null 9>&- &
    fi
  fi
  echo "updated $(date '+%m-%d %H:%M')" >> $W/status.new; mv $W/status.new $W/status.txt
  [ -n "$ONCE" ] && break
  sleep $EVERY 9>&-
done
alert "watchdog stop"
