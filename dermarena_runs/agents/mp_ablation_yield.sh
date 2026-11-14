#!/bin/bash
# mp_ablation_yield.sh <agent_lockdir> <agent_req> <work_lockdir> -- <command...>
# Run ablation work on a SURPLUS agent holder and hand the GPU back the moment the agent run needs it (owner 11-09: agents first; ablation
# only on resources the agents cannot use, without slowing them). "Needs it" = <agent_lockdir> exists and its owner job is not RUNNING
# (the holder that ran that agent work has gone). Then: stop the work (resumable), queue <agent_req> as 005_* in this holder's queue so it
# runs next, re-queue the current request (as 9<name>) to resume the ablation later if this holder is surplus again; exit 0.
# <work_lockdir>: one holder per ablation item (another RUNNING owner -> skip, exit 0). Needs $Q (holder queue) and $REQ_SELF from the req.
AL=$1; AREQ=$2; WL=$3; shift 4
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs
# <agent_lockdir> may be a ':'-separated list (11-09 17:45: Qwen shards 0 and 1 have separate locks): needed if ANY of them is stale
agent_needs() { local l o; for l in ${AL//:/ }; do [ -d "$l" ] || continue; o=$(cat $l/owner 2>/dev/null); [ -z "$o" ] && return 0
  [ "$(squeue -h -j "$o" -o %T 2>/dev/null)" != RUNNING ] && return 0; done; return 1; }
requeue() { cp "$AREQ" "$Q/005_$(basename $AREQ)"; cp "$REQ_SELF" "$Q/9$(basename $REQ_SELF)"; }   # 005_ runs next; 9<name> resumes the ablation after
if agent_needs; then echo "[yield] agent work needs this holder before starting: re-queueing it $(date)"; requeue; exit 0; fi
NOWAIT=1 bash $B/agents/mp_shard_lock.sh "$WL" "$SLURM_JOB_ID" || { echo "[yield] $WL taken by another running job: skip"; exit 0; }
setsid "$@" & P=$!
echo "[yield] ablation work started (pid $P) $(date): $*"
while kill -0 $P 2>/dev/null; do
  sleep ${YIELD_POLL:-120}
  if agent_needs; then
    echo "[yield] agent work needs this holder: stopping the ablation work (resumable) $(date)"
    # guarded.sh starts its command with setsid (own process group): stop the children's groups first, then ours
    for c in $(pgrep -P $P); do for g in $(pgrep -P $c) $c; do kill -TERM -- -$g 2>/dev/null; done; done; kill -TERM -- -$P 2>/dev/null
    sleep ${YIELD_GRACE:-30}; for c in $(pgrep -P $P); do kill -KILL -- -$c 2>/dev/null; done; kill -KILL -- -$P 2>/dev/null
    rm -rf "$WL"; requeue; exit 0
  fi
done
wait $P; rc=$?; rm -rf "$WL"; echo "[yield] ablation work ended rc=$rc $(date)"; exit $rc
