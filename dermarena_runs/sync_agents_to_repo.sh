#!/bin/bash
# Copy the LIVE agent-run operating scripts (benchmark_runs/agents/, not a git repo) into the DermAgent repo
# (dermarena_runs/agents/), next to the agent code they drive (baselines/MedAgent-Pro, scripts/run_dermarena_v1.py), so git
# matches what actually ran. Same pattern as tools/sync_to_repo.sh (DermArena). Run after changing any of them, then commit.
#   bash tools/sync_agents_to_repo.sh [--check]     --check: only report differences (exit 1 if any)
# Copies code, holder requests and the repro spec only: never outputs, logs, traces, the MedlinePlus KB data, backups or secrets.
set -uo pipefail
L=/fs04/scratch2/ub62/ssim0070/benchmark_runs/agents
R=/fs04/scratch2/ub62/ssim0070/DermAgent/dermarena_runs/agents
diffs=0
sync() {  # $1 live file  $2 repo file
  cmp -s "$1" "$2" 2>/dev/null && return
  diffs=$((diffs+1)); echo "differs: ${2#$R/}"
  [ "$MODE" = --check ] || { mkdir -p "$(dirname "$2")"; cp -p "$1" "$2"; }
}
MODE=${1:-}
shopt -s nullglob
for f in $L/*.sh $L/*.py $L/*.sbatch $L/*.md; do sync "$f" "$R/$(basename $f)"; done
for d in medagentpro_reqs ablation_reqs_h200_v2_20261110; do for f in $L/$d/*.req; do sync "$f" "$R/$d/$(basename $f)"; done; done
for f in $L/ops/*.sh $L/ops/*.py $L/ops/*.sbatch; do sync "$f" "$R/ops/$(basename $f)"; done
echo "$diffs file(s) $( [ "$MODE" = --check ] && echo differ || echo synced )"
[ "$MODE" = --check ] && [ $diffs -gt 0 ] && exit 1 || exit 0
