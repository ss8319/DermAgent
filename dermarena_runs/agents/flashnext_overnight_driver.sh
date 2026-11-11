#!/bin/bash
# Detached overnight driver for DermAgent x Qwen3.8-Flash-Next (owner 2026-11-06). No human needed.
#  1. wait for the Flash-Next server (ENDPOINT shows Flash-Next, /v1/models lists it)      [max 4 h]
#  2. release the canary job (01_canary.json.hold -> .json), wait for ALLDONE                [max 2 h]
#  3. GATE: >=95 % parsed over both tasks, 0 agent_request_error rows. Pass -> write 02_full.json (20 threads) and wait
#     for ALLDONE [max 14 h]. Fail -> write ALERT.txt, STOP workers, release the server (STOP_SERVER) so the H200 queue moves on.
#  4. after the full job: stage the Flash-Next judge req (095) into the H200 queue, STOP workers, STOP_SERVER.
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; D=$B/dx/dermagent-qwen3.8-flash-next; J=$D/jobs; L=$B/logs/flashnext_driver.log
say() { echo "$(date '+%F %T') $*" >> $L; }
say "driver start"
t0=$(date +%s)
until grep -q 'Qwen3.8-Flash-Next' $B/holders/h200x4/ENDPOINT 2>/dev/null && \
      curl -s -m 8 "$(cut -d' ' -f1 $B/holders/h200x4/ENDPOINT)/models" 2>/dev/null | grep -q 'Qwen3.8-Flash-Next'; do
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && { say "TIMEOUT waiting for the Flash-Next server"; echo "server never came up" > $D/ALERT.txt; exit 1; }
  sleep 30
done
say "server up: $(cat $B/holders/h200x4/ENDPOINT | cut -c1-120)"
[ -f $J/01_canary.json.hold ] && mv $J/01_canary.json.hold $J/01_canary.json; say "canary released"
t0=$(date +%s); until [ -f $J/01_canary.ALLDONE ]; do
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && { say "TIMEOUT canary"; echo "canary did not finish in 2 h" > $D/ALERT.txt; touch $J/STOP $J/STOP_SERVER; exit 1; }
  sleep 60; done
GATE=$(python3 -I - <<'PY'
import json,glob
rows=[json.loads(l) for t in ("T1.1","T1.2") for l in open(f"/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/dermagent-qwen3.8-flash-next/canary_v1/{t}/predictions.jsonl") if l.strip()]
n=len(rows); p=sum(bool(r.get("parsed")) for r in rows); e=sum(r.get("finish_reason")=="agent_request_error" for r in rows)
print(f"{'PASS' if n>=40 and p/n>=0.95 and e==0 else 'FAIL'} parsed {p}/{n} request_errors {e}")
PY
)
say "gate: $GATE"
if [[ "$GATE" != PASS* ]]; then
  echo "canary gate failed: $GATE (see dx/dermagent-qwen3.8-flash-next/canary_v1 and logs)" > $D/ALERT.txt
  touch $J/STOP $J/STOP_SERVER; say "gate failed -> STOP"; exit 1
fi
printf '%s\n' '{"tasks": "T1.1,T1.2", "out_tmpl": "/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/dermagent-qwen3.8-flash-next/{task}", "limit": null, "threads": 20}' > $J/02_full.json.tmp && mv $J/02_full.json.tmp $J/02_full.json
say "full run released"
t0=$(date +%s); until [ -f $J/02_full.ALLDONE ]; do
  [ $(( $(date +%s) - t0 )) -gt 50400 ] && { say "TIMEOUT full run (14 h)"; echo "full run exceeded 14 h" > $D/ALERT.txt; break; }
  sleep 120; done
say "full run done/stopped"
cp $B/holders/h200x4/staged/095_judge_dermagent_flashnext.req $B/holders/h200x4/queue/095_judge_dermagent_flashnext.req && say "judge 095 queued"
touch $J/STOP; sleep 20; touch $J/STOP_SERVER; say "STOP + STOP_SERVER touched"
