#!/bin/bash
# Detached gate: when the Flash-Next canary job is ALLDONE, release the full job only if parsed >= 95 % and no request errors.
D=/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/dermagent-qwen3.8-flash-next; J=$D/jobs
until [ -f $J/01_canary.ALLDONE ]; do sleep 30; done
python3 -I - <<PY > $J/gate.txt 2>&1
import json,glob
rows=[json.loads(l) for t in ("T1.1","T1.2") for f in glob.glob("$D/canary_v1/%s/predictions.p*.jsonl"%t) for l in open(f)]
n=len(rows); p=sum(r["parsed"] for r in rows); err=sum(r.get("finish_reason")=="agent_request_error" for r in rows)
dropped=sum(1 for t in ("T1.1","T1.2") for f in glob.glob("$D/canary_v1/%s/dropped.p*.jsonl"%t) for l in open(f) if l.strip())
ok = n>=40 and p/n>=0.95 and err==0 and dropped<=1
print(f"rows {n} parsed {p} ({100*p/max(n,1):.1f}%) request_errors {err} dropped {dropped} -> {'PASS' if ok else 'FAIL'}")
open("$J/GATE_"+("PASS" if ok else "FAIL"),"w").write("x")
PY
if [ -f $J/GATE_PASS ]; then
  printf '%s\n' '{"tasks": "T1.1,T1.2", "out_tmpl": "/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/dermagent-qwen3.8-flash-next/{task}", "limit": null, "threads": 20}' > $J/02_full.json.tmp && mv $J/02_full.json.tmp $J/02_full.json
fi
