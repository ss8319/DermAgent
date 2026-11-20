#!/bin/bash
# mp_canary_integrity.sh <canary dir>: exit 0 and touch <dir>/PASSED_INTEGRITY when both tasks' POSTCHECK.txt exist and every FAIL gate is a
# behaviour gate (no target in the paper: memory removal, no evidence, Uncertain, step shares, has_guideline). Otherwise touch FAILED_INTEGRITY.
# On PASS (owner 11-08: "keep those answers for the full run"): the canary's 40 answers (identical test rows, same code and settings) are
# copied into the full-run folders as predictions.pcanary.jsonl (+ traces, cases), so the full run skips them. Check the full run's
# RUN_META run_sig equals CANARY_RUN_SIG in SEEDED_FROM_CANARY; if not, delete predictions.pcanary.jsonl and those cases are re-run.
D=$1; rm -f $D/PASSED_INTEGRITY $D/FAILED_INTEGRITY
for t in T1.1 T1.2; do [ -s $D/$t/POSTCHECK.txt ] || { echo "no $D/$t/POSTCHECK.txt"; touch $D/FAILED_INTEGRITY; exit 1; }; done
bad=$(grep -h "\[mp-gate\] FAIL" $D/T1.1/POSTCHECK.txt $D/T1.2/POSTCHECK.txt | grep -v -E "steps removed by memory|candidates with no evidence|Uncertain dropped|step share|has_guideline")
grep -h "\[mp-gate\] FAIL\|\[mp-gate\] REPORT" $D/T1.1/POSTCHECK.txt $D/T1.2/POSTCHECK.txt
if [ -n "$bad" ]; then echo "INTEGRITY FAIL:"; echo "$bad"; touch $D/FAILED_INTEGRITY; exit 1; fi
for t in T1.1 T1.2; do
  F=$(dirname $D)/$t; mkdir -p $F/traces $F/cases
  if [ ! -s $F/predictions.pcanary.jsonl ]; then
    cat $D/$t/predictions.p*.jsonl > $F/predictions.pcanary.jsonl.tmp && mv $F/predictions.pcanary.jsonl.tmp $F/predictions.pcanary.jsonl
    cp -rn $D/$t/traces/. $F/traces/ 2>/dev/null; cp -rn $D/$t/cases/. $F/cases/ 2>/dev/null
    sig=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['run_sig'])" $D/$t/RUN_META.json)
    echo "seeded $(date -Is) from $D/$t | CANARY_RUN_SIG $sig | rows $(wc -l < $F/predictions.pcanary.jsonl)" > $F/SEEDED_FROM_CANARY
    cat $F/SEEDED_FROM_CANARY
  fi
done
echo "INTEGRITY PASS $D"; touch $D/PASSED_INTEGRITY
