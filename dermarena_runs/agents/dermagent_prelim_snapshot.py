"""Rebuild dx/dermagent-qwen3.5-27b-prelim/<task>/predictions.jsonl = contiguous prefix of completed cases per shard
(no straggler bias: the runner submits cases in file order, long cases finish last). Usage: python3 -I this.py [T1.1|T1.2]"""
import json, glob, sys
from pathlib import Path
task = sys.argv[1] if len(sys.argv) > 1 else "T1.1"
TD = {"T1.1": "T1.1_disease_screening", "T1.2": "T1.2_disease_confirmation"}[task]
B = Path("/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx")
rows = [json.loads(l) for l in open(f"/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks/{TD}/test.jsonl") if l.strip()]
done = {}
for f in glob.glob(str(B / f"dermagent-qwen3.5-27b/{task}/predictions.p*.jsonl")):
    for l in open(f):
        r = json.loads(l); done[r["_id"]] = r
shards = [[r["_id"] for i, r in enumerate(rows) if i % 4 == k] for k in range(4)]
cuts = [next((j for j, c in enumerate(s) if c not in done), len(s)) for s in shards]
m = cuts  # each shard keeps ITS OWN contiguous complete prefix (cases are assigned to shards by index % 4, so the union is unbiased)
sel = [c for s, mk in zip(shards, cuts) for c in s[:mk]]
out = B / f"dermagent-qwen3.5-27b-prelim/{task}"; out.mkdir(parents=True, exist_ok=True)
tmp = out / "predictions.jsonl.tmp"
tmp.write_text("".join(json.dumps(done[c], ensure_ascii=False) + "\n" for c in sorted(sel)))
tmp.rename(out / "predictions.jsonl")
print(f"{task}: prefixes {m} per shard -> {len(sel)} cases (completed rows total {len(done)}); unparsed {sum(not done[c]['parsed'] for c in sel)}")
