"""Preliminary DermAgent vs base paired comparison on the cases both have a verdict for (judge score 2 = exact).
Usage: python3 -I this.py <task>. Uses scores.jsonl of dx/dermagent-qwen3.5-27b-prelim/<task> and dx/qwen3.5-27b/<task>.
Unparsed = wrong (top1 0) in both arms; 'not_judged'/'no_prediction' rows are dropped (not yet run)."""
import json, sys, math
from pathlib import Path
task = sys.argv[1] if len(sys.argv) > 1 else "T1.1"
B = Path("/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx")
def load(slug):
    d = {}
    for l in open(B / slug / task / "scores.jsonl"):
        r = json.loads(l)
        if r["status"] in ("judged", "unparsed"):
            d[r["_id"]] = (r.get("top1") or 0, r.get("top5") or 0, r.get("corpus_status"), r["status"])
    return d
SLUG = sys.argv[2] if len(sys.argv) > 2 else "dermagent-qwen3.5-27b-prelim"
BASE = sys.argv[3] if len(sys.argv) > 3 else "qwen3.5-27b"
A, Bs = load(SLUG), load(BASE)
ids = sorted(set(A) & set(Bs)); n = len(ids)
def rate(d, i, k): return sum(d[x][i] == 2 for x in ids) / n
def mcn(i):
    b = sum(A[x][i] == 2 and Bs[x][i] != 2 for x in ids); c = sum(A[x][i] != 2 and Bs[x][i] == 2 for x in ids)
    m = b + c; kmin = min(b, c)
    p = min(1.0, 2 * sum(math.comb(m, k) for k in range(kmin + 1)) / 2 ** m) if m else 1.0
    d = (b - c) / n; se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0)) / n
    return b, c, d, d - 1.96 * se, d + 1.96 * se, p
print(f"{task}: paired cases n={n} (agent judged {len(A)}, base judged {len(Bs)}); unparsed agent {sum(A[x][3]=='unparsed' for x in ids)}")
for name, i in (("top-1 exact", 0), ("top-5 exact", 1)):
    b, c, d, lo, hi, p = mcn(i)
    print(f"  {name}: agent {100*rate(A,i,0):.1f}%  base {100*rate(Bs,i,0):.1f}%  diff {100*d:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}]  "
          f"agent-only right {b}, base-only right {c}, McNemar exact p={p:.3f}")
for cs in ("train_absent", "train_represented"):
    s = [x for x in ids if A[x][2] == cs]
    if s:
        a1 = sum(A[x][0] == 2 for x in s) / len(s); b1 = sum(Bs[x][0] == 2 for x in s) / len(s)
        print(f"  {cs} (n={len(s)}): top-1 agent {100*a1:.1f}% base {100*b1:.1f}%")
