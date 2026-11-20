"""Footnote check: DermAgent with STRICT parsing only (P3-rescued rows scored wrong) vs base, paired McNemar.
Usage: python3 -I this.py <agent_slug> <base_slug>   (prints headline-rule and strict-only side by side)"""
import json, sys, math, glob
B = "/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/"
slug, base = sys.argv[1], sys.argv[2]
def scores(s, t):
    return {r["_id"]: r for r in map(json.loads, open(f"{B}{s}/{t}/scores.jsonl")) if r["status"] in ("judged", "unparsed")}
def p3(s, t):
    return {json.loads(l)["_id"] for f in glob.glob(f"{B}{s}/{t}/predictions.p*.jsonl") for l in open(f)
            if l.strip() and json.loads(l).get("parse_rule") == "P3_trailing_top5_list"}
def mc(a, b):
    ids = sorted(set(a) & set(b)); n = len(ids)
    x = sum(a[i] and not b[i] for i in ids); y = sum(b[i] and not a[i] for i in ids); m = x + y
    p = min(1.0, 2 * sum(math.comb(m, k) for k in range(min(x, y) + 1)) / 2 ** m) if m else 1.0
    d = (x - y) / n; se = math.sqrt(max(m - (x - y) ** 2 / n, 0)) / n
    return n, sum(a[i] for i in ids) / n, sum(b[i] for i in ids) / n, x, y, d, d - 1.96 * se, d + 1.96 * se, p
for t in ("T1.1", "T1.2"):
    A, Bs, P = scores(slug, t), scores(base, t), p3(slug, t)
    for k, name in (("top1", "top-1"), ("top5", "top-5")):
        base_t = {i: (r.get(k) or 0) == 2 for i, r in Bs.items()}
        head = {i: (r.get(k) or 0) == 2 for i, r in A.items()}
        strict = {i: (v and i not in P) for i, v in head.items()}
        for lab, a in (("headline (P3 on)", head), ("strict only", strict)):
            n, ra, rb, x, y, d, lo, hi, p = mc(a, base_t)
            print(f"{t} {name} {lab:17s} n={n} agent {100*ra:.1f}% base {100*rb:.1f}% diff {100*d:+.2f} [{100*lo:+.2f},{100*hi:+.2f}] agent-only {x} base-only {y} p={p:.4f}  (P3 rows {len(P)})")
