"""DermAgent live load + risk check (backbone server + agent run), last N minutes.

Usage: python3 -I dermagent_load_check.py <holder_jobid> <task_dir> [minutes=10]
Reads holders/h200x4/out/server_<jobid>.log (SGLang decode lines) and <task_dir>/predictions.p*.jsonl.
Prints per backbone copy: running requests, queue, KV (full token) usage, evictions (retract), gen tok/s; and for
the run: cases/min, unparsed %, dropped calls, median wall s. RISK lines flag what needs action.
"""
import glob, json, re, statistics as S, sys
from datetime import datetime, timedelta, timezone

job, tdir = sys.argv[1], sys.argv[2]
mins = int(sys.argv[3]) if len(sys.argv) > 3 else 10
log = f"/fs04/scratch2/ub62/ssim0070/benchmark_runs/holders/h200x4/out/server_{job}.log"
now = datetime.now()
cut = now - timedelta(minutes=mins)
dec = {}
retract = 0
ts_re = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: (DP\d))?\]")
for l in open(log, errors="ignore"):
    m = ts_re.match(l)
    if not m:
        continue
    t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    if t < cut:
        continue
    if "retract" in l.lower():
        retract += 1
    if "Decode batch" in l:
        dp = m.group(2) or "DP0"
        g = lambda p: float(re.search(p, l).group(1)) if re.search(p, l) else None  # noqa: E731
        dec.setdefault(dp, []).append((g(r"#running-req: (\d+)"), g(r"#queue-req: (\d+)"),
                                       g(r"full token usage: ([\d.]+)") or g(r"token usage: ([\d.]+)"),
                                       g(r"gen throughput \(token/s\): ([\d.]+)")))
risks = []
print(f"== backbone, last {mins} min ({now:%H:%M})")
for dp, v in sorted(dec.items()):
    run = [x[0] for x in v if x[0] is not None]; q = [x[1] for x in v if x[1] is not None]
    kv = [x[2] for x in v if x[2] is not None]; tps = [x[3] for x in v if x[3] is not None]
    print(f"  {dp}: running mean {S.mean(run):.1f} max {max(run):.0f} | queue max {max(q):.0f} | KV mean {S.mean(kv):.2f} "
          f"max {max(kv):.2f} | gen tok/s mean {S.mean(tps):.0f}")
    if max(kv) > 0.92:
        risks.append(f"{dp} KV peak {max(kv):.2f} > 0.92 (evictions likely)")
    if max(q) > 0:
        risks.append(f"{dp} requests queued (max {max(q):.0f}): server saturated, more threads won't help")
    if max(run) >= 64:
        risks.append(f"{dp} running {max(run):.0f} >= cuda-graph-max-bs 64 (graphs off above this)")
print(f"  evictions (retract lines): {retract}")
if retract > 20:
    risks.append(f"{retract} evictions in {mins} min")

rows = [json.loads(l) for f in glob.glob(f"{tdir}/predictions.p*.jsonl") for l in open(f) if l.strip()]
rec = [r for r in rows if datetime.fromisoformat(r["timestamp"]).astimezone().replace(tzinfo=None) >= cut]
drop = sum(1 for f in glob.glob(f"{tdir}/dropped.p*.jsonl") for l in open(f) if l.strip())
print(f"== run {tdir.split('dermagent-qwen3.5-27b/')[-1]}: total rows {len(rows)}, last {mins} min {len(rec)} "
      f"({len(rec) / mins:.1f} cases/min)")
if rec:
    up = sum(not r.get("parsed") for r in rec)
    print(f"  unparsed (window) {up}/{len(rec)} = {100 * up / len(rec):.1f} % | median wall {S.median(r['wall_s'] for r in rec):.0f} s"
          f" | dropped calls total {drop}")
    if up / len(rec) > 0.06:
        risks.append(f"unparsed {100 * up / len(rec):.1f} % in window (> 6 %)")
if drop > 5:
    risks.append(f"{drop} dropped calls")
print("RISK: " + ("; ".join(risks) if risks else "none"))
