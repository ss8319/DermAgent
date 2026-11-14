"""DermAgent answer re-parse (P1, owner 2026-11-05) + regression over ALL saved DermAgent outputs.

P3 rule (owner 11-05): a normal final turn with no FINAL_ANSWER marker whose last numbered list starting at 1 has
>= 5 items -> its first 5 names (rule C7 of the base arm, restricted to a full top-5 list). Fairness: the base arm
needs the same rule (flagged to the owner).

P1 rule (owner): when the final reply IS the cap-final turn's answer (row.cap_final_answer) and holds no
'FINAL_ANSWER' marker, accept its LAST numbered list that starts at 1 (up to 5 names). That turn asked for exactly
"FINAL_ANSWER:\n1. ...\n5. ...", so the list is the requested answer. Nothing else changes: normal turns keep the strict
FINAL_ANSWER rule (a bare trailing list in a normal turn = base-arm rule C7, rejected 11-04).

Applied as a post-parse step (not inside run_dermarena_v1.py, whose code hash is part of the running full run's
run_sig): run --apply on a MERGED predictions.jsonl after the run (merge() rebuilds it from the shards).
The strict result is kept in 'prediction_strict' / 'unparsed_reason_strict'; 'parse_rule' names the rule used.

Usage:
  python3 -I dermagent_reparse.py check  <dir> [<dir> ...]   # regression: every row, old vs new; prints each change
  python3 -I dermagent_reparse.py apply  <task_dir>           # rewrite <task_dir>/predictions.jsonl (merged file)
"""
import glob, json, re, sys
from pathlib import Path

sys.path.insert(0, "/fs04/scratch2/ub62/ssim0070/DermAgent/scripts")
from run_dermarena_v1 import parse_final_answer, _ITEM  # noqa: E402  (the runner's own strict parser)

_FA_ANY = re.compile(r"FINAL[_ ]ANSWER", re.I)


def _last_list_from_1(text, cap=None):
    """Last contiguous numbered block whose first item is 1 -> names. Blank / bullet lines inside are allowed."""
    blocks, cur = [], None
    for line in text.splitlines():
        m = _ITEM.match(line)
        if m:
            num = int(m.group(1))
            name = m.group(2).strip().strip("*").strip().rstrip(";.,").strip()
            name = re.sub(r"^\*\*|\*\*$", "", name).strip()
            if num == 1:
                cur = [name]; blocks.append(cur)
            elif cur is not None and num == len(cur) + 1:
                cur.append(name)
            else:
                cur = None
        elif line.strip() and not line.strip().startswith(("-", "*")) and cur is not None:
            cur = None  # prose ends the block
    blocks = [b for b in blocks if b and all(b)]
    return blocks[-1] if blocks else []


def parse_v2(row):
    """-> (items, reason, rule)."""
    resp = row.get("agent_response") or ""
    if row.get("unparsed_reason") in ("agent_recursion_limit", "length") or row.get("finish_reason") == "length":
        return [], row.get("unparsed_reason"), "strict"
    items, why = parse_final_answer(resp)
    if items:
        return items, "ok", "strict"
    if row.get("cap_final_answer"):  # P1 dropped by the owner 11-05 (strict for cap-final replies)
        return [], why, "strict"
    # P3 (owner 11-05 "as long as it has the top 5"): a normal final turn with NO FINAL_ANSWER marker at all whose
    # LAST numbered list (starting at 1) has >= 5 items -> its first 5 names. Shorter lists stay unparsed.
    # Guard: a reply that contains a tool call written as text was an action, not an answer (11-05: urachal case,
    # list of anomaly types with "<- Most likely" on item 4 -> would misstate the model's ranking).
    if not _FA_ANY.search(resp) and not re.search(r"<tool_call>|<function=|<parameter=|['\"]arguments['\"]\s*:", resp):
        it = _last_list_from_1(resp)
        if len(it) >= 5:
            return it[:5], "ok", "P3_trailing_top5_list"
    return [], why, "strict"


def _old_items(row):
    try:
        return json.loads(row["prediction"])["diagnoses"] if row.get("prediction") else []
    except Exception:  # noqa: BLE001
        return []


def check(dirs):
    files = []
    for d in dirs:
        files += sorted(glob.glob(str(Path(d) / "**" / "predictions.p*.jsonl"), recursive=True)) or \
                 sorted(glob.glob(str(Path(d) / "**" / "predictions.jsonl"), recursive=True))
    n = same = gained = changed = lost = 0
    for f in files:
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l); n += 1
            old = _old_items(r); new, why, rule = parse_v2(r)
            if old == new:
                same += 1; continue
            kind = "GAINED" if (not old and new) else "LOST" if (old and not new) else "CHANGED"
            gained += kind == "GAINED"; lost += kind == "LOST"; changed += kind == "CHANGED"
            resp = r.get("agent_response") or ""
            print(f"\n[{kind}] {f.split('dermagent-qwen3.5-27b/')[-1]} {r['_id']} rule={rule} cap_final={r.get('cap_final_answer')}")
            print(f"   reply tail: {resp[-420:]!r}")
            print(f"   old: {old}\n   new: {new}")
    print(f"\nREGRESSION: {n} rows in {len(files)} files | unchanged {same} | gained {gained} | changed {changed} | lost {lost}")


def apply(task_dir):
    p = Path(task_dir) / "predictions.jsonl"
    rows = [json.loads(l) for l in p.open() if l.strip()]
    k = 0
    for r in rows:
        new, why, rule = parse_v2(r)
        if "prediction_strict" not in r:
            r["prediction_strict"], r["unparsed_reason_strict"] = r.get("prediction"), r.get("unparsed_reason")
        if rule != "strict":
            k += 1
        r["prediction"] = json.dumps({"diagnoses": new}, ensure_ascii=False) if new else ""
        r["parsed"], r["unparsed_reason"], r["n_items"], r["parse_rule"] = bool(new), (None if new else why), len(new), rule
    tmp = p.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.rename(p)
    print(f"[reparse] {p}: {len(rows)} rows, P1 applied to {k}")


if __name__ == "__main__":
    {"check": lambda a: check(a), "apply": lambda a: [apply(x) for x in a]}[sys.argv[1]](sys.argv[2:])
