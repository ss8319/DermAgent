#!/usr/bin/env python3
"""MedAgent-Pro post-check: what dx_check does not know about. Reads predictions.jsonl only, stdlib only.

  mp_postcheck.py <task_dir> <task> [--gate] [--n-input N]
Reports tool errors (CUDA separately), failed plans, candidates with no plan, memory removals, reflection terminations, capped steps,
Uncertain dropped, unclear memory / reflection / verdict replies and their defaults, the final call's parse rate, has_guideline share,
planned indicator counts, a by-has_guideline table, per-tool step share, LLM calls per case.
With --gate, each canary gate prints PASS or FAIL and any FAIL exits 3.
Limits are PROVISIONAL, the builder's judgement, not owner decisions; the owner sets them after real numbers. They only stop a broken
setup reaching the full run. Uncertain now means evidence LOST, not a score of 0.5: the gate limits how much is dropped.
"""
import argparse
import json
import sys
from pathlib import Path

GATES = [  # name, value key, comparison, limit
    ("rows written", "rows_pct", ">=", 100.0), ("parsed answers %", "parsed_pct", ">=", 95.0),
    ("final call parsed % of cases with candidates", "final_parsed_pct", ">=", 95.0),
    ("CUDA errors", "cuda_errors", "<=", 0), ("tool errors % of tool calls", "tool_error_pct", "<=", 1.0),
    ("failed plans % of candidates", "failed_plan_pct", "<=", 5.0), ("candidates with no plan %", "no_plan_pct", "<=", 5.0),
    ("candidates with no evidence %", "no_evidence_pct", "<=", 25.0),
    ("steps removed by memory % of planned", "memory_removed_pct", "<=", 50.0),
    ("unreliable % of reflections", "reflection_unreliable_pct", "<=", 40.0),
    ("Uncertain dropped % of summarised indicators", "uncertain_pct", "<=", 40.0),
    ("unclear verdicts % of summarised indicators", "unclear_pct", "<=", 2.0),
    ("memory replies on the default %", "memory_default_pct", "<=", 2.0),
    ("reflection replies on the default %", "reflection_default_pct", "<=", 2.0),
    ("has_guideline % of candidates", "has_guideline_pct", ">", 0),
    ("PanDerm step share %", "share_panderm", ">", 0), ("MAKE step share %", "share_make", ">", 0),
    ("DermoGPT step share %", "share_dermogpt", ">", 0), ("backbone VLM step share %", "share_backbone", ">", 0),
]

# owner 11-08 (#4): the paper sets no expected rate for these, so they are reported, not used to stop a run
REPORT_ONLY = {"reflection_unreliable_pct", "share_panderm"}


def by_guideline(rows):
    """Per-candidate behaviour split by has_guideline (the spec's reporting line); accuracy by has_guideline needs the judge's verdicts."""
    t = {}
    for r in rows:
        for c in r.get("candidate_results") or []:
            d = t.setdefault(bool(c.get("has_guideline")), {"candidates": 0, "no_evidence": 0, "indicators": 0, "uncertain": 0, "memory_removed": 0,
                                                            "terminated": 0, "capped": 0})
            d["candidates"] += 1
            d["no_evidence"] += bool(c.get("no_evidence"))
            d["indicators"] += len(c.get("indicators") or [])
            d["uncertain"] += len(c.get("dropped_uncertain") or [])
            d["memory_removed"] += len(c.get("memory_removed") or [])
            d["terminated"] += len(c.get("terminated") or [])
            d["capped"] += len(c.get("capped_steps") or [])
    return t


def indicator_counts(rows):
    """Planned final indicators per candidate with a plan: the 3-7 target is reported, never enforced."""
    c = {}
    for r in rows:
        for x in r.get("candidate_results") or []:
            if x.get("plan_status") == "ok":
                k = x.get("n_planned_indicators", 0)
                c[k] = c.get(k, 0) + 1
    return dict(sorted(c.items()))


def summarise(rows, n_input):
    n = max(1, len(rows))
    S = lambda k: sum(r.get(k) or 0 for r in rows)  # noqa: E731
    pct = lambda a, b: round(100 * a / max(1, b), 1)  # noqa: E731
    share = {}
    for r in rows:
        for k, v in (r.get("tool_steps") or {}).items():
            share[k] = share.get(k, 0) + v
    tot = sum(share.values())
    cands = S("n_candidates")
    calls = S("n_tool_calls") + S("n_tool_errors")
    out = {"rows": len(rows), "rows_pct": pct(len(rows), n_input or len(rows)), "parsed_pct": pct(sum(bool(r.get("parsed")) for r in rows), n),
           "tool_calls": S("n_tool_calls"), "tool_errors": S("n_tool_errors"), "cuda_errors": S("n_cuda_errors"),
           "tool_error_pct": pct(S("n_tool_errors"), calls), "candidates": cands, "failed_plans": S("n_plans_failed"),
           "failed_plan_pct": pct(S("n_plans_failed"), cands), "no_plan": S("n_no_plan"), "no_plan_pct": pct(S("n_no_plan"), cands),
           "plans_not_cached": S("n_plans_uncached"), "no_evidence": S("n_no_evidence"), "no_evidence_pct": pct(S("n_no_evidence"), cands),
           "unscored": S("n_unscored"), "unscored_pct": pct(S("n_unscored"), cands),
           "no_guideline_pct": pct(S("n_no_guideline"), cands), "has_guideline_pct": pct(cands - S("n_no_guideline"), cands),
           "steps_planned": S("n_steps_planned"), "steps_executed": S("n_steps_executed"), "memory_removed": S("n_steps_memory_removed"),
           "memory_removed_pct": pct(S("n_steps_memory_removed"), S("n_steps_planned")), "memory_calls": S("n_memory_calls"),
           "memory_reasked": S("n_memory_reasked"), "memory_default": S("n_memory_default"),
           "memory_default_pct": pct(S("n_memory_default"), S("n_memory_calls")), "reflections": S("n_reflections"),
           "reflection_unreliable": S("n_reflection_unreliable"), "reflection_unreliable_pct": pct(S("n_reflection_unreliable"), S("n_reflections")),
           "reflection_reasked": S("n_reflection_reasked"), "reflection_default": S("n_reflection_default"),
           "reflection_default_pct": pct(S("n_reflection_default"), S("n_reflections")), "steps_terminated": S("n_steps_terminated"),
           "steps_capped": S("n_steps_capped"), "indicators": S("n_indicators"), "uncertain_dropped": S("n_uncertain"),
           "uncertain_pct": pct(S("n_uncertain"), S("n_indicators") + S("n_uncertain")),
           "unclear_verdicts": S("n_unclear_verdicts"), "unclear_pct": pct(S("n_unclear_verdicts"), S("n_indicators") + S("n_uncertain")),
           "verdicts_reasked": S("n_verdicts_reasked"), "dropped_indicators": S("n_dropped_indicators"),
           "final_called": sum(bool(r.get("final_called")) for r in rows),
           "final_parsed_pct": pct(sum(bool(r.get("final_parsed")) for r in rows), sum(bool(r.get("final_called")) for r in rows)),
           "final_outside_candidates": S("n_final_outside_candidates"),
           "llm_calls_per_case": round(S("n_llm_calls") / n, 1), "cases_over_call_budget": sum(bool(r.get("call_budget_exceeded")) for r in rows),
           "mean_wall_s": round(S("wall_s") / n, 1)}
    for k in ("panderm", "make", "dermogpt", "backbone"):
        out[f"share_{k}"] = pct(share.get(k, 0), tot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("task")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--n-input", type=int, default=None)
    a = ap.parse_args()
    p = Path(a.task_dir) / "predictions.jsonl"
    rows = [json.loads(l) for l in p.open()] if p.exists() else []
    s = summarise(rows, a.n_input)
    print(f"[mp-post] {a.task}: " + " | ".join(f"{k} {v}" for k, v in s.items()))
    print(f"[mp-post] planned final indicators per candidate (target 3-7, not enforced): {indicator_counts(rows)}")
    for g, d in sorted(by_guideline(rows).items(), reverse=True):
        print(f"[mp-post] has_guideline={g}: " + " | ".join(f"{k} {v}" for k, v in d.items()))
    errs = {}
    for r in rows:
        for e in r.get("tool_errors") or []:
            k = f"{e.get('tool')}: {str(e.get('error'))[:100]}"
            errs[k] = errs.get(k, 0) + 1
    for k, v in sorted(errs.items(), key=lambda x: -x[1])[:10]:
        print(f"[mp-post] tool error x{v}: {k}")
    if not a.gate:
        return 0
    bad = 0
    for name, key, op, lim in GATES:
        v = s[key]
        ok = v >= lim if op == ">=" else v <= lim if op == "<=" else v > lim
        if key in REPORT_ONLY:
            print(f"[mp-gate] REPORT {a.task} {name}: {v} (reported only; was {op} {lim})")
            continue
        bad += not ok
        print(f"[mp-gate] {'PASS' if ok else 'FAIL'} {a.task} {name}: {v} (needs {op} {lim})")
    print(f"MEDAGENTPRO_CANARY_GATES {a.task} {'FAIL: ' + str(bad) + ' gate(s)' if bad else 'PASS'}")
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
