"""DermAgent run issue tracker: classify every unparsed row + backbone quirks, write ISSUES.md next to the run.

Usage: python3 -I dermagent_issue_tracker.py <run_dir> [T1.1 T1.2]
  run_dir = dx/dermagent-qwen3.5-27b (full run) or .../canary_vN. Reads predictions.p*.jsonl (live shards), so it
  works mid-run. Writes <run_dir>/ISSUES.md (overwrites) and prints the same table.
Causes are decided by fixed rules on the saved row (no guessing); 'other' rows are listed for a human to read.
"""
import glob, json, re, sys, collections as C, datetime as dt
from pathlib import Path

HANDLING = {  # cause -> (who, how it is handled)
    "request_error": ("agent", "Model-caused request error (context overflow / bad request), scored wrong, not retried. ALWAYS read the error: a burst = a new bug (061: critic system message)."),
    "recursion_limit": ("upstream", "30-step graph limit (upstream default). Scored wrong. Rate watched."),
    "cap_final_no_prefix": ("parser", "Cap-final reply is a numbered list without 'FINAL_ANSWER:'. Scored wrong (strict parser). OWNER DECISION if > 0.5 %."),
    "tool_call_as_text": ("model", "Qwen wrote a tool call as text in the final reply. Scored wrong."),
    "fa_mention_no_list": ("model", "Reply mentions FINAL_ANSWER (usually reasoning used as answer) but holds no numbered list. Scored wrong."),
    "empty": ("model", "Empty final reply after re-issues (EOS / unclosed think). Scored wrong."),
    "cap_unanswered": ("agent", "Ended on a refused tool batch without cap-final (should not happen with CAP_FINAL=1) -> investigate."),
    "length": ("model", "Hit the 32,768-token per-call budget. Scored wrong."),
    "list_no_prefix": ("parser", "Normal final turn ends with a ranked list but no 'FINAL_ANSWER:'. Scored wrong (same strict rule as the base arm; bare trailing list = C7, rejected 11-04)."),
    "json_answer": ("model", "Final reply is JSON (diagnosis_ranking / rank+diagnosis blocks) instead of the FINAL_ANSWER format. Scored wrong (strict). Possible parser rule P4 if it grows: OWNER DECISION."),
    "truncated_reply": ("model", "Final reply is a stub (a heading or ``` only, < 200 chars, no list): the model stopped early. Scored wrong."),
    "other": ("?", "Read the row; add a rule here."),
}
_NUM = re.compile(r"^\s*(?:\*\*)?\s*\d{1,2}\s*[\.\)\:]\s*\S", re.M)


def cause(r):
    resp = r.get("agent_response") or ""
    if r.get("finish_reason") == "agent_request_error":
        return "request_error"
    if r.get("finish_reason") == "agent_recursion_limit":
        return "recursion_limit"
    if r.get("finish_reason") == "agent_tool_call_cap":
        return "cap_unanswered"
    if r.get("unparsed_reason") == "length" or r.get("finish_reason") == "length":
        return "length"
    if not resp.strip():
        return "empty"
    if "<tool_call>" in resp or "<parameter=" in resp or "<function=" in resp:
        return "tool_call_as_text"
    if r.get("cap_final_answer") and "FINAL_ANSWER" not in resp.upper().replace(" ", "_") and _NUM.search(resp):
        return "cap_final_no_prefix"
    if re.search(r"FINAL[_ ]ANSWER", resp, re.I):
        return "fa_mention_no_list"
    if re.search(r'"diagnosis_ranking"|"rank"\s*:|"diagnoses"\s*:', resp):
        return "json_answer"
    if _NUM.search(resp):
        return "list_no_prefix"
    if len(resp.strip()) < 200 and not _NUM.search(resp):
        return "truncated_reply"
    return "other"


def main():
    run = Path(sys.argv[1])
    tasks = sys.argv[2:] or ["T1.1", "T1.2"]
    lines = [f"# Issue tracker — {run.name}  ({dt.datetime.now():%Y-%m-%d %H:%M})", ""]
    tot = C.Counter(); allrows = 0
    for t in tasks:
        rows = []
        for f in sorted(glob.glob(str(run / t / "predictions.p*.jsonl"))):
            rows += [json.loads(l) for l in open(f) if l.strip()]
        if not rows:
            continue
        n = len(rows); allrows += n
        bad = [r for r in rows if not r.get("parsed")]
        cc = C.Counter(cause(r) for r in bad); tot.update(cc)
        ri = C.Counter()
        for r in rows:
            for k, v in (r.get("backbone_reissues") or {}).items():
                ri[k] += v
        dropped = sum(1 for f in glob.glob(str(run / t / "dropped.p*.jsonl")) for l in open(f) if l.strip())
        lines += [f"## {t}: {n} rows · parsed {n - len(bad)} ({100 * (n - len(bad)) / n:.1f} %) · unparsed {len(bad)} "
                  f"({100 * len(bad) / n:.1f} %) · dropped calls {dropped}", "",
                  "| Cause | Count | % of rows | Owner | Handling |", "|---|---|---|---|---|"]
        for k, v in cc.most_common():
            lines.append(f"| {k} | {v} | {100 * v / n:.2f} | {HANDLING[k][0]} | {HANDLING[k][1]} |")
        capped = sum(bool(r.get("tool_call_cap_hit")) for r in rows)
        capf = sum(bool(r.get("cap_final_answer")) for r in rows)
        capf_ok = sum(bool(r.get("cap_final_answer")) and r.get("parsed") for r in rows)
        lines += ["", f"Recovered / handled (rows still parsed): tool-call cap hit {capped} ({100 * capped / n:.1f} %) — "
                  f"cap-final used {capf}, parsed {capf_ok}; answer taken from reasoning "
                  f"{sum(str(r.get('answer_source', '')).endswith('reasoning') for r in rows)}; backbone re-issues "
                  f"{dict(ri)}", ""]
        oth = [r["_id"] for r in bad if cause(r) == "other"]
        if oth:
            lines += [f"'other' rows to read: {oth[:20]}", ""]
    if allrows:
        lines += [f"**All tasks: {allrows} rows, unparsed {sum(tot.values())} ({100 * sum(tot.values()) / allrows:.1f} %)** — "
                  + ", ".join(f"{k} {v}" for k, v in tot.most_common())]
    txt = "\n".join(lines) + "\n"
    (run / "ISSUES.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
