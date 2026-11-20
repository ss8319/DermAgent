"""Persistent DermAgent (LangGraph agent) worker — HOLD-THE-ALLOCATION pattern.

Loads the 7 tools + LangGraph benchmark agent ONCE and stays resident, servicing
successive run requests via a filesystem queue — so iterating never reloads tools
and the H200 allocation is held until walltime (per workspace CLAUDE.md GPU rule).

Uses the SAME code path as scripts/run_dermarena_agent.py (create_tools →
create_benchmark_agent → BenchmarkRunner.run_open_diagnosis), so predictions are
identical to the per-job runs — only the load happens once.

Protocol (no networking):
  drop a request JSON into $DERMAGENT_REQ_DIR:
    {"task":"rds|rdc","benchmark":"<jsonl>","output":"<preds.jsonl>","limit":null}
  worker resumes (skips _ids already in output), writes predictions, then touches
    "<output>.done".
  drop a file named "STOP" to shut the worker down (or let walltime end the job).

Backbone (SGLang) must already be serving at $OPENAI_BASE_URL on the same node.
"""
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from skin_agent.benchmark_agent import (  # noqa: E402
    BenchmarkRunner, create_benchmark_agent, create_tools, OPEN_SET_DX_SYSTEM_PROMPT,
)
from skin_agent.tracing import TraceLogger  # noqa: E402

IMG_ROOT = Path(os.getenv("DERMARENA_IMG_ROOT", "/fs04/scratch2/ub62/ssim0070/dermarena_dx"))
INSTRUCTION = (
    "Read the following dermatology patient case and the attached image(s), then give "
    "the top 5 most likely diagnoses in ranked order (most likely first)."
)
REQ_DIR = os.environ.get("DERMAGENT_REQ_DIR",
                         "/fs04/scratch2/ub62/ssim0070/DermAgent/results/warm_langgraph/requests")
ENABLED_TOOLS = [t.strip() for t in os.environ.get(
    "ENABLED_TOOLS", "panderm,make,dermogpt_vqa,qwen_vqa,rag,text_rag,ontology"
).split(",") if t.strip()]
BACKBONE = os.environ.get("DERMAGENT_BACKBONE", "qwen/qwen3.5-27b")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "15"))
RECURSION_LIMIT = int(os.environ.get("RECURSION_LIMIT", "60"))

PROMPT_HASH = hashlib.sha256((OPEN_SET_DX_SYSTEM_PROMPT + INSTRUCTION).encode()).hexdigest()[:16]


def _case_text(row: dict, task: str) -> str:
    txt = row.get("case_report") or ""
    if task == "rdc":
        exam = row.get("examination_results")
        if exam and exam != "NA":
            txt += f"\n\nExamination results:\n{exam}"
    tables = row.get("tables") or []
    if tables:
        txt += "\n\nTables:\n" + "\n".join(
            (t.get("text_for_embedding") or t.get("caption") or json.dumps(t))[:1500]
            if isinstance(t, dict) else str(t) for t in tables)
    return txt


def _image_paths(row: dict) -> list:
    out = []
    for im in row.get("images") or []:
        p = im.get("image_path") or ""
        if not p:
            continue
        cand = Path(p) if Path(p).is_absolute() else IMG_ROOT / p
        if cand.exists():
            out.append(str(cand))
    return out


def _process(req: dict, runner: "BenchmarkRunner") -> None:
    task = req.get("task", "rds")
    bench = Path(req["benchmark"])
    out = Path(req["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in bench.open() if l.strip()]
    done = set()
    if out.exists():
        for l in out.open():
            try:
                done.add(json.loads(l)["_id"])
            except Exception:
                pass
    todo = [r for r in rows if r["_id"] not in done]
    if req.get("limit"):
        todo = todo[: req["limit"]]
    print(f"[worker] task={task} benchmark={len(rows)} done={len(done)} to-run={len(todo)}", flush=True)
    with out.open("a") as fout:
        for i, row in enumerate(todo, 1):
            cid = row["_id"]
            try:
                res = runner.run_open_diagnosis(
                    case_text=_case_text(row, task), image_paths=_image_paths(row),
                    instruction=INSTRUCTION, case_id=cid)
                rec = {"_id": cid, "prediction": res["response"], "model": f"DermAgent_{BACKBONE}",
                       "vision_used": res["vision_used"], "images_attached": res["n_images"],
                       "tools_used": res["tools_used"], "image_status": row.get("image_status"),
                       "prompt_hash": PROMPT_HASH, "timestamp": datetime.now(timezone.utc).isoformat()}
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                rec = {"_id": cid, "prediction": "", "model": f"DermAgent_{BACKBONE}",
                       "error": f"{type(e).__name__}: {e}", "prompt_hash": PROMPT_HASH,
                       "timestamp": datetime.now(timezone.utc).isoformat()}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[worker] [{i}/{len(todo)}] {cid} vision={rec.get('vision_used')} "
                  f"empty={not (rec['prediction'] or '').strip()}", flush=True)
    Path(str(out) + ".done").touch()
    print(f"[worker] request complete -> {out}", flush=True)


def main() -> None:
    os.makedirs(REQ_DIR, exist_ok=True)
    print(f"[worker] loading tools {ENABLED_TOOLS} + LangGraph agent (once) ...", flush=True)
    tools = create_tools(device="cuda", enabled_tools=ENABLED_TOOLS, preload=True)
    agent = create_benchmark_agent(tools=tools, model_name=BACKBONE, device="cuda",
                                   base_url=BASE_URL, enabled_tools=ENABLED_TOOLS,
                                   max_tool_calls=MAX_TOOL_CALLS, recursion_limit=RECURSION_LIMIT)
    runner = BenchmarkRunner(agent=agent, tools=tools,
                             trace_logger=TraceLogger(log_dir=os.path.join(REQ_DIR, "traces")),
                             model_name=BACKBONE, recursion_limit=RECURSION_LIMIT)
    print(f"[worker] READY — holding allocation, polling {REQ_DIR}", flush=True)
    while True:
        if os.path.exists(os.path.join(REQ_DIR, "STOP")):
            print("[worker] STOP received — exiting.", flush=True)
            break
        reqs = sorted(glob.glob(os.path.join(REQ_DIR, "*.json")))
        if not reqs:
            time.sleep(3)
            continue
        req_path = reqs[0]
        try:
            req = json.load(open(req_path))
        except Exception as e:  # noqa: BLE001
            print(f"[worker] bad request {req_path}: {e}", flush=True)
            os.remove(req_path)
            continue
        os.remove(req_path)  # claim it
        try:
            _process(req, runner)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"[worker] request FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
