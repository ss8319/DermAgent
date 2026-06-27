#!/usr/bin/env python3
"""Run DermAgent (open-set, multi-image mode) on a DermArena Dx benchmark.

Routes each case through BenchmarkRunner.run_open_diagnosis: all images attached
to the multimodal backbone, full tool set offered, the LLM autonomously decides
which tools to call on which image, and returns a free-text top-5 differential.

Backbone is configurable (GPT-4o, or Qwen3.5-27B via OpenRouter / a local server):
    --backbone qwen/qwen3.5-27b  --base-url https://openrouter.ai/api/v1
    (OPENAI_API_KEY in env = the backbone endpoint's key)

Tools run locally on a GPU; the backbone is an API call. Output schema matches
the DermArena dev1000 spec; resume-by-_id (skip _ids already in the output).

Example:
    OPENAI_API_KEY=$OPENROUTER_KEY python scripts/run_dermarena_agent.py \
        --benchmark /fs04/scratch2/ub62/ssim0070/dermarena_dx/stratified/MM_RDS_benchmark_dev1000.jsonl \
        --task rds --backbone qwen/qwen3.5-27b --base-url https://openrouter.ai/api/v1 \
        --enabled-tools panderm,make,rag,ontology --device cuda \
        --output .../mm_rds_dev1000_predictions.jsonl --limit 1
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from skin_agent.benchmark_agent import (
    BenchmarkRunner, create_benchmark_agent, create_tools, OPEN_SET_DX_SYSTEM_PROMPT,
)
from skin_agent.tracing import TraceLogger

IMG_ROOT = Path(os.getenv("DERMARENA_IMG_ROOT", "/fs04/scratch2/ub62/ssim0070/dermarena_dx"))

INSTRUCTION = (
    "Read the following dermatology patient case and the attached image(s), then give "
    "the top 5 most likely diagnoses in ranked order (most likely first)."
)


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
            if isinstance(t, dict) else str(t) for t in tables
        )
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--task", choices=["rds", "rdc", "dxtest"], default="rds")
    ap.add_argument("--backbone", default="gpt-4o")
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    ap.add_argument("--enabled-tools", default="panderm,make,rag,ontology")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    prompt_hash = hashlib.sha256((OPEN_SET_DX_SYSTEM_PROMPT + INSTRUCTION).encode()).hexdigest()[:16]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in args.benchmark.open() if l.strip()]
    done = set()
    if args.output.exists():
        for l in args.output.open():
            try:
                done.add(json.loads(l)["_id"])
            except Exception:
                pass
    todo = [r for r in rows if r["_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[adapter] benchmark={len(rows)} done={len(done)} to-run={len(todo)} "
          f"| backbone={args.backbone} base_url={args.base_url} tools={args.enabled_tools}")

    enabled = [t.strip() for t in args.enabled_tools.split(",") if t.strip()]
    tools = create_tools(device=args.device, enabled_tools=enabled)
    agent = create_benchmark_agent(tools=tools, model_name=args.backbone,
                                   device=args.device, base_url=args.base_url,
                                   enabled_tools=enabled)
    runner = BenchmarkRunner(agent=agent, tools=tools,
                             trace_logger=TraceLogger(log_dir=str(args.output.parent / "traces")),
                             model_name=args.backbone)

    with args.output.open("a") as fout:
        for i, row in enumerate(todo, 1):
            cid = row["_id"]
            imgs = _image_paths(row)
            try:
                res = runner.run_open_diagnosis(
                    case_text=_case_text(row, args.task), image_paths=imgs,
                    instruction=INSTRUCTION, case_id=cid)
                rec = {
                    "_id": cid, "prediction": res["response"], "model": f"DermAgent_{args.backbone}",
                    "vision_used": res["vision_used"], "images_attached": res["n_images"],
                    "tools_used": res["tools_used"], "image_status": row.get("image_status"),
                    "prompt_hash": prompt_hash, "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                import traceback; traceback.print_exc()
                rec = {"_id": cid, "prediction": "", "model": f"DermAgent_{args.backbone}",
                       "error": f"{type(e).__name__}: {e}", "prompt_hash": prompt_hash,
                       "timestamp": datetime.now(timezone.utc).isoformat()}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i}/{len(todo)}] {cid} vision={rec.get('vision_used')} "
                  f"tools={rec.get('tools_used')} pred_head={rec['prediction'][:80]!r}")
    print(f"[adapter] done -> {args.output}")


if __name__ == "__main__":
    main()
