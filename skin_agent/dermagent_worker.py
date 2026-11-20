"""Persistent DermAgent+MedGemma inference worker.

Loads PanDerm + MAKE + DermoGPT + MedGemma ONCE, then stays resident and services
inference requests via a filesystem queue — so iterating doesn't reload the models.

Protocol (no networking needed):
  - drop a request JSON into $DERMAGENT_REQ_DIR:
      {"task":"rds","benchmark":"<jsonl>","limit":1,"output":"<preds.jsonl>",
       "model":"qwen/qwen3.5-27b","max_tokens":8000,"max_tool_images":3}
  - worker writes predictions to "output", then touches "<output>.done".
  - drop a file named "STOP" to shut the worker down.
"""
import glob
import json
import os
import sys
import time

EVAL = "/fs04/scratch2/ub62/ssim0070/DermArena/dataset_collection/eval"
if EVAL not in sys.path:
    sys.path.insert(0, EVAL)
from infer_dx_agent import _load_tools, predict_case  # noqa: E402

REQ_DIR = os.environ.get("DERMAGENT_REQ_DIR", "/fs04/scratch2/ub62/ssim0070/DermAgent/results/warm/requests")
os.makedirs(REQ_DIR, exist_ok=True)

print("[worker] loading tools (PanDerm + MAKE + DermoGPT + MedGemma) ...", flush=True)
tools = _load_tools("dermagent", device="cuda", use_medgemma=True)
for name, t in tools.items():
    if hasattr(t, "preload"):
        try:
            t.preload()
            print(f"[worker]   preloaded {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[worker]   {name} preload deferred: {e}", flush=True)
print(f"[worker] READY — polling {REQ_DIR}", flush=True)

while True:
    if os.path.exists(os.path.join(REQ_DIR, "STOP")):
        print("[worker] STOP received — exiting.", flush=True)
        break
    reqs = sorted(glob.glob(os.path.join(REQ_DIR, "*.json")))
    if not reqs:
        time.sleep(2)
        continue
    req_path = reqs[0]
    try:
        req = json.load(open(req_path))
    except Exception as e:  # noqa: BLE001
        print(f"[worker] bad request {req_path}: {e}", flush=True)
        os.remove(req_path)
        continue
    os.remove(req_path)  # claim it
    t0 = time.time()
    rows = [json.loads(l) for l in open(req["benchmark"]) if l.strip()]
    if req.get("limit"):
        rows = rows[: req["limit"]]
    out = req["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            rec = predict_case(row, req["task"], tools, "dermagent",
                               req.get("max_tool_images", 3), req.get("max_tokens", 8000),
                               req["model"])
            f.write(json.dumps(rec) + "\n")
            f.flush()
    open(out + ".done", "w").write("ok")
    print(f"[worker] served {os.path.basename(req_path)} "
          f"({req['task']}, {len(rows)} case) -> {out}  [{time.time()-t0:.0f}s]", flush=True)
