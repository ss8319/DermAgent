#!/usr/bin/env python3
"""DermAgent (open-set LangGraph agent) on the DermArena v1 paper benchmark — T1.1 / T1.2 test split.

Same inputs as the base-model arm (benchmark_runs/dx/run_dx_model_shared.sh -> dx_infer.py):
  - case text = dx_infer.case_text(row, task)  (T1.1: case_report + tables; T1.2: + "Examination results:")
  - images    = row["images"] in row order, from the release root; non-RGB / non-JPEG/PNG converted to RGB PNG
                exactly like dx_infer.image_parts (converted copies cached on node-local disk)
  - backbone  = the same local SGLang model with its creator sampling (DERMAGENT_SAMPLING_JSON, set by the .req)
The agent keeps its own system prompt (OPEN_SET_DX_SYSTEM_PROMPT) and FINAL_ANSWER block. Its numbered FINAL_ANSWER
list is converted, deterministically, to the benchmark JSON {"diagnoses": [...]} so dx_check / judge_dx.sh score it
unchanged. No FINAL_ANSWER list (tool-call cap hit, empty reply, budget hit) -> prediction "" -> unparsed = wrong.

Concurrency: one process holds one copy of the tools; --threads cases run at once in it. Each tool CLASS gets a lock
(the tools are shared torch / transformers / embedded-Qdrant objects, not thread-safe); the backbone calls overlap.
Several processes split the cases with --shard k/n, each writing predictions.p<k>.jsonl; --merge joins them into
predictions.jsonl (what dx_check / the judge read). Resume: _ids already in any shard file are skipped.
Policy = dx_infer: a case that raises (server error, crash) is not written (logged to dropped.p<k>.jsonl) and is
retried once in pass 2; everything else is written once (one sample per case).

Usage:
  run_dermarena_v1.py --task T1.1 --input <test.jsonl> --out-dir <dx/<slug>/T1.1> --shard 0/4 --threads 12
  run_dermarena_v1.py --merge --out-dir <dx/<slug>/T1.1>
"""
import argparse
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DC = Path(os.environ.get("DERMARENA_DC", "/fs04/scratch2/ub62/ssim0070/DermArena/dataset_collection"))
RELEASE = Path(os.environ.get("DX_IMAGE_ROOT", "/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1"))
CONV_DIR = Path(os.environ.get("DERMAGENT_IMG_CACHE", "/tmp/dermagent_rgb_cache"))

INSTRUCTION = (
    "Read the following dermatology patient case and the attached image(s), then give "
    "the top 5 most likely diagnoses in ranked order (most likely first)."
)
DERM_MODALITIES = {"clinical_photo", "dermoscopy"}
TASK_DIR = {"T1.1": "T1.1_disease_screening", "T1.2": "T1.2_disease_confirmation"}
_FA = re.compile(r"FINAL[_ ]ANSWER\s*:?", re.I)
_ITEM = re.compile(r"^\s*(?:\*\*)?\s*(\d{1,2})\s*[\.\)\:]\s*(.+?)\s*$")


# --------------------------------------------------------------------------------------------- helpers
def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return None


def parse_final_answer(text):
    """Numbered items after the LAST 'FINAL_ANSWER' marker -> up to 5 names. ([] if none)."""
    if not text:
        return [], "empty"
    ms = list(_FA.finditer(text))
    if not ms:
        return [], "no_final_answer_block"
    tail = text[ms[-1].end():]
    items = []
    for line in tail.splitlines():
        m = _ITEM.match(line)
        if not m:
            if items and line.strip() and not line.strip().startswith(("-", "*")):
                break  # list ended (synthesis text after the block)
            continue
        name = m.group(2).strip().strip("*").strip().rstrip(";.,").strip()
        name = re.sub(r"^\*\*|\*\*$", "", name).strip()
        if name and not re.fullmatch(r"<[^<>]*>", name):
            items.append(name)
        if len(items) == 5:
            break
    return items, ("ok" if items else "no_numbered_items")


def _last_list_from_1(text):
    """Last contiguous numbered block whose first item is 1 -> names (all items)."""
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
            cur = None
    blocks = [b for b in blocks if b and all(b)]
    return blocks[-1] if blocks else []


def parse_answer(text, cap_final):
    """Owner 11-05: strict FINAL_ANSWER parse, then P3 for NORMAL final turns only (no FINAL_ANSWER marker, no text
    tool call: last numbered list from 1 with >= 5 items -> first 5). Cap-final replies: strict only (P1 dropped).
    Returns (items, reason, rule). Regression-tested: agents/dermagent_reparse.py check."""
    items, why = parse_final_answer(text)
    if items:
        return items, why, "strict"
    t = text or ""
    # P1 (cap-final list without the label) DROPPED by the owner 11-05 ("strict is fine": the agent may not have
    # consolidated its top 5 when the budget ran out) -> a cap-final reply needs an explicit FINAL_ANSWER block.
    if not cap_final and not re.search(r"FINAL[_ ]ANSWER", t, re.I) and not re.search(
            r"<tool_call>|<function=|<parameter=|['\"]arguments['\"]\s*:", t):
        it = _last_list_from_1(t)
        if len(it) >= 5:
            return it[:5], "ok", "P3_trailing_top5_list"
    return [], why, "strict"


_conv_lock = threading.Lock()


def rgb_path(fp: Path) -> str:
    """dx_infer.image_parts rule: RGB JPEG/PNG used byte-for-byte; anything else -> RGB PNG copy."""
    from PIL import Image
    with Image.open(fp) as I:
        if I.mode == "RGB" and I.format in ("JPEG", "PNG"):
            return str(fp)
        out = CONV_DIR / (hashlib.md5(str(fp).encode()).hexdigest() + ".png")
        with _conv_lock:
            if not out.exists():
                CONV_DIR.mkdir(parents=True, exist_ok=True)
                # per-process temp name + atomic replace: the cache dir is shared by every runner on the node, and
                # _conv_lock only covers this process (a shared temp name let two writers leave a torn PNG behind)
                tmp = out.with_name(f"{out.stem}.{os.getpid()}.{threading.get_ident()}.tmp.png")
                I.convert("RGB").save(tmp, format="PNG")
                os.replace(tmp, out)
        return str(out)


def _strip_images(content):
    if isinstance(content, list):
        out = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "image_url":
                u = (c.get("image_url") or {}).get("url", "")
                out.append({"type": "image_url", "image_url": f"<base64 image, {len(u)} chars, sha {hashlib.md5(u.encode()).hexdigest()[:10]}>"})
            else:
                out.append(c)
        return out
    return content


def serialise_messages(msgs):
    out = []
    for m in msgs:
        d = {"type": type(m).__name__, "content": _strip_images(getattr(m, "content", None))}
        for k in ("name", "tool_call_id", "tool_calls", "additional_kwargs", "response_metadata", "usage_metadata"):
            v = getattr(m, k, None)
            if v:
                d[k] = v
        out.append(d)
    return out


def lock_tool_classes(tools):
    """One lock per tool class: _run executes serially per tool type, other work (LLM calls) overlaps."""
    for t in tools:
        cls = type(t)
        if getattr(cls, "_dermarena_locked", False):
            continue
        orig = cls._run
        lk = threading.Lock()

        def make(orig_fn, lock):
            @functools.wraps(orig_fn)  # keeps the signature LangChain inspects (args schema, run_manager)
            def _run(self, *args, **kwargs):
                with lock:
                    return orig_fn(self, *args, **kwargs)
            return _run

        cls._run = make(orig, lk)
        cls._dermarena_locked = True


def done_ids(out_dir: Path, run_sig=None):
    """_ids that already have a row. With run_sig, only rows of that run_sig count: a row written with other settings
    (e.g. RUN_META unreadable at start, DX_ALLOW_META_CHANGE, a seeded canary) is re-run, never resumed into this run."""
    ids = set()
    for f in list(out_dir.glob("predictions.p*.jsonl")) + [out_dir / "predictions.jsonl"]:
        if f.exists():
            for l in f.open():
                try:
                    r = json.loads(l)
                    if run_sig is None or r.get("run_sig") == run_sig:
                        ids.add(r["_id"])
                except Exception:  # noqa: BLE001
                    pass
    return ids


def merge(out_dir: Path, run_sig=None):
    """One row per _id. run_sig (default: RUN_META.json's) wins over a row of other settings for the same _id; any row
    of other settings that is still kept is reported as MIXED, never hidden."""
    if run_sig is None:
        try:
            run_sig = json.loads((out_dir / "RUN_META.json").read_text()).get("run_sig")
        except Exception:  # noqa: BLE001
            run_sig = None
    rows = {}
    for f in sorted(out_dir.glob("predictions.p*.jsonl")):
        for l in f.open():
            try:
                r = json.loads(l)
                cid = r["_id"]
            except Exception:  # noqa: BLE001
                continue
            if cid not in rows or (run_sig and rows[cid].get("run_sig") != run_sig and r.get("run_sig") == run_sig):
                rows[cid] = r
    other = {}
    for r in rows.values():
        if run_sig and r.get("run_sig") != run_sig:
            other[r.get("run_sig")] = other.get(r.get("run_sig"), 0) + 1
    if other:
        print(f"[merge] {out_dir}: MIXED run_sig — {sum(other.values())} row(s) not from run_sig {run_sig}: {other}", flush=True)
    # per-process temp name: several shard runners may merge the same folder at once (a shared name lets two writers
    # interleave into one file and then race on the rename)
    tmp = out_dir / f"predictions.jsonl.tmp{os.getpid()}.{threading.get_ident()}"
    with tmp.open("w") as fo:
        for r in rows.values():
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_dir / "predictions.jsonl")
    print(f"[merge] {out_dir}: {len(rows)} rows -> predictions.jsonl")


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="T1.1,T1.2")
    ap.add_argument("--input-tmpl", default=str(RELEASE / "tasks/{task_dir}/test.jsonl"),
                    help="input path; {task} / {task_dir} are filled (canary: benchmark_runs/dx/canary20_{task}.jsonl)")
    ap.add_argument("--out-tmpl", default=None, help="output dir per task, e.g. .../dx/<slug>/{task}")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="max cases for THIS shard (canary)")
    ap.add_argument("--backbone", default=os.environ.get("DERMAGENT_BACKBONE", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--slug", default=os.environ.get("DERMAGENT_SLUG", "dermagent-qwen3.5-27b"))
    ap.add_argument("--enabled-tools", default=os.environ.get(
        "ENABLED_TOOLS", "panderm,make,dermogpt_vqa,qwen_vqa,rag,text_rag,ontology"))
    ap.add_argument("--qdrant-path", default=os.environ.get("QDRANT_PATH", str(PROJECT_ROOT / "qdrant_storage")))
    ap.add_argument("--max-tool-calls", type=int, default=int(os.environ.get("MAX_TOOL_CALLS", 20)))
    ap.add_argument("--recursion-limit", type=int, default=int(os.environ.get("RECURSION_LIMIT", 60)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--service", default=None,
                    help="persistent worker: load tools ONCE, then run job files (*.json) dropped in this dir until STOP")
    a = ap.parse_args()
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    if a.merge:
        for t in tasks:
            merge(Path(a.out_tmpl.format(task=t)))
        return

    k, n = (int(x) for x in a.shard.split("/"))
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(DC / "dx_benchmark"))
    os.environ.setdefault("DX_IMAGE_ROOT", str(RELEASE))
    import dx_infer  # noqa: E402  (benchmark's own case_text: identical text to the base arm)
    from skin_agent.benchmark_agent import (  # noqa: E402
        BenchmarkRunner, create_benchmark_agent, create_tools, OPEN_SET_DX_SYSTEM_PROMPT,
        OPEN_SET_DX_SYSTEM_PROMPT_UPSTREAM_FULL,
    )
    from skin_agent.tracing import TraceLogger  # noqa: E402

    prompt_variant = os.environ.get("DERMAGENT_OPEN_PROMPT") or "open_set_v0"
    sys_prompt = OPEN_SET_DX_SYSTEM_PROMPT_UPSTREAM_FULL if prompt_variant == "upstream_full" else OPEN_SET_DX_SYSTEM_PROMPT
    critic = {"enable_critic": os.environ.get("DERMAGENT_ENABLE_CRITIC", "0") == "1",
              "critic_image_reinject": os.environ.get("DERMAGENT_CRITIC_IMAGE_REINJECT", "0") == "1",
              "critic_llm_summary": os.environ.get("DERMAGENT_CRITIC_LLM_SUMMARY", "0") == "1",
              "max_critic_retries": int(os.environ.get("DERMAGENT_MAX_CRITIC_RETRIES", "2"))}
    sampling = json.loads(os.environ.get("DERMAGENT_SAMPLING_JSON") or "{}")
    base_meta = {
        "system": "DermAgent (open-set LangGraph, skin_agent.benchmark_agent.run_open_diagnosis)",
        "slug": a.slug, "backbone": a.backbone, "base_url": os.environ.get("OPENAI_BASE_URL"),
        "sampling": sampling, "max_tokens_per_call": int(os.environ.get("DERMAGENT_MAX_TOKENS", "8192")),
        "enabled_tools": a.enabled_tools, "max_tool_calls": a.max_tool_calls, "recursion_limit": a.recursion_limit,
        "prompt_variant": prompt_variant, "critic": critic,
        "orchestration": {k: os.environ.get(k) for k in (
            "DERMAGENT_ATTACH_IMAGES", "DERMAGENT_USER_MSG", "DERMAGENT_MIDCONV_SYSTEM_AS_USER",
            "DERMAGENT_EOS_RETRIES", "DERMAGENT_REISSUE_THINK_TOOLCALL", "DERMAGENT_IMAGE_MODALITY_TAGS", "DERMAGENT_TOOL_IMAGES", "DERMAGENT_REASONING_FALLBACK", "DERMAGENT_OPEN_SHORTLIST",
            "DERMAGENT_REINJECT_ALL_DERM", "DERMAGENT_PASS_REASONING",
            "DERMAGENT_REISSUE_INTERRUPTED", "DERMAGENT_CAP_FINAL", "DERMAGENT_THINK_TOOLCALL_EXTRACT",
            "RECURSION_LIMIT", "DERMAGENT_SANITIZE_SPECIAL_TOKENS", "DERMAGENT_STRIP_NOTOOL_TOOLCALLS", "DERMAGENT_CASERAG_INDEX")}, "threads_per_process": a.threads, "shards": n,
        "prompt_sha": hashlib.sha256((sys_prompt + INSTRUCTION).encode()).hexdigest()[:16],
        "code": {p: sha_file(PROJECT_ROOT / p) for p in
                 ("skin_agent/benchmark_agent.py", "skin_agent/tools/skin_tools.py", "scripts/run_dermarena_v1.py")},
        "case_text_from": str(DC / "dx_benchmark/dx_infer.py"), "case_text_sha": sha_file(DC / "dx_benchmark/dx_infer.py"),
        "git_head": subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
    }
    run_sig = hashlib.sha256(json.dumps({x: base_meta[x] for x in
                                         ("backbone", "sampling", "max_tokens_per_call", "enabled_tools",
                                          "max_tool_calls", "prompt_sha", "prompt_variant", "critic", "orchestration", "code")}, sort_keys=True).encode()
                             ).hexdigest()[:12]

    def build_plan(tasks, input_tmpl, out_tmpl, limit, threads):
      plan = []
      for task in tasks:
        inp = Path(input_tmpl.format(task=task, task_dir=TASK_DIR[task]))
        out_dir = Path(out_tmpl.format(task=task))
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(l) for l in inp.open() if l.strip()]
        mine = [r for i, r in enumerate(rows) if i % n == k]
        have = done_ids(out_dir, run_sig)
        todo = [r for r in mine if r["_id"] not in have]
        if limit:
            todo = todo[: limit]
        meta = dict(base_meta, task=task, threads_per_process=threads, input={"path": str(inp), "sha": sha_file(inp)}, run_sig=run_sig)
        mp = out_dir / "RUN_META.json"
        try:  # several shards start together: tolerate a file another shard is writing right now
            prev = json.loads(mp.read_text()).get("run_sig") if mp.exists() else None
        except Exception:  # noqa: BLE001
            prev = None
        if prev not in (None, run_sig) and not os.environ.get("DX_ALLOW_META_CHANGE"):
            raise SystemExit(f"[v1] {mp} has a different run_sig — refusing to mix settings (DX_ALLOW_META_CHANGE=1)")
        if k == 0:
            tmp = mp.with_name(f"RUN_META.json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(meta, indent=1))
            os.replace(tmp, mp)  # atomic
        print(f"[v1] {task} shard {k}/{n}: {len(mine)} cases, {len(mine) - len(todo)} done, {len(todo)} to run "
              f"| threads {threads} | run_sig {run_sig}", flush=True)
        plan.append((task, out_dir, todo))
      return plan

    if not a.service:
        plan = build_plan(tasks, a.input_tmpl, a.out_tmpl, a.limit, a.threads)
        if not any(todo for _, _, todo in plan):
            return

    enabled = [t.strip() for t in a.enabled_tools.split(",") if t.strip()]
    t0 = time.time()
    tools = create_tools(device="cuda", enabled_tools=enabled, preload=True, qdrant_path=a.qdrant_path)
    # Hard gate (2026-11-05): preload prints "loaded successfully" even when a tool's own init failed
    # (PanDerm text tower + Qwen3-Embedding were dangling cache links) -> the agent silently ran without them.
    bad = [t.name for t in tools
           if getattr(t, "_load_failed", False) or (hasattr(t, "is_initialized") and not t.is_initialized)]
    print(f"[v1] tool load check: {[(t.name, 'OK' if t.name not in bad else 'FAILED') for t in tools]}", flush=True)
    if bad:
        raise SystemExit(f"[v1] TOOLS FAILED TO LOAD: {bad} — refusing to run (fix weights/cache first)")
    runners = {}

    def get_runner(cap):
        """One agent graph per tool-call cap (tools are shared; only the LangGraph + LLM client are rebuilt)."""
        if cap not in runners:
            agent = create_benchmark_agent(tools=tools, model_name=a.backbone, device="cuda",
                                           base_url=os.environ.get("OPENAI_BASE_URL"), enabled_tools=enabled,
                                           max_tool_calls=cap, recursion_limit=a.recursion_limit,
                                           enable_critic=critic["enable_critic"] or None,
                                           critic_image_reinject=critic["critic_image_reinject"],
                                           critic_llm_summary=critic["critic_llm_summary"],
                                           max_critic_retries=critic["max_critic_retries"])
            lock_tool_classes(tools)  # idempotent; after bind_tools (schemas built from the original _run)
            r = BenchmarkRunner(agent=agent, tools=tools,
                                trace_logger=TraceLogger(log_dir=str(CONV_DIR / "unused_traces")),
                                model_name=a.backbone, recursion_limit=a.recursion_limit)
            r._v1_cap = cap
            runners[cap] = r
        return runners[cap]

    def sig_for(cap):
        base_meta["max_tool_calls"] = cap
        return hashlib.sha256(json.dumps({x: base_meta[x] for x in
                                          ("backbone", "sampling", "max_tokens_per_call", "enabled_tools",
                                           "max_tool_calls", "prompt_sha", "prompt_variant", "critic", "orchestration",
                                           "code")}, sort_keys=True).encode()).hexdigest()[:12]

    runner = get_runner(a.max_tool_calls)
    print(f"[v1] tools + agent loaded in {time.time() - t0:.0f}s", flush=True)

    if not a.service:
        for task, out_dir, todo in plan:
            if todo:
                run_task(task, out_dir, todo, k, n, a.threads, runner, dx_infer, run_sig,
                         base_meta["max_tokens_per_call"], a.backbone)
        return

    # ---- persistent worker (CLAUDE.md: load once, hold the allocation) ----------------------------------------
    # Job file <jobdir>/<NN_name>.json: {"tasks": "T1.1,T1.2", "input_tmpl": ..., "out_tmpl": ..., "limit": null,
    # "threads": 12}. Every shard runs its share and writes <name>.p<k>.done; shard 0 then waits for all n,
    # merges the shards, runs the Dx postcheck and writes <name>.ALLDONE. STOP in the jobdir ends the worker.
    jd = Path(a.service)
    jd.mkdir(parents=True, exist_ok=True)
    print(f"[v1] SERVICE ready (shard {k}/{n}) — polling {jd}", flush=True)
    while not (jd / "STOP").exists():
        jobs = [j for j in sorted(jd.glob("*.json")) if not (jd / f"{j.stem}.p{k}.done").exists()]
        if not jobs:
            time.sleep(30)
            continue
        jf = jobs[0]
        try:
            job = json.loads(jf.read_text())
        except Exception as e:  # noqa: BLE001 — half-written job file: retry next poll
            print(f"[v1] SERVICE cannot read {jf.name}: {e}", flush=True)
            time.sleep(10)
            continue
        jtasks = [t.strip() for t in job.get("tasks", "T1.1,T1.2").split(",") if t.strip()]
        thr = int(job.get("threads", a.threads))
        cap = int(job.get("max_tool_calls") or a.max_tool_calls)  # per-job override (cap-20 check, owner 11-05)
        runner = get_runner(cap)
        run_sig = sig_for(cap)  # build_plan reads run_sig/base_meta from this scope at call time
        print(f"[v1] SERVICE job {jf.stem}: {jtasks} threads {thr} max_tool_calls {cap} run_sig {run_sig}", flush=True)
        plan = build_plan(jtasks, job.get("input_tmpl", str(RELEASE / "tasks/{task_dir}/test.jsonl")),
                          job["out_tmpl"], job.get("limit"), thr)
        for task, out_dir, todo in plan:
            if todo and not (jd / "STOP").exists():
                run_task(task, out_dir, todo, k, n, thr, runner, dx_infer, run_sig, base_meta["max_tokens_per_call"], a.backbone)
        (jd / f"{jf.stem}.p{k}.done").write_text(now())
        if k == 0:  # merge + postcheck once every shard has finished this job
            while not all((jd / f"{jf.stem}.p{x}.done").exists() for x in range(n)) and not (jd / "STOP").exists():
                time.sleep(20)
            for task, out_dir, _ in plan:
                merge(out_dir)
                inp = job.get("input_tmpl", str(RELEASE / "tasks/{task_dir}/test.jsonl")).format(
                    task=task, task_dir=TASK_DIR[task])
                r = subprocess.run(["python3", str(DC / "dx_benchmark/dx_check.py"), "post", str(out_dir), task, inp],
                                   capture_output=True, text=True)
                (out_dir / "POSTCHECK.txt").write_text(r.stdout + r.stderr)
                print(f"[v1] SERVICE postcheck {jf.stem} {task}:\n{r.stdout[-1500:]}", flush=True)
            (jd / f"{jf.stem}.ALLDONE").write_text(now())
            print(f"[v1] SERVICE job {jf.stem} ALLDONE", flush=True)
    print("[v1] SERVICE STOP seen — exiting", flush=True)


def run_task(task, out_dir, todo, k, n, threads, runner, dx_infer, run_sig, max_tok, backbone):
    t0 = time.time()
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    out_f = (out_dir / f"predictions.p{k}.jsonl").open("a")
    drop_f = (out_dir / f"dropped.p{k}.jsonl").open("a")
    wlock = threading.Lock()
    stats = {"done": 0, "parsed": 0, "dropped": 0}
    total = len(todo)

    stop_file = Path(os.environ.get("DERMAGENT_STOP_FILE", "/nonexistent/STOP"))

    def one(row):
        cid = row["_id"]
        if stop_file.exists():  # operator stop: skip the remaining cases, workers exit, the allocation is kept
            return
        t1 = time.time()
        # owner 2026-11-05: the backbone gets ALL images (+ case + tables) exactly like the base arm; the image
        # tools get only clinical photos / dermoscopy (release 'modality' label; 'unknown' = not usable).
        imgs, mods, usable, missing = [], [], [], 0
        derm_only = os.environ.get("DERMAGENT_TOOL_IMAGES", "all") == "derm"
        for im in row.get("images") or []:
            fp = RELEASE / im["image_path"]
            if not fp.exists():
                missing += 1
                continue
            imgs.append(rgb_path(fp))
            mods.append(im.get("modality"))
            usable.append((im.get("modality") in DERM_MODALITIES) if derm_only else True)
        n_all = len(imgs)
        try:
            res = runner.run_open_diagnosis(case_text=dx_infer.case_text(row, task), image_paths=imgs,
                                            instruction=INSTRUCTION, case_id=f"{task}_{cid}",
                                            image_modalities=mods, tool_usable=usable)
        except Exception as e:  # noqa: BLE001 — failed call: not written, retried in pass 2
            if type(e).__name__ == "GraphRecursionError":  # the AGENT ran out of steps: its answer, = wrong
                rec = {"_id": cid, "task": task, "examinee": f"DermAgent/{backbone}", "system": "dermagent",
                       "prediction": "", "parsed": False, "unparsed_reason": "agent_recursion_limit", "n_items": 0,
                       "finish_reason": "agent_recursion_limit", "answer_source": "agent_final_answer",
                       "agent_response": "", "error": str(e)[:500], "max_tokens": max_tok, "call_attempts": 1,
                       "corpus_status": row.get("corpus_status"), "image_status": row.get("image_status"),
                       "wall_s": round(time.time() - t1, 1), "run_sig": run_sig, "timestamp": now()}
                with wlock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    stats["done"] += 1
                print(f"[v1] {task} p{k} RECURSION_LIMIT {cid} (unparsed)", flush=True)
                return
            if type(e).__name__ == "BadRequestError" or "context length" in str(e).lower() \
                    or "maximum context" in str(e).lower():  # review 11-05: model-caused, not infra -> no pass-2 retry
                rec = {"_id": cid, "task": task, "examinee": f"DermAgent/{backbone}", "system": "dermagent",
                       "prediction": "", "parsed": False, "unparsed_reason": "agent_request_error", "n_items": 0,
                       "finish_reason": "agent_request_error", "answer_source": "agent_final_answer",
                       "agent_response": "", "error": str(e)[:500], "max_tokens": max_tok, "call_attempts": 1,
                       "corpus_status": row.get("corpus_status"), "image_status": row.get("image_status"),
                       "wall_s": round(time.time() - t1, 1), "run_sig": run_sig, "timestamp": now()}
                with wlock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    stats["done"] += 1
                print(f"[v1] {task} p{k} REQUEST_ERROR {cid} (unparsed): {str(e)[:160]}", flush=True)
                return
            with wlock:
                drop_f.write(json.dumps({"_id": cid, "error": f"{type(e).__name__}: {e}",
                                         "tb": traceback.format_exc()[-3000:], "timestamp": now()}) + "\n")
                drop_f.flush()
                stats["dropped"] += 1
            print(f"[v1] {task} p{k} DROPPED {cid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        finally:
            try:  # free the per-case checkpoint (MemorySaver keeps every thread's messages + base64 images)
                cp = getattr(runner.agent, "checkpointer", None)
                if cp is not None and hasattr(cp, "delete_thread"):
                    cp.delete_thread(f"open_{task}_{cid}")
            except Exception:  # noqa: BLE001
                pass
        msgs = res.get("messages") or []
        ai = [m for m in msgs if type(m).__name__ == "AIMessage"]
        last = ai[-1] if ai else None
        fin = (getattr(last, "response_metadata", {}) or {}).get("finish_reason") if last else None
        n_tool_calls = sum(len(getattr(m, "tool_calls", None) or []) for m in ai)
        ended_on_tool_call = bool(last is not None and getattr(last, "tool_calls", None))
        cap_final = any((getattr(m, "response_metadata", {}) or {}).get("cap_final") for m in ai)
        executed = {getattr(m, "tool_call_id", None) for m in msgs if type(m).__name__ == "ToolMessage"
                    and not str(m.content).startswith("Not executed: the tool-call budget")}
        n_refused = sum(1 for m in ai for tc in (getattr(m, "tool_calls", None) or []) if tc.get("id") not in executed)
        comp_tok = sum(((getattr(m, "usage_metadata", None) or {}).get("output_tokens") or 0) for m in ai)
        reissues = {}
        for m in ai:
            rm = getattr(m, "response_metadata", {}) or {}
            if rm.get("reissue_reason"):
                reissues[rm["reissue_reason"]] = reissues.get(rm["reissue_reason"], 0) + int(rm.get("reissues", 1))
        items, why, prule = parse_answer(res.get("response") or "", cap_final)
        if fin == "length":
            items, why, prule = [], "length", "strict"
        n_think_tc = sum(int((getattr(m, "response_metadata", {}) or {}).get("n_think_toolcalls") or 0) for m in ai)
        rec = {
            "_id": cid, "task": task, "examinee": f"DermAgent/{backbone}", "system": "dermagent",
            "prediction": json.dumps({"diagnoses": items}, ensure_ascii=False) if items else "",
            "parsed": bool(items), "unparsed_reason": None if items else why, "n_items": len(items),
            "finish_reason": "agent_tool_call_cap" if ended_on_tool_call else fin,
            "tool_call_cap_hit": bool(n_refused or ended_on_tool_call or cap_final), "n_tool_calls_refused": n_refused,
            "cap_final_answer": cap_final, "parse_rule": prule, "think_toolcalls_extracted": n_think_tc,
            "max_tool_calls": getattr(runner, "_v1_cap", None),
            "answer_source": "agent_final_answer" + (
                "_from_reasoning" if (getattr(last, "response_metadata", {}) or {}).get("answer_source") == "reasoning"
                else ""), "agent_response": res.get("response") or "",
            "tools_used": res.get("tools_used"), "n_tool_calls": n_tool_calls, "n_llm_calls": len(ai),
            "completion_tokens": comp_tok, "max_tokens": max_tok, "call_attempts": 1, "backbone_reissues": reissues,
            "n_images_sent": res.get("n_images"), "n_images_case": n_all,
            "n_images_tool_usable": sum(usable), "images_modality": mods, "n_images_missing": missing,
            "corpus_status": row.get("corpus_status"), "image_status": row.get("image_status"),
            "wall_s": round(time.time() - t1, 1), "run_sig": run_sig, "timestamp": now(),
        }
        (trace_dir / f"{cid}.json").write_text(json.dumps(
            {"_id": cid, "task": task, "run_sig": run_sig, "record": rec,
             "messages": serialise_messages(msgs)}, ensure_ascii=False, indent=1, default=str))
        with wlock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            stats["done"] += 1
            stats["parsed"] += bool(items)
            el = time.time() - t0
            print(f"[v1] {task} p{k} [{stats['done']}/{total}] {cid} parsed={bool(items)} tools={n_tool_calls} "
                  f"llm={len(ai)} {rec['wall_s']}s | parse {stats['parsed']}/{stats['done']} "
                  f"| {stats['done'] / max(el, 1) * 60:.2f} cases/min", flush=True)

    for p in (1, 2):
        if p == 2:
            have = done_ids(out_dir, run_sig)
            todo = [r for r in todo if r["_id"] not in have]
            if not todo:
                break
            print(f"[v1] {task} p{k} pass 2: retrying {len(todo)} dropped case(s)", flush=True)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for f in as_completed([ex.submit(one, r) for r in todo]):
                f.result()
    print(f"[v1] DONE {task} shard {k}/{n}: {stats} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
