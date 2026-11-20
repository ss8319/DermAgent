#!/usr/bin/env python3
"""Index build of the MedlinePlus knowledge base, after medlineplus_kb.py on CPU. Summaries need the backbone server.

  summaries  one-sentence summary per document from the backbone server (paper: 'a one-sentence summary is pre-generated
             for each document and used as its index'), through mp_llm, full trace kept per row. Per backbone, so each
             arm uses only its own model. Resumable: documents with a summary are skipped; failures are retried in up to
             MP_KB_PASSES rounds (default 3), then logged.
  embed      the embedder (MP_EMBEDDER; default ada-002 through OpenRouter, no GPU) over the chunk file of MP_CHUNKS (default 300
             tokens) -> index/emb__<embedder>__<profile>/shard_*.npy, unit norm. Shared by all backbones. Resumable: finished shards
             are skipped; each shard is written atomically. The Qwen3-Embedding index (index/emb/) is never written here.
  finalize   refuses unless every document has a summary and every chunk an embedding; writes
             index/INDEX_MANIFEST_<backbone>__<embedder>__<profile>.json with the embedder, chunk settings and chunk file hash;
             its content goes into kb_sha, the plan namespace and run_sig.
Env for summaries: OPENAI_BASE_URL, MP_MODEL, MP_SAMPLING_JSON, MP_MAX_TOKENS, MP_EOS_RETRIES (see run_medagentpro_v1.sh).
Env for embed: OPENROUTER_API_KEY (the real key, never printed), MP_EMBEDDER, MP_CHUNKS, MP_EMBED_RETRIES, MP_EMBED_TIMEOUT.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from medlineplus_kb import (CHUNK_PROFILES, KB_DEFAULT, chunk_profile_name, chunks_manifest_path, chunks_path,  # noqa: E402
                            sha_path, utc)

SUMMARY_PROMPT = ("Write ONE sentence that summarises the document below. Start with the name of the disease or topic, "
                  "and include any other names given for it. Return only that sentence.\n\n"
                  "Title: {title}\nOther names: {alt}\n\nDocument:\n{text}")
SHARD = 4096


def summaries_path(kb, backbone):
    import kb_rag
    return Path(kb) / "index" / f"summaries_{kb_rag.index_slug(backbone)}.jsonl"


def load_docs(kb):
    return [json.loads(l) for l in (Path(kb) / "docs.jsonl").open()]


def done_summaries(p):
    out = {}
    if p.exists():
        for l in p.open():
            try:
                r = json.loads(l)
            except Exception:  # noqa: BLE001  torn last line after a kill: that document is simply redone
                continue
            if r.get("summary"):
                out[r["doc_id"]] = r["summary"]
    return out


def clean_summary(text):
    t = " ".join((text or "").split())
    return re.sub(r"^.*</think>", "", t).strip()


def cmd_summaries(kb, backbone, workers, client=None, passes=None):
    """Up to `passes` rounds over the documents still without a summary, so one bad call does not end the step."""
    import mp_llm
    passes = passes or int(os.environ.get("MP_KB_PASSES", "3"))
    p = summaries_path(kb, backbone)
    p.parent.mkdir(parents=True, exist_ok=True)
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key="local",
                        timeout=float(os.environ.get("MP_TIMEOUT", "3600")))
    lock = threading.Lock()
    out = p.open("a")
    stat = {"ok": 0, "failed": 0}

    def one(d):
        calls = mp_llm.start_case_trace()
        msg = SUMMARY_PROMPT.replace("{title}", d["title"]).replace("{alt}", "; ".join(d.get("alt_names", [])) or "none") \
            .replace("{text}", d["text"])
        try:
            comp = mp_llm.mp_create(client, [{"role": "user", "content": msg}], module="kb_summary")
            s = clean_summary(comp.choices[0].message.content)
            fin = comp.choices[0].finish_reason
        except Exception as e:  # noqa: BLE001  server trouble: not recorded as done, the next pass redoes it
            print(f"[kb-index] {d['doc_id']} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        if not (s and fin == "stop"):
            print(f"[kb-index] {d['doc_id']} unusable reply: finish {fin}, {len(s)} characters", flush=True)
            return
        with lock:
            out.write(json.dumps({"doc_id": d["doc_id"], "title": d["title"], "summary": s, "backbone": backbone,
                                  "created": utc(), "trace": calls}, ensure_ascii=False) + "\n")
            out.flush()
            stat["ok"] += 1
            if stat["ok"] % 200 == 0:
                print(f"[kb-index] summaries {stat['ok']} this run", flush=True)

    for k in range(passes):
        have = done_summaries(p)
        docs = [d for d in load_docs(kb) if d["doc_id"] not in have]
        print(f"[kb-index] summaries pass {k + 1}/{passes}: {len(have)} done, {len(docs)} to do, workers {workers}", flush=True)
        if not docs:
            break
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(one, d) for d in docs]):
                f.result()
    out.close()
    have = done_summaries(p)
    stat["failed"] = sum(1 for d in load_docs(kb) if d["doc_id"] not in have)
    print(f"[kb-index] summaries done {stat}", flush=True)
    return stat


def cmd_embed(kb, embedder=None, shard=SHARD, batch=128):
    import numpy as np
    import kb_rag
    kb = Path(kb)
    emb = embedder or kb_rag.make_embedder()
    if emb.name != kb_rag.embedder_name():
        raise SystemExit(f"[kb-index] embedder {emb.name!r} != MP_EMBEDDER {kb_rag.embedder_name()!r}")
    d = kb_rag.emb_dir(kb)
    d.mkdir(parents=True, exist_ok=True)
    texts = [json.loads(l)["text"] for l in chunks_path(kb).open()]
    n_shards = (len(texts) + shard - 1) // shard
    t0, n_new = time.time(), 0
    for k in range(n_shards):
        f = d / f"shard_{k:05d}_of_{n_shards:05d}_size{shard}.npy"
        if f.exists():
            continue
        v = emb.encode(texts[k * shard:(k + 1) * shard], query=False, batch_size=batch).astype(getattr(emb, "store_dtype", np.float16))
        tmp = f.with_name(f.name + f".tmp{os.getpid()}.npy")
        np.save(tmp, v)
        os.replace(tmp, f)
        n_new += 1
        print(f"[kb-index] embed shard {k + 1}/{n_shards} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[kb-index] embed done: {n_shards} shards, {n_new} new, {len(texts)} chunks", flush=True)


def cmd_finalize(kb, backbone, shard=SHARD):
    import numpy as np
    import kb_rag
    kb = Path(kb)
    p = summaries_path(kb, backbone)
    have = done_summaries(p)
    docs = load_docs(kb)
    missing = [d["doc_id"] for d in docs if d["doc_id"] not in have]
    if missing:
        raise SystemExit(f"[kb-index] {len(missing)} documents lack a summary (e.g. {missing[:5]}): rerun summaries")
    profile = chunk_profile_name()
    cfile = chunks_path(kb)
    n_chunks = sum(1 for _ in cfile.open())
    chunk_meta = {"chunk_profile": profile, "chunking": {k: v for k, v in CHUNK_PROFILES[profile].items() if k != "file"},
                  "chunks_path": cfile.name}
    cm = chunks_manifest_path(kb)
    if CHUNK_PROFILES[profile]["unit"] == "token":
        if not cm.exists():
            raise SystemExit(f"[kb-index] {cm} missing: run medlineplus_kb.py chunk-tokens")
        cman = json.loads(cm.read_text())
        if cman["chunks_sha256"] != sha_path(cfile) or cman["docs_sha256"] != sha_path(kb / "docs.jsonl"):
            raise SystemExit(f"[kb-index] {cfile.name} does not match {cm.name}: rerun chunk-tokens, then embed")
    n_shards = (n_chunks + shard - 1) // shard
    shards, n = [], 0
    edir = kb_rag.emb_dir(kb)
    for k in range(n_shards):
        f = edir / f"shard_{k:05d}_of_{n_shards:05d}_size{shard}.npy"
        if not f.exists():
            raise SystemExit(f"[kb-index] missing {f}: rerun embed")
        a = np.load(f, mmap_mode="r")
        if not np.isfinite(np.asarray(a[:8], dtype=np.float32)).all():
            raise SystemExit(f"[kb-index] non-finite values in {f}")
        n += len(a)
        shards.append({"path": str(f.relative_to(kb)), "rows": len(a), "dim": int(a.shape[1]), "sha256": sha_path(f)})
    if n != n_chunks:
        raise SystemExit(f"[kb-index] {n} embeddings for {n_chunks} chunks")
    embed = kb_rag.embedder_name()
    man = {"created": utc(), "backbone": backbone, "embed_model": embed,
           "query_instruct": kb_rag.QUERY_INSTRUCT if embed == kb_rag.EMBED_MODEL else None,
           "summary_prompt": SUMMARY_PROMPT, "docs_sha256": sha_path(kb / "docs.jsonl"),
           "chunks_sha256": sha_path(cfile), **chunk_meta,
           "summaries": {"path": str(p.relative_to(kb)), "n": len(have), "sha256": sha_path(p)},
           "embeddings": {"n": n, "shards": shards}}
    mp = kb_rag.index_manifest_path(kb, backbone)
    if mp.exists():      # same index: leave the file alone
        old = json.loads(mp.read_text())
        if {k: v for k, v in old.items() if k != "created"} == {k: v for k, v in man.items() if k != "created"}:
            print(f"[kb-index] {mp} unchanged; kb_sha {kb_rag.kb_sha(kb, backbone)}", flush=True)
            return
    tmp = mp.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, mp)
    print(f"[kb-index] {mp}: {len(have)} summaries, {n} embeddings; kb_sha {kb_rag.kb_sha(kb, backbone)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summaries", "embed", "finalize"])
    ap.add_argument("--kb", default=KB_DEFAULT)
    ap.add_argument("--backbone", default=os.environ.get("MP_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    if a.cmd == "summaries":
        sys.exit(1 if cmd_summaries(a.kb, a.backbone, a.workers)["failed"] else 0)
    elif a.cmd == "embed":
        cmd_embed(a.kb, batch=a.batch)
    else:
        cmd_finalize(a.kb, a.backbone)
