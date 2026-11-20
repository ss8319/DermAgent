"""Runtime side of the MedlinePlus knowledge base (MedAgent-Pro ICLR 2026 camera-ready Sec 3.2, Figure 3).

  query -> keyword filter over the one-sentence document summaries -> vector retrieval of the top 5 chunks inside the
  remaining documents -> the VLM (backbone) summarises those chunks into a guideline.

Needs the index written by kb_index_build.py. Without it KnowledgeBase() refuses to start.
Upstream's RAG.py is a live URL fetch with LangChain + FAISS; the paper's two-step method is implemented here on a frozen
snapshot. Search is exact (numpy dot product over the rows of the filtered documents), not an ANN index.
Index = one embedder + one chunk profile: its own embedding directory and manifest, so the Qwen3-Embedding index stays.
Default: openai/text-embedding-ada-002 through OpenRouter (the OpenAIEmbeddings default of both code branches; owner exception
to the paper's PubMedBERT) over 300-token chunks. Only MedlinePlus text and 'How to diagnose <disease>?' leave the cluster.
GUIDELINE_PROMPT is the MIMIC branch's RAG.py qa_prompt without its 'Focus on what can be assessed from chest X-ray images' line.
"""
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from medlineplus_kb import chunk_profile_name, doc_tokens, keyword_match, keywords

ADA_MODEL = "openai/text-embedding-ada-002"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "Qwen3-Embedding-8B"
EMBED_PATH = ("/fs04/scratch2/ub62/ssim0070/DermAgent/model-weights/models--Qwen--Qwen3-Embedding-8B/snapshots/"
              "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af")
QUERY_INSTRUCT = "Instruct: Given a disease name, retrieve passages describing how that disease is diagnosed\nQuery: "
TOP_CHUNKS = 5
NO_DATA = "No relevant data found."
RAG_QUESTION = "How to diagnose {disease}?"     # upstream Task_level.py: rag_prompt = "How to diagnose glaucoma?"
GUIDELINE_PROMPT = (
    "You are a medical RAG expert. Based on the retrieved context, "
    "provide a concise clinical guideline for the queried condition.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Instructions:\n"
    "1. Summarise the key diagnostic indicators and clinical criteria.\n"
    "2. List the indicators in a structured way.\n"
    "3. If the context is insufficient, state 'No relevant data found.'\n\n"
    "Answer:"
)


def index_slug(backbone):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", backbone)


class LocalEmbedder:
    """Qwen3-Embedding-8B through sentence-transformers (last-token pooling + normalise, from the model's own modules.json)."""
    name = EMBED_MODEL

    def __init__(self, path=EMBED_PATH, device=None):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(path, device=device, tokenizer_kwargs={"padding_side": "left"},
                                     model_kwargs={"torch_dtype": "bfloat16"})
        self.lock = threading.Lock()

    def encode(self, texts, query=False, batch_size=32):
        with self.lock:
            kw = {"prompt": QUERY_INSTRUCT} if query else {}
            v = self.m.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, **kw)
        return np.asarray(v, dtype=np.float32)


class HttpEmbedder:
    """Same model behind an OpenAI-compatible /v1/embeddings endpoint (e.g. SGLang --is-embedding)."""
    name = EMBED_MODEL

    def __init__(self, base_url, model=None):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key="local", timeout=600)
        self.model = model or f"Qwen/{EMBED_MODEL}"

    def encode(self, texts, query=False, batch_size=64):
        out = []
        texts = [(QUERY_INSTRUCT + t) if query else t for t in texts]
        for i in range(0, len(texts), batch_size):
            r = self.c.embeddings.create(model=self.model, input=texts[i:i + batch_size])
            out += [d.embedding for d in r.data]
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class EmbedError(RuntimeError):
    """The embedding request failed for good. The text never holds the API key."""


class OpenRouterEmbedder:
    """openai/text-embedding-ada-002 through OpenRouter. Batched; transient errors (429, 408, 5xx, connection, timeout, a 200 with
    an error body or a short result) are retried with backoff; anything else (400, 401, 403) stops at once. ada-002 takes no
    query instruction, so queries and chunks are sent as they are. Vectors are stored float32: ada-002 similarities sit in a
    narrow band, where float16 rounding could reorder close chunks."""
    name = ADA_MODEL
    store_dtype = np.float32
    TRANSIENT = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}

    def __init__(self, api_key=None, url=None, model=ADA_MODEL, retries=None, timeout=None, sleep=time.sleep):
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY") or ""
        if not self.key or self.key == "local":     # the GPU holder exports OPENROUTER_API_KEY=local for its own server
            raise SystemExit("[kb] OPENROUTER_API_KEY is unset or 'local': the embedder needs the real key (run_medagentpro_v1.sh loads it)")
        self.url = url or os.environ.get("MP_OPENROUTER_EMBED_URL") or OPENROUTER_EMBED_URL
        self.model = model
        self.retries = int(os.environ.get("MP_EMBED_RETRIES", "6")) if retries is None else retries
        self.timeout = float(os.environ.get("MP_EMBED_TIMEOUT", "120")) if timeout is None else timeout
        self.sleep = sleep

    def _post(self, texts):
        body = json.dumps({"model": self.model, "input": list(texts)}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _batch(self, texts):
        last = None
        for attempt in range(self.retries + 1):
            if attempt:
                self.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random() / 2))
            try:
                d = self._post(texts)
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code not in self.TRANSIENT:
                    raise EmbedError(f"embedding request refused: {last}") from None
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as e:
                last = type(e).__name__
                continue
            rows = d.get("data") if isinstance(d, dict) else None
            if not isinstance(rows, list) or len(rows) != len(texts):
                last = f"bad reply: {len(rows) if isinstance(rows, list) else 'no'} rows for {len(texts)} texts"
                continue
            rows = sorted(rows, key=lambda x: x.get("index", 0))
            return [x["embedding"] for x in rows]
        raise EmbedError(f"embedding failed after {self.retries + 1} tries: {last}")

    def encode(self, texts, query=False, batch_size=128):
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            out += self._batch(texts[i:i + batch_size])
        v = np.asarray(out, dtype=np.float32)
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        if v.ndim != 2 or not np.isfinite(v).all() or (nrm == 0).any():
            raise EmbedError(f"unusable embeddings: shape {v.shape}")
        return v / nrm


def embedder_name():
    return os.environ.get("MP_EMBEDDER") or ADA_MODEL


def make_embedder():
    """MP_EMBEDDER: ada-002 through OpenRouter (default), or Qwen3-Embedding-8B (MP_EMBED_URL endpoint, else local weights)."""
    n = embedder_name()
    if n == ADA_MODEL:
        return OpenRouterEmbedder()
    if n == EMBED_MODEL:
        u = os.environ.get("MP_EMBED_URL")
        return HttpEmbedder(u) if u else LocalEmbedder()
    raise SystemExit(f"[kb] unknown MP_EMBEDDER {n!r}")


def legacy_index(embed, profile):
    return embed == EMBED_MODEL and profile == "char400o50"


def index_tag(embed=None, profile=None):
    """Index identity in file names: '' for the original Qwen3-Embedding + 400-character index, whose files stay where they are."""
    embed, profile = embed or embedder_name(), chunk_profile_name(profile)
    return "" if legacy_index(embed, profile) else f"__{index_slug(embed)}__{profile}"


def emb_dir(kb, embed=None, profile=None):
    tag = index_tag(embed, profile)
    return Path(kb) / "index" / (f"emb{tag}" if tag else "emb")


def index_manifest_path(kb, backbone, embed=None, profile=None):
    return Path(kb) / "index" / f"INDEX_MANIFEST_{index_slug(backbone)}{index_tag(embed, profile)}.json"


def kb_sha(kb, backbone, embed=None, profile=None):
    """The hash that goes into run_sig and the plan-cache namespace: snapshot manifest + index manifest (summaries of this
    backbone, embedder, chunk settings, embeddings). Content only: 'created' is left out, so rerunning finalize on the same
    index keeps every cached plan."""
    h = hashlib.sha256()
    for p in (Path(kb) / "MANIFEST.json", index_manifest_path(kb, backbone, embed, profile)):
        if not p.exists():
            raise SystemExit(f"[kb] {p} missing: build the knowledge base first (medlineplus_kb.py, kb_index_build.py)")
        d = json.loads(p.read_text())
        d.pop("created", None)
        h.update(json.dumps(d, sort_keys=True).encode())
    return h.hexdigest()[:16]


class KnowledgeBase:
    def __init__(self, kb, backbone, embedder=None):
        self.kb = Path(kb)
        self.sha = kb_sha(kb, backbone)
        man = json.loads(index_manifest_path(kb, backbone).read_text())
        if embedder is not None and getattr(embedder, "name", None) != man.get("embed_model"):
            raise SystemExit(f"[kb] embedder {getattr(embedder, 'name', None)!r} != index embedder {man.get('embed_model')!r}")
        self.docs = {}
        for l in (self.kb / "docs.jsonl").open():
            d = json.loads(l)
            self.docs[d["doc_id"]] = {"title": d["title"], "url": d["url"], "source": d["source"]}
        self.summ = {}
        for l in (self.kb / man["summaries"]["path"]).open():
            r = json.loads(l)
            if r.get("summary"):
                self.summ[r["doc_id"]] = r["summary"]
        missing = set(self.docs) - set(self.summ)
        if missing:
            raise SystemExit(f"[kb] {len(missing)} documents have no summary: rerun kb_index_build.py summaries")
        self.tokens = {k: doc_tokens(v) for k, v in self.summ.items()}
        self.inv = {}
        for k, toks in self.tokens.items():
            for t in toks:
                self.inv.setdefault(t, set()).add(k)
        self.chunks = [json.loads(l) for l in (self.kb / man.get("chunks_path", "chunks.jsonl")).open()]
        self.rows_by_doc = {}
        for c in self.chunks:
            self.rows_by_doc.setdefault(c["doc_id"], []).append(c["row"])
        self.emb = np.concatenate([np.load(self.kb / s["path"]) for s in man["embeddings"]["shards"]])
        if len(self.emb) != len(self.chunks):
            raise SystemExit(f"[kb] {len(self.emb)} embeddings for {len(self.chunks)} chunks")
        self.embedder = embedder

    def keyword_filter(self, disease):
        """Documents whose one-sentence summary holds every content word of the disease name."""
        kw = keywords(disease)
        s = None
        for k in kw:
            s = set(self.inv.get(k, ())) if s is None else s & self.inv.get(k, set())
            if not s:
                return kw, []
        return kw, sorted(d for d in (s or ()) if keyword_match(kw, self.tokens[d]))

    def retrieve(self, disease):
        """-> dict with keywords, filtered documents and the top-5 chunks. No document after the filter = no chunks."""
        kw, doc_ids = self.keyword_filter(disease)
        out = {"query": RAG_QUESTION.replace("{disease}", disease), "keywords": kw, "n_docs_after_filter": len(doc_ids),
               "docs": [{"doc_id": d, "title": self.docs[d]["title"], "summary": self.summ[d]} for d in doc_ids[:50]],
               "chunks": []}
        rows = [r for d in doc_ids for r in self.rows_by_doc.get(d, [])]
        if not rows:
            return out
        q = self.embedder.encode([out["query"]], query=True)[0]
        sims = self.emb[rows].astype(np.float32) @ q
        for j in np.argsort(-sims, kind="stable")[:TOP_CHUNKS]:
            c = self.chunks[rows[int(j)]]
            d = self.docs[c["doc_id"]]
            out["chunks"].append({"row": c["row"], "doc_id": c["doc_id"], "title": d["title"], "url": d["url"],
                                  "score": round(float(sims[int(j)]), 6), "text": c["text"]})
        return out


def guideline_messages(disease, chunks):
    ctx = "\n\n".join(c["text"] for c in chunks)
    q = RAG_QUESTION.replace("{disease}", disease)
    return [{"role": "user", "content": GUIDELINE_PROMPT.replace("{context}", ctx).replace("{question}", q)}]


def is_no_data(text):
    t = (text or "").strip().strip('"').strip()
    return not t or t.lower().rstrip(".") == NO_DATA.lower().rstrip(".")
