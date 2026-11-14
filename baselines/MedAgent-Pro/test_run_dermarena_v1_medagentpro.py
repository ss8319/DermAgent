#!/usr/bin/env python3
"""CPU-only offline test of the per-candidate MedAgent-Pro runner: NO GPU, NO model, NO cluster job, NO external API.

A local HTTP stub speaks the OpenAI chat protocol (the real openai client talks to it); PanDerm / MAKE / DermoGPT are fakes
behind the REAL Dermatology/tools/skin_tools bridges; the run's embedder is a bag-of-words fake, and a second local stub stands in
for OpenRouter's embeddings endpoint. The knowledge base is a small fixture that goes through the REAL parse, chunk (real cl100k_base
tokenizer), manifest, summary, embed, finalize and retrieve code. Three real canary cases run end to end.

  case A  T1.1 pmid_41573439_1  6 images (4 clinical)  full flow; each new stage's normal, re-ask, default and fault paths
  case B  T1.2 pmid_32923268_1  4 images (1 clinical)  no usable top-10 -> unparsed = wrong, no plan, no tool, no final call
  case C  T1.2 pmid_22937450_1  2 images (histo/IHC)   no derm image; plan fallbacks (relaxed tries, no plan, not stored)
Usage: <dermagent python> test_run_dermarena_v1_medagentpro.py [--work-dir DIR]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CANARY = Path("/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx")
TIKTOKEN_CACHE = Path("/fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/medagentpro_iclr_diff_20261107/tiktoken_cache")
CASES = {"A": ("T1.1", "pmid_41573439_1"), "B": ("T1.2", "pmid_32923268_1"), "C": ("T1.2", "pmid_22937450_1")}
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5,
            "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}}
BACKBONE = "mock/backbone"
CANDS_A = ["Psoriasis", "psoriasis.", "Lupus erythematosus, discoid", "Hand, foot and mouth disease", "Lichen planus",
           "Zebra fever", "Scabies", "Unplannable disease", "Deciderfail disease", "Tinea corporis", "Nummular eczema",
           "Pityriasis rosea"]
CANDS_C = ["Psoriasis", "Zebra fever", "Unplannable disease", "Deciderfail disease", "Imageonly disease", "Flaky disease",
           "Bundled disease", "Metastep disease", "Allmeta disease", "Indicatorfail disease"]
THRESH = {"psoriasis": .2, "lupus erythematosus discoid": .5, "hand foot and mouth disease": .5, "lichen planus": .1,
          "zebra fever": .7, "scabies": .3, "tinea corporis": .9, "nummular eczema": .4, "flaky disease": .95}
FINAL_A = ["Scabies", "Psoriasis", "Nummular eczema", "Lichen planus", "Tinea corporis"]
MARGIN_A = ["Lichen planus", "Scabies", "Psoriasis", "Lupus erythematosus, discoid", "Nummular eczema"]
LEAK_TITLE = "Secret journal article title about guttate flares"


def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", s.casefold()).split())


def plan_for(token):
    q = lambda i, t, a, inp, o="final indicator": {  # noqa: E731
        "id": i, "tool": [t], "action_type": "qualitative", "action": a, "input_type": inp, "output_type": o,
        "output_path": "diagnosis.json"}
    z = lambda i, t, a: {"id": i, "tool": [t], "action_type": "quantitative", "action": a, "input_type": [0],  # noqa: E731
                         "output_type": "final indicator", "output_path": "diagnosis.json"}
    return [z(1, 2, f"Classify skin image for {token}"), z(2, 3, "Annotate scale"),
            q(3, 1, f"Assess whether zero-shot scores support {token}", [1]), q(4, 1, "Judge scale from the concept output", [2]),
            q(5, 4, "Assess lesion morphology", [0]), q(6, 1, "Assess history of onset", [0]),
            z(7, 3, "Annotate erythema"), q(8, 1, "Judge stripe pattern", [7]), q(9, 1, "Assess onset timing detail", [6])]


class Mock:
    def __init__(self, prefixes):
        self.prefixes = prefixes
        self.lock = threading.Lock()
        self.count = {}
        self.log = []
        self.fail_summary_of = None
        self.bad_token_requests = 0
        self.cot_once = True
        self.jitter = False
        self.final_mode = "ok"
        self.force_500 = None               # module name: its next call answers HTTP 500

    def which_case(self, text):
        for c, p in self.prefixes.items():
            if p in text:
                return c
        m = re.search(r"observation for case ([A-C])", text)
        return m.group(1) if m else None

    def respond(self, body):
        msgs = body["messages"]
        sysmsg = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        user = msgs[-1]["content"]
        parts = user if isinstance(user, list) else [{"type": "text", "text": user}]
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        n_img = sum(1 for p in parts if p.get("type") == "image_url")
        case = self.which_case(text)
        if "planning assistant" in sysmsg:
            mod = "planner"
        elif "reliable and accurate" in sysmsg:
            mod = "reflection"
        elif "Assign clinical importance weights" in sysmsg:
            mod = "weights"
        elif "make a decision" in sysmsg:
            mod = "vlm"
        elif "summarize" in sysmsg:
            mod = "summary"
        elif "You are a medical RAG expert" in text:
            mod = "guideline"
        elif "Summarize the disease-specific clinical indicators" in text:
            mod = "indicators"
        elif "Which steps does this patient's data support?" in text:
            mod = "memory"
        elif "was analysed separately for this patient" in text:
            mod = "final"
        elif "Write ONE sentence" in text:
            mod = "kb_summary"
        elif "10 most likely diagnoses" in text:
            mod = "candidates"
        else:
            mod = "unknown"
        if "<|image_pad|>" in text:             # what the real server does: the literal is counted as an image token
            self.bad_token_requests += 1
            return (500, None, None, None, 0)
        if self.force_500 == mod:
            self.force_500 = None
            return (500, None, None, None, 0)
        m = re.search(r"Goal: Diagnose potential (.+)", text) or re.search(r"^Disease: (.+)$", text, re.M) or \
            re.search(r"Step 1: \w+ .*? for (.+)", text) or re.search(r"^Step: .*? for (.+)$", text, re.M)
        disease = m.group(1).strip() if m else None
        with self.lock:
            key = (mod, case, disease)
            self.count[key] = self.count.get(key, 0) + 1
            n = self.count[key]
            self.log.append({"module": mod, "case": case, "disease": disease, "n_images": n_img, "text": text, "system": sysmsg,
                             "params": {k: v for k, v in body.items() if k != "messages"}})
        R = lambda content, reasoning="(mock thinking)", fin="stop", toks=50: (200, content, reasoning, fin, toks)  # noqa: E731
        d = norm(disease or "")
        if mod == "kb_summary":
            title = re.search(r"Title: (.+)", text).group(1)
            if title == "Scabies" and self.fail_summary_of:
                self.fail_summary_of -= 1
                return (500, None, None, None, 0)
            return R(f"{title} is a condition described in MedlinePlus.")
        if mod == "guideline":
            if "How to diagnose Scabies" in text:
                return R("No relevant data found.")
            return R("1. Take a history. 2. Examine the skin. 3. Confirm the typical signs.")
        if mod == "indicators":
            if d == "indicatorfail disease":
                return R("I cannot list them.")
            if d == "scabies" and n == 1:
                return R("There are several.")                     # unparsed once, then the retry works
            return R(json.dumps({"indicators": [f"Is sign {i} of {disease} present?" for i in (1, 2, 3)]}))
        if mod == "candidates":
            if case == "A" and n == 1:
                return R("", None, "stop", 1)          # immediate-EOS blank -> mp_create re-issues
            if case == "B":
                return R("I would need more information.")
            return R(json.dumps({"diagnoses": CANDS_A if case == "A" else CANDS_C}))
        if mod == "planner":
            p = plan_for(disease)
            if norm(disease) == "unplannable disease":
                return R("I cannot write this plan.")                          # never JSON: no plan at all
            if disease == "Allmeta disease":
                return R(json.dumps([{"id": 1, "tool": [1], "action_type": "qualitative", "action": f"Diagnose potential {disease}",
                                      "input_type": [0], "output_type": "final indicator", "output_path": "diagnosis.json"}]))
            if disease == "Bundled disease":
                p[3]["action"] = "assess scale, crust"                         # word check rejects until the relaxed tries
                return R(json.dumps(p))
            if disease == "Metastep disease":
                p.append({"id": 10, "tool": [1], "action_type": "qualitative", "action": f"Synthesize findings for final diagnosis of {disease}",
                          "input_type": [3, 4], "output_type": "final indicator", "output_path": "diagnosis.json"})
                return R(json.dumps(p))
            if disease == "Scabies" and n == 1:      # the real plan on disk looked like this: bare verbs
                return R(json.dumps([dict(p[0], action="classify"), dict(p[1], action="annotate"),
                                     dict(p[2], action="assess", input_type=[0, 1, 2], id=3)]))
            if disease == "Zebra fever" and n == 1:
                p[1]["action"] = "Annotate zebra stripes"                  # no MAKE concept: rejected, no silent VLM fall-through
            if disease == "Tinea corporis":
                p[0].update(tool=[2, 3], action=f"Annotate scale for {disease}")   # one step, two tools
            if disease == "Imageonly disease":
                return R(json.dumps([p[0], dict(p[2], id=2, input_type=[1]), dict(p[4], id=3)]))
            if disease == "Flaky disease" and n == 1:
                return (400, None, None, None, 0)                          # a request failure says nothing about the disease
            if disease == "Psoriasis" and n == 1:
                p[4]["action"] = "assess lesion morphology and border"      # upstream validator rejects
            if disease == "Lichen planus" and n == 1:
                p[2]["input_type"] = [0]                                   # classify step has no judgement: post-check rejects
            if d == "lupus erythematosus discoid":                          # 12 steps: the cap of 10 applies
                p += [{"id": 10, "tool": [1], "action_type": "qualitative", "action": "Assess nail pitting", "input_type": [0],
                       "output_type": "final indicator", "output_path": "diagnosis.json"},
                      {"id": 11, "tool": [1], "action_type": "qualitative", "action": "Assess scalp plaques", "input_type": [0],
                       "output_type": "final indicator", "output_path": "diagnosis.json"},
                      {"id": 12, "tool": [1], "action_type": "qualitative", "action": "Assess scalp scarring", "input_type": [11],
                       "output_type": "final indicator", "output_path": "diagnosis.json"}]
            if d == "nummular eczema":
                p[5]["action"] = "Assess vague onset history"               # summarised Uncertain: step 9 depends on it
            return R(json.dumps(p))
        if mod == "weights":
            if d == "deciderfail disease":
                return R("I am unable to weigh these indicators.")
            keys = json.loads(re.search(r"Indicator keys \(use EXACTLY\): (\[.*\])", text).group(1))
            ws = {k: {"weight": 1 / len(keys)} for k in keys}
            if d == "psoriasis":                                  # paraphrased keys must keep their weight
                ws = {"The " + k.upper() + " check": v for k, v in ws.items()}
            return R("```json\n" + json.dumps({"indicators": ws, "threshold": THRESH.get(d, .5)}) + "\n```")
        if mod == "memory":
            ids = [int(x) for x in re.findall(r"^Step (\d+):", text, re.M)]
            if d == "hand foot mouth disease":                                  # norm() drops the &
                return R(json.dumps({"supported_steps": []}))                   # nothing supported: a clear reply, honoured
            if d == "lichen planus":
                return R(json.dumps({"supported_steps": [i for i in ids if i != 2]}))    # step 4 depends on 2
            if d == "nummular eczema" and n == 1:
                return R("The plan looks fine to me.")                         # unclear once, then parsed
            if d == "zebra fever":
                return R(json.dumps({"supported_steps": [99]}))                # unknown step, twice: default keep all
            if d == "tinea corporis" and n == 1:
                return (400, None, None, None, 0)                              # request failure counts as an unclear reply
            return R(json.dumps({"supported_steps": ids}))
        if mod == "reflection":
            if d == "psoriasis":
                return R("No")                                                 # Terminate: step 1 and step 3
            if d == "zebra fever":
                return R("Hard to say from this.")                             # unclear twice: default reliable
            if d == "scabies" and n == 1:
                return R(None, "The user asks whether the scores look right. Let me think step by step. " * 6)
            if d == "nummular eczema":
                return (400, None, None, None, 0)                              # both tries: default reliable
            return R("Yes")
        if mod == "final":
            if self.final_mode == "prose":
                return R("All of these seem possible.")
            if self.final_mode == "400":
                return (400, None, None, None, 0)
            names = re.findall(r"^Candidate: (.+)$", text, re.M)
            if self.jitter:
                return R(json.dumps({"diagnoses": names[::-1][:5]}))
            if case == "A":
                if n == 1:
                    return R("I would rank these by likelihood.")              # unparsed once, the retry works
                return R(json.dumps({"diagnoses": FINAL_A}))
            return R(json.dumps({"diagnoses": names[:5]}))
        if mod == "vlm":
            tail = " The model echoed <|image_pad|> here." if "history of onset" in text else ""
            return R(f"Mock VLM observation for case {case}: erythematous scaly plaques.{tail}")
        if mod == "summary":
            if "onset timing" in text:
                return R("No")
            if "history of onset" in text:
                return R("Yes.")
            if "vague onset" in text:
                return R("Uncertain")
            if "zero-shot scores" in text:
                return R("**Yes**")
            if "Judge scale" in text:
                return R("No")
            if "nail pitting" in text or "scalp plaques" in text or "scalp scarring" in text:
                return R("Yes")
            if "morphology" in text:
                if self.cot_once:                   # blank content, thinking only: the fallback copies it in, the runner re-asks
                    self.cot_once = False
                    return R(None, "The user asks about morphology. Let me think step by step. " * 12)
                return R("uncertain")
            if "stripe" in text:
                return R("Cannot assess from the available text.")
            return R("???")
        return R("unexpected request")


def serve(mock):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            code, content, reasoning, fin, toks = mock.respond(body)
            if code != 200:
                out = json.dumps({"error": {"message": "mock 500", "type": "server_error"}}).encode()
            else:
                out = json.dumps({
                    "id": "mock", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": fin,
                                 "message": {"role": "assistant", "content": content, "reasoning_content": reasoning}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": toks, "total_tokens": 100 + toks}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ----------------------------------------------------------------------------------------------- fake tools
class FakePanDerm:
    is_initialized = True
    calls = []

    def preload(self):
        return self

    def _run(self, image_path, candidate_diseases=None):       # only the load gate calls it
        d = [x.strip() for x in candidate_diseases.split(",")]
        p = fake_softmax(self, image_path, d)
        rows = sorted(zip(d, p), key=lambda x: -x[1])[:5]
        return "Zero-shot Classification Results:\nImage: IMG\n----\n" + "\n".join(
            f"{i + 1}. {n}: {v:.2%}" for i, (n, v) in enumerate(rows))


def fake_softmax(tool, image_path, diseases):
    FakePanDerm.calls.append({"image": image_path, "diseases": list(diseases)})
    w = np.arange(len(diseases), 0, -1, dtype=float)
    return list(w / w.sum())


class FakeMAKE:
    is_initialized = True
    calls = []
    fail_once = True

    def preload(self):
        return self

    def _run(self, image_path, top_k=5, target_concepts=None):
        FakeMAKE.calls.append({"image": image_path, "concepts": target_concepts})
        if len(FakeMAKE.calls) == 3 and FakeMAKE.fail_once:
            FakeMAKE.fail_once = False
            return "Error in make_concept_annotator (concept annotation): [RuntimeError] mock"
        c = [x.strip() for x in target_concepts.split(",")]
        return ("Concept Annotation Results (Target Mode):\nImage: IMG\nEvaluated concepts: 1\n----------\n" +
                "\n".join(f"{i + 1}. {x}: 0.7{i}12" for i, x in enumerate(c)) + "\n\nDetected concepts (score > 0.5): x")


class FakeDermoGPT:
    is_initialized = True
    calls = []

    def preload(self):
        return self

    def _run(self, image_path, query=None):
        FakeDermoGPT.calls.append({"image": image_path, "query": query})
        return "DermoGPT VQA Response:\nA well demarcated scaly plaque on the skin."


class EmbedStub:
    """Stands in for OpenRouter's /embeddings. `script` holds what the next calls do: an HTTP code, 'short', or 'shuffle'."""

    def __init__(self):
        self.script, self.calls, self.auth = [], [], set()
        self.fail_from = None
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.auth.add(self.headers.get("Authorization"))
                stub.calls.append(b)
                act = stub.script.pop(0) if stub.script else None
                if stub.fail_from is not None and len(stub.calls) > stub.fail_from:
                    act = 503
                if isinstance(act, int):
                    out, code = b'{"error": {"message": "stub"}}', act
                else:
                    rows = [{"index": i, "embedding": list(FakeEmbedder().encode([t])[0]) + [0.0] * 1472} for i, t in enumerate(b["input"])]
                    if act == "short":
                        rows = rows[:-1]
                    if act == "shuffle":
                        rows = rows[::-1]
                    out, code = json.dumps({"data": [{k: (list(map(float, v)) if k == "embedding" else v) for k, v in r.items()} for r in rows],
                                            "model": b["model"]}).encode(), 200
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/api/v1/embeddings"


class FakeEmbedder:
    name = "fake"
    n_texts = 0

    def encode(self, texts, query=False, batch_size=32):
        FakeEmbedder.n_texts += len(texts)
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)


# ----------------------------------------------------------------------------------------------- knowledge-base fixture
def ency_html(title, summary, sections):
    s = "".join(f'<section><div class="section"><div class="section-header"><div class="section-title"><h2>{h}</h2></div></div>'
                f'<div class="section-body" id="s">{b}</div></div></section>' for h, b in sections)
    return (f'<html><div id="mplus-header">Navigation Menu</div><div id="d-article"><div class="page-title"><h1>{title}</h1></div>'
            f'<div class="main"><div id="ency_summary"><p>{summary}</p></div>{s}</div></div>'
            f'<div id="mplus-footer">Footer text</div></html>')


def topic_xml(rows):
    t = "".join(f'<health-topic title="{a}" url="https://medlineplus.gov/{b}.html" id="{i}" language="English">'
                f'<also-called>{c}</also-called><full-summary>&lt;p&gt;{d}&lt;/p&gt;</full-summary>'
                f'<site title="Article: {LEAK_TITLE}" url="https://www.ncbi.nlm.nih.gov/pubmed/42952621"/>'
                f'<site title="ClinicalTrials.gov: {a}" url="https://clinicaltrials.gov/search?cond={b}"/></health-topic>'
                for i, (a, b, c, d) in enumerate(rows))
    return f'<?xml version="1.0"?><health-topics total="{len(rows)}">{t}</health-topics>'


def build_kb_fixture(kb):
    import medlineplus_kb as K
    long = " ".join(f"Psoriasis sentence {i} describes red scaly plaques on elbows and knees and how doctors examine them."
                    for i in range(120))
    xml = topic_xml([("Psoriasis", "psoriasis", "Plaque psoriasis", long),
                     ("Scabies", "scabies", "Itch mite", "Scabies is an itchy mite infestation of the skin."),
                     ("Lichen Planus", "lichenplanus", "LP", "Lichen planus causes itchy flat purple bumps."),
                     ("Hand, Foot, and Mouth Disease", "hfmd", "HFMD", "Viral illness with mouth sores and a rash."),
                     ("Discoid lupus erythematosus", "dle", "DLE", "A scarring autoimmune skin disease."),
                     ("Tinea corporis", "tinea", "Ringworm", "A fungal skin infection."),
                     ("Nummular eczema", "nummular", "Discoid eczema", "Coin shaped eczema patches.")])
    topic_page = ('<html><section id="topsum_section"><p>Summary</p></section><h2>Clinical Trials</h2>'
                  '<a href="https://clinicaltrials.gov/search?cond=Psoriasis">ClinicalTrials.gov: Psoriasis</a>'
                  '<h2>Journal Articles</h2><a href="https://www.ncbi.nlm.nih.gov/pubmed/42827493">Article: ' + LEAK_TITLE + '</a>'
                  '<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/">full text</a></html>')
    page = ency_html("Psoriasis", "Psoriasis is a skin condition.", [
        ("Causes", "<p>The immune system plays a part.</p>"), ("Exams and Tests", "<p>A skin biopsy may be done.</p>"),
        ("Alternative Names", "<p>Plaque psoriasis; Guttate psoriasis</p>"), ("Images", '<a href="x">Psoriasis - arm</a>'),
        ("References", "<p>Dinulos JGH. Psoriasis. In: <em>Habif's Clinical Dermatology.</em> Philadelphia, PA: Elsevier; 2021:chap 8. PMID: 7654321</p>"),
        ("Review Date 8/5/2024", "<p>Updated by a reviewer.</p>")])
    files = {"xml/mplus_topics_2026-11-06.xml": xml, "topics/psoriasis.html": topic_page, "ency/article/000434.htm": page}
    with (kb / "fetch_log.jsonl").open("w") as lf:
        for rel, body in files.items():
            p = kb / "raw" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
            lf.write(json.dumps({"url": f"https://medlineplus.gov/{rel}", "path": rel, "status": 200, "bytes": len(body),
                                 "sha256": hashlib.sha256(body.encode()).hexdigest(), "retrieved": "2026-11-07T00:00:00+00:00"}) + "\n")
    K.cmd_parse(kb)
    K.cmd_chunk(kb)
    K.cmd_manifest(kb)
    K.cmd_chunk_tokens(kb)
    return files


# ----------------------------------------------------------------------------------------------- test
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=None)
    a = ap.parse_args()
    W = Path(a.work_dir or tempfile.mkdtemp(prefix="mp_offline_test_"))
    W.mkdir(parents=True, exist_ok=True)
    rows = {}
    for letter, (task, cid) in CASES.items():
        for l in open(CANARY / f"canary20_{task}.jsonl"):
            r = json.loads(l)
            if r["_id"] == cid:
                rows[letter] = r
    assert len(rows) == 3, rows.keys()
    for task in ("T1.1", "T1.2"):
        with open(W / f"in_{task}.jsonl", "w") as f:
            for letter, (t, _) in CASES.items():
                if t == task:
                    f.write(json.dumps(rows[letter]) + "\n")
    cfg = W / "cfg"
    shutil.copytree(HERE / "Dermatology/dermarena_dx", cfg)

    mock = Mock({c: r["case_report"][:80] for c, r in rows.items()})
    srv = serve(mock)
    os.environ.update({
        "OPENAI_BASE_URL": f"http://127.0.0.1:{srv.server_port}/v1", "OPENAI_API_KEY": "local", "MP_DATA_ROOT": str(cfg),
        "MP_MODEL": BACKBONE, "MP_SAMPLING_JSON": json.dumps(SAMPLING), "MP_MAX_TOKENS": "32768", "MP_EOS_RETRIES": "2",
        "MP_REASONING_FALLBACK": "1", "MP_SANITIZE_SPECIAL_TOKENS": "1", "MP_SERVER_ARGS": "--tp-size 2 --context-length 131072",
        "MP_HTTP_RETRIES": "0", "MP_TIMEOUT": "60", "MP_CALL_BUDGET": "40", "MP_EMBEDDER": "fake",
        "TIKTOKEN_CACHE_DIR": os.environ.get("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE)),
        "DERMAGENT_IMG_CACHE": str(W / "rgb_cache")})
    sys.path.insert(0, str(HERE))
    import kb_index_build as KI
    import medlineplus_kb as K
    import run_dermarena_v1_medagentpro as R
    M = R.mods()
    kb_rag, st = M["kb_rag"], M["skin_tools"]
    st._get_panderm = lambda: FakePanDerm()
    st._get_make = lambda: FakeMAKE()
    st._get_dermogpt = lambda: FakeDermoGPT()
    st._panderm_softmax = fake_softmax
    st._get_rag = lambda: (_ for _ in ()).throw(AssertionError("Case RAG must not be used"))
    kb_rag.make_embedder = lambda: FakeEmbedder()

    ok = []

    def check(cond, msg):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + msg, flush=True)

    # --- 0. units
    print("=== 0. units")
    for reply, want in [("Yes.", "Yes"), ("**No**", "No"), ("uncertain", "Uncertain"), ("yes, it is present", "Yes"),
                        ("Cannot assess this.", "Uncertain"), ("", "Uncertain"), (None, "Uncertain"), ("Not present", "Uncertain"),
                        ("<think>maybe no</think>\nYes", "Yes"), ("x" * 300, "Uncertain"), ("No evidence to assess", "Uncertain"),
                        ("Yes or No", "Uncertain"), ("No. It is absent.", "No"), ("Yes - clearly", "Yes")]:
        check(R.normalise_answer(reply)[0] == want, f"normalise_answer({str(reply)[:30]!r}) -> {want}")
    for action, token, keep, want in [("classify", "Psoriasis", True, False), ("annotate", "Psoriasis", False, False),
                                      ("assess", "Psoriasis", False, False), ("Assess the skin lesion image", "Psoriasis", False, False),
                                      ("Assess Psoriasis", "Psoriasis", False, False), ("Judge output of step 1", "Psoriasis", True, False),
                                      ("Classify the image for Psoriasis", "Psoriasis", True, True), ("Assess silvery scale", "Psoriasis", False, True),
                                      ("Assess itching at night", "Scabies", False, True)]:
        check(R.names_finding(action, token, keep) == want, f"content gate: {action!r} -> {want}")
    check(R.plan_token("Hand, foot and mouth disease") == "Hand foot & mouth disease" and
          R.plan_token("A / B, C") == "A/B C", "plan_token removes the three separators the validator rejects")
    check(R.normalise_name("  Behçet's   DISEASE. ") == R.normalise_name("behçets disease") and
          R.normalise_name("Hand, foot-and-mouth") == "hand foot and mouth", "normalise_name: case, punctuation, whitespace")
    cm = R.ConceptMap()
    check(cm.map("Annotate scale")[0] == ["scale"] and cm.map("Quantify erythematous skin")[0] == ["erythema"],
          "MAKE map: phrase match on key and synonym")
    check(cm.map("Annotate zebra stripes")[0] == [] and cm.map("annotate gel")[0] == [],
          "MAKE map: no concept -> none; artefact concepts excluded")
    check(cm.map("Annotate pustules")[0] == ["pustule"] and cm.map("Annotate pustule of Pustule disease", "pustule disease")[0] ==
          ["pustule"], "MAKE map: nearest term; disease name itself is ignored")
    long = " ".join(f"word{i}" for i in range(400))
    ch = K.chunk_text(long + "\n\n" + long)
    within = [(a_, b_) for a_, b_ in zip(ch, ch[1:]) if a_.split()[-1] != "word399"]     # not across the paragraph break
    check(max(map(len, ch)) <= 400 and len(ch) > 5 and all(a_.split()[-1] in b_.split()[:10] for a_, b_ in within) and
          all(len(set(a_.split()) & set(b_.split())) * 7 <= 60 for a_, b_ in within),
          "chunk_text: <= 400 characters; neighbours overlap by about 50 characters")
    check(" ".join(ch).count("word399") >= 2 and K.chunk_text("short text") == ["short text"], "chunk_text: nothing lost, short text unchanged")
    count = K.token_counter("cl100k_base")
    prose = " ".join(f"Sentence {i} says the plaques on the elbows, knees and scalp are red and scaly." for i in range(300))
    tk = K.chunk_text_tokens(prose + "\n\n" + prose, count, 300, 50)
    within = [(a_, b_) for a_, b_ in zip(tk, tk[1:]) if "Sentence 0 " not in b_[:40]]
    ov = [max(k for k in range(1, 400) if a_[-k:] == b_[:k]) if any(a_[-k:] == b_[:k] for k in range(1, 400)) else 0 for a_, b_ in within]
    check(len(tk) > 10 and max(map(count, tk)) <= 300 and min(map(count, tk[:-1])) >= 250,
          f"300-token chunks by the real cl100k_base count: {min(map(count, tk))}-{max(map(count, tk))} tokens")
    check(all(40 <= count(a_[-k:]) <= 50 for (a_, _), k in zip(within, ov)), "neighbouring chunks share about 50 tokens")
    check(all(f"Sentence {i} says" in " ".join(tk) for i in (0, 150, 299)) and K.chunk_text_tokens("short text", count, 300, 50) == ["short text"],
          "token chunks: nothing lost, short text unchanged")
    check(max(map(count, K.chunk_text_tokens("x" * 5000, count, 300, 50))) <= 300, "one very long word is still split under the limit")
    for act, want in [("Diagnose potential Yellow Nail Syndrome", True), ("synthesize_final_diagnosis", True), ("Integrate all indicators for final diagnosis", True),
                      ("assess Sweet syndrome diagnosis", True), ("determine diagnosis of Rosacea", True), ("determine_drug_hypersensitivity_diagnosis", True),
                      ("classify potential LCH diagnosis based on all prior indicators", True), ("Determine diagnostic likelihood of Papular Urticaria", True),
                      ("assess Carcinosarcoma diagnosis based on probability scores", False), ("Assess clinical history for prior primary melanoma diagnosis", False),
                      ("judge if abscess supports Dermal Sinus Tract diagnosis", False), ("Is silvery scale present?", False),
                      ("Assess whether zero-shot scores support Psoriasis", False), ("determine abnormality consistent with diabetic dermopathy", False)]:
        check(R.is_diagnosis_step(act) == want, f"diagnosis-step check: {act!r} -> {want}")
    check(R.names_finding("Does the patient have Psoriasis?", "Psoriasis", False) is False and R.names_finding("Is silvery scale present?", "Psoriasis", False),
          "question form: 'Does the patient have X?' names no finding, 'Is silvery scale present?' does")
    dec = R.make_decider(M["Pro_Decider"])("x")
    items3 = [(1, "a", "A"), (2, "b", "B")]
    w_ok = R.parse_weights(dec, '{"indicators": {"a": {"weight": 0.75}, "b": {"weight": 0.25}}, "threshold": 0.4}', "stop", items3)
    check(w_ok["by_step"] == {"1": .75, "2": .25} and w_ok["threshold"] == .4, "parse_weights: keys become step ids")
    for raw, fin, why in [('{"indicators": {"a": {"weight": 1}, "b": {"weight": 1}}}', "stop", "no threshold"), ("prose", "stop", "no JSON"),
                          ('{"indicators": {"a": {"weight": 0}, "b": {"weight": 0}}, "threshold": 0.4}', "stop", "all zero"),
                          ('{"indicators": {"a": {"weight": 1}}, "threshold": 0.4}', "stop", "an indicator without a weight"),
                          ('{"indicators": {"a": {"weight": 1}, "b": {"weight": 1}}, "threshold": 0.4}', "length", "hit max_tokens")]:
        try:
            R.parse_weights(dec, raw, fin, items3)
            check(False, f"parse_weights refuses: {why}")
        except (ValueError, TypeError, AttributeError, KeyError, json.JSONDecodeError):
            check(True, f"parse_weights refuses, no default invented: {why}")
    ws_ = {"by_step": {"1": .5, "2": .3, "3": .2}, "threshold": .4}
    rs_ = R.risk_score(dec, [{"indicator_name": "x", "step": 1, "if_abnormal": "Yes"}, {"indicator_name": "y", "step": 3, "if_abnormal": "No"}], ws_)
    check(abs(rs_["score"] - 5 / 7) < 1e-9 and abs(sum(rs_["weights"].values()) - 1) < 1e-9 and rs_["diagnosis"] == "Positive",
          "risk_score: weights rescaled over the survivors, Yes 1, No 0")
    check(R.risk_score(dec, [{"indicator_name": "x", "step": 9, "if_abnormal": "Yes"}], ws_) is None, "risk_score: survivors without weight give no score")
    check([x[1] for x in R.plan_indicators([{"id": 1, "output_type": "final indicator", "action": "Is it red?"}, {"id": 2, "output_type": "final indicator", "action": "Is it red?"},
                                            {"id": 3, "output_type": "intermediate result", "action": "x"}])] == ["is_it_red_step_1", "is_it_red_step_2"],
          "a repeated action gets a unique weight key")
    check(R.parse_reflection("Yes") == "reliable" and R.parse_reflection("No.") == "unreliable" and R.parse_reflection("Uncertain") is None and
          R.parse_reflection("") is None and R.parse_reflection("maybe") is None, "reflection reply: Yes, No, anything else unclear")
    for txt, fin, want in [('{"supported_steps": [1, 3]}', "stop", {1, 3}), ('{"supported_steps": []}', "stop", set()), ('{"supported_steps": [7]}', "stop", None),
                           ('{"supported_steps": "all"}', "stop", None), ("prose", "stop", None), ('{"supported_steps": [1]}', "length", None),
                           ('{"supported_steps": ["Step 1", "step 3"]}', "stop", {1, 3}), ('{"supported_steps": ["2", "step_3"]}', "stop", {2, 3}),
                           ('{"supported_steps": ["Step 9"]}', "stop", None), ('{"supported_steps": ["first"]}', "stop", None), ('{"supported_steps": [true]}', "stop", None)]:
        check(R.parse_memory(txt, fin, {1, 2, 3})[0] == want, f"memory reply {txt!r} ({fin}) -> {want}")
    pm, pc = K.extract_ids('href="https://www.ncbi.nlm.nih.gov/pubmed/42952621" x https://pubmed.ncbi.nlm.nih.gov/123456/ '
                           'PMID: 99 PMC1234567 https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/ ?term=%22x%22 pubmed/?term=2020')
    check(pm == {"42952621", "123456", "99"} and pc == {"PMC1234567", "PMC7654321"}, f"extract_ids finds links and citations: {pm} {pc}")

    # --- 1. knowledge base through the real code
    print("=== 1. knowledge base fixture")
    kb = W / "kb"
    kb.mkdir()
    build_kb_fixture(kb)
    docs = [json.loads(l) for l in (kb / "docs.jsonl").open()]
    blob = (kb / "docs.jsonl").read_text() + (kb / "chunks.jsonl").read_text()
    check(len(docs) == 8 and "Discoid eczema" in blob, "parse: 7 topics + 1 encyclopedia page")
    check(not any(x in blob for x in ("pubmed", "42952621", "42827493", "PMC1234567", "clinicaltrials", LEAK_TITLE, "7654321",
                                      "Dinulos", "Elsevier", "Navigation Menu", "Footer text", "Updated by a reviewer")),
          "indexed text holds no Journal Articles, Clinical Trials, References, review credits or page furniture")
    check(LEAK_TITLE in (kb / "raw/topics/psoriasis.html").read_text() and (kb / "raw/ency/article/000434.htm").exists(),
          "raw HTML kept in the snapshot")
    check(not K.check_docs(docs), "post-parse scan finds nothing forbidden")
    check(K.check_docs([{"doc_id": "x", "text": "See PMID 123 in Elsevier; 2021:chap 8"}]).keys() >= {"pmid", "publisher", "chap"},
          "the scan itself fires on citation text")
    enc = next(d for d in docs if d["source"] == "ency_article")
    check(enc["alt_names"] == ["Plaque psoriasis", "Guttate psoriasis"] and "References" in enc["sections_dropped"],
          "encyclopedia parse: alternative names kept, References dropped")
    # positive control: a zero in the real report only counts if a planted real test article is found
    kb2 = W / "kb_control"
    (kb2 / "raw/topics").mkdir(parents=True)
    t_row = json.loads(open(f"{K.RELEASE_TASKS}/T1.1_disease_screening/test.jsonl").readline())
    (kb2 / "raw/topics/control.html").write_text(
        f'<a href="https://www.ncbi.nlm.nih.gov/pubmed/{t_row["pmid"]}">Article: x</a> <a href="https://pmc.ncbi.nlm.nih.gov/articles/'
        f'{t_row["pmcid"]}/">y</a> <p>{t_row["title"]}</p>')
    K.cmd_ids(kb2)
    rep = json.loads((kb2 / "reports/pmid_pmcid_match.json").read_text())
    check(rep["n_pmid_matches"] == 1 and rep["n_pmcid_matches"] == 1 and rep["dermarena_by_split"]["test"]["pmid_matches"] == 1 and
          {m["kind"] for m in rep["test_val_matches"]} == {"pmid", "pmcid"} and "test" in rep["title_matches"][0]["splits"],
          "ID matcher positive control: planted test article found by PMID, PMCID and title")
    mock.fail_summary_of = 2
    from openai import OpenAI
    cl = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key="x", max_retries=0)
    s1 = KI.cmd_summaries(kb, BACKBONE, 2, client=cl, passes=1)
    n1 = len(mock.log)
    check(s1["failed"] == 1 and s1["ok"] == 7, "summaries: with one pass, a server error leaves that document undone")
    try:
        KI.cmd_finalize(kb, BACKBONE, shard=16)
        check(False, "finalize refuses with a summary missing")
    except SystemExit as e:
        check("lack a summary" in str(e), f"finalize refuses with a summary missing: {e}")
    s2 = KI.cmd_summaries(kb, BACKBONE, 2, client=cl)
    check(s2 == {"ok": 1, "failed": 0} and len(mock.log) == n1 + 2, "summaries retry: the same document fails once more, the next pass gets it")
    n_chunks = sum(1 for _ in (kb / "chunks.jsonl").open())
    n_chunks = sum(1 for _ in K.chunks_path(kb).open())
    KI.cmd_embed(kb, embedder=FakeEmbedder(), shard=4)
    first = FakeEmbedder.n_texts
    edir = kb_rag.emb_dir(kb)
    victim = sorted(edir.glob("shard_*.npy"))[0]
    victim.unlink()
    KI.cmd_embed(kb, embedder=FakeEmbedder(), shard=4)
    check(first == n_chunks and FakeEmbedder.n_texts - first == 4, "embed resume: only the missing shard is redone")
    check(edir.name == "emb__fake__tok300o50_cl100k_base" and not (kb / "index/emb").exists(), "embeddings go to the index's own directory, not index/emb")
    KI.cmd_finalize(kb, BACKBONE, shard=4)
    sha0 = kb_rag.kb_sha(kb, BACKBONE)
    idx = kb_rag.index_manifest_path(kb, BACKBONE)
    d0 = json.loads(idx.read_text())
    idx.write_text(json.dumps(dict(d0, created="2030-01-01"), indent=1))     # as if finalize had been rerun later
    check(kb_rag.kb_sha(kb, BACKBONE) == sha0, "rerunning finalize keeps kb_sha, so cached plans are not orphaned")
    KI.cmd_finalize(kb, BACKBONE, shard=4)
    check(json.loads(idx.read_text())["created"] == "2030-01-01", "finalize on an unchanged index leaves the manifest alone")
    man = json.loads(idx.read_text())
    check(man["embed_model"] == "fake" and man["chunking"] == {"unit": "token", "tokenizer": "cl100k_base", "size": 300, "overlap": 50} and
          man["chunks_path"] == "chunks_tok300o50_cl100k_base.jsonl" and man["query_instruct"] is None, "index manifest names the embedder and the chunk settings")
    check(idx.name == "INDEX_MANIFEST_mock_backbone__fake__tok300o50_cl100k_base.json" and
          kb_rag.index_manifest_path(kb, BACKBONE, kb_rag.EMBED_MODEL, "char400o50").name == "INDEX_MANIFEST_mock_backbone.json",
          "new index has its own manifest; the Qwen3-Embedding manifest keeps its old name")
    print("=== 1b. chunks in a new file; leak scan re-runs on them")
    c_old = K.sha_path(kb / "chunks.jsonl")
    m_old = K.sha_path(kb / "MANIFEST.json")
    cm0 = json.loads(K.chunks_manifest_path(kb).read_text())
    K.cmd_chunk_tokens(kb)
    check(K.sha_path(kb / "chunks.jsonl") == c_old and K.sha_path(kb / "MANIFEST.json") == m_old, "chunk-tokens leaves chunks.jsonl and MANIFEST.json alone")
    check(cm0["chunks_sha256"] == K.sha_path(K.chunks_path(kb)) and cm0["tokens_max"] <= 300 and cm0["n_chunks"] == n_chunks, "chunk manifest: hash, counts")
    try:
        K.cmd_chunk_tokens(kb, "char400o50")
        check(False, "chunk-tokens refuses the legacy profile")
    except SystemExit:
        check(True, "chunk-tokens refuses the legacy profile, so chunks.jsonl cannot be overwritten")
    good = K.chunks_path(kb).read_text()
    K.chunks_path(kb).write_text(good + good.splitlines()[0] + "\n")
    try:
        KI.cmd_finalize(kb, BACKBONE, shard=4)
        check(False, "finalize refuses a chunk file that changed after chunk-tokens")
    except SystemExit as e:
        check("does not match" in str(e), f"finalize refuses a chunk file that changed after chunk-tokens: {str(e)[:70]}")
    K.chunks_path(kb).write_text(good)
    kl = W / "kb_leak"
    shutil.copytree(kb, kl)
    (kl / "reports").mkdir(exist_ok=True)
    (kl / "reports/leak_scan.json").write_text("{\"old\": true}")
    probe = W / "leak_probe.jsonl"
    probe.write_text(json.dumps({"_id": "planted", "case_report": "Patient note. Psoriasis sentence 5 describes red scaly plaques on elbows and knees and how doctors examine them."}) + "\n")
    keep = K.dermarena_files
    K.dermarena_files = lambda: [("T1.1_x", "test", probe)]
    K.cmd_leakscan(kl, "tok300o50_cl100k_base")
    K.dermarena_files = keep
    lrep = json.loads((kl / "reports/leak_scan_tok300o50_cl100k_base.json").read_text())
    check(lrep["n_hits_in_indexed"] >= 1 and lrep["chunk_profile"] == "tok300o50_cl100k_base" and (kl / "reports/leak_scan_hits_tok300o50_cl100k_base.jsonl").exists(),
          "leak scan positive control: planted case text found in the new chunks, reported under the profile's name")
    check(json.loads((kl / "reports/leak_scan.json").read_text()) == {"old": True}, "the earlier leak_scan.json is left as it was")
    print("=== 1c. OpenRouter embedder (local stub, fake key)")
    stub = EmbedStub()
    secret = "sk-or-TESTKEY-not-real"
    E = lambda **k: kb_rag.OpenRouterEmbedder(api_key=secret, url=stub.url, sleep=lambda s_: None, retries=3, **k)  # noqa: E731
    stub.script = [429, 500, "short", "shuffle"]
    v = E().encode(["psoriasis scale", "itchy mite burrow", "flat purple bumps"], batch_size=128)
    check(len(stub.calls) == 4 and v.shape == (3, 1536) and abs(float(np.linalg.norm(v[0])) - 1) < 1e-5,
          "429, 500 and a short result are retried; 1536 dims; unit norm")
    check(float(v[0] @ E().encode(["psoriasis scale"])[0]) > .999, "a reply out of order is put back by its index")
    check(stub.calls[0]["model"] == "openai/text-embedding-ada-002" and stub.auth == {f"Bearer {secret}"}, "model name and bearer key reach the endpoint")
    stub.calls.clear()
    E().encode([f"text {i}" for i in range(5)], batch_size=2)
    check([len(c_["input"]) for c_ in stub.calls] == [2, 2, 1], "batched")
    stub.calls.clear()
    E().encode(["How to diagnose Psoriasis?"], query=True)
    check(stub.calls[0]["input"] == ["How to diagnose Psoriasis?"], "a query goes as it is: no instruction, nothing but the disease question")
    for script, want, label in [([401], 1, "401 stops at once"), ([503] * 4, 4, "503 gives up after the retries")]:
        stub.calls.clear()
        stub.script = list(script)
        try:
            E().encode(["x"])
            check(False, label)
        except kb_rag.EmbedError as e:
            check(len(stub.calls) == want and secret not in str(e) and secret not in repr(e), f"{label}, and the key is not in the error: {e}")
    for bad in ("", "local"):
        os.environ["OPENROUTER_API_KEY"] = bad
        try:
            kb_rag.OpenRouterEmbedder()
            check(False, f"key {bad!r} refused")
        except SystemExit as e:
            check(True, f"key {bad!r} refused (the holder exports 'local'): {str(e)[:60]}")
    os.environ.pop("OPENROUTER_API_KEY")
    ka = W / "kb_ada"
    shutil.copytree(kb, ka, ignore=shutil.ignore_patterns("emb__*", "INDEX_MANIFEST*"))
    (ka / "index/emb").mkdir()
    old = {"INDEX_MANIFEST_mock_backbone.json": "{}", "emb/shard_00000_of_00001_size4096.npy": "qwen"}
    for k_, v_ in old.items():
        (ka / "index" / k_).write_text(v_)
    os.environ["MP_EMBEDDER"] = kb_rag.ADA_MODEL
    ada = E()
    stub.calls.clear()
    stub.script = []
    stub.fail_from = 3
    try:
        KI.cmd_embed(ka, embedder=ada, shard=4, batch=2)
        check(False, "a failing endpoint stops the embed step")
    except kb_rag.EmbedError:
        done1 = sorted(p_.name for p_ in kb_rag.emb_dir(ka).glob("*.npy"))
        check(len(done1) == 1, f"a failing endpoint stops the embed step; the finished shard stays: {done1}")
    stub.fail_from = None
    stub.calls.clear()
    KI.cmd_embed(ka, embedder=ada, shard=4, batch=2)
    resent = {t_ for c_ in stub.calls for t_ in c_["input"]}
    first4 = {json.loads(l)["text"] for l in list(K.chunks_path(ka).open())[:4]}
    check(not resent & first4 and len(list(kb_rag.emb_dir(ka).glob("*.npy"))) == (n_chunks + 3) // 4, "index build resumes: finished chunks are not sent again")
    KI.cmd_finalize(ka, BACKBONE, shard=4)
    a_man = json.loads(kb_rag.index_manifest_path(ka, BACKBONE).read_text())
    a_emb = np.load(ka / a_man["embeddings"]["shards"][0]["path"])
    check(a_man["embed_model"] == kb_rag.ADA_MODEL and a_emb.dtype == np.float32 and a_emb.shape[1] == 1536, "ada index: manifest names the model; float32, 1536 dims")
    check(all((ka / "index" / k_).read_text() == v_ for k_, v_ in old.items()), "the Qwen3-Embedding index files are not touched")
    a_sha = kb_rag.kb_sha(ka, BACKBONE)
    os.environ["MP_EMBEDDER"] = "fake"
    shutil.copy(kb / "index" / idx.name, ka / "index" / idx.name)
    check(a_sha != kb_rag.kb_sha(ka, BACKBONE), "kb_sha differs by embedder, so plan namespace and run_sig do too")
    stub.srv.shutdown()
    base = kb_rag.KnowledgeBase(kb, BACKBONE, FakeEmbedder())
    r = base.retrieve("Psoriasis")
    check(len(r["chunks"]) == 5 and {c["title"] for c in r["chunks"]} <= {"Psoriasis"} and r["n_docs_after_filter"] == 2 and
          all(count(c["text"]) <= 300 for c in r["chunks"]) and max(count(c["text"]) for c in r["chunks"]) > 200,
          "retrieve: keyword filter then top-5 chunks, only inside the filtered documents")
    check(base.retrieve("Zebra fever")["chunks"] == [], "retrieve: no document after the filter = no chunks")

    # --- 2. plan cache under concurrency
    print("=== 2. plan cache concurrency")
    pc = R.PlanCache(W / "race", "ns")
    wins, seen = [], []

    def race(i):
        e, ours = pc.write("psoriasis", {"who": i, "pad": "x" * 2_000_000})
        wins.append(ours)
        seen.append(e["who"])

    th = [threading.Thread(target=race, args=(i,)) for i in range(8)]
    [t.start() for t in th]
    [t.join() for t in th]
    check(sum(wins) == 1 and len(set(seen)) == 1, "8 threads: one writer wins, all read the same complete entry")
    code = ("import sys,json;sys.path.insert(0,%r);import run_dermarena_v1_medagentpro as R;"
            "c=R.PlanCache(%r,'ns2');e,o=c.write('k',{'who':sys.argv[1],'pad':'y'*3000000});print(e['who'],o)") % (str(HERE), str(W / "race"))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    ps = [subprocess.Popen([sys.executable, "-c", code, str(i)], stdout=subprocess.PIPE, text=True, env=env) for i in range(4)]
    outs = [p.communicate()[0].strip().split()[-2:] for p in ps]
    check(len({o[0] for o in outs}) == 1 and sum(o[1] == "True" for o in outs) == 1 and not list((W / "race/ns2/_tmp").iterdir()),
          f"4 processes: one winner, no torn file, no temp left: {outs}")

    # --- 3. run
    def run(*argv):
        sys.argv = ["runner", *argv]
        R.main()

    out, cache = W / "out", W / "plan_cache"
    common = ["--kb", str(kb), "--plan-cache", str(cache), "--backbone", BACKBONE]
    io_args = ["--tasks", "T1.1,T1.2", "--input-tmpl", str(W / "in_{task}.jsonl"), "--out-tmpl", str(out / "{task}")]
    print("=== 3. run 3 cases")
    n_before = len(mock.log)
    FakePanDerm.calls.clear()
    FakeMAKE.calls.clear()
    FakeDermoGPT.calls.clear()
    run(*common, *io_args, "--shard", "0/1", "--threads", "3")
    print("=== 4. merge")
    run("--merge", *io_args)
    log = mock.log[n_before:]

    recs = {}
    for t in ("T1.1", "T1.2"):
        for l in open(out / t / "predictions.jsonl"):
            x = json.loads(l)
            recs[x["_id"]] = x
    A, B, C = (recs.get(CASES[x][1]) for x in "ABC")
    check(A and B and C and len(recs) == 3, "3 records merged")
    trA = json.loads((out / "T1.1/traces" / f"{A['_id']}.json").read_text())
    pcs = {p["name"]: p for p in trA["per_candidate"]}
    print(json.dumps({k: {f: r[f] for f in ("parsed", "unparsed_reason", "n_candidates", "n_plans_new", "n_plans_cached", "n_plans_uncached",
                                            "n_plans_failed", "n_no_plan", "n_no_guideline", "n_unscored", "n_no_evidence", "n_llm_calls", "n_tool_calls",
                                            "n_tool_errors", "call_budget_exceeded")} for k, r in recs.items()}, indent=1))
    print("A prediction:", A["prediction"])
    print("A margin ranking:", [(x["name"], x["margin"]) for x in trA["margin_ranking"]])

    # candidates
    check(A["backbone_reissues"] == {"immediate_eos": 1}, "A: immediate-EOS blank re-issued once")
    check(A["candidates"] == [c for c in CANDS_A if c not in ("psoriasis.", "Pityriasis rosea")] and A["n_candidates"] == 10,
          "A: top-10 kept with original names; duplicate after normalisation removed; list stops at 10")
    cr = [x for x in log if x["module"] == "candidates" and x["case"] == "A"]
    check(cr[-1]["n_images"] == 6 and rows["A"]["case_report"][:200] in cr[-1]["text"] and "10 most likely diagnoses" in cr[-1]["text"],
          "A: candidates call = all 6 images + case text")
    # stage 1: plans, cache, guideline, indicators, weights
    check(A["n_plans_new"] == 8 and A["n_plans_cached"] == 0 and A["n_plans_uncached"] == 2, "A: 8 plans stored, 2 not stored (no plan; no weights)")
    check(C["n_plans_cached"] == 2 and C["n_plans_new"] == 3 and C["n_plans_uncached"] == 5,
          "C: 2 from the cache, 3 new, 5 not stored (no plan x2, no weights, indicators failed, 400)")
    files = {json.loads(p.read_text())["key"]: json.loads(p.read_text()) for p in cache.glob("*/*.json")}
    ps = files["psoriasis"]
    check(ps["status"] == "ok" and [x["ok"] for x in ps["attempts"]] == [False, True] and ps["attempts"][0]["stage"] == "validator" and
          "bundle" in ps["attempts"][0]["error"], "Psoriasis: validator rejection re-asked")
    pl_all = [x for x in log if x["module"] == "planner" and x["disease"] == "Psoriasis"]
    check(next(x for x in log if x["module"] == "planner" and norm(x["disease"]) == "unplannable disease" and "not valid JSON" in x["text"]) and
          "rejected" not in pl_all[0]["text"] and "Your previous plan was rejected: Qualitative step id=5 seems to bundle" in pl_all[1]["text"] and
          ps["attempts"][1]["retry_reason"] == pl_all[1]["text"].split("rejected: ")[1].split("\nReturn a corrected plan.")[0],
          "item 5: the first prompt is unchanged, the retry states the rejection reason")
    lp = files["lichen planus"]
    check([x.get("stage") for x in lp["attempts"]] == ["postcheck", None] and "Classify skin" in lp["attempts"][0]["error"],
          "B7: 'classify' tool step without a judgement rejected by the post-check, then re-asked")
    check(all(s["output_type"] == "intermediate result" for s in lp["plan"] if s["action_type"] == "quantitative"),
          "B7: tool steps labelled final indicator become intermediate")
    hf = files["hand foot and mouth disease"]
    lu = files["lupus erythematosus discoid"]
    check(hf["status"] == "ok" and len(hf["attempts"]) == 1 and "Hand foot & mouth disease" in hf["plan"][2]["action"] and
          lu["status"] == "ok" and len(lu["attempts"]) == 1 and "Lupus erythematosus discoid" in lu["plan"][2]["action"],
          "B7: names with comma / 'and' pass the unchanged validator first time through the sanitised token")
    check(ps["has_guideline"] and len(ps["retrieval"]["chunks"]) == 5 and ps["guideline"].startswith("1. Take") and ps["llm_calls"] and
          ps["plan"] and ps["plan_sha"] and ps["schema"] == R.PLAN_SCHEMA, "cache entry stores retrieved text, guideline, plan and trace")
    gl = next(x for x in log if x["module"] == "guideline")
    check("You are a medical RAG expert" in gl["text"] and "Summarise the key diagnostic indicators" in gl["text"] and "List the indicators in a structured way" in gl["text"] and
          "X-ray" not in gl["text"] and "chest" not in gl["text"].lower() and "How to diagnose" in gl["text"], "item 3: MIMIC guideline wording, without the chest X-ray line")
    ig = next(x for x in log if x["module"] == "indicators" and x["disease"] == "Psoriasis")
    check("Clinical guideline:\n1. Take" in ig["text"] and "3-7 indicators" in ig["text"] and "answered with Yes or No" in ig["text"] and ig["n_images"] == 0,
          "item 4 + A2: indicators from G; 3-7, Yes/No questions")
    iz = next(x for x in log if x["module"] == "indicators" and x["disease"] == "Zebra fever")
    check("Clinical guideline" not in iz["text"] and "3-7 indicators" in iz["text"], "item 4: no G -> indicators from the disease name alone")
    check(ps["indicators"][0] == "Is sign 1 of Psoriasis present?" and ps["indicator_status"] == "ok" and ps["indicator_step"]["prompt"] == ig["text"], "indicators and their prompt stored in the entry")
    check(len(files["scabies"]["indicator_step"]["attempts"]) == 2 and files["scabies"]["indicator_status"] == "ok", "unparsed indicator reply re-asked")
    pl = pl_all[-1]
    check("RAG text:\n1. Take a history" in pl["text"] and "Clinical indicators:\n- Is sign 1 of Psoriasis present?" in pl["text"] and "3-7 final indicators" in pl["text"] and
          "answered with Yes or No" in pl["text"] and "Goal: Diagnose potential Psoriasis" in pl["text"], "item 4 + A2: planner receives G, I and both indicator rules; the goal names the disease")
    check(not any("5 most likely" in x["text"] or "Yize" in x["text"] for x in log if x["module"] == "planner"),
          "planner has no old global-goal text")
    zf = files["zebra fever"]
    zp = [x for x in log if x["module"] == "planner" and x["disease"] == "Zebra fever"][-1]
    check(not zf["has_guideline"] and zf["guideline_status"] == "no_document" and zf["status"] == "ok" and "RAG text:\nClinical indicators:\n- Is sign 1" in zp["text"],
          "no guideline found: planner gets the indicators alone")
    check(zf["attempts"][0]["stage"] == "postcheck" and "names no MAKE concept" in zf["attempts"][0]["error"] and len(zf["attempts"]) == 2,
          "MAKE step with no mappable concept rejected, Planner re-asked")
    check(files["scabies"]["attempts"][0]["stage"] == "postcheck" and "vacuous" in files["scabies"]["attempts"][0]["error"],
          "bare-verb plan (classify / annotate / assess) rejected, Planner re-asked")
    check(ps["tool_steps"] == {"panderm": 1, "make": 2, "backbone": 5, "dermogpt": 1}, f"per-tool step count stored: {ps['tool_steps']}")
    trC = json.loads((out / "T1.2/traces" / f"{C['_id']}.json").read_text())
    pcC = {p["name"]: p for p in trC["per_candidate"]}
    check("flaky disease" not in files and pcC["Flaky disease"]["plan_status"] == "transient",
          "a 400 during planning is not stored; that candidate has no plan in this case only")
    check(pcs["Zebra fever"]["has_guideline"] is False and
          next(x for x in A["candidate_results"] if x["name"] == "Zebra fever")["has_guideline"] is False, "has_guideline=false recorded per case")
    sc = files["scabies"]
    check(not sc["has_guideline"] and sc["guideline_status"] == "no_relevant_data" and sc["retrieval"]["chunks"],
          "guideline step says 'No relevant data found' -> has_guideline false, retrieved text still stored")
    # item 5: fallbacks, nothing failed is stored
    bd = files["bundled disease"]
    check(bd["status"] == "ok" and bd["relaxed"] and len(bd["attempts"]) == 6 and [x["relaxed"] for x in bd["attempts"]] == [False] * 5 + [True] and
          bd["plan"][3]["action"] == "assess scale, crust", "item 5: after the normal tries the word check is skipped and the plan stands")
    ms = files["metastep disease"]
    check(ms["status"] == "ok" and ms["relaxed"] and "is the diagnosis itself" in ms["attempts"][0]["error"] and not any(s_["id"] == 10 for s_ in ms["plan"]) and
          ms["plan_notes"] and ms["n_final_indicators"] == 6, "item 5: a diagnosis step is rejected first, then removed since real indicators remain")
    check("allmeta disease" not in files and pcC["Allmeta disease"]["plan_status"] == "failed" and
          pcC["Allmeta disease"]["no_evidence_reason"] == "no_plan", "item 5: only diagnosis steps -> no plan, not stored, handled as no evidence")
    check("unplannable disease" not in files and pcs["Unplannable disease"]["plan_status"] == "failed" and pcs["Unplannable disease"]["score"] is None and
          sum(1 for x in log if x["module"] == "planner" and norm(x["disease"]) == "unplannable disease") == 14,
          "item 5: a failed plan is not stored as final: tried again (5 + 2) on the next case")
    fdir = next(cache.glob("*/_failed"))
    failed_files = [json.loads(p_.read_text()) for p_ in fdir.glob("*.json")]
    check(sum(e_["key"] == "unplannable disease" for e_ in failed_files) == 2 and all(not e_["cacheable"] for e_ in failed_files), "failed entries kept for audit in _failed/, never read as plans")
    ifl = next(e_ for e_ in failed_files if e_["key"] == "indicatorfail disease")
    check("indicatorfail disease" not in files and ifl["indicator_status"] == "unparsed" and ifl["status"] == "ok" and pcC["Indicatorfail disease"]["plan_status"] == "ok",
          "indicators unparsed: the plan from G serves this case, the entry is not stored")
    # item 11: weights
    w = ps["weights"]
    check(w["status"] == "ok" and w["threshold"] == .2 and abs(sum(w["by_step"].values()) - 1) < 1e-9 and set(w["by_step"]) == {"3", "4", "5", "6", "8", "9"},
          "item 11: weights by indicator step and threshold cached with the plan")
    wr = [x for x in log if x["module"] == "weights"]
    wp = next(x for x in wr if x["disease"] == "Psoriasis")
    check("Clinical guideline:\n1. Take" in wp["text"] and "decision threshold in [0, 1]" in wp["text"] and "role" not in wp["text"] and "CXR" not in wp["system"] + wp["text"] and
          "X-ray" not in wp["system"] + wp["text"], "item 11: MIMIC wording, guideline in, no roles, no chest X-ray text, threshold range [0, 1]")
    check(all(x["n_images"] == 0 and not any(r_["case_report"][:80] in x["text"] for r_ in rows.values()) and "Mock VLM" not in x["text"] for x in wr), "item 11: no patient data in the weights call")
    check("Clinical guideline" not in next(x for x in wr if x["disease"] == "Zebra fever")["text"], "item 11: no G -> from the indicators alone")
    check(sum(1 for x in wr if x["disease"] == "Psoriasis") == 1, "item 11: one weights call per disease, not per patient")
    check(len(w["info"]["remapped"]) == 6, "paraphrased indicator keys keep their weight (6 remapped)")
    dfc = pcs["Deciderfail disease"]
    check("deciderfail disease" not in files and dfc["plan_source"] == "uncached" and dfc["plan_status"] == "ok" and dfc["score"] is None and dfc["evidence"],
          "weights unparsed: no score in the comparison arm, entry not stored, the evidence still reaches the final call")
    # level 2: a candidate where nothing was removed
    sc_steps = {s["id"]: s for s in pcs["Scabies"]["steps"]}
    check(isinstance(sc_steps[1]["output"], dict) and len(sc_steps[1]["output"]["images"]) == 4 and
          len(sc_steps[1]["output"]["images"][0]["scores"]) == 10, "B2/B8: PanDerm step stored under step_1 as JSON, all images, all 10 scores")
    dj = json.loads((out / "T1.1/cases" / A["_id"] / "cand_00/diagnosis.json").read_text())
    check(not {k for k in dj if not k.startswith(("step_", "_raw_step_", "_raw_"))} and "step_1" in dj and "step_2" in dj,
          "B2: every key in diagnosis.json is a step key")
    p3 = sc_steps[3]["prompt"]
    check(all(R.panderm_name(c) in p3 or c in p3 for c in A["candidates"]) and "MAKE" not in p3,
          "B6: indicator step sees all 10 PanDerm scores, and only its own dependency (no MAKE output)")
    p4 = sc_steps[4]["prompt"]
    check("MAKE concept presence" in p4 and "PanDerm" not in p4 and rows["A"]["case_report"][:200] in p4,
          "B2/B4: step without input 0 gets its own tool output + the case text")
    v4 = [x for x in log if x["module"] == "vlm" and "Judge scale" in x["text"] and x["case"] == "A"]
    check(v4 and all(x["n_images"] == 6 for x in v4), "B4: and all 6 images")
    check(sc_steps[2]["make_map"] == {"concepts": ["scale"], "how": "phrase"} and sc_steps[2]["output"]["concepts"] == ["scale"],
          "MAKE in target mode with the mapped concept")
    check(all(c["concepts"] != "" and c["concepts"] is not None for c in FakeMAKE.calls), "MAKE never ran in default mode")
    check(not any("zebra stripes" in x["text"] for x in log if x["module"] == "vlm"), "no MAKE step falls through to the VLM")
    ti = {s_["id"]: s_ for s_ in pcs["Tinea corporis"]["steps"]}
    check(ti[1]["output"]["tool"] == "multi" and [x["tool"] for x in ti[1]["output"]["parts"]] == ["panderm", "make"] and
          "PanDerm zero-shot" in ti[3]["prompt"] and "MAKE concept presence" in ti[3]["prompt"],
          "two-tool step [2, 3]: both outputs kept and both reach the judgement step")
    check(not any(m_ in x["text"] for x in log if x["module"] in ("vlm", "summary", "memory", "reflection", "final", "weights", "indicators", "planner", "candidates") for m_ in
                  ("clinical_photo", "dermoscopy)", "histopathology)", "(radiology")), "image modality never reaches the backbone through labels")
    check(all(im["label"].startswith("image ") and "(" not in im["label"] for im in trA["images"]), "tool labels are 'image k'")
    check(sc_steps[5]["kinds"] == ["dermogpt"] and "A well demarcated" in sc_steps[5]["output"] and
          not any("Assess lesion morphology" in x["text"] for x in log if x["module"] == "vlm"),
          "B3: DermoGPT step routed by tool id; the backbone VLM did not take it")
    derm_paths = {im["path"] for im in trA["images"] if im["tool_usable"]}
    smoke = R.smoke_image()                      # the load gate's own calls are in the fakes' logs
    seen = {n_: {c["image"] for c in k_.calls} - {smoke} for n_, k_ in
            (("panderm", FakePanDerm), ("make", FakeMAKE), ("dermogpt", FakeDermoGPT))}
    check(len(derm_paths) == 4 and all(v == derm_paths for v in seen.values()),
          "PanDerm, MAKE and DermoGPT saw exactly the 4 derm images of case A, none of its 2 other images, nothing from C")
    pan_A = [c for c in FakePanDerm.calls if c["image"] in derm_paths and len(c["diseases"]) == 10]
    check(len(pan_A) == 4 and all("," not in d for c in pan_A for d in c["diseases"]),
          "PanDerm ran once per image for the case (not per candidate), comma-free names")
    n_dermo_A = len({(c["image"], c["query"]) for c in FakeDermoGPT.calls})
    check(n_dermo_A == len(FakeDermoGPT.calls), "same image + question never sent twice within a case")
    check(A["n_tool_errors"] == 1 and A["tool_errors"][0]["tool"] == "make" and "mock" in A["tool_errors"][0]["error"],
          "tool error text is caught and recorded per case")
    check(mock.bad_token_requests == 0 and "<|image_pad|>" in sc_steps[6]["output"] and
          "[image_pad]" in next(x for x in log if x["module"] == "vlm" and "onset timing" in x["text"])["text"],
          "literal <|image_pad|> in model output is rewritten before the next request (no HTTP 500, no dropped case)")
    # items 6-7: memory
    mem = [x for x in log if x["module"] == "memory" and x["case"] == "A"]
    m0 = next(x for x in mem if x["disease"] == "Scabies")
    check(m0["n_images"] == 6 and rows["A"]["case_report"][:200] in m0["text"] and "image 1, image 2, image 3, image 4, image 5, image 6" in m0["text"] and
          "Step 1: Classify skin image for Scabies" in m0["text"] and "Needs: Clinical or dermoscopic skin image" in m0["text"] and "output of step 7" in m0["text"],
          "items 6-7: memory call sees the case text, all images as 'image k', and the plan with what each step needs")
    lpc = pcs["Lichen planus"]
    check({x["step"]: x["reason"] for x in lpc["memory_removed"]} == {2: "not_supported_by_patient_data", 4: "depends_on_step_2"} and
          lpc["memory"]["how"] == "parsed" and not any(s_.get("output") for s_ in lpc["steps"] if s_["id"] in (2, 4)) and
          [s_["status"] for s_ in lpc["steps"] if s_["id"] in (2, 4)] == ["memory_removed", "memory_removed"], "items 6-7: a removed step and its dependant do not run, and are recorded")
    check(pcs["Nummular eczema"]["memory"]["how"] == "parsed_reasked" and len(pcs["Nummular eczema"]["memory"]["replies"]) == 2 and
          pcs["Nummular eczema"]["memory"]["replies"][0]["parse"] != "ok", "memory fallback: unclear reply re-asked once")
    zm = pcs["Zebra fever"]["memory"]
    check(zm["how"] == "default_keep_all" and len(zm["replies"]) == 2 and zm["supported"] == list(range(1, 10)) and not pcs["Zebra fever"]["memory_removed"] and
          "unknown_steps" in zm["replies"][0]["parse"], "memory default (mine): unclear twice -> keep every step, recorded")
    tm = pcs["Tinea corporis"]["memory"]
    check("error" in tm["replies"][0] and tm["how"] == "parsed_reasked", "memory fault: a 400 is treated as an unclear reply")
    hm = pcs["Hand, foot and mouth disease"]
    check(hm["memory"]["supported"] == [] and len(hm["memory_removed"]) == 9 and not hm["evidence"] and hm["no_evidence_reason"] == "no_indicator_left" and hm["score"] is None,
          "memory supports nothing: a clear reply is honoured, nothing runs, candidate has no evidence")
    check(A["n_memory_calls"] == 9 and A["n_memory_default"] == 1 and A["n_memory_reasked"] == 2 and A["n_steps_memory_removed"] == 11, f"memory counts in the record: {A['n_memory_calls']}, {A['n_memory_default']}, {A['n_memory_reasked']}, {A['n_steps_memory_removed']}")
    # items 8-9, A3: reflection
    rf = [x for x in log if x["module"] == "reflection"]
    r1 = next(x for x in rf if x["disease"] == "Psoriasis")
    check(r1["n_images"] == 4, "A3: reflection sees the step's input images: the 4 derm images, not the other 2")
    labels = [im["label"] for im in trA["images"] if im["tool_usable"]]
    check(all(l_ in r1["text"] for l_ in labels) and pcs["Psoriasis"]["reflection"][0]["images"] == labels, f"A3: the prompt names those images: {labels}")
    check(all(t_ in r1["text"] for t_ in ("Input quality:", "blurry, occluded, cropped, low-resolution", "Tool output:", "Scores: does each value lie in a valid range",
                                          "consistent with what the tool reports", "Text: is each statement made only when clear visual or textual clues", "Determine if the result is reliable and accurate?")) and
          "Answer with only one word (Yes or No)" in r1["system"], "A3: all four E.4 checks, numeric one reworded; Examiner wording with reliable for confidential")
    check("Clinical guideline:\n1. Take" in r1["text"] and "PanDerm zero-shot" in r1["text"] and rows["A"]["case_report"][:80] not in r1["text"], "item 8: reflection gets G and the tool output, no case text for a tool step")
    check("Clinical guideline" not in pcs["Zebra fever"]["reflection"][0]["prompt"].split("Check the result")[0], "item 8: no G -> none in the prompt")
    pz = pcs["Psoriasis"]
    check(pz["reflection"][0]["verdict"] == "unreliable" and {x["step"]: x["reason"] for x in pz["terminated"]} == {1: "reflection_unreliable", 3: "depends_on_step_1"} and
          not any(i["step"] == 3 for i in pz["indicators"]), "item 9: Terminate drops that step and its dependant, recorded")
    check({i["step"] for i in pz["indicators"]} == {4, 6, 9}, "item 9: other indicators continue; no down-weighting")
    check(len([x for x in rf if x["disease"] == "Psoriasis"]) == 1 and len(pz["reflection"]) == 3 and all(r_["step"] in (1, 2, 7) for r_ in pz["reflection"]), "item 8: one call per intermediate step, none for indicators")
    zr = pcs["Zebra fever"]["reflection"][0]
    check(zr["how"] == "default_reliable" and zr["verdict"] == "reliable" and len(zr["replies"]) == 2, "reflection default (mine): unclear twice -> reliable, recorded")
    sr = pcs["Scabies"]["reflection"][0]
    check(sr["how"] == "reasked" and sr["replies"][0]["verdict"] is None and sr["replies"][0]["answer_source"] == "reasoning", "reflection fallback: thinking copied by the fallback is not an answer; re-asked")
    nr = pcs["Nummular eczema"]["reflection"][0]
    check(nr["how"] == "default_reliable" and all("error" in x for x in nr["replies"]), "reflection fault: a 400 twice falls to the default")
    check(A["n_reflection_default"] >= 2 and A["n_reflection_reasked"] >= 1 and A["n_reflection_unreliable"] == 1 and A["n_steps_terminated"] >= 2, "reflection counts in the record")
    # item 10: cap
    lc = pcs["Lupus erythematosus, discoid"]
    check([x["step"] for x in lc["capped_steps"]] == [11, 12] and lc["n_executed"] == 10 and lc["capped_steps"][0]["reason"] == "over_cap_10" and lc["capped_steps"][1]["reason"] == "depends_on_step_11" and
          [s_["status"] for s_ in lc["steps"]][-2:] == ["capped", "capped"] and not any("scalp" in x["text"] for x in log if x["module"] == "vlm"), "item 10: 10 executed steps; later steps excluded, recorded, never called")
    # item 12: indicators and comparison score
    vals = {i["step"]: i["if_abnormal"] for i in pcs["Scabies"]["indicators"]}
    check(vals == {3: "Yes", 4: "No", 6: "Yes", 9: "No"}, f"item 12: only Yes / No survive: {vals}")
    du = {x["step"]: x["how"] for x in pcs["Psoriasis"]["dropped_uncertain"]}
    check(du == {5: "exact_reasked", 8: "unclear_after_reask"}, f"item 12: Uncertain dropped and recorded: {du}")
    check(A["n_uncertain"] >= 2 and A["n_unclear_verdicts"] >= 1, "dropped Uncertain and unclear verdicts counted per case")
    s8 = next(s_ for s_ in pcs["Psoriasis"]["steps"] if s_["id"] == 8)
    check(len(s8["summary_replies"]) == 2 and s8["status"] == "dropped_uncertain", "'Cannot assess' asked twice, then dropped as Uncertain")
    cot = next(s_ for s_ in pcs["Psoriasis"]["steps"] if s_["id"] == 5)
    check(cot["summary_replies"][0]["answer_source"] == "reasoning" and cot["summary_replies"][0]["how"] == "unclear_reply",
          "chain of thought copied by the reasoning fallback is not taken as Uncertain: re-asked, real answer used")
    ne = pcs["Nummular eczema"]
    check({x["step"]: x["reason"] for x in ne["terminated"]} == {9: "depends_on_step_6"} and 6 in {x["step"] for x in ne["dropped_uncertain"]} and
          not any("onset timing" in x["text"] and x["disease"] == "Nummular eczema" for x in log if x["module"] == "vlm"), "Fig 4: analysis built on an Uncertain indicator stops")
    sm = [x for x in log if x["module"] == "summary"][0]
    check("please provide a brief summary. The task is: " in sm["text"] and "Does this patient have the abnormal?" in sm["text"] and
          "(Yes, No or Uncertain)" in sm["text"] and "most likely diagnosis" not in sm["text"], "B1: upstream's summary sentence verbatim, Uncertain enabled")
    pr = next(x for x in A["candidate_results"] if x["name"] == "Psoriasis")
    check(abs(pr["score"] - 1 / 3) < 1e-9 and pr["threshold"] == .2 and abs(pr["margin"] - (1 / 3 - .2)) < 1e-6 and abs(sum(pr["weights"].values()) - 1) < 1e-9,
          f"item 11-12: score over the survivors, weights rescaled, Yes 1, No 0: {pr['score']}")
    check(not any(x["module"] == "pro_decider" for x in trA["llm_calls"]), "no per-patient decider call any more")
    # items 13a-e: final call
    fin = [x for x in log if x["module"] == "final" and x["case"] == "A"]
    check(len(fin) == 2 and trA["final"]["status"] == "ok" and len(trA["final"]["attempts"]) == 2 and trA["final"]["attempts"][0]["parse"] != "ok" and A["final_attempts"] == 2, "13b: an unparsed reply is asked again, as in the top-10 step")
    ft = fin[-1]
    check(json.loads(A["prediction"])["diagnoses"] == FINAL_A and A["parsed"] and A["n_items"] == 5 and A["parse_rule"] == "final_call" and A["final_parsed"], "13a: the final call is the output")
    check(ft["n_images"] == 0 and not any(r_["case_report"][:80] in ft["text"] for r_ in rows.values()) and ft["system"] == "", "13c: no images, no case text, no role prompt")
    check('{"diagnoses": ["<most likely>", "<2nd>", "<3rd>", "<4th>", "<5th>"]}' in ft["text"] and "5 most likely diagnoses" in ft["text"], "13b: base-arm output format")
    check(all(f"Candidate: {c}" in ft["text"] for c in A["candidates"]), "13d: every candidate is listed")
    blocks = {b_.split("\n")[0][11:]: b_ for b_ in ft["text"].split("\n\n") if b_.startswith("Candidate: ")}
    check(blocks["Unplannable disease"].endswith("no evidence") and blocks["Hand, foot and mouth disease"].endswith("no evidence") and A["n_no_evidence"] == 2, "13d: no plan, or no indicator left -> marked 'no evidence'")
    sb = blocks["Scabies"]
    check("Indicator: Assess whether zero-shot scores support Scabies\n  Abnormal: Yes" in sb and "Mock VLM observation" in sb and "PanDerm zero-shot" in sb, "13a: indicators with verdict and evidence, and the tool results they rest on")
    check("Assess lesion morphology" not in sb and "Judge stripe" not in sb and "Annotate erythema" not in sb, "dropped Uncertain indicators, and tool steps feeding only those, are not evidence")
    check("Classify skin image" not in blocks["Psoriasis"] and "zero-shot scores support" not in blocks["Psoriasis"], "a terminated branch is not evidence")
    check(not any(w_ in ft["text"] for w_ in ("weight", "margin", "score -")), "13c: no weights or margin in the final prompt")
    scored = [r_ for r_ in trA["per_candidate"] if r_.get("score") is not None]
    check(scored and all(f"Risk score: {r_['score']:.2f} (threshold {r_['threshold']:.2f}): {r_['verdict']}" in blocks[r_["name"]] for r_ in scored), "13a: each scored candidate's risk result reaches the final call")
    check("Risk score" not in blocks["Unplannable disease"] and "Risk score" not in blocks["Deciderfail disease"], "no risk line without a score")
    check(trA["final"]["prompt"].replace("<|image_pad|>", "[image_pad]") == ft["text"], "final prompt saved in full")
    check(A["agent_response"] == trA["final"]["attempts"][-1]["reply"], "final reply saved in full")
    check(A["margin_top5"] == MARGIN_A and [x["name"] for x in trA["margin_ranking"]][:5] == MARGIN_A and A["margin_top5"] != json.loads(A["prediction"])["diagnoses"],
          "13e: score-minus-threshold ranking kept as a comparison arm; the tie goes to the backbone's order")
    check([x["name"] for x in trA["margin_ranking"]][-3:] == ["Hand, foot and mouth disease", "Unplannable disease", "Deciderfail disease"], "comparison arm: unscored candidates last")
    check([x["name"] for x in trA["ranking"]] == FINAL_A and trA["ranking"][0]["backbone_rank"] == 6, "trace ranking = the final call's")
    check(A["final_top1_has_guideline"] is False and A["n_final_outside_candidates"] == 0, "the top answer's has_guideline is recorded")
    # full traces
    ts = pcs["Psoriasis"]
    check(ts["memory"]["prompt"] and ts["memory"]["replies"] and ts["reflection"][0]["prompt"] and ts["reflection"][0]["replies"] and "steps" in ts, "memory and reflection prompts and replies saved per candidate")
    check(len(pcs["Scabies"]["memory"]["prompt"]) > len(rows["A"]["case_report"][:200]) and rows["A"]["case_report"] in trA["case_text"], "prompts are not cut")
    rc = next(x for x in A["candidate_results"] if x["name"] == "Psoriasis")
    check("prompt" not in json.dumps(rc["memory"]) and rc["reflection"][0] == {"step": 1, "verdict": "unreliable", "how": "exact"}, "predictions row stays compact; trace holds the full text")
    # B unparsed
    check(not B["parsed"] and B["prediction"] == "" and B["unparsed_reason"] == "candidates_unparsed" and B["n_llm_calls"] == 2 and B["n_tool_calls"] == 0 and B["n_candidates"] == 0 and not B["final_called"],
          "B: no usable top-10 after 2 attempts -> unparsed, nothing guessed, no plan, no tool, no final call")
    # C no derm image
    c_ps = pcC["Psoriasis"]
    cv = {i["step"]: (i["if_abnormal"], i["how"]) for i in c_ps["indicators"]}
    check({d_["step"]: d_["reason"] for d_ in c_ps["dropped_steps"]} == {3: "no_derm_image", 4: "no_derm_image", 5: "no_derm_image", 8: "no_derm_image"},
          f"C no derm image (Eq. 3): image-tool indicators dropped and recorded: {c_ps['dropped_steps']}")
    check(sorted(cv) == [6, 9] and cv[6] == ("Yes", "exact"), "C no derm image: only the text-based indicators survive")
    cps = next(x for x in C["candidate_results"] if x["name"] == "Psoriasis")
    check(abs(cps["score"] - 0.5) < 1e-9 and C["n_dropped_indicators"] > 0, f"score over the remaining indicators only: {cps['score']}")
    check(not c_ps["reflection"], "a tool step that could not run gets no reflection call")
    io = pcC["Imageonly disease"]
    check(io["plan_status"] == "ok" and io["score"] is None and io["indicators"] == [] and len(io["dropped_steps"]) == 2 and io["no_evidence_reason"] == "no_indicator_left",
          "no indicator left: unscored, marked no evidence")
    check(C["n_tool_calls"] == 0 and C["n_images_tool_usable"] == 0 and C["n_images_sent"] == 2, "C: no image tool ran; both images still go to the backbone")
    cvlm = [x for x in log if x["module"] == "vlm" and x["case"] == "C"]
    check(cvlm and all(x["n_images"] == 2 for x in cvlm) and not any("Judge scale" in x["text"] for x in cvlm), "C: no VLM call on the blocked steps")
    check(C["parsed"] and C["n_items"] == 5, "C: final call answers from the candidates it has")
    # traces, budget, meta
    check(len(trA["llm_calls"]) == A["n_llm_calls"] and all("messages" in x and "usage" in x and "reasoning_content" in x for x in trA["llm_calls"]) and
          {x["module"] for x in trA["llm_calls"]} == {"candidates", "guideline", "indicators", "planner", "weights", "memory", "gpt_decider", "reflection", "summary", "final"},
          "every LLM call saved in full, all stages through mp_llm")
    check(A["call_budget_exceeded"] and A["n_candidates"] == 10 and len(trA["per_candidate"]) == 10, f"budget guard logged ({A['n_llm_calls']} > 40), no candidate cut")
    p = [x for x in log if x["module"] == "final"][-1]["params"]
    check(p.get("model") == BACKBONE and p.get("top_k") == 20 and p.get("max_tokens") == 32768 and p.get("temperature") == 1.0,
          "creator sampling + 32768 budget reach the server")
    import dx_parse
    for r in recs.values():
        if r["parsed"]:
            it, _ = dx_parse.parse(r["task"], r["prediction"], r["finish_reason"])
            check(it == json.loads(r["prediction"])["diagnoses"], f"{r['_id']}: dx_parse re-parse == recorded items")
    meta = json.loads((out / "T1.1/RUN_META.json").read_text())
    check(meta["run_sig"] == A["run_sig"] and meta["kb_manifest_sha"] and meta["toolset_sha"] and meta["sampling"] == SAMPLING, "RUN_META carries run_sig, KB and toolset hashes")
    check(meta["server_args"] == os.environ["MP_SERVER_ARGS"] and any(k.endswith("eval/utils.py") for k in meta["code"]) and
          any(k.endswith("prompts/dx_top5.txt") for k in meta["code"]), "run_sig inputs include server args, eval/utils.py, dx_top5.txt")
    check(meta["embed_model"] == "fake" and meta["chunk_profile"] == "tok300o50_cl100k_base" and meta["chunking"]["size"] == 300 and meta["step_cap"] == 10, "RUN_META names the embedder, chunk settings and step cap")
    tj = json.loads((cfg / "toolset.json").read_text())
    check(not any("coding" in t["type"].lower() for t in tj) and any("dermogpt" in str(t.get("command")) and "vlm" in t["type"].lower() for t in tj),
          "toolset: no coding tool; DermoGPT is a domain VLM")

    print("=== 5. infrastructure error in each new stage: case dropped, retried in pass 2")
    (W / "in_drop_T1.1.jsonl").write_text(json.dumps(rows["A"]) + "\n")
    (W / "in_drop_T1.2.jsonl").write_text(json.dumps(rows["C"]) + "\n")
    rows_d = W / "in_drop_{task}.jsonl"
    for mod_ in ("memory", "reflection", "final"):
        mock.force_500 = mod_
        run(*common, "--tasks", "T1.1", "--input-tmpl", str(rows_d), "--out-tmpl", str(W / f"out_drop_{mod_}/{{task}}"), "--shard", "0/1", "--threads", "1")
        dd = W / f"out_drop_{mod_}/T1.1"
        dropped = [json.loads(l) for l in open(dd / "dropped.p0.jsonl")]
        check(len(dropped) == 1 and "InfraError" in dropped[0]["error"] and list((dd / "traces").glob("*.dropped_*.json")) and
              sum(1 for _ in open(dd / "predictions.p0.jsonl")) == 1, f"HTTP 500 in {mod_} -> case dropped with partial trace, written in pass 2")
    run(*common, "--tasks", "T1.2", "--input-tmpl", str(rows_d), "--out-tmpl", str(W / "out_c2/{task}"), "--shard", "0/1", "--threads", "1")
    flaky = [json.loads(p_.read_text()) for p_ in cache.glob("*/*.json") if json.loads(p_.read_text())["key"] == "flaky disease"]
    check(len(flaky) == 1 and flaky[0]["status"] == "ok", "the plan hit by the 400 was made and stored on the next case")

    print("=== 5b. final call unparsed: wrong, never guessed")
    for mode, why, fin_ in (("prose", "final_unparsed", "stop"), ("400", "final_unparsed", "agent_request_error")):
        mock.final_mode = mode
        run(*common, "--tasks", "T1.2", "--input-tmpl", str(rows_d), "--out-tmpl", str(W / f"out_final_{mode}/{{task}}"), "--shard", "0/1", "--threads", "1")
        mock.final_mode = "ok"
        u = json.loads(open(W / f"out_final_{mode}/T1.2/predictions.p0.jsonl").readline())
        check(not u["parsed"] and u["prediction"] == "" and u["unparsed_reason"] == why and u["finish_reason"] == fin_ and u["final_called"] and not u["final_parsed"] and u["margin_top5"],
              f"final {mode}: unparsed = wrong; the comparison arm is recorded but does not fill in")

    print("=== 6. resume: nothing left to run")
    n = len(mock.log)
    run(*common, *io_args, "--shard", "0/1", "--threads", "3")
    check(len(mock.log) == n, "resume skips all 3 done cases (no LLM call)")

    print("=== 7. resume with a changed setting must be refused")
    man0 = (kb / "MANIFEST.json").read_text()
    ENV = lambda k_, v_, old_: (lambda: os.environ.__setitem__(k_, v_), (lambda: os.environ.__setitem__(k_, old_)) if old_ is not None else (lambda: os.environ.pop(k_)))  # noqa: E731
    PATCH = lambda name_, add=" x": (lambda: setattr(R, name_, getattr(R, name_) + add), lambda: setattr(R, name_, getattr(R, name_)[:-len(add)]))  # noqa: E731
    for label, (change, undo) in [
        ("sampling", ENV("MP_SAMPLING_JSON", json.dumps(dict(SAMPLING, temperature=0.5)), json.dumps(SAMPLING))),
        ("toolset", (lambda: (cfg / "toolset.json").write_text((cfg / "toolset.json").read_text() + "\n"),
                     lambda: (cfg / "toolset.json").write_text((cfg / "toolset.json").read_text().rstrip("\n")))),
        ("server args", ENV("MP_SERVER_ARGS", "--tp-size 4", "--tp-size 2 --context-length 131072")),
        ("knowledge-base manifest", (lambda: (kb / "MANIFEST.json").write_text(man0.replace('"n_files": 3', '"n_files": 4')),
                                     lambda: (kb / "MANIFEST.json").write_text(man0))),
        ("step cap", ENV("MP_STEP_CAP", "9", None)), ("final attempts", ENV("MP_FINAL_ATTEMPTS", "3", None)),
        ("relaxed attempts", ENV("MP_PLAN_RELAXED_ATTEMPTS", "1", None)), ("indicator attempts", ENV("MP_INDICATOR_ATTEMPTS", "3", None)),
        ("weights attempts", ENV("MP_WEIGHTS_ATTEMPTS", "3", None)),
        ("memory prompt", PATCH("MEMORY_PROMPT")), ("reflection prompt", PATCH("REFLECT_PROMPT")), ("reflection system", PATCH("REFLECT_SYSTEM")),
        ("final prompt", PATCH("FINAL_PROMPT")), ("indicator prompt", PATCH("INDICATOR_PROMPT")), ("indicator prompt, no G", PATCH("INDICATOR_PROMPT_NO_G")),
        ("planner indicator rules", PATCH("PLANNER_INDICATORS")), ("plan retry note", PATCH("PLAN_RETRY_NOTE")), ("weights prompt", PATCH("WEIGHTS_PROMPT")),
        ("weights system", PATCH("WEIGHTS_SYSTEM")), ("post-check version", PATCH("POSTCHECK_VERSION")), ("plan schema", PATCH("PLAN_SCHEMA"))]:
        change()
        try:
            run(*common, *io_args, "--shard", "0/1", "--threads", "3")
            check(False, f"{label} change refused")
        except SystemExit as e:
            check("different run_sig" in str(e), f"{label} change refused: {str(e)[-90:]}")
        undo()
    ns_args = (BACKBONE, "k", "t", "s")
    ns0 = R.namespace(*ns_args)
    for label, (change, undo) in [("indicator prompt", PATCH("INDICATOR_PROMPT")), ("weights prompt", PATCH("WEIGHTS_PROMPT")), ("plan schema", PATCH("PLAN_SCHEMA")),
                                  ("planner rules", PATCH("PLANNER_INDICATORS")), ("embedder", ENV("MP_EMBEDDER", "other", "fake")),
                                  ("chunk profile", ENV("MP_CHUNKS", "char400o50", None))]:
        change()
        ns1 = R.namespace(*ns_args)
        undo()
        check(ns1 != ns0, f"plan namespace changes with the {label}: old plans cannot be mixed in")
    check(R.namespace(*ns_args) == ns0, "namespace is stable when nothing changed")
    cache_old = cache / "28f8ed5023ee"
    cache_old.mkdir()
    entry = json.loads(next(p_ for p_ in cache.glob("*/*.json") if json.loads(p_.read_text())["key"] == "psoriasis").read_text())
    (cache_old / R.PlanCache(cache, "28f8ed5023ee").path("psoriasis").name).write_text(json.dumps({"key": "psoriasis", "status": "failed", "plan": None}))
    run(*common, "--tasks", "T1.1", "--input-tmpl", str(rows_d), "--out-tmpl", str(W / "out_oldns/{task}"), "--shard", "0/1", "--threads", "1")
    o = json.loads(open(W / "out_oldns/T1.1/predictions.p0.jsonl").readline())
    check(next(x for x in o["candidate_results"] if x["name"] == "Psoriasis")["plan_sha"] == entry["plan_sha"], "an old namespace in the same root is never read")
    shutil.rmtree(cache_old)

    print("=== 8. owner's plan reader, REVIEWED inside the namespace")
    run("--print-plans", "2", "--plan-cache", str(cache))
    ns_dir = next(d for d in cache.iterdir() if d.is_dir() and list(d.glob("*.json")))
    full_args = [*common, *io_args, "--shard", "0/1", "--threads", "3", "--require-reviewed"]
    try:
        run(*full_args)
        check(False, "full run refused without REVIEWED")
    except SystemExit as e:
        check("REVIEWED" in str(e) and ns_dir.name in str(e), f"full run refused without <namespace>/REVIEWED: {str(e)[:70]}")
    (cache / "REVIEWED").write_text("")
    try:
        run(*full_args)
        check(False, "a REVIEWED file outside the namespace does not count")
    except SystemExit:
        check(True, "a REVIEWED file outside the namespace does not count")
    (ns_dir / "REVIEWED").write_text("")
    run(*full_args)
    check(True, "with <namespace>/REVIEWED the run starts")
    run("--print-plans", "1", "--plan-cache", str(cache))

    print("=== 9. rank stability, text only: re-runs the final call")
    n = len(mock.log)
    rs = ["--rank-stability", "3", "--backbone", BACKBONE, "--tasks", "T1.1", "--out-tmpl", str(out / "{task}"), "--threads", "2"]
    run(*rs)
    r0 = json.loads((out / "T1.1/rank_stability.json").read_text())
    new = mock.log[n:]
    check(r0["n_cases"] == 1 and r0["same_top5_order_pct"] == 100.0 and r0["mean_top5_overlap_pct"] == 100.0 and new and
          {x["module"] for x in new} == {"final"} and all(x["n_images"] == 0 for x in new) and len(new) == 3,
          f"same replies -> top 5 unchanged; text-only final calls, the saved prompt: {r0}")
    mock.jitter = True
    run(*rs)
    r1 = json.loads((out / "T1.1/rank_stability.json").read_text())
    mock.jitter = False
    check(r1["same_top1_pct"] == 0.0 and r1["mean_top5_overlap_pct"] < 100.0 and r1["cases_with_any_top1_change_pct"] == 100.0,
          f"changed final replies are measured: {r1}")
    mock.final_mode = "prose"
    run(*rs)
    mock.final_mode = "ok"
    r2 = json.loads((out / "T1.1/rank_stability.json").read_text())
    check(r2["n_unparsed_repeats"] == 3, "an unparsed repeat is counted")
    row = json.loads(open(out / "T1.1/rank_stability_cases.jsonl").readline())
    check(len(row["repeats"]) == 3 and row["llm_calls"], "per-case repeats and full LLM trace saved")

    print("=== 10. postcheck and canary gates")
    pc_py = str(HERE / "mp_postcheck.py")
    g = subprocess.run([sys.executable, pc_py, str(out / "T1.1"), "T1.1", "--gate", "--n-input", "1"], capture_output=True, text=True)
    print(g.stdout)
    check("tool_errors 1" in g.stdout and "cuda_errors 0" in g.stdout and "failed_plans 1" in g.stdout and "no_plan 1" in g.stdout and "unscored 3" in g.stdout and
          "uncertain_pct" in g.stdout, "POSTCHECK reports tool errors, CUDA, failed plans, no-plan candidates, unscored, Uncertain %")
    check(all(k in g.stdout for k in ("memory_removed", "memory_default", "steps_terminated", "reflection_default", "steps_capped", "uncertain_dropped", "final_parsed_pct",
                                      "has_guideline_pct", "no_evidence")) and "has_guideline=True:" in g.stdout and "has_guideline=False:" in g.stdout and
          "planned final indicators per candidate" in g.stdout, "POSTCHECK reports every new stage, the has_guideline split and the indicator count")
    check(g.returncode == 3 and "[mp-gate] FAIL" in g.stdout and "MEDAGENTPRO_CANARY_GATES T1.1 FAIL" in g.stdout, "a broken canary prints FAIL and exits 3")
    clean = W / "clean"
    clean.mkdir()
    (clean / "predictions.jsonl").write_text(json.dumps({
        "parsed": True, "n_tool_calls": 200, "n_tool_errors": 0, "n_cuda_errors": 0, "n_candidates": 10, "n_plans_failed": 0, "n_unscored": 0,
        "n_indicators": 50, "n_uncertain": 5, "n_unclear_verdicts": 0, "n_llm_calls": 100, "final_called": True, "final_parsed": True,
        "n_memory_calls": 10, "n_reflections": 20, "n_steps_planned": 60, "n_no_guideline": 3,
        "tool_steps": {"panderm": 1, "make": 1, "dermogpt": 1, "backbone": 3}}) + "\n")
    g = subprocess.run([sys.executable, pc_py, str(clean), "T1.1", "--gate", "--n-input", "1"], capture_output=True, text=True)
    check(g.returncode == 0 and "MEDAGENTPRO_CANARY_GATES T1.1 PASS" in g.stdout, "a clean canary passes every gate")
    row0 = json.loads((clean / "predictions.jsonl").read_text())
    for field, val, gate in [("n_uncertain", 40, "Uncertain dropped"), ("n_memory_default", 1, "memory replies on the default"),
                             ("n_reflection_default", 1, "reflection replies on the default"), ("final_parsed", False, "final call parsed"),
                             ("n_no_plan", 3, "candidates with no plan"), ("n_steps_memory_removed", 40, "steps removed by memory")]:
        (clean / "predictions.jsonl").write_text(json.dumps(dict(row0, **{field: val})) + "\n")
        g = subprocess.run([sys.executable, pc_py, str(clean), "T1.1", "--gate", "--n-input", "1"], capture_output=True, text=True)
        check(g.returncode == 3 and f"[mp-gate] FAIL T1.1 {gate}" in g.stdout, f"gate fires: {gate}")
    for field, val, gate in [("n_reflection_unreliable", 12, "unreliable % of reflections"),
                             ("tool_steps", {"make": 1, "dermogpt": 1, "backbone": 3}, "PanDerm step share %")]:   # owner 11-08 #4
        (clean / "predictions.jsonl").write_text(json.dumps(dict(row0, **{field: val})) + "\n")
        g = subprocess.run([sys.executable, pc_py, str(clean), "T1.1", "--gate", "--n-input", "1"], capture_output=True, text=True)
        check(g.returncode == 0 and f"[mp-gate] REPORT T1.1 {gate}" in g.stdout, f"report only, does not stop the run: {gate}")
    (clean / "predictions.jsonl").write_text(json.dumps({
        "parsed": True, "n_tool_calls": 200, "n_tool_errors": 1, "n_cuda_errors": 1, "n_candidates": 10, "n_indicators": 50,
        "tool_steps": {"panderm": 1, "make": 1, "dermogpt": 1, "backbone": 3}}) + "\n")
    g = subprocess.run([sys.executable, pc_py, str(clean), "T1.1", "--gate", "--n-input", "1"], capture_output=True, text=True)
    check(g.returncode == 3 and "[mp-gate] FAIL T1.1 CUDA errors" in g.stdout, "one CUDA error fails the canary")
    srv.shutdown()
    print(f"\n{sum(ok)}/{len(ok)} checks passed | work dir {W}")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
