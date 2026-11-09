#!/usr/bin/env python3
"""MedlinePlus knowledge base for MedAgent-Pro on DermArena (paper v3 Sec 3.2). CPU steps only; stdlib only.

  download   frozen snapshot -> <kb>/raw/ (health-topics XML, English topic pages, encyclopedia index + pages)
  ids        every PMID / PMCID linked or cited ANYWHERE in the raw snapshot, matched to DermArena (all tasks, splits);
             runs on the raw files, i.e. before any stripping -> <kb>/reports/pmid_pmcid_match.json
  parse      raw -> <kb>/docs.jsonl = the text that gets indexed. WHITELIST: a topic document is its XML title + summary;
             an encyclopedia document is its summary + prose sections. So 'Journal Articles' (MEDLINE/PubMed), 'Clinical
             Trials', every other link list, 'References' and review credits never enter it. Raw HTML stays in raw/.
             Then verifies: no PubMed / PMC / trials link, no PMID, no reference-list text in docs.jsonl.
  chunk      docs.jsonl -> <kb>/chunks.jsonl (upstream RAG.py sizes: 400 characters, overlap 50). Legacy Qwen3-Embedding index.
  chunk-tokens  docs.jsonl -> <kb>/chunks_<profile>.jsonl + index/CHUNKS_<profile>.json (paper: 300-token chunks; owner addendum A1:
             50-token overlap; tokens = the embedder's tokenizer, cl100k_base for ada-002).
             chunks.jsonl, MANIFEST.json and the existing reports are not touched. Needs tiktoken; the run itself does not.
  manifest   <kb>/MANIFEST.json: URL, retrieval time, sha256 of every raw file, sha256 of docs/chunks
  coverage   name match of DermArena gold diagnoses and Qwen top-5 predictions -> <kb>/reports/coverage.json
  leakscan   10-word shingle overlap with all test + val case text, all tasks -> <kb>/reports/leak_scan*.json(l).
             With --chunks <profile> the indexed view is that chunk file and the reports are leak_scan_<profile>.json(l).

Run with `python3 -I` : the snapshot is downloaded data, never on sys.path. Polite client: one request at a time,
MP_KB_DELAY seconds apart (default 1.0, i.e. <= 60/min, under NLM's published 85/min), a research User-Agent without a
personal address, backoff on 429/5xx, resumable (a file already on disk is not fetched again).
"""
import argparse
import glob
import hashlib
import html as html_mod
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

KB_DEFAULT = "/fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/medlineplus_kb"
UA = "DermArena-benchmark-research-client/1.0 (academic research; non-commercial)"
SITE = "https://medlineplus.gov"
RELEASE_TASKS = "/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks"
QWEN_T11 = "/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/qwen3.5-27b/T1.1/predictions.jsonl"
CHUNK_SIZE, CHUNK_OVERLAP = 400, 50      # upstream RAG.py: RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
# A chunk profile = one chunk file. The profile's settings reach the index manifest, so kb_sha, the plan namespace and run_sig.
CHUNK_PROFILES = {
    "char400o50": {"unit": "character", "tokenizer": None, "size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP, "file": "chunks.jsonl"},
    "tok300o50_cl100k_base": {"unit": "token", "tokenizer": "cl100k_base", "size": 300, "overlap": 50,
                              "file": "chunks_tok300o50_cl100k_base.jsonl"},
}
DEFAULT_CHUNK_PROFILE = "tok300o50_cl100k_base"
DELAY = float(os.environ.get("MP_KB_DELAY", "1.0"))


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Fetcher:
    def __init__(self, kb):
        self.kb = Path(kb)
        self.log = (self.kb / "fetch_log.jsonl").open("a")
        self.last = 0.0
        self.n = 0

    def get(self, url, rel):
        """Fetch url into raw/<rel> unless present. Returns the path, or None after a logged failure."""
        out = self.kb / "raw" / rel
        if out.exists() and out.stat().st_size:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            wait = DELAY - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            status, body, err = None, None, None
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as r:
                    status, body = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, err = e.code, str(e)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            if status == 200 and body:
                tmp = out.with_name(out.name + f".tmp{os.getpid()}")
                tmp.write_bytes(body)
                os.replace(tmp, out)
                self.log.write(json.dumps({"url": url, "path": rel, "status": 200, "bytes": len(body),
                                           "sha256": hashlib.sha256(body).hexdigest(), "retrieved": utc()}) + "\n")
                self.log.flush()
                self.n += 1
                if self.n % 100 == 0:
                    print(f"[kb] {utc()} fetched {self.n} (last {rel})", flush=True)
                return out
            if status in (404, 410):
                break
            time.sleep(min(120, 10 * 2 ** attempt))   # 429 / 5xx / network: back off
        self.log.write(json.dumps({"url": url, "path": rel, "status": status, "error": err, "retrieved": utc()}) + "\n")
        self.log.flush()
        print(f"[kb] FAILED {url}: {status} {err}", flush=True)
        return None


def latest_topics_url(html):
    names = sorted(set(re.findall(r"mplus_topics_\d{4}-\d{2}-\d{2}\.xml", html)))
    if not names:
        raise SystemExit("[kb] no mplus_topics_*.xml link on xml.html")
    return f"{SITE}/xml/{names[-1]}", names[-1]


def topic_urls(xml_path):
    """English topics only (the file holds English and Spanish)."""
    out = []
    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag == "health-topic":
            if el.get("language") == "English":
                out.append(el.get("url"))
            el.clear()
    return sorted(set(out))


def cmd_download(kb):
    kb = Path(kb)
    kb.mkdir(parents=True, exist_ok=True)
    f = Fetcher(kb)
    p = f.get(f"{SITE}/xml.html", "xml/xml.html")
    url, name = latest_topics_url(p.read_text(errors="ignore"))
    existing = sorted((kb / "raw/xml").glob("mplus_topics_*.xml"))   # frozen: a resume keeps the first file taken
    xml_path = existing[0] if existing else f.get(url, f"xml/{name}")
    # encyclopedia index -> article + patient-instruction pages
    a = f.get(f"{SITE}/ency/encyclopedia_A.htm", "ency_index/encyclopedia_A.htm")
    # the A page does not link to itself, so it is added by hand
    letters = sorted({"encyclopedia_A.htm"} | set(re.findall(r'href="(?:\.\./ency/)?(encyclopedia_[A-Za-z0-9\-]+\.htm)"',
                                                             a.read_text(errors="ignore"))))
    pages = set()
    for name_ in letters:
        q = f.get(f"{SITE}/ency/{name_}", f"ency_index/{name_}")
        if q:
            pages |= set(re.findall(r'href="((?:article|patientinstructions)/\d+\.htm)"', q.read_text(errors="ignore")))
    print(f"[kb] {len(letters)} index pages, {len(pages)} encyclopedia pages listed", flush=True)
    topics = topic_urls(xml_path)
    print(f"[kb] {len(topics)} English health topics in {xml_path.name}", flush=True)
    for u in topics:
        rel = u.split("medlineplus.gov/", 1)[-1].strip("/").replace("/", "__")
        f.get(u, f"topics/{rel}")
    for pg in sorted(pages):
        f.get(f"{SITE}/ency/{pg}", f"ency/{pg}")
    print(f"[kb] DOWNLOAD DONE {utc()} new files {f.n}", flush=True)



# ----------------------------------------------------------------------------------------------- PMID / PMCID extraction
_PMID_PATTERNS = [
    re.compile(r"(?:ncbi\.nlm\.nih\.gov/pubmed|pubmed\.ncbi\.nlm\.nih\.gov)(?:/|%2F)+(\d{1,9})(?![\d])", re.I),
    re.compile(r"\bPMID\s*[:#=]?\s*(\d{1,9})\b", re.I),
    re.compile(r"(?:[?&]|&amp;)(?:list_uids|uid|pmid)=(\d{1,9})\b", re.I),
]
_PMC_PATTERN = re.compile(r"\bPMC\s?(\d{4,9})\b", re.I)


def extract_ids(text):
    """(set of PMIDs, set of PMCIDs) in one raw file: links and printed citations alike."""
    pmids = set()
    for p in _PMID_PATTERNS:
        pmids |= {str(int(x)) for x in p.findall(text)}
    return pmids, {f"PMC{int(x)}" for x in _PMC_PATTERN.findall(text)}


def raw_files(kb):
    for sub in ("xml", "topics", "ency_index", "ency"):
        for f in sorted((Path(kb) / "raw" / sub).rglob("*")):
            if f.is_file():
                yield sub, f


def dermarena_files():
    """(task dir, split, path) for every jsonl of every task."""
    for f in sorted(glob.glob(f"{RELEASE_TASKS}/*/*.jsonl")):
        p = Path(f)
        yield p.parent.name, p.name.split(".")[0].split("_")[0], p


def row_id(r):
    return r.get("_id") or r.get("item_id") or r.get("case_uid")


def dermarena_ids():
    """PMID -> rows and PMCID -> rows; T2.1 rows carry no pmid field, so case_uid/_id 'pmid_<n>_<k>' is read as well."""
    by_pmid, by_pmc = defaultdict(list), defaultdict(list)
    per_split = defaultdict(lambda: {"rows": 0, "pmids": set(), "pmcids": set()})
    for task, split, p in dermarena_files():
        for l in p.open():
            r = json.loads(l)
            pm = str(r.get("pmid") or "").strip()
            if not pm:
                m = re.match(r"pmid_(\d+)_", str(r.get("case_uid") or r.get("_id") or ""))
                pm = m.group(1) if m else ""
            pc = str(r.get("pmcid") or "").strip().upper()
            info = {"task": task, "file": p.name, "split": split, "id": row_id(r), "pmid": pm, "pmcid": pc,
                    "diagnosis": r.get("diagnosis"), "title": r.get("title")}
            s = per_split[split]
            s["rows"] += 1
            if pm.isdigit():
                pm = str(int(pm))
                by_pmid[pm].append(info)
                s["pmids"].add(pm)
            if pc.startswith("PMC"):
                by_pmc[pc].append(info)
                s["pmcids"].add(pc)
    return by_pmid, by_pmc, per_split


def link_title(text, ident):
    """The anchor text next to a PubMed link (topic pages print 'Article: <title>')."""
    m = re.search(rf"/pubmed/{ident}\b[^>]*>\s*([^<]{{0,200}})", text)
    return html_mod.unescape(m.group(1)).strip() if m else None


def title_matches(kb, by_pmid, by_pmc):
    """Article titles of DermArena found word for word in any raw page, references included."""
    titles = {}
    for rows in list(by_pmid.values()) + list(by_pmc.values()):
        for r in rows:
            t = r.get("title")
            if t:
                w = re.findall(r"[a-z0-9]+", t.lower())
                if len(w) >= 6:
                    titles.setdefault(" ".join(w), {"title": t, "splits": set()})["splits"].add(r["split"])
    first = defaultdict(list)
    for k in titles:
        first[" ".join(k.split()[:6])].append(k)
    hits = defaultdict(set)
    for _, f in raw_files(kb):
        w = re.findall(r"[a-z0-9]+", html_mod.unescape(re.sub(r"<[^>]+>", " ", f.read_text(errors="ignore"))).lower())
        rel = str(f.relative_to(kb / "raw"))
        for i in range(len(w) - 5):
            for k in first.get(" ".join(w[i:i + 6]), ()):
                if " ".join(w[i:i + len(k.split())]) == k:
                    hits[k].add(rel)
    return [{"title": titles[k]["title"], "splits": sorted(titles[k]["splits"]), "files": sorted(v)[:10]} for k, v in hits.items()]


def cmd_ids(kb):
    kb = Path(kb)
    where_pmid, where_pmc = defaultdict(set), defaultdict(set)
    per_source = Counter()
    n_files = Counter()
    titles = {}
    for sub, f in raw_files(kb):
        t = f.read_text(errors="ignore")
        n_files[sub] += 1
        pm, pc = extract_ids(t)
        per_source[f"{sub}_pmid_mentions"] += len(pm)
        per_source[f"{sub}_pmcid_mentions"] += len(pc)
        rel = str(f.relative_to(kb / "raw"))
        for x in pm:
            where_pmid[x].add(rel)
            if x not in titles:
                titles[x] = link_title(t, x)
        for x in pc:
            where_pmc[x].add(rel)
    by_pmid, by_pmc, per_split = dermarena_ids()
    hit_pmid, hit_pmc = set(where_pmid) & set(by_pmid), set(where_pmc) & set(by_pmc)
    title_hits = title_matches(kb, by_pmid, by_pmc)
    splits = {}
    for sp, s in sorted(per_split.items()):
        splits[sp] = {"rows": s["rows"], "distinct_pmids": len(s["pmids"]), "distinct_pmcids": len(s["pmcids"]),
                      "pmid_matches": len(s["pmids"] & set(where_pmid)),
                      "pmcid_matches": len(s["pmcids"] & set(where_pmc))}
    matches = []
    for kind, hits, where, by in (("pmid", hit_pmid, where_pmid, by_pmid), ("pmcid", hit_pmc, where_pmc, by_pmc)):
        for x in sorted(hits):
            rows = by[x]
            matches.append({"kind": kind, "id": x, "snapshot_title": titles.get(x), "snapshot_files": sorted(where[x]),
                            "splits": sorted({r["split"] for r in rows}),
                            "article_title": next((r["title"] for r in rows if r.get("title")), None),
                            "rows": [{k: r[k] for k in ("task", "file", "split", "id", "diagnosis")} for r in rows]})
    out = {"created": utc(), "n_files_scanned": dict(n_files), "distinct_pmids_in_snapshot": len(where_pmid),
           "distinct_pmcids_in_snapshot": len(where_pmc), "mentions_by_source": dict(per_source),
           "dermarena_by_split": splits, "n_pmid_matches": len(hit_pmid), "n_pmcid_matches": len(hit_pmc),
           "matches": matches,
           "test_val_matches": [m for m in matches if set(m["splits"]) & {"test", "val"}],
           "title_matches_note": "whole DermArena article title (>= 6 words) found in a raw page. Encyclopedia references "
                                 "cite by text with no PMID, so ID matching alone cannot see them.",
           "title_matches": title_hits}
    (kb / "reports").mkdir(exist_ok=True)
    (kb / "reports/pmid_pmcid_match.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("matches", "test_val_matches")}, indent=1))
    print(f"[kb] test/val ID matches: {len(out['test_val_matches'])}; whole-title matches (any split): {len(title_hits)}, "
          f"test/val: {sum(1 for t in title_hits if set(t['splits']) & {'test', 'val'})}")
    for m in out["test_val_matches"]:
        print(f"  {m['kind']} {m['id']} {m['splits']} {m['snapshot_files'][:3]} :: {m['snapshot_title'] or m['article_title']}")


# ----------------------------------------------------------------------------------------------- parse -> indexed text
class _Text(HTMLParser):
    """Visible text with block structure kept as newlines (chunking splits on them)."""
    BLOCK = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "section", "br", "tr", "table"}
    SKIP = {"script", "style", "button", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_text(fragment):
    p = _Text()
    p.feed(fragment)
    t = "".join(p.out).replace("\xa0", " ")
    lines = [" ".join(x.split()) for x in t.split("\n")]
    out, blank = [], False
    for x in lines:
        if x:
            out.append(x)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()


# Sections of an encyclopedia page that are not prose about the condition. 'references' = textbook / journal citations.
ENCY_DROP = ("references", "review date", "images", "related medlineplus health topics", "patient instructions",
             "clinical trials", "journal articles", "find an expert", "learn more")


def parse_ency(raw, rel):
    """-> dict or None. Only the summary and the kept <section> bodies of #d-article are read."""
    a = raw.find('id="d-article"')
    if a < 0:
        return None
    body = raw[a:]
    end = body.find('id="mplus-footer"')
    body = body[:end] if end > 0 else body
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    title = html_text(m.group(1)) if m else ""
    parts = re.split(r'<div class="section">', body)
    summary = ""
    ms = re.search(r'id="ency_summary"[^>]*>(.*)', parts[0], re.S)
    if ms:
        summary = html_text(ms.group(1))
    kept, dropped, alt = [], [], []
    for part in parts[1:]:
        mh = re.search(r"<h2[^>]*>(.*?)</h2>", part, re.S)
        h = html_text(mh.group(1)) if mh else ""
        sec = part.split("</section>")[0]
        mb = re.search(r'class="section-body"[^>]*>(.*)', sec, re.S)
        text = html_text(mb.group(1)) if mb else ""
        if any(h.lower().startswith(d) for d in ENCY_DROP):
            dropped.append(h)
            continue
        if h.lower() == "alternative names":
            alt = [x.strip() for x in re.split(r"[;\n]", text) if x.strip()]
        if text:
            kept.append(f"{h}\n{text}")
    text = "\n\n".join(([summary] if summary else []) + kept)
    src = "ency_patientinstructions" if "patientinstructions" in rel else "ency_article"
    return {"doc_id": rel.replace("/", "_").replace(".htm", ""), "source": src, "url": f"{SITE}/ency/{rel}",
            "title": title, "alt_names": alt, "text": text, "sections_dropped": dropped}


def parse_topics(xml_path):
    """English topics from the official XML: title, also-called, full-summary. <site> link lists are not read."""
    docs = []
    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag != "health-topic":
            continue
        if el.get("language") == "English":
            fs = el.find("full-summary")
            text = html_text(fs.text or "") if fs is not None else ""
            docs.append({"doc_id": f"topic_{el.get('id')}", "source": "topic", "url": el.get("url"),
                         "title": el.get("title") or "", "alt_names": [x.text for x in el.findall("also-called") if x.text],
                         "text": text, "sections_dropped": ["<all link lists: not read>"],
                         "groups": [x.text for x in el.findall("group") if x.text]})
        el.clear()
    return docs


# Must not survive into docs.jsonl. Strings that only appear in a citation / link list.
FORBIDDEN = [("pubmed", re.compile(r"pubmed|medline/", re.I)), ("pmid", re.compile(r"\bPMID\b")),
             ("pmcid", re.compile(r"\bPMC\d{4,}")), ("clinicaltrials", re.compile(r"clinicaltrials\.gov|\bNCT\d{8}\b", re.I)),
             ("article_link", re.compile(r"^Article:", re.M)), ("doi", re.compile(r"doi\.org|\bdoi:", re.I)),
             ("et_al", re.compile(r"\bet al\b")), ("publisher", re.compile(r"\b(?:Elsevier|Saunders|Wolters Kluwer|Lippincott)\b")),
             ("chap", re.compile(r"\bchap \d+"))]


def check_docs(docs):
    bad = defaultdict(list)
    for d in docs:
        for name, rx in FORBIDDEN:
            m = rx.search(d["text"])
            if m:
                bad[name].append({"doc_id": d["doc_id"], "context": d["text"][max(0, m.start() - 60):m.end() + 60]})
    return bad


def cmd_parse(kb):
    kb = Path(kb)
    xmls = sorted((kb / "raw/xml").glob("mplus_topics_*.xml"))
    docs = parse_topics(xmls[0])
    n_topic = len(docs)
    fails = []
    for f in sorted((kb / "raw/ency").rglob("*.htm")):
        rel = str(f.relative_to(kb / "raw/ency"))
        d = parse_ency(f.read_text(errors="ignore"), rel)
        if d is None or not d["text"].strip():
            fails.append(rel)
        else:
            docs.append(d)
    empty = [d["doc_id"] for d in docs if not d["text"].strip()]
    docs = [d for d in docs if d["text"].strip()]
    for d in docs:
        d["sha256"] = hashlib.sha256(d["text"].encode()).hexdigest()
    with (kb / "docs.jsonl.tmp").open("w") as fo:
        for d in docs:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(kb / "docs.jsonl.tmp", kb / "docs.jsonl")
    bad = check_docs(docs)
    src = Counter(d["source"] for d in docs)
    dropped = Counter(h.split(" ")[0] if h.lower().startswith("review date") else h
                      for d in docs for h in d["sections_dropped"])
    rep = {"created": utc(), "n_docs": len(docs), "by_source": dict(src), "n_topics_in_xml": n_topic,
           "empty_docs_excluded": empty, "unparsable_pages": fails, "sections_dropped": dict(dropped.most_common()),
           "total_chars": sum(len(d["text"]) for d in docs), "forbidden_found": {k: len(v) for k, v in bad.items()},
           "forbidden_examples": {k: v[:20] for k, v in bad.items()}}
    (kb / "reports").mkdir(exist_ok=True)
    (kb / "reports/parse.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "forbidden_examples"}, indent=1)[:3000])


# ----------------------------------------------------------------------------------------------- chunk
def _flat(text, seps, size, length=len):
    """Recursive pieces: (piece, the separator that joins it to the next)."""
    if length(text) <= size:
        return [(text, seps[0])]
    sep = seps[0]
    pieces = list(text) if sep == "" else text.split(sep)
    out = []
    for p in pieces:
        if length(p) > size and len(seps) > 1:
            out += _flat(p, seps[1:], size, length)
        else:
            out.append((p, sep))
    return out


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """= RecursiveCharacterTextSplitter defaults (separators paragraph, line, word, character): pieces are merged up
    to `size`, and the next chunk starts with the last pieces of the previous one, up to `overlap` characters."""
    pieces = [(p, s) for p, s in _flat(text, ["\n\n", "\n", " ", ""], size) if p.strip()]
    chunks, cur, n = [], [], 0
    for p, s in pieces:
        add = len(p) + (len(cur[-1][1]) if cur else 0)
        if cur and n + add > size:
            chunks.append("".join(x + (y if i < len(cur) - 1 else "") for i, (x, y) in enumerate(cur)).strip())
            while cur and (n > overlap or n + add > size):
                n -= len(cur[0][0]) + len(cur[0][1])
                cur.pop(0)
            n = max(n, 0)
            add = len(p) + (len(cur[-1][1]) if cur else 0)
        cur.append((p, s))
        n += add
    if cur:
        chunks.append("".join(x + (y if i < len(cur) - 1 else "") for i, (x, y) in enumerate(cur)).strip())
    return [c for c in chunks if c]


def chunk_profile_name(name=None):
    name = name or os.environ.get("MP_CHUNKS") or DEFAULT_CHUNK_PROFILE
    if name not in CHUNK_PROFILES:
        raise SystemExit(f"[kb] unknown chunk profile {name!r}: {sorted(CHUNK_PROFILES)}")
    return name


def chunks_path(kb, name=None):
    return Path(kb) / CHUNK_PROFILES[chunk_profile_name(name)]["file"]


def chunks_manifest_path(kb, name=None):
    return Path(kb) / "index" / f"CHUNKS_{chunk_profile_name(name)}.json"


def token_counter(encoding):
    """Token count of a string with the embedder's tokenizer. Special-token text in a document is counted as plain text."""
    import tiktoken
    enc = tiktoken.get_encoding(encoding)
    return lambda t: len(enc.encode(t, disallowed_special=()))


def chunk_text_tokens(text, count, size, overlap):
    """chunk_text with tokens in place of characters: same separators, same merge, same overlap rule. A piece is counted
    together with the separator before it (' word' is what the tokenizer sees), which follows the real count closely but not
    exactly (a word opening a chunk can take one token more), so near the limit the joined text itself is counted."""
    pieces = [(p, s) for p, s in _flat(text, ["\n\n", "\n", " ", ""], size, count) if p.strip()]

    def join(cur):
        return "".join(x + (y if i < len(cur) - 1 else "") for i, (x, y, _) in enumerate(cur)).strip()

    chunks, cur, n = [], [], 0
    for p, s in pieces:
        add = count((cur[-1][1] if cur else "") + p)
        if cur and (n + add > size or (n + add > size - 16 and count(join(cur + [(p, s, add)])) > size)):
            chunks.append(join(cur))
            while cur and (n > overlap or n + add > size):
                n -= cur.pop(0)[2]
            n = max(n, 0)
            add = count((cur[-1][1] if cur else "") + p)
        cur.append((p, s, add))
        n += add
    if cur:
        chunks.append(join(cur))
    return [c for c in chunks if c]


def cmd_chunk_tokens(kb, name=None, count=None):
    """Writes the profile's own chunk file and its small manifest. The legacy chunks.jsonl and MANIFEST.json stay as they are:
    a running job reads them, and kb_sha (hence its run_sig) depends on MANIFEST.json."""
    kb = Path(kb)
    name = chunk_profile_name(name)
    prof = CHUNK_PROFILES[name]
    if prof["unit"] != "token":
        raise SystemExit(f"[kb] {name} is not a token profile: use `chunk`")
    count = count or token_counter(prof["tokenizer"])
    out = kb / prof["file"]
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    n, lens, too_long = 0, [], []
    with tmp.open("w") as fo:
        for l in (kb / "docs.jsonl").open():
            d = json.loads(l)
            for i, c in enumerate(chunk_text_tokens(d["text"], count, prof["size"], prof["overlap"])):
                k = count(c)
                if k > prof["size"]:
                    too_long.append((d["doc_id"], i, k))
                fo.write(json.dumps({"row": n, "doc_id": d["doc_id"], "i": i, "text": c}, ensure_ascii=False) + "\n")
                n += 1
                lens.append(k)
    if too_long:
        tmp.unlink()
        raise SystemExit(f"[kb] {len(too_long)} chunks over {prof['size']} tokens, e.g. {too_long[:5]}")
    os.replace(tmp, out)
    lens.sort()
    man = {"created": utc(), "profile": name, "chunking": {k: v for k, v in prof.items() if k != "file"}, "path": prof["file"],
           "n_chunks": n, "tokens_median": lens[len(lens) // 2], "tokens_max": lens[-1], "tokens_total": sum(lens),
           "docs_sha256": sha_path(kb / "docs.jsonl"), "chunks_sha256": sha_path(out)}
    mp = chunks_manifest_path(kb, name)
    mp.parent.mkdir(parents=True, exist_ok=True)
    t2 = mp.with_suffix(f".tmp{os.getpid()}")
    t2.write_text(json.dumps(man, indent=1))
    os.replace(t2, mp)
    print(f"[kb] {name}: {n} chunks, tokens median {man['tokens_median']}, max {man['tokens_max']}, total {man['tokens_total']}")
    return man


def cmd_chunk(kb):
    kb = Path(kb)
    n = 0
    lens = []
    with (kb / "chunks.jsonl.tmp").open("w") as fo:
        for l in (kb / "docs.jsonl").open():
            d = json.loads(l)
            for i, c in enumerate(chunk_text(d["text"])):
                fo.write(json.dumps({"row": n, "doc_id": d["doc_id"], "i": i, "text": c}, ensure_ascii=False) + "\n")
                n += 1
                lens.append(len(c))
    os.replace(kb / "chunks.jsonl.tmp", kb / "chunks.jsonl")
    lens.sort()
    print(f"[kb] {n} chunks, length median {lens[len(lens) // 2]}, max {lens[-1]}")


# ----------------------------------------------------------------------------------------------- manifest
def sha_path(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_manifest(kb):
    kb = Path(kb)
    log = {}
    failed = []
    for l in (kb / "fetch_log.jsonl").open():
        r = json.loads(l)
        if r.get("status") == 200:
            log[r["path"]] = r
        else:
            failed.append(r)
    files = []
    for sub, f in raw_files(kb):
        rel = str(f.relative_to(kb / "raw"))
        if f.name.endswith(".tmp") or ".tmp" in f.name:
            continue
        r = log.get(rel, {})
        sha = sha_path(f)
        if r and r.get("sha256") != sha:
            raise SystemExit(f"[kb] {rel}: sha256 differs from the fetch log")
        files.append({"path": rel, "url": r.get("url"), "retrieved": r.get("retrieved"), "sha256": sha,
                      "bytes": f.stat().st_size})
    still_failed = [r for r in failed if r["path"] not in log]
    m = {"source": "MedlinePlus (National Library of Medicine): health topics XML, topic pages, A.D.A.M. Medical Encyclopedia",
         "created": utc(), "client": UA, "delay_s": DELAY, "n_files": len(files),
         "retrieval_first": min(x["retrieved"] for x in files if x["retrieved"]),
         "retrieval_last": max(x["retrieved"] for x in files if x["retrieved"]),
         "files_not_retrieved": still_failed, "docs_sha256": sha_path(kb / "docs.jsonl"),
         "chunks_sha256": sha_path(kb / "chunks.jsonl"), "chunking": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
         "files": files}
    (kb / "MANIFEST.json.tmp").write_text(json.dumps(m, indent=0))
    os.replace(kb / "MANIFEST.json.tmp", kb / "MANIFEST.json")
    print(f"[kb] MANIFEST: {len(files)} files, sha256 {sha_path(kb / 'MANIFEST.json')[:16]}, not retrieved {len(still_failed)}")


# ----------------------------------------------------------------------------------------------- names (shared with kb_rag)
def normalise_name(name):
    """Case, punctuation, whitespace. Apostrophes vanish ("Behcet's" = "behcets"); other punctuation becomes a space."""
    s = unicodedata.normalize("NFKC", str(name or "")).casefold()
    s = re.sub(r"['‘’`]", "", s)
    s = re.sub(r"[^\w\s]|_", " ", s)
    return " ".join(s.split())


STOP = set("""of the and with without due to in on a an or by for from at as is
disease diseases disorder disorders syndrome syndromes condition conditions type types form forms variant variants
acute chronic primary secondary congenital acquired idiopathic localized localised generalized generalised
skin cutaneous dermal systemic nos unspecified other associated related induced severe mild""".split())


def stem(t):
    return t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t


def keywords(name):
    """Content words of a disease name. All words if every word is generic."""
    toks = normalise_name(name).split()
    kw = [stem(t) for t in toks if t not in STOP]
    return kw or [stem(t) for t in toks]


def doc_tokens(text):
    return {stem(t) for t in normalise_name(text).split()}


def keyword_match(kw, tokens):
    """Runtime rule of the keyword filter: every content word of the query is in the document's index text."""
    return bool(kw) and all(k in tokens for k in kw)


def first_sentence(text):
    m = re.search(r"(.+?[.!?])(\s|$)", text.replace("\n", " "))
    return m.group(1) if m else text[:200]


# ----------------------------------------------------------------------------------------------- coverage
def gold_rows(task):
    return [json.loads(l) for l in open(f"{RELEASE_TASKS}/{task}/test.jsonl")]


def cmd_coverage(kb):
    kb = Path(kb)
    docs = [json.loads(l) for l in (kb / "docs.jsonl").open()]
    exact = {}
    for d in docs:
        for n in [d["title"]] + d.get("alt_names", []):
            exact.setdefault(normalise_name(n), d["doc_id"])
    disease_docs = [d for d in docs if d["source"] != "ency_patientinstructions"]
    # Proxy for the runtime keyword filter (real one-sentence summaries need the GPU step): title + alternative names +
    # the document's first sentence stand in for the summary.
    index = [doc_tokens(" ".join([d["title"]] + d.get("alt_names", []) + [first_sentence(d["text"])])) for d in disease_docs]
    inv = defaultdict(set)
    for i, toks in enumerate(index):
        for t in toks:
            inv[t].add(i)

    def kw_hit(name):
        kw = keywords(name)
        if not kw:
            return False
        s = None
        for k in kw:
            s = inv.get(k, set()) if s is None else s & inv.get(k, set())
            if not s:
                return False
        return True

    def stats(names):
        """names: list of raw strings, one per occurrence."""
        c = Counter(normalise_name(n) for n in names if normalise_name(n))
        tot = sum(c.values())
        e = {n for n in c if n in exact}
        k = {n for n in c if n in e or kw_hit(n)}
        return {"occurrences": tot, "distinct_names": len(c),
                "exact_name_match": {"distinct": len(e), "distinct_pct": round(100 * len(e) / max(1, len(c)), 1),
                                     "occurrences_pct": round(100 * sum(c[n] for n in e) / max(1, tot), 1)},
                "keyword_proxy_match": {"distinct": len(k), "distinct_pct": round(100 * len(k) / max(1, len(c)), 1),
                                        "occurrences_pct": round(100 * sum(c[n] for n in k) / max(1, tot), 1)}}

    rep = {"created": utc(), "note": "exact = normalised name equals a document title or alternative name. keyword proxy = "
           "the runtime keyword rule applied to title + alternative names + first sentence, standing in for the "
           "one-sentence summaries that the GPU step writes.", "n_docs": len(docs),
           "n_exact_names": len(exact)}
    allg = []
    for t in ("T1.1_disease_screening", "T1.2_disease_confirmation"):
        rows = gold_rows(t)
        rep[f"gold_diagnosis_{t}"] = stats([r.get("diagnosis") for r in rows])
        rep[f"gold_label_title_{t}"] = stats([r.get("disease_label_title") for r in rows])
        allg += rows
    preds, top1 = [], []
    n_rows = n_parsed = 0
    for l in open(QWEN_T11):
        r = json.loads(l)
        n_rows += 1
        try:
            items = json.loads(r["prediction"]).get("diagnoses") or []
        except Exception:  # noqa: BLE001
            try:
                sys.path.insert(0, "/fs04/scratch2/ub62/ssim0070/DermArena/dataset_collection/dx_benchmark")
                import dx_parse
                items = dx_parse.parse("T1.1", r.get("prediction"), r.get("finish_reason"))[0] or []
            except Exception:  # noqa: BLE001
                items = []
        items = [x for x in items if isinstance(x, str)][:5]
        n_parsed += bool(items)
        preds += items
        top1 += items[:1]
    rep["qwen_top5_T1.1"] = dict(stats(preds), rows=n_rows, rows_with_items=n_parsed)
    rep["qwen_top1_T1.1"] = stats(top1)
    (kb / "reports").mkdir(exist_ok=True)
    (kb / "reports/coverage.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ----------------------------------------------------------------------------------------------- leak scan
SHINGLE = 10
TEXT_FIELDS = ("case_report", "examination_results", "tables", "narrative", "history", "question", "options")


def words(text):
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())


def shingles(ws):
    return (" ".join(ws[i:i + SHINGLE]) for i in range(len(ws) - SHINGLE + 1))


def field_text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) if v else ""


def cmd_leakscan(kb, profile=None):
    """Both views: the indexed text (what the agent can read) and the full visible text of every raw page. With a chunk
    profile the indexed text is that profile's chunk file, and the reports get its name, so earlier reports stay."""
    kb = Path(kb)
    sources = {}
    if profile:
        for l in chunks_path(kb, profile).open():
            d = json.loads(l)
            sources[("indexed", f"{d['doc_id']}#{d['i']}")] = d["text"]
    else:
        for l in (kb / "docs.jsonl").open():
            d = json.loads(l)
            sources[("indexed", d["doc_id"])] = d["text"]
    tag = f"_{profile}" if profile else ""
    for sub, f in raw_files(kb):
        if sub in ("topics", "ency"):
            sources[("raw", str(f.relative_to(kb / "raw")))] = html_text(f.read_text(errors="ignore"))
    for x in sorted((kb / "raw/xml").glob("mplus_topics_*.xml")):
        for _, el in ET.iterparse(x, events=("end",)):
            if el.tag == "health-topic":
                t = " ".join(html_text(c.text or "") for c in el.iter("full-summary"))
                t += " " + " ".join(s.get("title") or "" for s in el.iter("site"))
                sources[("raw", f"xml:{el.get('language')}:{el.get('id')}")] = t
                el.clear()
    table = {}   # shingle hash -> bit 1 indexed, 2 raw
    for (view, _), t in sources.items():
        bit = 1 if view == "indexed" else 2
        for s in shingles(words(t)):
            h = hash(s)
            table[h] = table.get(h, 0) | bit
    print(f"[kb] {len(sources)} source texts, {len(table)} distinct {SHINGLE}-word shingles", flush=True)
    hits, seen_text = [], {}
    scanned = Counter()
    for task, split, p in dermarena_files():
        if split not in ("test", "val"):
            continue
        for l in p.open():
            r = json.loads(l)
            for fld in TEXT_FIELDS:
                t = field_text(r.get(fld))
                if not t:
                    continue
                scanned[f"{task}/{split}"] += 1
                key = hashlib.md5(t.encode()).hexdigest()
                if key not in seen_text:
                    ws = words(t)
                    found = [(i, table[hash(" ".join(ws[i:i + SHINGLE]))]) for i in range(len(ws) - SHINGLE + 1)
                             if hash(" ".join(ws[i:i + SHINGLE])) in table]
                    spans = []
                    for i, bits in found:     # merge overlapping shingles into maximal spans
                        if spans and i <= spans[-1][1]:
                            spans[-1][1] = i + SHINGLE
                            spans[-1][2] |= bits
                        else:
                            spans.append([i, i + SHINGLE, bits])
                    seen_text[key] = [{"text": " ".join(ws[a:b]), "n_words": b - a, "in_indexed": bool(c & 1),
                                       "in_raw": bool(c & 2)} for a, b, c in spans]
                for sp in seen_text[key]:
                    hits.append(dict(sp, task=task, file=p.name, split=split, id=row_id(r), field=fld,
                                     pmcid=r.get("pmcid"), diagnosis=r.get("diagnosis")))
    # name the source document of each distinct span (second, small pass)
    spans = {h["text"] for h in hits}
    wanted = defaultdict(set)
    for s in spans:
        ws = s.split()
        wanted[" ".join(ws[:SHINGLE])].add(s)
    src = defaultdict(set)
    for (view, name), t in sources.items():
        for sh in shingles(words(t)):
            if sh in wanted:
                for s in wanted[sh]:
                    src[s].add(f"{view}:{name}")
    for h in hits:
        h["sources"] = sorted(src.get(h["text"], []))[:10]
    (kb / "reports").mkdir(exist_ok=True)
    with (kb / f"reports/leak_scan_hits{tag}.jsonl").open("w") as fo:
        for h in hits:
            fo.write(json.dumps(h, ensure_ascii=False) + "\n")
    by = Counter((h["task"], h["split"], "indexed" if h["in_indexed"] else "raw_only") for h in hits)
    distinct = Counter(h["text"] for h in hits)
    rep = {"created": utc(), "chunk_profile": profile, "shingle_words": SHINGLE, "fields": TEXT_FIELDS, "n_source_texts": len(sources),
           "n_distinct_shingles": len(table), "text_fields_scanned": dict(scanned), "n_hit_rows": len(hits),
           "n_distinct_spans": len(distinct), "n_distinct_case_ids": len({h["id"] for h in hits}),
           "n_hits_in_indexed": sum(h["in_indexed"] for h in hits),
           "by_task_split": {" ".join(k): v for k, v in sorted(by.items())},
           "longest_spans": [{"n_words": len(s.split()), "n_rows": c, "text": s} for s, c in
                             sorted(distinct.items(), key=lambda x: -len(x[0].split()))[:40]],
           "most_frequent_spans": [{"n_rows": c, "text": s} for s, c in distinct.most_common(40)]}
    (kb / f"reports/leak_scan{tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("longest_spans", "most_frequent_spans")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "ids", "parse", "chunk", "chunk-tokens", "manifest", "coverage", "leakscan"])
    ap.add_argument("--kb", default=KB_DEFAULT)
    ap.add_argument("--chunks", default=None, help="chunk profile: for chunk-tokens, and for leakscan on that chunk file")
    a = ap.parse_args()
    if a.cmd == "chunk-tokens":
        cmd_chunk_tokens(a.kb, a.chunks)
    elif a.cmd == "leakscan":
        cmd_leakscan(a.kb, a.chunks)
    else:
        {"download": cmd_download, "ids": cmd_ids, "parse": cmd_parse, "chunk": cmd_chunk, "manifest": cmd_manifest,
         "coverage": cmd_coverage}[a.cmd](a.kb)
