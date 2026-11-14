#!/usr/bin/env python3
"""fn_refcheck.py (11-13) - does a NEW Flash-Next server compute the same model as the H200 reference server?

Why: the A100 is not on Qwen3.8-Flash-Next's supported BF16 hardware list, so its sparse-attention decode path is untested.
Before any MedAgent-Pro case runs on a new server we compare it with the H200 server that produced cases so far.

  record  <url> <out_dir>   (on the H200, once) 20 fixed prompts (15 short, 5 long ~10k tokens to reach the sparse-attention path):
                            greedy continuation (48 tokens) + teacher-forced per-token logprobs of prompt+continuation.
  compare <url> <ref_dir>   (on the new server) same requests; PASS if
                            - teacher-forced: median over prompts of mean |dlogprob| < 0.05 and every prompt < 0.20
                            - greedy continuation identical in >= 14 / 20 (near-ties may flip on other hardware)
                            exit 0 PASS, 1 FAIL, 2 cannot run. Writes <ref_dir>/compare_<host>_<time>.json.
Uses SGLang's native /generate (text in, logprobs out), temperature 0, so sampling settings play no part.
"""
import json, os, socket, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

R = "/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks/T1.1_disease_screening/test.jsonl"


def post(url, body, timeout=3600):
    req = urllib.request.Request(url.rstrip("/").removesuffix("/v1") + "/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def prompts():
    rows = [json.loads(l) for l in open(R) if l.strip()]
    text = lambda r: (r.get("case_report") or r.get("case_text") or r.get("text") or json.dumps(r)[:4000])
    short = [f"Clinical case:\n{text(r)[:3000]}\n\nThe single most likely diagnosis is" for r in rows[:15]]
    long_ = []
    start = 100
    for n in (15, 40, 80, 140, 200):          # ~4k .. ~40k tokens: real MedAgent-Pro calls reach tens of thousands of tokens
        block = "\n\n".join(text(r)[:6000] for r in rows[start: start + n]); start += n
        long_.append(f"Clinical cases:\n{block[:200000]}\n\nFor the last case, the single most likely diagnosis is")
    return short + long_


def one(url, p, cont=None):
    if cont is None:
        g = post(url, {"text": p, "sampling_params": {"max_new_tokens": 48, "temperature": 0}})
        cont = g["text"]
    t = post(url, {"text": p + cont, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                   "return_logprob": True, "logprob_start_len": 0})
    lp = [x[0] for x in t["meta_info"]["input_token_logprobs"] if x[0] is not None]
    return cont, lp


def main():
    mode, url, d = sys.argv[1:4]
    os.makedirs(d, exist_ok=True)
    ps = prompts()
    if mode == "record":
        def rec(ip):
            i, p = ip; cont, lp = one(url, p)
            print(f"[refcheck] record {i + 1}/{len(ps)} tokens={len(lp)} cont={cont[:60]!r}", flush=True)
            return {"i": i, "prompt": p, "cont": cont, "lp": lp}
        with ThreadPoolExecutor(len(ps)) as ex:
            out = list(ex.map(rec, enumerate(ps)))
        json.dump({"url": url, "host": url.split("//")[1].split(":")[0], "time": time.strftime("%F %T"), "items": out},
                  open(f"{d}/reference.json", "w"))
        print("[refcheck] reference saved", flush=True)
        return 0
    ref = json.load(open(f"{d}/reference.json"))["items"]
    def cmp(it):
        cont, _ = one(url, it["prompt"])                       # greedy on the new server
        _, lp = one(url, it["prompt"], it["cont"])             # teacher-forced on the reference continuation
        n = min(len(lp), len(it["lp"]))
        dl = statistics.fmean(abs(a - b) for a, b in zip(lp[:n], it["lp"][:n])) if n else 9.9
        print(f"[refcheck] {it['i']}: same={cont == it['cont']} mean|dlp|={dl:.4f} n={n}", flush=True)
        return {"i": it["i"], "same_greedy": cont == it["cont"], "mean_abs_dlogprob": dl, "n_tok": n,
                "len_match": len(lp) == len(it["lp"])}
    with ThreadPoolExecutor(len(ref)) as ex:
        res = list(ex.map(cmp, ref))
    med = statistics.median(r["mean_abs_dlogprob"] for r in res)
    worst = max(r["mean_abs_dlogprob"] for r in res)
    same = sum(r["same_greedy"] for r in res)
    ok = med < 0.05 and worst < 0.20 and same >= 14 and all(r["len_match"] for r in res)
    summ = {"verdict": "PASS" if ok else "FAIL", "median_mean_abs_dlogprob": med, "worst": worst, "same_greedy": same,
            "n": len(res), "server": url, "host": socket.gethostname(), "time": time.strftime("%F %T"), "items": res}
    json.dump(summ, open(f"{d}/compare_{url.split('//')[1].split(':')[0]}_{time.strftime('%m%d_%H%M')}.json", "w"), indent=1)
    print(f"[refcheck] {summ['verdict']}: median |dlogprob| {med:.4f}, worst {worst:.4f}, greedy identical {same}/{len(res)}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[refcheck] cannot run: {e!r}", flush=True)
        sys.exit(2)
