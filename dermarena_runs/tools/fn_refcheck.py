#!/usr/bin/env python3
"""fn_refcheck.py (11-13) - does a NEW Flash-Next server compute the same model as the H200 reference server?

Why: the A100 is not on Qwen3.8-Flash-Next's supported BF16 hardware list, so its sparse-attention decode path is untested.
Before any MedAgent-Pro case runs on a new server we compare it with the H200 server that produced cases so far.

  record  <url> <out_dir>   (on the H200, once) 20 fixed prompts (15 short, 5 long ~10k tokens to reach the sparse-attention path):
                            greedy continuation (48 tokens) + teacher-forced per-token logprobs of prompt+continuation.
  compare <url> <ref_dir>   (on the new server) same requests; PASS if
                            - teacher-forced: median over prompts of mean |dlogprob| < 0.10 and every prompt < 0.20 (11-13 recalibrated, see compare())
                            - greedy continuation identical in >= 10 / 20 (H200 vs itself: 10/15)
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


TAIL = int(os.environ.get("FN_REFCHECK_TAIL", "1024"))


def one(url, p, cont=None, start=0):
    if cont is None:
        g = post(url, {"text": p, "sampling_params": {"max_new_tokens": 48, "temperature": 0}})
        cont = g["text"]
    t = post(url, {"text": p + cont, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                   "return_logprob": True, "logprob_start_len": start})
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
        # 11-13 20:15: long prompts score only their LAST TAIL tokens (logprob_start_len), aligned to the reference from the end:
        # full-length input logprobs (up to 38k x full vocab) OOM'd the 80 GB A100 pp2 server even one request at a time. The tail
        # tokens still attend over the whole long context, which is the path this check exists for.
        R0 = len(it["lp"]); start = max(0, R0 + 1 - TAIL) if R0 > 2 * TAIL else 0
        _, lp = one(url, it["prompt"], it["cont"], start)      # teacher-forced on the reference continuation
        ref_lp = it["lp"][-len(lp):] if start else it["lp"]
        n = min(len(lp), len(ref_lp))
        dl = statistics.fmean(abs(a - b) for a, b in zip(lp[:n], ref_lp[:n])) if n else 9.9
        print(f"[refcheck] {it['i']}: same={cont == it['cont']} mean|dlp|={dl:.4f} n={n}", flush=True)
        return {"i": it["i"], "same_greedy": cont == it["cont"], "mean_abs_dlogprob": dl, "n_tok": n,
                "len_match": abs(len(lp) - (R0 + 1 - start if start else R0)) <= (1 if start else 0)}
    # 11-13 20:10: one request at a time (env FN_REFCHECK_THREADS): 20 parallel teacher-forced requests (input logprobs over
    # up to 38k tokens -> full-vocab logits) OOM'd the 80 GB A100 pp2 server at mem-fraction 0.92; normal runs never ask for them.
    with ThreadPoolExecutor(int(os.environ.get("FN_REFCHECK_THREADS", "1"))) as ex:
        res = list(ex.map(cmp, ref))
    med = statistics.median(r["mean_abs_dlogprob"] for r in res)
    worst = max(r["mean_abs_dlogprob"] for r in res)
    same = sum(r["same_greedy"] for r in res)
    # 11-13 20:10 RECALIBRATED before any full A100 result, from the H200's own run-to-run noise (reference_v1_short.json
    # 11:48 vs reference.json 12:00, same 15 prompts, same server): greedy identical 10/15; mean|dlp| 0 where the batch
    # happened to compute identically, 0.051-0.083 where it did not. The old bar (median < 0.05, greedy >= 14/20) was as
    # strict as H200-vs-H200. A broken kernel gives |dlp| >> 0.2 and garbled text; the 0.20 per-prompt cap stays.
    ok = med < 0.10 and worst < 0.20 and same >= 10 and all(r["len_match"] for r in res)
    # 11-13 21:05 (owner OK'd the pp2 A100 server as noise: "OK kinda sounds like it cud be just noise"): a FAIL caused ONLY by the
    # per-prompt cap (median, greedy and token lengths all pass, worst < 0.5; pp2 had 0.33 on the highest-entropy prompt) exits 3
    # = PASS_NOISE_CAP, which callers accept only while dx/<slug>/servers/A100_REFCHECK_OVERRIDE exists.
    noise_cap = (not ok) and med < 0.10 and same >= 10 and all(r["len_match"] for r in res) and worst < 0.5
    summ = {"verdict": "PASS" if ok else ("PASS_NOISE_CAP" if noise_cap else "FAIL"), "median_mean_abs_dlogprob": med, "worst": worst, "same_greedy": same,
            "n": len(res), "server": url, "host": socket.gethostname(), "time": time.strftime("%F %T"), "items": res}
    json.dump(summ, open(f"{d}/compare_{url.split('//')[1].split(':')[0]}_{time.strftime('%m%d_%H%M')}.json", "w"), indent=1)
    print(f"[refcheck] {summ['verdict']}: median |dlogprob| {med:.4f}, worst {worst:.4f}, greedy identical {same}/{len(res)}",
          flush=True)
    return 0 if ok else (3 if noise_cap else 1)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[refcheck] cannot run: {e!r}", flush=True)
        sys.exit(2)
