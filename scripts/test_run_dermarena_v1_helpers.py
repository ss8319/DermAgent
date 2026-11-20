#!/usr/bin/env python3
"""CPU-only offline test of the resume / merge / end-marker / image-cache helpers in run_dermarena_v1.py (no GPU, no model, no server).
Usage: <dermagent python> scripts/test_run_dermarena_v1_helpers.py
"""
import io
import json
import multiprocessing as mp
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_TMP = tempfile.mkdtemp(prefix="v1_helpers_")
os.environ.setdefault("DERMAGENT_IMG_CACHE", str(Path(_TMP) / "rgb_cache"))   # spawned workers inherit the parent's
import run_dermarena_v1 as v1  # noqa: E402

N = {"ok": 0}


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL {msg}")
    N["ok"] += 1
    print(f"PASS {msg}")


def write(p, rows):
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _convert(src):
    return v1.rgb_path(Path(src))


def main():
    d = Path(_TMP) / "out"
    d.mkdir()
    (d / "RUN_META.json").write_text(json.dumps({"run_sig": "NEW"}))
    write(d / "predictions.p0.jsonl", [{"_id": "a", "run_sig": "OLD", "prediction": "old-a"},
                                       {"_id": "b", "run_sig": "NEW", "prediction": "new-b"}])
    write(d / "predictions.p1.jsonl", [{"_id": "a", "run_sig": "NEW", "prediction": "new-a"},
                                       {"_id": "c", "run_sig": "OLD", "prediction": "old-c"}])
    with (d / "predictions.p1.jsonl").open("a") as f:
        f.write('{"_id": "torn", "run_sig": "NEW", "predi')          # torn last line: never counted

    # F1 (PR #4 review): resume must count every readable row, whatever its run_sig. The run_sig hashes the runner code,
    # so a per-row filter re-ran whole finished folders after any runner edit (and ignored META_OVERRIDES).
    check(v1.done_ids(d) == {"a", "b", "c"}, "done_ids counts every readable row of any run_sig, skips torn lines")
    import inspect
    check(list(inspect.signature(v1.done_ids).parameters) == ["out_dir"], "done_ids has no run_sig filter")

    buf = io.StringIO()
    with redirect_stdout(buf):
        v1.merge(d)
    rows = {json.loads(l)["_id"]: json.loads(l) for l in (d / "predictions.jsonl").open()}
    check(set(rows) == {"a", "b", "c"}, "merge keeps one row per _id")
    check(rows["a"]["prediction"] == "old-a", "merge keeps the first row per _id in shard order (unchanged rule)")
    check("MIXED run_sig" in buf.getvalue(), "merge reports a folder with rows of several run_sigs")
    check(not list(d.glob("predictions.jsonl.tmp*")), "merge leaves no temp file")

    write(d / "predictions.p0.jsonl", [{"_id": "a", "run_sig": "NEW"}, {"_id": "b", "run_sig": "NEW"}])
    write(d / "predictions.p1.jsonl", [{"_id": "c", "run_sig": "NEW"}])
    buf = io.StringIO()
    with redirect_stdout(buf):
        v1.merge(d)
    check("MIXED" not in buf.getvalue(), "merge is quiet when every row has one run_sig")

    # F2: shard_left reads each output folder once, not once per case
    calls = {"n": 0}
    real = v1.done_ids

    def counting(od):
        calls["n"] += 1
        return real(od)
    v1.done_ids = counting
    try:
        left = v1.shard_left([("T1.1", d, [{"_id": x} for x in ("a", "b", "c", "z", "y")])])
    finally:
        v1.done_ids = real
    check(left == {"T1.1": ["z", "y"]} and calls["n"] == 1, "shard_left: missing ids in plan order, one scan per task")

    # F3 / F4: end markers of a service job (2 shards)
    def scenario(name, left0, left1, stop_after=None):
        jd = Path(_TMP) / f"jobs_{name}"
        jd.mkdir()
        merged = []
        plan = [("T1.1", d, [])]
        with redirect_stdout(io.StringIO()):
            r1 = v1.finish_job(jd, "J", 1, 2, plan, left1, "SIG", lambda p: merged.append(1), poll=0)
            if stop_after:
                (jd / "J.p1.done").unlink(missing_ok=True)
                (jd / "J.p1.incomplete").unlink(missing_ok=True)
                (jd / "STOP").write_text("x")
            r0 = v1.finish_job(jd, "J", 0, 2, plan, left0, "SIG", lambda p: merged.append(1), poll=0)
        return jd, merged, r0, r1
    jd, merged, r0, r1 = scenario("all_done", {"T1.1": []}, {"T1.1": []})
    check((jd / "J.ALLDONE").exists() and not (jd / "J.ENDED_INCOMPLETE").exists() and merged == [1],
          "all shards done -> merge once + ALLDONE")
    jd, merged, r0, r1 = scenario("shard1_short", {"T1.1": []}, {"T1.1": ["x"]})
    check((jd / "J.p1.incomplete").exists() and (jd / "J.ENDED_INCOMPLETE").exists() and not (jd / "J.ALLDONE").exists()
          and merged == [1], "F3: an incomplete shard -> merge + ENDED_INCOMPLETE (waiters get a terminal marker)")
    jd, merged, r0, r1 = scenario("shard0_short", {"T1.1": ["x"]}, {"T1.1": []})
    check((jd / "J.p0.incomplete").exists() and (jd / "J.ENDED_INCOMPLETE").exists() and merged == [1],
          "F4: shard 0 incomplete still merges + postchecks + writes the end marker")
    jd, merged, r0, r1 = scenario("stop_wait", {"T1.1": []}, {"T1.1": []}, stop_after=True)
    check((jd / "J.p0.done").exists() and not (jd / "J.ALLDONE").exists() and not (jd / "J.ENDED_INCOMPLETE").exists(),
          "STOP while a shard is open -> no end marker (job not finished)")
    jd = Path(_TMP) / "jobs_stop_left"
    jd.mkdir()
    (jd / "STOP").write_text("x")
    with redirect_stdout(io.StringIO()):
        r = v1.finish_job(jd, "J", 0, 1, [("T1.1", d, [])], {"T1.1": ["x"]}, "SIG", lambda p: None, poll=0)
    check(r is False and not list(jd.glob("J.*")), "STOP with cases left -> nothing written, job stays open")

    # scripts/wait_job_end.sh: bounded wait, accepts both end markers
    import subprocess
    w = str(HERE / "wait_job_end.sh")
    jd = Path(_TMP) / "jobs_wait"
    jd.mkdir()
    rc = lambda: subprocess.run(["bash", w, str(jd), "J", "3", "1"], capture_output=True).returncode  # noqa: E731
    check(rc() == 3, "wait_job_end: no marker -> timeout rc 3")
    (jd / "J.ENDED_INCOMPLETE").write_text("{}")
    check(rc() == 2, "wait_job_end: ENDED_INCOMPLETE -> rc 2")
    (jd / "J.ALLDONE").write_text("x")
    check(rc() == 0, "wait_job_end: ALLDONE -> rc 0")

    from PIL import Image
    src = Path(_TMP) / "img.png"
    Image.new("RGBA", (256, 256), (10, 200, 30, 255)).save(src)
    with mp.get_context("spawn").Pool(4) as pool:
        outs = pool.map(_convert, [str(src)] * 8)
    check(len(set(outs)) == 1 and outs[0] != str(src), "concurrent conversions from several processes share one cache path")
    with Image.open(outs[0]) as im:
        im.load()
        check(im.mode == "RGB" and im.size == (256, 256), "cached PNG is complete and RGB")
    check(not list(Path(os.environ["DERMAGENT_IMG_CACHE"]).glob("*.tmp.png")), "no temp PNG left in the cache")
    print(f"\n{N['ok']}/{N['ok']} checks passed | work dir {_TMP}")


if __name__ == "__main__":
    main()
