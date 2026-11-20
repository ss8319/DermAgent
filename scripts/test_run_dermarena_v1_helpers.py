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

    # F3 / F4: end markers of a service job (2 shards). stop() always has a deadline, so a regressed wait loop fails the
    # test instead of spinning forever.
    import time as _t

    def stopper(jd, secs=5.0):
        t_end = _t.time() + secs
        def stop():
            if _t.time() > t_end:
                raise SystemExit("FAIL finish_job wait loop did not exit within the deadline")
            return (jd / "STOP").exists()
        return stop

    def fj(jd, k, left, merged, n=2):
        with redirect_stdout(io.StringIO()):
            return v1.finish_job(jd, "J", k, n, [("T1.1", d, [])], left, "SIG", lambda p: merged.append(k),
                                 stop=stopper(jd), poll=0.01)

    def fresh(name):
        jd = Path(_TMP) / f"jobs_{name}"
        jd.mkdir()
        (jd / "J.json").write_text("{}")
        return jd

    def markers(jd):
        return sorted(p.name for p in jd.glob("J.*") if p.name != "J.json")

    jd, m = fresh("all_done"), []
    r1, r0 = fj(jd, 1, {"T1.1": []}, m), fj(jd, 0, {"T1.1": []}, m)
    check(r1 and r0 and m == [0] and markers(jd) == ["J.ALLDONE", "J.p0.done", "J.p1.done"],
          "all shards done -> shard 0 merges once + ALLDONE")
    jd, m = fresh("shard1_short"), []
    r1, r0 = fj(jd, 1, {"T1.1": ["x"]}, m), fj(jd, 0, {"T1.1": []}, m)
    check(r1 and r0 and m == [0] and markers(jd) == ["J.ENDED_INCOMPLETE", "J.p0.done", "J.p1.incomplete"],
          "F3: an incomplete shard -> merge + ENDED_INCOMPLETE (waiters get a terminal marker)")
    check(json.loads((jd / "J.ENDED_INCOMPLETE").read_text())["shards"] == {"0": "done", "1": "incomplete"},
          "ENDED_INCOMPLETE records each shard's state")
    jd, m = fresh("shard0_short"), []
    r1, r0 = fj(jd, 1, {"T1.1": []}, m), fj(jd, 0, {"T1.1": ["x"]}, m)
    check(r1 and r0 and m == [0] and markers(jd) == ["J.ENDED_INCOMPLETE", "J.p0.incomplete", "J.p1.done"],
          "F4: shard 0 incomplete still merges + postchecks + writes the end marker")
    jd, m = fresh("one_shard"), []
    r0 = fj(jd, 0, {"T1.1": ["x"]}, m, n=1)
    check(r0 and m == [0] and markers(jd) == ["J.ENDED_INCOMPLETE", "J.p0.incomplete"], "n=1 incomplete -> merge + end marker")

    # R1 (fix review): STOP during shard 0's wait -> no end marker; after restart shard 0 picks the job up again and finishes
    jd, m = fresh("stop_wait"), []
    (jd / "STOP").write_text("x")
    r0 = fj(jd, 0, {"T1.1": []}, m)
    check(r0 and m == [0] and markers(jd) == ["J.p0.done"], "STOP while shard 1 is open -> partial merge, no end marker")
    (jd / "STOP").unlink()
    check([p.name for p in v1.open_jobs(jd, 0)] == ["J.json"], "R1: shard 0 still owns a job without an end marker")
    check([p.name for p in v1.open_jobs(jd, 1)] == ["J.json"], "shard 1 still has its share to do")
    fj(jd, 1, {"T1.1": []}, m)
    check(v1.open_jobs(jd, 1) == [], "shard 1 closes its share with .p1.done")
    r0 = fj(jd, 0, {"T1.1": []}, m)
    check(r0 and m == [0, 0] and (jd / "J.ALLDONE").exists() and v1.open_jobs(jd, 0) == [],
          "R1: after restart shard 0 re-merges and writes ALLDONE; the job is then closed")

    # R2: a retry ends with exactly one end marker
    jd, m = fresh("retry"), []
    fj(jd, 1, {"T1.1": ["x"]}, m), fj(jd, 0, {"T1.1": []}, m)
    (jd / "J.p1.incomplete").unlink()
    (jd / "J.ENDED_INCOMPLETE").unlink()          # the retry hint printed by finish_job
    check([p.name for p in v1.open_jobs(jd, 0)] == ["J.json"], "R2: after the retry hint shard 0 owns the job again")
    fj(jd, 1, {"T1.1": []}, m), fj(jd, 0, {"T1.1": []}, m)
    check(markers(jd) == ["J.ALLDONE", "J.p0.done", "J.p1.done"], "R2: successful retry -> ALLDONE only, stale markers gone")

    jd, m = fresh("stop_left"), []
    (jd / "STOP").write_text("x")
    r = fj(jd, 0, {"T1.1": ["x"]}, m, n=1)
    check(r is False and m == [] and markers(jd) == [], "STOP with cases left -> nothing written, job stays open")

    # wiring: both runners use the shared helpers and no done_ids() call sits in a per-case comprehension (F2)
    import re as _re
    for src in (HERE / "run_dermarena_v1.py", HERE.parent / "baselines/MedAgent-Pro/run_dermarena_v1_medagentpro.py"):
        txt = src.read_text()
        check("finish_job(" in txt and "shard_left(" in txt and "open_jobs(" in txt
              and not _re.search(r"\bfor\b[^\n]*\bin\b[^\n]*done_ids\(", txt.split("def main")[-1]),
              f"{src.name}: service loop uses open_jobs/shard_left/finish_job, no done_ids per case")

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
    import shutil
    shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n{N['ok']}/{N['ok']} checks passed (work dir removed)")


if __name__ == "__main__":
    main()
