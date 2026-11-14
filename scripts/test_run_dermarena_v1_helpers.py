#!/usr/bin/env python3
"""CPU-only offline test of the resume / merge / image-cache helpers in run_dermarena_v1.py (no GPU, no model, no server).
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

    check(v1.done_ids(d) == {"a", "b", "c"}, "done_ids without run_sig counts every readable row (old behaviour)")
    check(v1.done_ids(d, "NEW") == {"a", "b"}, "done_ids(run_sig) skips rows of other settings and torn lines")

    buf = io.StringIO()
    with redirect_stdout(buf):
        v1.merge(d)
    rows = {json.loads(l)["_id"]: json.loads(l) for l in (d / "predictions.jsonl").open()}
    check(set(rows) == {"a", "b", "c"}, "merge keeps one row per _id")
    check(rows["a"]["prediction"] == "new-a", "merge prefers the RUN_META run_sig row over an earlier row of other settings")
    check("MIXED run_sig" in buf.getvalue() and "'OLD': 1" in buf.getvalue(), "merge reports the kept row of other settings")
    check(not list(d.glob("predictions.jsonl.tmp*")), "merge leaves no temp file")

    write(d / "predictions.p1.jsonl", [{"_id": "a", "run_sig": "NEW"}, {"_id": "c", "run_sig": "NEW"}])
    buf = io.StringIO()
    with redirect_stdout(buf):
        v1.merge(d)
    check("MIXED" not in buf.getvalue(), "merge is quiet when every kept row has the run_sig")

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
