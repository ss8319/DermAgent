"""Planning-only probe for MedAgent-Pro (no tools, no patients): build plans for N diseases through the real runner code
(get_plan -> guideline, indicators, planner + checks, weights) and report which tools the plans use. Plans go into the REAL plan
cache under the current namespace, so a later canary reuses them. Usage (via run_medagentpro_v1.sh plan-probe):
  python3 mp_plan_probe.py --kb KB --plan-cache CACHE --backbone B --from-ns OLDNS --n 20 --seed 0 --threads 20"""
import argparse, json, random, sys, collections, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, "/fs04/scratch2/ub62/ssim0070/DermAgent/baselines/MedAgent-Pro")
import run_dermarena_v1_medagentpro as R

ap = argparse.ArgumentParser()
for x in ("--kb", "--plan-cache", "--backbone", "--from-ns"):
    ap.add_argument(x, required=True)
ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=20)
a = ap.parse_args()
R.check_backbone_env(a.backbone)
M = R.mods()
task_cfg, toolset = R.load_task_cfg()
kb_sha = M["kb_rag"].kb_sha(a.kb, a.backbone)
ns = R.namespace(a.backbone, kb_sha, R.sha_path(R.DATA_ROOT / "toolset.json")[:16], R.sha_path(R.DATA_ROOT / "task.json")[:16])
old = Path(a.plan_cache) / a.from_ns
diseases = sorted({json.loads(p.read_text())["disease"] for p in old.glob("*.json")})
random.Random(a.seed).shuffle(diseases)
diseases = diseases[:a.n]
client = R.make_client()
decider, planner = R.make_decider(M["Pro_Decider"])(R.API_KEY), M["Planner"](R.API_KEY)
decider.client = planner.client = client
embedder = M["kb_rag"].make_embedder()
kb = M["kb_rag"].KnowledgeBase(a.kb, a.backbone, embedder)
ctx = {"client": client, "decider": decider, "planner": planner, "kb": kb, "kb_sha": kb_sha, "cache": R.PlanCache(a.plan_cache, ns),
       "concepts": R.ConceptMap(), "toolset": toolset, "tool_by_id": {int(t["id"]): t for t in toolset if "id" in t},
       "task_cfg": task_cfg, "backbone": a.backbone, "max_tokens": int(R.os.environ.get("MP_MAX_TOKENS", "0")) or None}
print(f"[probe] namespace {ns} (old {a.from_ns}) | {len(diseases)} diseases | threads {a.threads}", flush=True)

def one(d):
    calls = []
    e, how = R.get_plan(ctx, d, calls)
    return d, e, how

new_k, old_k = collections.Counter(), collections.Counter()
rows = []
with cf.ThreadPoolExecutor(a.threads) as ex:
    for d, e, how in ex.map(one, diseases):
        oe = R.PlanCache(a.plan_cache, a.from_ns).read(R.normalise_name(d))
        nk = {k: v for k, v in (e.get("tool_steps") or {}).items() if v}
        ok = {k: v for k, v in ((oe or {}).get("tool_steps") or {}).items() if v}
        new_k.update(nk); old_k.update(ok)
        pd = [s["action"] for s in (e.get("plan") or []) if any(R.tool_kind(ctx["tool_by_id"][int(t)]) == "panderm" for t in s["tool"])]
        rows.append((d, e.get("status"), how, nk, ok, pd))
        print(f"[probe] {d} | {e.get('status')} {how} | new {nk} | old {ok} | panderm steps {pd}", flush=True)
n_pd = sum(1 for r in rows if r[5])
print(f"[probe] SUMMARY plans {len(rows)} | with a PanDerm step: {n_pd} | tool steps new {dict(new_k)} | old {dict(old_k)}", flush=True)
