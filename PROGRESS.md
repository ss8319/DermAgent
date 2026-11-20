# DermAgent / MedAgent-Pro — Reproduction Progress Log

**Engineer:** Claude (ML eng. role) · **Goal:** faithful-as-feasible reproduction of the **DermAgent** and **MedAgent-Pro** rows of Table 1 (+ Table 2 LOO ablation), then run both agentic baselines on **DermArena Task 1 (Final Diagnosis / RDS)**.

This file is an **exact, replayable log**: every command is the canonical form of what was actually run, with the observed result. Steps are chronological. `✅` = verified working this session; `🔄` = in progress; `⬜` = not started.

---

## 0. Environment (facts, not assumptions)

| Fact | Value |
|------|-------|
| Repo root (`$ROOT`) | `/fs04/scratch2/ub62/ssim0070/DermAgent` |
| Cluster | Monash M3. **Login node has internet; GPU compute nodes do NOT.** ⇒ all weights/repos must be pre-fetched on the login node. |
| GPUs (Slurm) | `gpu` partition: A100×2, L40S×4, A40×4; `m3h`: H100×4. None on login node. |
| Python used | `/fs04/scratch2/ub62/ssim0070/dermagent/bin/python` (torch 2.x, openai, pandas, numpy 2.2.6) **+ added** `scikit-learn nltk rouge-score` |
| HF cache | `HF_HOME=/fs04/scratch2/ub62/ssim0070/.hf_cache` (on scratch ⇒ readable from compute nodes) |
| API endpoint | OpenAI-compatible aggregator `https://api2.aigcbest.top/v1`, serving `gpt-4o` (key from `DermArena_data/.env`) |

Shorthand used below:
```bash
ROOT=/fs04/scratch2/ub62/ssim0070/DermAgent
PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
export HF_HOME=/fs04/scratch2/ub62/ssim0070/.hf_cache
cd $ROOT
```

---

## 1. Repo state ✅

Repo was already cloned at `$ROOT` (git HEAD `2c25c5a "Update citation and add arXiv badge"`, clean tree). Scripts are thin wrappers; **no external assets shipped** (`data/` holds only CSV split definitions; no images, no model weights, no RAG indexes).

---

## 2. Python environment ✅

```bash
# dermagent (renamed from trl-env) had torch+openai+pandas+numpy but not sklearn/nltk — added them:
/fs04/scratch2/ub62/ssim0070/dermagent/bin/python -m pip install -q scikit-learn nltk rouge-score
/fs04/scratch2/ub62/ssim0070/dermagent/bin/python -c "import sklearn,nltk; print('sklearn',sklearn.__version__)"
# -> sklearn 1.7.2
```

**Gotcha (do not use `conda/envs/lmms-eval`):** it has all the libs but a **corrupted numpy** — `import numpy; numpy.__version__` raises `AttributeError: module 'numpy' has no attribute '__version__'`, which crashes `pandas` import. `dermagent (renamed from trl-env)` is clean.

---

## 3. API key / endpoint ✅

`.env` written to `$ROOT/.env` (the GPT-4o model wrapper auto-loads it; `OpenAI()` reads both vars):
```bash
grep -E '^OPENAI_API_KEY|^OPENAI_BASE_URL' /fs04/scratch2/ub62/ssim0070/DermArena_data/.env > $ROOT/.env
# OPENAI_API_KEY='sk-...'         (aggregator key)
# OPENAI_BASE_URL="https://api2.aigcbest.top/v1"
```

---

## 4. Dataset wiring — HAM10000 ✅

**No download/scrape needed** — a colleague's copy holds all 642 benchmark images:
`/fs04/scratch2/ub62/xieji/VL_Data/open_clip_downstream/HAM10000_clean/ISIC2018/*.jpg`.

The benchmark harness expects images at `datasets/ham10000/images/{image_id}.jpg` (see `benchmark/datasets/base.py::load_dataset` → `data_root/image_subdir/{image_id}{image_ext}`). Wired via symlinks:

```bash
cd $ROOT
mkdir -p datasets/ham10000/images
ln -sf $ROOT/data/ham10000/HAM10000_benchmark_500.csv datasets/ham10000/HAM10000_benchmark_500.csv
$PY - <<'EOF'
import csv,os,glob
base="/fs04/scratch2/ub62/xieji/VL_Data/open_clip_downstream/HAM10000_clean"
paths={os.path.basename(p):p for p in glob.glob(base+"/**/*.jpg",recursive=True)}
rows=list(csv.DictReader(open("data/ham10000/HAM10000_benchmark_500.csv")))
dst="datasets/ham10000/images"; n=0
for r in rows:
    src=paths.get(r['image_id']+".jpg")
    if src:
        d=os.path.join(dst,r['image_id']+".jpg")
        if not os.path.islink(d): os.symlink(src,d)
        n+=1
print("symlinked",n,"/",len(rows))   # -> symlinked 642 / 642
EOF
```
Result: **642/642** benchmark image IDs resolved.

---

## 5. GPT-4o harness validation (smoke test) ✅ — the "is everything wired" gate

```bash
cd $ROOT
PYTHONPATH=benchmark $PY benchmark/run.py \
  --model gpt4o --dataset HAM10000_500 --max-samples 3 \
  --device cpu --output-dir ./results/smoke
```
**Observed:** loaded `.env`, OpenAI client → `https://api2.aigcbest.top/v1`, 3 images processed via API, parsed predictions, metrics computed, CSV+JSON written to `results/smoke/task1_diagnosis/HAM10000_500/`. Accuracy on the 3-sample smoke = **0.667** (sanity only — not a reportable number).

**Conclusion:** images + harness + API are correctly wired end-to-end. This is the cheapest validation and it passes.

**Run convention (important):**
- Run from `$ROOT` with `PYTHONPATH=benchmark` (the `from datasets import ...` / `from models import ...` in `benchmark/run.py` require `benchmark/` on `sys.path`; relative data paths resolve from `$ROOT`).
- Use `--device cpu` for the `gpt4o` model (it's an API call; no GPU needed).

### (Optional) full GPT-4o HAM10000 baseline — paused
A full 642-sample run (`--model gpt4o --dataset HAM10000_500`, no `--max-samples`) was started, then **paused at 25/642 to cap API spend** pending budget agreement. Paper reference for this row: HAM10000 acc **0.4891**. Re-run by dropping `--max-samples`.

---

## 6. Phase 0 — pre-fetch agentic-baseline dependencies ✅

**Decision (user-approved):** skip the Derm1M **Case-RAG** for v1 (avoids the hundreds-of-GB Derm1M download + GPU-hours Qdrant build). So Phase 0 fetches only what PanDerm + MAKE + DermoGPT need.

```bash
cd $ROOT
export HF_HOME=/fs04/scratch2/ub62/ssim0070/.hf_cache
HF=/fs04/scratch2/ub62/ssim0070/dermagent/bin/hf

# 6.1 DermoGPT-RL — already in HF cache (models--mendicant04--DermoGPT-RL); symlink it
mkdir -p model-weights
DGPT=$(ls -d $HF_HOME/hub/models--mendicant04--DermoGPT-RL/snapshots/*/ | head -1)
ln -sfn "$DGPT" model-weights/DermoGPT-RL          # -> 4 safetensors shards present

# 6.2 PanDerm + MAKE weights -> HF cache (reachable from compute nodes)
$HF download redlessone/DermLIP_PanDerm-base-w-PubMed-256
$HF download xieji-x/MAKE

# 6.3 External code (open_clip forks) -> repo root
git clone --depth 1 https://github.com/SiyuanYan1/Derm1M.git Derm1M   # provides Derm1M/src
git clone --depth 1 https://github.com/SiyuanYan1/MAKE.git   MAKE     # provides MAKE/src + concept_annotation/term_lists/ConceptTerms.json
```
**Completed** (`.repro_logs/phase0.log`): PanDerm DermLIP (9 files) + MAKE weights (10 files) in HF cache; `Derm1M/` and `MAKE/` cloned. Verified layout: `Derm1M/src/open_clip`, `MAKE/src/open_clip`, `MAKE/concept_annotation/term_lists/ConceptTerms.json`. Already-present: DermoGPT-RL (cache), Qwen3-VL-8B-Instruct (cache), HAM10000 images.

### 6.4 Gate — import both open_clip forks ✅
```bash
# First attempt failed: ModuleNotFoundError: No module named 'ftfy'  (open_clip dep)
$PY -m pip install -q ftfy regex timm
# Retry:
$PY -c "import sys; sys.path.insert(0,'Derm1M/src'); import open_clip; print(hasattr(open_clip,'create_model_from_pretrained'))"  # -> True
$PY -c "import sys; sys.path.insert(0,'MAKE/src');   import open_clip; print(hasattr(open_clip,'create_model_and_transforms'))"   # -> True
```
**Gotcha:** the two forks each ship their *own* `open_clip` package with the same name. They must never be on `sys.path` simultaneously — `skin_agent/tools/skin_tools.py` handles this by inserting `Derm1M/src` first for PanDerm/RAG, then stripping any `Derm1M` path before importing MAKE's fork. `timm` is required (PanDerm uses a timm/BEiT visual backbone).

---

## 7. Infra: GPU + compute-node internet ✅

- **Compute nodes on `gpu`/`desktop` partitions HAVE outbound internet** (contradicts the "no compute-node internet" assumption). Verified with a 5-min diagnostic job (`.repro_logs/diag.sbatch`): from node `m3g112`, `curl` reached `api2.aigcbest.top` (HTTP 401 = server responded) and `huggingface.co` (HTTP 200). ⇒ the agents' mid-pipeline GPT-4o calls work from the GPU node.
- **Launch:** `sbatch` to `--partition=desktop --qos=desktopq` (idle T4/L4/A40, allocates instantly) — the `gpu` partition (L40S/A100) was queue-backed. Bake env into the job: `export HF_HOME=...`, `source` the `OPENAI_*` vars from `$ROOT/.env`.
- **Efficiency note (for full scale):** these agents are API-bound, not GPU-bound (~50 GPT-4o calls/sample, ~1 GPU call) → GPU ~95% idle. For full runs, switch to **precompute tool outputs (1 GPU batch) → cache → run API orchestration on a CPU node**. Tools are deterministic per image, so caching is exact. Not done for smokes.

## 8. Phase 1 — MedAgent-Pro smoke ✅ (pipeline validated)

Required patches (all documented in-code with `REPRO PATCH` comments):
1. **`Decider/__init__.py`** imported 5 non-vendored deciders (Janus/BioMedClip/Qwen/InternVL/Gemma) → crashed `from Decider import ...`. Commented them out (only GPT/Pro/MultiClass exist & are used).
2. **Model id:** all components hardcoded `chatgpt-4o-latest`, which the proxy **rejects** (429 "account deactivated"); `gpt-4o` works. `sed -i 's/chatgpt-4o-latest/gpt-4o/g'` across `*.py Decider/*.py`.
3. **Case-RAG dropped:** removed tool id 4 (`rag_retrieve`) from `task1_diagnosis/toolset.json` (kept ids 1,2,3,5); removed the tool-4 line from the task-1 `domain_hint`.
4. **Planner web-RAG disabled** in `Derm_Task_level.py`: upstream `RAG_Module.query()` does `requests.get()` on **dermnetnz.org** (off-limits) + needs faiss/embeddings. Set `rag_result=""`; plan still GPT-4o-generated from toolset + domain hint.
5. **Planner validator vs LLM:** the anti-bundling guard rejects `and`/commas in a qualitative `action`; tightened the task-1 hint to force the final step's action to the single phrase `classify the skin lesion`, plus a 3-try retry loop.

Data prep: built `datasets/ham10000/HAM10000_mp_smoke3.csv` with `filename` (=`image_id`.jpg) + `diag` (full class name) — the columns `Derm_Case_level`/`Derm_Evaluator` require.

Run (login-node planner = API-only; GPU job = tools + API):
```bash
# Planner (login node):
cd baselines/MedAgent-Pro
set -a; source <(grep -E '^OPENAI_(API_KEY|BASE_URL)=' ../../.env | sed "s/['\"]//g"); set +a
PYTHONPATH=.:../.. $PY Derm_Task_level.py --tasks 1 --model gpt-4o   # -> plan.json (9 steps)
# Case-level + eval (GPU job): .repro_logs/medagentpro_smoke.sbatch  (desktop/L4)
#   Derm_Case_level.py --task 1 --csv-path ...HAM10000_mp_smoke3.csv --image-dir ...images --max-samples 3 --output-tag smoke3
#   Derm_Evaluator.py  --task 1 --record-dir Dermatology/task1_diagnosis/record/smoke3 --csv-path ...HAM10000_mp_smoke3.csv
```
**Result:** 3/3 samples produced `final_diagnosis.json` (structured `{diagnosis,confidence,reasoning}`); evaluator ran (metrics.json). Smoke accuracy 0/3 is **not meaningful** — the first 3 HAM rows are all melanocytic nevi and the agent over-called melanoma; a class-balanced sample is needed for any real number. **Gate (pipeline runs end-to-end): PASS.**

## 9. Phase 2 — DermAgent smoke 🔄 (running)

Shipped-bug patch: `scripts/run_task1_ham10000_500_agent.py` called `create_tools(..., qwen_model_id=...)`, but `create_tools()` has **no such param** (would `TypeError`). **Qwen3-VL is not wired into the agent at all** — `create_tools` only handles `panderm/make/dermogpt_vqa/rag/text_rag/ontology`; the VLM tool is DermoGPT. Removed the dead kwarg.

Run (`.repro_logs/dermagent_smoke.sbatch`, desktop/L4, ~20 GB — DermoGPT 16 + PanDerm/MAKE 4):
```bash
$PY scripts/run_task1_ham10000_500_agent.py --device cuda --model-name gpt-4o \
   --enabled-tools panderm,make,dermogpt_vqa,ontology --max-samples 3
```

## 9b. Qwen3-VL wiring (paper-vs-code gap fix) ✅

`Qwen3VLTool` shipped (`skin_tools.py`, `name="qwen_vqa"`) but was never instantiated — `create_tools()` had no branch and runners passed an unsupported `qwen_model_id` kwarg (`TypeError`). Patched `skin_agent/benchmark_agent.py`: import `Qwen3VLTool`, add `qwen_model_id` param + a `qwen_vqa` branch; restored the runner's `qwen_model_id` plumbing. Also `base.py::ImageQueryInput.query` now has a default (the agent sometimes called `qwen_vqa` without a query → `ValidationError` before `_run`).
**Re-smoke** (`panderm,make,qwen_vqa,ontology`, L4): Qwen3VLTool created + loaded + **invoked by the agent (4 calls, 2 real inferences ~34–46 s)**. So Qwen3-VL now genuinely runs inside the agent. (Smoke acc 0/2 — all-nevi sample, not meaningful.)

## 10. Phase 3 + 4 — DermArena Task-1 (RDS) ✅ smoke-validated

Benchmarks live in `DermArena_data/` (not `dataset_collection/data/benchmark/`): `MM_RDS_benchmark_bound.jsonl` (**219**, image-bound) etc. Adapter built **in the DermArena repo** (branch `agentic-baselines-eval`, pushed to `ss8319/RareArena`): `dataset_collection/eval/infer_dx_agent.py`.

**Design:** these single-image closed-set agents can't natively do DermArena's case-report/open-set-ranked task, so the adapter applies their *core mechanism* — run the specialist vision tools (PanDerm/MAKE/DermoGPT) on each case's bound image, feed the evidence + DermArena's exact case+image prompt to the GPT-4o backbone → open-set **top-5 ranked differential**. Reuses `infer_dx.py`'s prompt/image construction; emits `{_id, prediction}` JSONL that `score_dx.py` scores unchanged. (Does NOT reproduce plan-execute-reflect/critic — documented deviation.)

**Run (GPU adapter + login-node scoring):**
```bash
# Phase 3 (GPU): tools + gpt-4o -> predictions
python infer_dx_agent.py --task rds --benchmark .../MM_RDS_benchmark_bound.jsonl \
    --baseline dermagent --output dermagent_rds_pred.jsonl --device cuda --limit 2
# Phase 4 (login node, API-only): DermArena judge
#   point OPENROUTER_* at the SAME metered aigcbest proxy; --model gpt-4o
python score_dx.py --benchmark .../MM_RDS_benchmark_bound.jsonl \
    --predictions dermagent_rds_pred.jsonl --output-dir scores --model gpt-4o
```
**Gotchas:** (1) `score_dx`/`llm_client` write a ledger to a hardcoded `/mnt/hdd/...` → set `LLM_LEDGER_PATH` to a writable path. (2) `score_dx` resume treats pre-existing per-case JSONs as done → clear the output dir to re-score.

**2-case smoke (DermAgent-tooluse):** both cases produced valid ranked differentials; `pmid_41694845_1` (GT *lichen sclerosus*) → **top-1 exact hit (score 2)**; `pmid_39881935_1` (GT *hairy cell leukemia*, non-visual) → miss. **top-1 recall 0.5, top-5 0.5.** Pipeline (Phase 3 → Phase 4) end-to-end validated.

**Not yet run (needs budget):** full 219-case runs for both baselines + the `--baseline medagentpro` variant + comparison vs plain `infer_dx.py` GPT-4o. ~$5–15/baseline; beyond the $0.90 smoke cap.

See `REPRO_STATUS.md` for the full dependency map and walls (Guideline-RAG non-reproducible — DermNet scrape off-limits).
