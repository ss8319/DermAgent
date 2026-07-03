# DermAgent / MedAgent-Pro Reproduction — Status Note

_Working dir: `/fs04/scratch2/ub62/ssim0070/DermAgent`. Cluster: Monash M3 (login node has internet; GPUs via Slurm: H100/A100/L40S/A40)._
_Last updated: 2026-06-13._

## TL;DR

| Item | State |
|------|-------|
| Repo orientation + dependency map | ✅ Done |
| Asset inventory (datasets, weights, repos, API) | ✅ Done — most assets already on disk |
| Harness wired + validated (GPT-4o, HAM10000, 3-sample smoke) | ✅ **End-to-end PASS** |
| GPT-4o full HAM10000_500 baseline | 🟡 Running (real number pending) |
| MedAgent-Pro agent baseline | 🔶 Feasible (light path exists) — needs 2 repo clones + 1 GPU; **not yet run** |
| DermAgent full agent | 🔶 Partially feasible — **Guideline-RAG is a hard wall** (see below) |
| Table 2 LOO ablation | 🔶 Same walls as DermAgent full |
| DermArena task-fit decision | ✅ Done — see §5 |

## 1. Environment & wiring (DONE)

- **Python env**: reused `/fs04/scratch2/ub62/ssim0070/trl-env` (torch, openai, pandas, numpy 2.2.6) + installed `scikit-learn`, `nltk`, `rouge-score`. (`conda/envs/lmms-eval` has the libs but its numpy is corrupted — `np.__version__` missing.)
- **`.env`** created in repo root from `DermArena_data/.env`:
  - `OPENAI_API_KEY` + `OPENAI_BASE_URL=https://api2.aigcbest.top/v1` (OpenAI-compatible aggregator serving `gpt-4o`). This is the GPT-4o path for both the benchmark harness and the agents' reasoning backbone.
- **Datasets dir**: created `datasets/ham10000/{HAM10000_benchmark_500.csv, images/}`; 642/642 benchmark images symlinked from a colleague's copy (`/fs04/scratch2/ub62/xieji/VL_Data/open_clip_downstream/HAM10000_clean`). **No scraping/download needed.**
- **Run convention**: `cd DermAgent && PYTHONPATH=benchmark <py> benchmark/run.py ...` (the `datasets`/`models` imports require `benchmark/` on the path; relative data paths resolve from repo root).

**Smoke test (3 samples, GPT-4o, HAM10000):** images → API (aigcbest) → parsed preds → metrics → CSV/JSON all succeeded. Confirms images + harness + API are correctly wired (the hand-off's step-1 goal).

## 2. Asset inventory

| Asset | Needed by | Status |
|-------|-----------|--------|
| HAM10000 images (642) | all | ✅ xieji copy, symlinked |
| SNU / Derm7pt / SkinCon / SkinCAP images | GPT-4o & agent runs on those tasks | ✅ present (xieji `VL_Data/*`, `gdrive_download/dermobench_release/imgs/*`) — not yet wired |
| API key + gpt-4o endpoint | GPT-4o + both agents | ✅ `DermArena_data/.env` |
| DermoGPT-RL weights | DermAgent | ✅ HF cache `models--mendicant04--DermoGPT-RL` (symlink into `model-weights/`) |
| Qwen3-VL-8B-Instruct | DermAgent | ✅ HF cache |
| PanDerm DermLIP (`redlessone/DermLIP_PanDerm-base-w-PubMed-256`) | both agents | ⬇️ download needed (have `PanDerm2`, **not** this exact id) |
| MAKE weights (`xieji-x/MAKE`) | both agents | ⬇️ download needed |
| Qwen3-Embedding-8B | DermAgent Text-RAG | ⬇️ ~16 GB download needed |
| `Derm1M/src/` (open_clip fork) | both agents (PanDerm + Case-RAG encoder) | ⬇️ clone needed |
| `MAKE/src/` + `ConceptTerms.json` | both agents (MAKE) | ⬇️ clone needed |
| Derm1M **dataset** (413K imgs) + Qdrant Case-RAG | both agents (`rag_retrieve`) | 🧱 **Heavy** — hundreds of GB + GPU-hours to embed/build |
| `RAG/dermnet_chunks_cleaned.json`, `mayo_chunks_cleaned.json` | DermAgent Guideline-RAG | 🧱 **Hard wall** — not shipped, no builder; reconstructing = scraping DermNet (off-limits per our licensing rules) |

## 3. Per-system setup state & reproducibility verdict (Table 1)

- **GPT-4o (General MLLM row, paper HAM10000 0.4891)** — ✅ **Reproducible API-only.** Full 642-sample run in flight; SNU/Derm7pt/SkinCon/SkinCAP also runnable once images are wired (all present). No GPU needed. Expect close-but-not-exact (proxy model version + nondeterminism). _This also covers the GPT-5.2 row and the simple MLLM rows that have local weights (Qwen3-VL, DermoGPT-RL); LLaVA-Med/HuatuoGPT/Hulu-Med/SkinVL need their own weights._
- **MedAgent-Pro (paper HAM10000 0.5763)** — 🔶 **Feasible, not yet run.** `plan.json` is generated at runtime by `Derm_Task_level.py` (GPT-4o planner over `toolset.json`); Case-RAG is just tool id 4 of 5. **Two fidelity tiers:**
  - *Full*: build Derm1M Case-RAG → most faithful (heavy).
  - *Reduced (no Derm1M)*: drop tool 4 from `toolset.json` so the planner omits `rag_retrieve` → runs PanDerm + MAKE + GPT-4o decider on 1 GPU in hours. Documented deviation from paper.
  - Either tier needs: clone `Derm1M/src` + `MAKE/src`, download PanDerm + MAKE weights, 1× A40/L40S/A100.
- **DermAgent (Ours rows)** — 🔶 **Partially reproducible, with a fidelity ceiling.** All local heavy models are available (DermoGPT-RL, Qwen3-VL); PanDerm/MAKE/Qwen3-Embedding downloadable. **But Guideline-RAG cannot be faithfully reproduced from this repo** (chunk JSONs absent, builder absent, source = DermNet scrape = off-limits). Case-RAG also requires the heavy Derm1M build. So a DermAgent run here would be *Case-RAG-only or no-RAG*, i.e. an ablation-grade config, not the headline number. Peak ~50 GB → 2×24 GB or 1×80 GB.

## 4. Table 2 (LOO ablation) verdict

🔶 Same walls as DermAgent full. The `w/o Guideline-RAG` and `w/o Case-RAG` rows are runnable-by-construction (they *disable* the blocked component), but the **full-agent reference rows (0.1948 w/ Critic, 0.1727 w/o)** require both RAGs, so the headline ablation deltas can't be reproduced faithfully without the Derm1M build + the (blocked) Guideline-RAG. Partial ablation (Ontology/PanDerm/MAKE/DermoGPT toggles, Case-RAG-only) is feasible once the base agent runs.

## 5. Greater goal — which DermArena QA tasks suit these 2 baselines?

**DermArena's 5 tasks are text-heavy, multi-figure clinical *case-report* reasoning with free-text / ranked outputs, LLM-judge scored** (Dx top-5 differential, RDC post-test Dx, DxTest test recommendation, PathQA [not built], Tx treatment plan).

**Both baselines are, by construction, single-image → closed-set pipelines**: input = ONE image path + a fixed option/concept list; output = one disease label / per-concept yes-no / one caption. They have **no channel to ingest the clinical narrative text**, no multi-image aggregation, and a closed label space. Their dermatology tools (PanDerm zero-shot CLS, MAKE concepts, Case-RAG image-similarity, Ontology) are all image-centric.

**Verdict:**
| DermArena task | Fit | Why |
|----------------|-----|-----|
| **1. Final Diagnosis (RDS)** | **Best (still needs adaptation)** | Only task conceptually aligned (it's diagnosis). DermAgent's GPT-4o backbone + Ontology (hypernyms ↔ ICD-11 family-match metric) + PanDerm/Case-RAG could add image-grounded signal. Requires: (a) feed case figure(s) as the image, (b) unlock the closed decider to open-set free-text ranked output. |
| **2. Post-Test Dx (RDC)** | Weak | Same as T1 but the discriminating signal is *exam-results text* the image-only agents cannot read → they ignore exactly what the task measures. |
| **3. DxTest selection** | Unsuitable | Knowledge/reasoning task; vision tools add ~nothing → reduces to the bare GPT-4o backbone, agent scaffolding is dead weight. |
| **4. PathQA** | N/A | Not generated in v0.1; pathology is also outside the agents' dermoscopy/clinical-photo domain. |
| **5. Treatment planning (Tx)** | Unsuitable | Same as T3 — pure reasoning; no role for image-classification tools. |

**Recommendation:** if we run these baselines on DermArena, target **Task 1 (Final Diagnosis / RDS) only**, after (i) selecting a representative figure per case and (ii) replacing the closed-set decider with open-set ranked generation. Caveat to flag in any writeup: because the agents discard the case narrative, they may *underperform a plain GPT-4o* that reads the full report — so their value on DermArena is questionable, and this should inform whether full agentic reproduction is worth the cost.

## 6. Exact blockers for skipped items

- **Derm1M Case-RAG** — needs full Derm1M dataset download (hundreds of GB) + GPU embedding of 413K images + Qdrant build. Hours–days + large disk. Gates *faithful* MedAgent-Pro and DermAgent.
- **Guideline-RAG (DermAgent only)** — `RAG/dermnet_chunks_cleaned.json` + `mayo_chunks_cleaned.json` not in repo, no builder; faithful reconstruction requires scraping DermNet → **off-limits**. Hard ceiling on DermAgent fidelity.
- **External code** — `Derm1M/src`, `MAKE/src` not vendored (clone needed). `DermoGPT` weights present in HF cache.

## 7. Next steps — agentic baselines → DermArena QA (the plan)

**Strategy:** smoke-test the agent stack cheaply on HAM10000 (proves tools/GPU/plan/decider work), then adapt + point it at **DermArena Task 1 (Final Diagnosis / RDS)** — the only DermArena task these agents fit (§5). Skip the heavy Derm1M Case-RAG build for v1 (drop the RAG tool); revisit only if results justify it.

### Phase 0 — Fetch deps (FREE: bandwidth/disk only; no API, no GPU)
- Symlink `model-weights/DermoGPT-RL` → HF cache `models--mendicant04--DermoGPT-RL`.
- `huggingface-cli download` PanDerm `redlessone/DermLIP_PanDerm-base-w-PubMed-256` + `xieji-x/MAKE` → `model-weights/`.
- `git clone` Derm1M (need `src/` open_clip fork) + MAKE (need `src/` + `concept_annotation/term_lists/ConceptTerms.json`) into repo root.
- Defer Qwen3-Embedding-8B (Text-RAG) — only for DermAgent, and Guideline-RAG is walled anyway.
- **Gate:** `import open_clip` from both forks succeeds.

### Phase 1 — MedAgent-Pro smoke (CHEAP: 1× A40/L40S + ~$0.5 API, 3 samples)
- Edit `Dermatology/task1_diagnosis/toolset.json`: drop tool id 4 (`rag_retrieve`) → planner omits Case-RAG → no Derm1M build.
- `Derm_Task_level.py --tasks 1` (generate `plan.json`) → `Derm_Case_level.py --task 1 --image-col image_id --csv-path .../HAM10000_benchmark_500.csv --image-dir datasets/ham10000/images --max-samples 3`.
- **Gate:** `final_diagnosis.json` produced; `Derm_Evaluator.py` runs.

### Phase 2 — DermAgent smoke (1×80 GB or 2×24 GB + ~$0.5 API, 3 samples)
- `create_tools(enabled_tools=["panderm","make","qwen_vqa"])` (drop `"rag"`); `run_task1_ham10000_500_agent.py --max-samples 3 --device cuda`.
- **Gate:** tools load, Plan-Execute-Reflect emits a prediction + trace.

### Phase 3 — DermArena adapter (ENGINEERING; the real work)
- **Prereq:** DermArena `MM_RDS_benchmark.jsonl` must be built (DermArena projector pipeline). **Currently not on disk — blocker for the end goal.**
- Write `infer_dx_agent.py` (mirrors `infer_dx.py` IO): per RDS case → resolve primary figure (`figures[].local_image_path` under `DermArena_data/images`), run the agent, emit `{model}_dx_predictions.jsonl` in the schema `score_dx.py` expects. Then reuse `score_dx.py` + `metric_dx.py` unchanged.
- **Adapt each agent's decider to open-set ranked output** (free-text top-5 differential; remove the fixed class list). MedAgent-Pro: swap `MultiClass_Decider` closed candidate set → ranked free-text. DermAgent: add an "open-set" dataset mode (options=[]).
- **3 design choices to confirm with user:**
  1. *Feed the case narrative text into the agent?* **Rec: yes** (agents' qualitative/decider steps already accept text context). Otherwise they're guaranteed to underperform plain GPT-4o, which makes the comparison uninteresting. Optionally also run image-only as a contrast to isolate vision-tool value.
  2. *Case-RAG for v1?* **Rec: off** (skip Derm1M build); add later only if needed.
  3. *Which figure when a case has several?* **Rec: primary clinical/dermoscopic figure**; optionally loop all + aggregate.

### Phase 4 — DermArena smoke → full run
- Adapter on 3 cases → `score_dx.py` → `metric_dx.py`. Gate.
- Full 80-case RDS per baseline. **Headline question:** do the vision tools beat text-only GPT-4o on top-1/top-5 recall? If not, that's the finding.
- **Cost:** ~$5–15 API per baseline for 80 cases (planner+qual+decider calls × cases) + GPU hours.

**Rough total to first DermArena number:** Phase 0 free · Phases 1–2 smoke ≈ $1 + a couple GPU-hours · Phase 4 ≈ $5–15/baseline. No Derm1M build, no DermNet scrape.
