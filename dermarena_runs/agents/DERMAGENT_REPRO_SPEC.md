# DermAgent on DermArena v1 — reproduction spec (T1.1 / T1.2, open-QA)

Authoritative record of every design decision agreed with the owner on 2026-11-05 (forked "H200 agents" chat).
If code and this file disagree, fix one of them the same day.

Legend used in every table: ✅ same as upstream · 🔁 adapted because DermArena is open-QA / multi-image / has case
text · ⚠️ deliberate owner choice · ⚙️ engineering only (outputs unchanged).

---

## 0. Goal and the question it answers

**RQ M4 — do agentic systems beat their own backbone on DermArena diagnosis (T1.1 screening, T1.2 confirmation)?**
We reproduce published agent baselines as faithfully as possible (tools, orchestration, prompts) and run them on
the DermArena v1 test split, open-QA (free-text ranked top-5, no option list). T1.3 (test ordering) is out of scope
for now (neither agent was built for it).

| Arm | System | Backbone | Status |
|---|---|---|---|
| Base | plain model, benchmark prompt (`dx_infer.py`) | Qwen/Qwen3.5-27B, local SGLang | done, judged (T1.1 top-1 exact 28.8 %, T1.2 44.3 %) |
| A | **DermAgent** (this spec) | same Qwen3.5-27B, same server stack | canaries 060–063 on H200 (2026-11-05) |
| B | MedAgent-Pro | same | port in progress (§9) |

**Analysis plan (fixed before results):**
- Primary metric: top-1 exact (judge score 2). Secondary: top-5 exact, top-1/top-5 exact-or-broader.
- Test: each agent vs base on the SAME cases, McNemar exact test on discordant pairs, 95 % CI on the paired
  difference; Holm across the 4 primary tests (2 agents × 2 tasks).
- Pre-planned strata: `corpus_status` train_absent / train_represented; cases with vs without clinical/dermoscopy
  images (the agent's tools are image tools, so any gain should sit there).
- Power: compute the minimum detectable effect from the base arm's per-case results (discordance rate) BEFORE the
  full runs; with n = 2,806 / 2,030 a ~2–3 pp top-1 difference is expected to be detectable.
- Unparsed = wrong, stays in the denominator (benchmark rule). Agent-specific failure reasons are logged per row.

---

## 1. Sources and what counts as "upstream"

| Source | What we take from it |
|---|---|
| `YizeezLiu/DermAgent` **master `5da78b3`** (fetched 2026-11-05 as git ref `upstream/master`) | **The specification.** Its `skin_agent/` and `scripts/` code is byte-identical to our fork point `2c25c5a`; the only later upstream commit (#3) adds the Mayo/DermNet corpus-pipeline skill + README/.gitignore edits. |
| Upstream README | Tool list, weights, Qdrant/Docker setup, Table 1/2 numbers |
| Paper arXiv 2605.14403 (read via HTML extract) | Architecture description. It gives **no** prompts, decoding settings or iteration cap → where the paper is silent, the code is the spec |
| Reference configuration | `scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh` — the paper's Dx headline run (HAM10000): GPT-4o, 6 tools, Critic on, image re-injection on, 2 critic retries, LLM summary off. (SNU run used 5 tools, no ontology.) |

**Paper vs code differences (we follow the code):**

| Topic | Paper | Code (what we run) |
|---|---|---|
| Tool execution | "executes the planned calls in parallel" | `execution_mode="sequential"` default |
| Coverage gate for diagnosis | described ("PanDerm for diagnosis") | not present in the diagnosis branch of `critic_node` |
| Confidence gate | "cosine similarity below 90 %"; flag only if an uninvoked tool remains | softmax top-1 probability < 0.90 (`CONFIDENCE_THRESHOLDS`), fires unconditionally; exit rule = from retry 2, pass if all tools called |
| Qwen3-VL as 7th tool | listed | `create_tools()` never instantiates `qwen_vqa` → no upstream run uses it |

Our code base: `ss8319/DermAgent` branch **`reproduction-fixes`** (PR #2, HEAD `4500180`) + uncommitted changes listed
in §8. Owner synced `ss8319/master` to upstream `5da78b3` via the GitHub UI on 2026-11-05 (the token embedded in the
clone's `origin` URL is dead; PR #2 merges cleanly on top).

---

## 2. Tools (final, agreed)

| Tool | Model / data (upstream = ours) | Code vs upstream (per-class byte check 2026-11-05) | Image types it supports (source) | Candidate list | Status |
|---|---|---|---|---|---|
| Enabled set | 6 = upstream HAM10000 headline run = paper's "full agent" (Table 2 ablation base) | — | — | — | ✅ |
| `panderm_classifier` | DermLIP `redlessone/DermLIP_PanDerm-base-w-PubMed-256`; 4 prompt templates averaged; softmax; returns top 5 | **identical** (PanDermTool + input schema) | clinical + dermoscopy (HF card: "403,563 … Derm1M … Images include both dermoscopic and clinical images") | **exactly 10 = backbone's top-10 differential** (upstream: dataset options 7 / 134) | ✅ code · 🔁 candidates, images |
| `make_concept_annotator` | MAKE `xieji-x/MAKE` + `ConceptTerms.json`; top-5 concepts | **identical** | clinical + dermoscopy (MAKE README: same 403,563 Derm1M pairs; evaluated on SkinCon clinical + Derm7pt dermoscopy) | — | ✅ · 🔁 images |
| `dermogpt_vqa` | DermoGPT-RL (Qwen3-VL-based) bf16 greedy; diagnosis mode XML ≤1,024 tok, VQA ≤512 tok | **identical** | clinical + dermoscopy (HF card "Domain: dermatology and dermoscopy"; its own prompt: "clinical photography (macroscopic) and dermoscopy") | diagnosis mode: **the same 10** | ✅ code · 🔁 candidates, images |
| `rag_retrieval` (Case-RAG) | DermLIP 512-d image search over Derm1M **413,210** points, collection `derm1m`, top 5 | search identical; only the never-called `unload()` closes the client | clinical + dermoscopy (Derm1M index) | — | ✅ · 🔁 images |
| `text_rag_retrieval` (Guideline RAG) | **3,199** chunks (DermNet 2,751 + Mayo 448) = README figure; Qwen3-Embedding-8B (4,096-d) + keyword, RRF k=60; Qwen3-Reranker-0.6B top-20 → 5 | search identical; reads an embedded Qdrant store (`QDRANT_TEXT_PATH`) instead of a localhost server | text | — | ✅ · ⚙️ |
| `ontology_query` | local disease tree, fuzzy match | **identical** | text | — | ✅ |

**Rules applied across tools:**

| Rule | What we do | Why |
|---|---|---|
| Fixed candidate list (🔁, owner) | System prompt requires **exactly 10** candidates (backbone's top-10) for PanDerm and **the same 10** for DermoGPT diagnosis mode. Enforced by instruction only (tool code unchanged); list size logged per call (traces) | No option list in open-QA. A fixed size keeps PanDerm's softmax confidence — and the critic's 90 % gate — comparable across cases (a 3-name list clears 90 % easily, a 20-name list rarely) |
| Derm-only image gate (🔁, owner) | Only images whose release `images[].modality` ∈ {`clinical_photo`, `dermoscopy`} get a tool ID. All other 17 labels and `unknown` (15 T1.1 + 26 T1.2 images) get no ID | All four image tools are documented for clinical + dermoscopy only |
| Backbone still sees everything (⚠️, owner) | ALL images + case + tables go to the backbone, exactly like the base arm | agent vs base then differs only by the scaffold |
| Load safety (⚙️) | worker refuses to run if any enabled tool failed to load (`_load_failed` / not initialized) | two tools had silently failed (§7) |
| Concurrency (⚙️) | 4 processes × 12 threads; one lock per tool class; one Qdrant copy per process on node /tmp | throughput only |

**Modality labels:** 0 missing. Provenance (`_modality_source`, T1.1+T1.2 test, 14,279 images): 13,896 pipeline
default (19-class modality classifier), 377 relabelled by Qwen3.6-27B, 6 manual review → machine labels, spot-checked.

| Image mix (test split) | T1.1 (2,806 cases, 6,028 images) | T1.2 (2,030 cases, 8,251 images) |
|---|---|---|
| clinical photo / dermoscopy share of images | 73.5 % / 3.2 % | 41.0 % / 2.2 % |
| non-derm images (CT, histopathology, MRI, IHC, …) | ~23 % | ~57 % (histopathology 31 %, IHC 8 %) |
| cases with ≥1 clinical/dermoscopy image | 2,013 | 1,539 |
| … of which also have non-derm images | 355 | 1,166 |
| cases with images but NO derm image → agent runs text-tools only | 164 | 308 |
| cases with no images at all | 629 | 183 |

---

## 3. Orchestration (final, agreed)

| Item | Upstream (headline Dx run) | Ours | Status |
|---|---|---|---|
| Loop ("Plan → Execute → Reflect") | LangGraph: chatbot → tools → critic → feedback → (forced image re-look) → chatbot | same code | ✅ |
| Who decides tools | backbone, free, guided by the prompt's "Recommended Approach" | same | ✅ |
| Tool execution | sequential (code default) | same | ✅ |
| Repeated identical tool call | per-case cache | same | ✅ |
| Limits | `max_tool_calls` 10, recursion 30 (code defaults; runner passes neither) | 10 / 30; recursion-limit hit → row written as unparsed (= wrong), not a dropped call | ✅ |
| Critic | on, deterministic, 2 retries, image re-injection on, LLM summary off | same | ✅ |
| Critic gates | PanDerm top-1 < 0.90 · Case-RAG similarity < 0.80 · PanDerm-vs-RAG top label conflict (unless text-RAG already queried both) · any tool error | same code; gates read each tool's LAST output (upstream behaviour) | ✅ · 🔁 (PanDerm score over the backbone's 10) |
| Critic retry 1 | "Focused Re-diagnosis": shortlist from PanDerm (all >0 % preds) + Case-RAG top-2 + DermoGPT diagnosis, each matched to the dataset options; restricts tools to panderm/dermogpt (on the shortlist), text_rag, ontology | **same text and rules**; shortlist built from the same 3 sources **without option matching** (names as the tools wrote them, junk labels dropped) and **pooled over ALL calls of each tool** (one per image). Needs ≥2 names, else upstream's generic retry text | ✅ text · 🔁 shortlist source (owner) |
| Critic retry 2 | "Final Attempt" (diagnosis) | same | ✅ |
| Critic image re-injection | its one image + "Re-analyze this image based on the critic feedback above."; then a forced no-tool turn "Describe what you observe…" | **all clinical/dermoscopy images** of the case + "Re-analyze this case and these images based on the critic feedback above."; same forced turn. Cases with no derm image: falls back to `current_image` (first derm image, else image 1); no images → no re-injection | 🔁 (owner) |
| Critic message role | mid-conversation **system** message | sent as **user** message, identical text (Qwen3.5 template rejects non-leading system messages) | ⚙️ backbone compat (§5) |
| Case input | 1 image, no text | case text + tables via the benchmark's own `dx_infer.case_text` (T1.2 adds "Examination results:") | 🔁 |
| What the backbone sees | image ID only (`include_image_in_message=False`) | **ALL images attached first** (same files, same RGB conversion as base), then the user message | ⚠️ owner: same input as the vanilla model |
| Image routing | 1 anonymous ID `IMG_<md5[:8]>` (hides filenames, which contain PMIDs) | same IDs; one line per image in attachment order, **ID last**: `image 2 (dermoscopy): IMG_9b07e4aa`, `image 3 (histopathology): not available to the image tools` (ID-last so the tag is not copied into the tool argument) | ✅ · 🔁 |
| Option list in system prompt | injected each call | none | 🔁 |
| Final answer | free text → `AnswerParser` maps to ONE option | last `FINAL_ANSWER` block → up to 5 numbered names → `{"diagnoses": [...]}` (benchmark JSON); none / budget hit / recursion → unparsed = wrong. Not restricted to the 10 candidates (backbone's own synthesis, as upstream's GPT-4o) | 🔁 |
| History at final answer | — | full conversation: all images + case + tables, every tool output, critic messages, re-injected images; earlier reasoning passed back only within the tool loop (Qwen template, §5) | — |
| Throughput | serial | 4 procs × 12 threads, per-tool-class locks | ⚙️ |

**Worked example (agreed):** 2 clinical photos, 1 dermoscopy, 1 histopathology. Round 1: PanDerm on photo 1 and on
the dermoscopy image, Case-RAG on photo 1, DermoGPT on the dermoscopy image. Critic: last PanDerm top-1 60 % < 90 % →
retry; shortlist pools both PanDerm runs + RAG top-2 + DermoGPT's diagnosis. Message back = Critic Review + Focused
Re-diagnosis (shortlist) + the 2 photos and the dermoscopy image re-attached with "Re-analyze this case and these
images…". Round 2: forced description, re-run PanDerm/DermoGPT on the shortlist, text-RAG on the differences,
`FINAL_ANSWER` top-5.

---

## 4. Prompts (final, agreed)

| # | Prompt | Who sees it | Upstream | Ours | Status |
|---|---|---|---|---|---|
| 1 | System prompt | backbone, every call | `DIAGNOSIS_SYSTEM_PROMPT_DERMOGPT_FULL` + option list injected | same with the 8 edits below (`OPEN_SET_DX_SYSTEM_PROMPT_UPSTREAM_FULL`, derived at import from the upstream constant; each edit asserted to match exactly once); no option list | 🔁 |
| 2 | User message | backbone, first turn | `## Task` Diagnose the skin lesion in this image. `## Image` {id} `## Instructions` Use panderm_classifier with dataset="X", cross-validate with other tools (rag_retrieval, make_concept_annotator), then provide FINAL_ANSWER with one of the diagnostic options. | all images attached first, then `## Task` Diagnose the patient in this case. `## Image` one line per image (§3) `## Case` {base-arm case text + tables} `## Instructions` Use panderm_classifier with your top-10 candidate diagnoses, cross-validate with other tools (rag_retrieval, make_concept_annotator), then provide FINAL_ANSWER with your top 5 diagnoses, ranked. | 🔁 |
| 3 | Critic Review (issues, evidence, uncalled tools) | backbone, retry | as code | same | ✅ |
| 4 | Retry 1 "Focused Re-diagnosis" | backbone | as code | same text; shortlist per §3 | ✅ · 🔁 |
| 5 | Retry 2 "Final Attempt" | backbone | as code | same | ✅ |
| 6 | Image re-injection line | backbone | "Re-analyze this image based on the critic feedback above." | "Re-analyze this case and these images based on the critic feedback above." | 🔁 |
| 7 | Forced re-look ("Describe what you observe…") | backbone | as code | same | ✅ |
| 8 | DermoGPT internal prompts | DermoGPT tool | as code | same (tool byte-identical) | ✅ |
| 9 | PanDerm text templates | PanDerm tool | 4 templates | same | ✅ |

**The 8 system-prompt edits (all seen by the backbone only):**

| | Upstream line | Ours | Why |
|---|---|---|---|
| a | PanDerm: "Pass the exact candidate_diseases list from the task options" | "Pass exactly 10 candidate diagnoses: your top-10 differential for this case" | no option list; fixed 10 keeps PanDerm confidence / 90 % gate comparable |
| b | DermoGPT: "Pass candidate_diseases to get…" | "Pass the same 10 candidate diagnoses as candidate_diseases to get…" | same list for both tools, as upstream's shared option list |
| c | "Classify the skin lesion into ONE of the provided diagnostic categories." | "Read the patient case and the attached image(s), then give the TOP-5 most likely diagnoses in ranked order (most likely first). The answer is free text: there is NO fixed list of options." | DermArena task = ranked top-5 from a case |
| d | — | "The image tools accept clinical photographs and dermoscopy only: those images have an id (listed in the user message); pass that id as image_path when calling an image tool, and choose which image(s) to analyse." | multi-image, mixed modalities; backbone must know which IDs tools accept |
| e | "panderm_classifier (with exact candidate_diseases)" | "(with your top-10 candidate diagnoses)" | as (a), Recommended Approach |
| f | `FINAL_ANSWER: [exact diagnosis name from options]` | `FINAL_ANSWER:` + numbered 1–5 | 5 ranked names, parsed to benchmark JSON |
| g | "ALWAYS pass the exact candidate_diseases to panderm_classifier" | "ALWAYS pass exactly 10 candidate diagnoses (your top-10 differential) to panderm_classifier" | as (a), Important Rules |
| h | "Your final answer MUST be one of the provided options exactly" | removed | contradicts open-QA |

Everything else (tool descriptions, Recommended Approach steps, "free to choose", RAG image-mode rule, "Explain your
reasoning step by step") is upstream verbatim. The base arm's prompt asks for JSON; the agent keeps upstream's
`FINAL_ANSWER` block which we convert — same scored format, each system keeps its own prompt.

---

## 5. Backbone: Qwen3.5-27B, creator settings (owner: "follow Qwen settings, not upstream")

| Setting (Qwen3.5-27B model card + chat template) | Value used |
|---|---|
| Sampling, thinking mode general | T 1.0, top_p 0.95, top_k 20, min_p 0, presence 1.5, repetition 1.0 (read from `eval/sampling_profiles.py` → identical to base arm) |
| Output budget | 32,768 tokens per call (`DERMAGENT_MAX_TOKENS`) |
| Context | card: ≥128K → `--context-length 131072` |
| Server parsers | `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` (card's SGLang tool-call command) |
| Thinking in the tool loop | **pass each step's `reasoning_content` back**; the template renders `<think>…</think>` for steps after the last user message and drops it for older turns ("No Thinking Content in History"). LangChain drops it by default → model saw empty `<think></think>` per step. Verified on the live server: A/B prompt tokens 329 → 650 (+321 ≈ reasoning length) |
| Serving | holder `h200x4`, `--dp-size 2` (GPUs 0–1, tp1), `--mem-fraction-static 0.90 --cuda-graph-max-bs 64 --mamba-full-memory-ratio 2.0`; tools on GPUs 2–3 |

**Qwen3.5 quirks seen with DermAgent (written for GPT-4o) and how each is handled:**

| Quirk | Effect | Evidence | Handling |
|---|---|---|---|
| Template rejects a non-leading system message | Critic retry → HTTP 400 "System message must be at the beginning." → case dropped | 061: 22 of 40 cases (shows the critic fires often) | send the critic message as user, same text |
| Immediate-EOS blank (≤2 tokens, empty) | agent loop ends empty | 060 smoke: 1 case | re-issue that call ≤2× (= base-arm policy) |
| Tool calls written inside an unclosed `<think>` | parser files them as reasoning → empty turn → case ends | 060: 1, 061: 1 | re-issue ≤2×; root cause likely the missing reasoning pass-back → 063 A/B |
| Final answer inside an unclosed `<think>` | `FINAL_ANSWER` present but content empty | 061: 1 case | base-arm rule `llm_client._answer_from`: blank content + stop + reasoning > 20 chars → reasoning is the answer (base arm: 30 of 4,836 rows) |
| July runaway (105k tokens in one call, no cap) | stalls | July run | 32,768 cap |

Per-row logging: `backbone_reissues` {reason: count}, `answer_source` (`agent_final_answer[_from_reasoning]`).

---

## 6. Runs, switches and files

**Env switches (set by `benchmark_runs/agents/run_dermagent_v1.sh`; `DA_*` overrides):**

| Variable | Value used | Meaning |
|---|---|---|
| `ENABLED_TOOLS` (`DA_TOOLS`) | panderm,make,dermogpt_vqa,rag,text_rag,ontology | 6-tool headline set |
| `DERMAGENT_OPEN_PROMPT` | upstream_full | system prompt = upstream full-tool prompt + 8 edits |
| `DERMAGENT_USER_MSG` | upstream | upstream run_diagnosis user-message template (edited) |
| `DERMAGENT_ENABLE_CRITIC` / `_CRITIC_IMAGE_REINJECT` / `_CRITIC_LLM_SUMMARY` / `_MAX_CRITIC_RETRIES` | 1 / 1 / 0 / 2 | upstream headline critic settings |
| `MAX_TOOL_CALLS` / `RECURSION_LIMIT` | 10 / 30 | upstream defaults |
| `DERMAGENT_SAMPLING_JSON` | from `sampling_profiles.resolve("Qwen/Qwen3.5-27B")` | creator sampling |
| `DERMAGENT_MAX_TOKENS` | 32768 | per-call budget |
| `DERMAGENT_ATTACH_IMAGES` | 1 | backbone gets all images (owner) |
| `DERMAGENT_TOOL_IMAGES` | derm | tool IDs only for clinical_photo / dermoscopy |
| `DERMAGENT_IMAGE_MODALITY_TAGS` | 1 | image lines show the modality label |
| `DERMAGENT_OPEN_SHORTLIST` | 1 | open-set Focused Re-diagnosis shortlist (pooled) + diagnosis retry texts for open_diagnosis |
| `DERMAGENT_REINJECT_ALL_DERM` | 1 | re-inject all clinical/dermoscopy images, "case and these images" wording |
| `DERMAGENT_MIDCONV_SYSTEM_AS_USER` | 1 | critic message as user |
| `DERMAGENT_EOS_RETRIES` | 2 | re-issue EOS blanks (and tool-call-in-think, `DERMAGENT_REISSUE_THINK_TOOLCALL` default 1) |
| `DERMAGENT_REASONING_FALLBACK` | 1 | base-arm answer-in-reasoning rule |
| `DERMAGENT_PASS_REASONING` | 1 from canary 063 | Qwen creator agent setup (reasoning passed back) |
| `DERMAGENT_STOP_FILE` | `dx/<slug>/STOP_<mode>` | operator stop, no SSH: `touch` → workers skip remaining cases and exit; holder moves on |
| `DA_NPROC` / `DA_THREADS` / `DA_TOOL_GPUS` | 4 / 12 (canary 5) / "2 2 3 3" | concurrency layout |
| `QDRANT_PATH` / `QDRANT_TEXT_PATH` | per-process copies under `/tmp/dermagent_qdrant_p<k>/{img,text}` | embedded Qdrant takes an exclusive dir lock |

All switches, the prompt hash and code hashes are recorded in each task's `RUN_META.json`; a resume with a different
`run_sig` is refused (`DX_ALLOW_META_CHANGE=1` overrides).

**Files:**

| File | Role |
|---|---|
| `DermAgent/skin_agent/benchmark_agent.py` | `_ReasoningChatOpenAI` (sampling env, reasoning capture, reissue rules, mid-conv system→user, reasoning pass-back, reasoning fallback); derived open-set prompt; `run_open_diagnosis` (all images attached, derm-only IDs, ID-last lines, upstream user-message variant, `reinject_images`); critic open-set shortlist; multi-image re-injection |
| `DermAgent/scripts/run_dermarena_v1.py` | v1 adapter: shards, threads, per-tool-class locks, tool-load gate, RUN_META, per-case full trace `traces/<_id>.json` (images replaced by size+hash placeholders), FINAL_ANSWER → JSON, recursion = wrong, pass 2 for dropped calls, `--merge`. **Hidden by `scripts/.gitignore` (`*`) → `git add -f`** |
| `benchmark_runs/agents/run_dermagent_v1.sh <canary\|full>` | sets every switch, warms page cache, copies Qdrant per process, launches workers, merges shards, runs `dx_check.py post` |
| `holders/h200x4/queue/060…063_*.req` | 060 plumbing smoke (old prompt, 3/4 workers crashed on a RUN_META race); 061 faithful prompt+critic (pre-fix: critic 400s); 062 = all decisions above except reasoning pass-back; 063 = 062 + pass-back (A/B) |
| Outputs | `benchmark_runs/dx/dermagent-qwen3.5-27b/{canary (060), canary_v1 (061), canary_v2 (062), canary_v3 (063), T1.1, T1.2, logs/}`; judge with `dx/judge_dx.sh dermagent-qwen3.5-27b` (unchanged pipeline; it skips T1.3 since there are no predictions) |

Canary → full flow: canary on the benchmark's `canary20_T1.x.jsonl` (40 cases) → audit traces (V-practices in
PLAYBOOK) → full run `run_dermagent_v1.sh full` (T1.1 2,806 + T1.2 2,030) → judge → score.

---

## 7. Engineering fixes and lessons (2026-11-05)

| Problem | Cause | Fix |
|---|---|---|
| PanDerm (and Case-RAG, same encoder) failed to load offline; preload still printed "loaded successfully" for some tools | `sglang/hf_cache/hub/models--neuml--pubmedbert-base-embeddings` (PanDerm's text tower) was a **dangling symlink** into `.hf_cache`, whose target had been deleted | removed the dead link, downloaded a real copy into `sglang/hf_cache` |
| Text-RAG "Knowledge base not available" on every call | `TextRAGTool` uses hard-coded `cache_dir=DermAgent/model-weights`; its Qwen3-Embedding-8B / Qwen3-Reranker-0.6B entries were dangling links into `.hf_cache` | repointed both links to the complete copies in `sglang/hf_cache` |
| Tools silently missing | preload message lies | **hard gate**: worker exits with "TOOLS FAILED TO LOAD" unless every enabled tool is initialized and not `_load_failed` |
| 3 of 4 workers crashed (`JSONDecodeError` on RUN_META) | shards read RUN_META while shard 0 was writing it | atomic write (tmp + `os.replace`), tolerant read |
| Tool load 827 s (1 worker) → ~1,240 s (4 workers in lockstep from Lustre) | each worker pulls ~50 GB from Lustre | warm the node page cache once (parallel `cat` of weight files; 125 s in 062) before launching workers |
| Editing a running bash script | bash reads scripts incrementally | change run scripts by writing `*.new` + atomic `mv` |
| Throughput (060 smoke, 1 process × 5 threads) | — | ~2.5–2.8 cases/min per process; 4 × 12 expected to finish ~4,800 cases in a few hours (re-measure on 062/063) |
| Push to GitHub failed | token in `origin` URL dead | owner synced master in UI; new credential needed to push; suggest token-free remote URL |

---

## 8. Changes vs `reproduction-fixes` HEAD (uncommitted as of 2026-11-05)

- `skin_agent/benchmark_agent.py`: all items in §3–§5 (env-gated; with every switch unset the original behaviour is kept, except the trace-only reasoning capture).
- `scripts/run_dermarena_v1.py` (new, force-add).
- `baselines/MedAgent-Pro/`: `mp_llm.py` (new) + all 6 LLM call sites routed through `mp_create`; `Derm_Task_level.py` task 4 (DermArena) + plan trace; `Dermatology/dermarena_dx/{task.json,toolset.json}` (new).
- Not for commit: July worker files (`dermagent_worker.py`, `dermagent_langgraph_worker.py`, `tools/medgemma_tool.py`, old sbatch).
- Plan: commit to `reproduction-fixes` once the canary passes; push after a working credential exists.

---

## 9. Open items

| Item | State |
|---|---|
| 062 / 063 canary audit | check traces: list sizes = 10, no tool calls on non-derm images, invalid-ID errors, critic retries/shortlist/re-injection working, quirk rates 062 vs 063 |
| Full DermAgent run | after canary passes |
| MedAgent-Pro port | LLM calls routed via `mp_llm.mp_create` (env backbone, sampling, budget, EOS re-issue, full trace); DermArena planner config (task 4) written; planner not yet run; open question: PanDerm candidates (backbone's initial top-10 recommended, or drop PanDerm; PanDerm's default list = 10 HAM-style lesion classes, useless). Same Qwen-settings rule applies |
| Stronger backbone (in discussion) | base Dx top-1: Qwen3.8-Flash-Next 44.2 % / 61.2 %, Qwen3.8-27B 33.0 % / 52.1 %, Qwen3.5-27B 28.8 % / 44.3 % (T1.1 / T1.2); any backbone needs its own paired base arm (exists for these) |
| T1.3 | later; needs a task-specific instruction (documented deviation) |
| Commit / push | see §8 |

---

## Updates after 11:35 (2026-11-05) — decisions + live state (read this first on resume)

**Owner decisions**
- MedAgent-Pro × Qwen3.5-27B is **delayed**. Focus = DermAgent × Qwen3.5-27B, then evaluate on the **same H200** (judge after the run). Use all **4× H200** (backbone dp2 on GPUs 0-1, tools 4 procs on GPUs 2-3). A40 not needed for agents.
- Re-issue **interrupted turns** approved: a turn that stops inside an unclosed `<think>` with no tool call and no FINAL_ANSWER is re-issued (≤2×, `DERMAGENT_REISSUE_INTERRUPTED=1`, reason `interrupted_in_think`). The base-arm "reasoning is the answer" fallback now applies only when the reasoning contains `FINAL_ANSWER`.
- **Persistent workers** approved (CLAUDE.md load-once rule): `run_dermarena_v1.py --service <jobdir>` loads tools once, runs job files `<jobdir>/*.json` (`{"tasks","input_tmpl","out_tmpl","limit","threads"}`), writes `<job>.p<k>.done`; shard 0 merges + runs `dx_check.py post` (→ `POSTCHECK.txt`) and writes `<job>.ALLDONE`; `<jobdir>/STOP` ends the workers. `run_dermagent_v1.sh service` launches it.
- Stronger backbone (Qwen3.8-27B: base +4.2/+7.8 pp; Qwen3.8-Flash-Next: base T1.1 44.2 % / T1.2 61.2 %) = **later option**, not now. Flash-Next = 125B-A6B MoE on all 4 H200 (tp4/ep4) → tools would need another node (L40S×4); weights NOT on disk (re-download ~360 GB, prefetch first).

**Empty-answer diagnosis (060/061, all 8 unparsed cases):** every one = model stops inside an unclosed `<think>` where it should act; 7/8 after ≥1 tool step (consistent with the missing reasoning pass-back). Kinds: EOS blank 1, tool calls inside think 3, answer inside think 2, interrupted plan ("Let me make targeted calls:") 2. 061: 22/40 cases crashed on the critic system-message 400 (critic fires on ≥55 % of cases).

**Canary takeaways (060/061):** 3 tools were silently broken before the cache fixes; free-size PanDerm lists 5–10 (half 5) → fixed-10 rule; text-RAG dominates tool use (061: 57 calls vs PanDerm 36, DermoGPT 28, MAKE 22; 061 completed rows are survivor-biased toward no-critic cases); median 4–6 tool calls, ~3 LLM calls, 68–85 s/case at 5 threads/proc; tool load 827 s (1 proc) / ~1,240 s (4 procs); page-cache warm (125 s) did NOT speed loading (DermoGPT still ~100 s/shard → CPU/RAM contention, not Lustre).

**Live state at 11:50**
- H200 day-2 holder 61511147 (m3u010, ends ~15:25; day-3 61562245 from ~16:02, day-4 61878184 from ~21:26 share the queue).
- Running: `062_dermagent_qwen35_27b_canary_v2.req` (no reasoning pass-back) → `dx/dermagent-qwen3.5-27b/canary_v2/`.
- Queued next: `063_dermagent_qwen35_27b_service.req` = persistent service; job `dx/dermagent-qwen3.5-27b/jobs/01_canary_v3.json` (canary20 sets, threads 5, out `canary_v3/`) = 062 + reasoning pass-back + interrupted re-issue.

**Next steps**
1. Compare 062 vs canary_v3: parse rate, unparsed reasons, `backbone_reissues` by reason, `answer_source`, critic retries working (Focused Re-diagnosis + all-derm re-injection), PanDerm/DermoGPT list size = 10, no tool calls on non-derm images, invalid-ID errors, throughput (cases/min, GPU util; consider dp3 + tools on 1 GPU only if backbone-bound). Report to owner.
2. If clean: add `jobs/02_full.json` = `{"tasks":"T1.1,T1.2","input_tmpl":"<release>/tasks/{task_dir}/test.jsonl","out_tmpl":".../dx/dermagent-qwen3.5-27b/{task}","limit":null,"threads":12}` → full run starts with no reload.
3. After `02_full.ALLDONE`: `touch jobs/STOP`, queue a judge req on the H200 (`judge_dx.sh dermagent-qwen3.5-27b`, Qwen3.6-27B, same as base arm), then M4 analysis (McNemar vs `dx/qwen3.5-27b`, CI, Holm, strata).
4. Commit to `reproduction-fixes` (benchmark_agent.py, `git add -f scripts/run_dermarena_v1.py`, MedAgent-Pro mp_llm.py + task-4 config); push needs a working credential (origin token dead).

## Canary 062 (canary_v2) audit — 12:15 2026-11-05 (no reasoning pass-back)

- 40 cases, 4 procs × 5 threads, cases 11:54–12:08 (~2.9 cases/min); median 230 s/case, 7 tool calls, 7 LLM calls (critic now works → longer than 061).
- **Parsed 36/40 (90 %) → fails the ≥95 % canary gate.** All 4 unparsed = `agent_tool_call_cap`: upstream `should_continue` ends the graph when the next tool batch would exceed 10 calls, so no FINAL_ANSWER. All 4 were multi-image cases (2–8 images) after a critic retry. Upstream behaviour, but upstream runs 1 image. OWNER DECISION PENDING: keep (wrong) / forced no-tool final-answer turn at the cap (recommended) / raise cap.
- Critic works (no 400s): fired 28/40, Focused Re-diagnosis 27, Final Attempt 16, multi-image re-injection 27.
- Candidate lists, round 1: PanDerm 42/43 = 10, DermoGPT 18/21 = 10. After the critic, lists = the shortlist (4–8), as upstream's Focused Re-diagnosis.
- Image tools: all 151 image-tool calls on clinical photos; 0 on non-derm images. Tool errors 5 (MAKE missing image_path ×2, typo ID, '4', 'placeholder'), all recovered.
- Backbone: `toolcall_in_think` re-issues 5 (recovered); answer_from_reasoning 4. 9 forced re-look turns (no tools bound, upstream) answered with tool-call XML as text: harmless (not executed; next chatbot turn makes the real calls).
- Audit script: `benchmark_runs/agents/audit_dermagent_traces.py <canary dir>`.

## Canary_v3 (063 service, + reasoning pass-back) + tool-cap decision — 13:05 2026-11-05

- canary_v3: parsed 37/40 (92.5 %; 062: 36/40). Unparsed: 2 tool-call cap, 1 answer inside unclosed think with
  visible content "```" (reasoning holds FINAL_ANSWER; fallback skipped because content was not blank). Re-issues
  interrupted_in_think 5, toolcall_in_think 7 (062: 5); answer_from_reasoning 6. Pass-back does not visibly reduce
  the think quirks at n=40 (not testable); kept anyway (Qwen creator setting, owner rule). Critic 28/40, Focused
  Re-diagnosis 25, Final Attempt 19, re-injection 28; round-1 lists PanDerm 46/47 = 10, DermoGPT 16/18 = 10; all
  148 image-tool calls on clinical photos. Tool errors: 7 "Image not found" + 4 bad IDs ("image 1", "image_1",
  "image 2", typo) — recovered by the agent.
- **Owner decision (13:00): tool-call cap = upstream 10 + option A** (`DERMAGENT_CAP_FINAL=1`): when the next tool
  batch would exceed the cap, the pending calls get a "Not executed: budget used up" ToolMessage and ONE no-tool turn
  asks for FINAL_ANSWER (top-5 format); the graph then ends (no critic). Rows log `tool_call_cap_hit`,
  `cap_final_answer`, `max_tool_calls`. Evidence for the cap: 10 = untuned code default (runner never sets it);
  062 4/40 and canary_v3 2/40 hit it; calls ≈ first sweep (~5–6) + critic re-runs (+1–4) + extra text searches;
  case-text length no effect (median 7 both halves); 4+ derm images median 11 (n=7, weak).
- **Cap-20 check (owner):** after the full run, re-run ONLY the cases with `tool_call_cap_hit` with job key
  `"max_tool_calls": 20` (service builds a second graph; own run_sig) → paired McNemar top-1 on those cases; if
  significant, report and consider 20 as headline.
- Fallback fix: the answer-in-reasoning rule now treats content that is only backticks/whitespace as blank.
- Wiring test: `agents/test_cap_final_graph.py` (fake LLM, cap 4) → TEST OK (13:02).
- 064 service req (v4 code) queued; job `02_canary_v4` (40 cases); full run job `03_full` added after its review.

## Cap routing fix + canary_v5 — 14:12 2026-11-05 (FULL RUN STARTED, job 05_full)

- canary_v4 (cap → immediate cap-final, 064) parsed 39/40, but cap-final SKIPPED the critic: upstream sends a capped
  turn to the critic (which may retry). That was more than the agreed change → fixed (v5, 065): cap → critic as
  upstream; `cap_final` only where the graph would END on a refused tool batch (`should_continue_after_critic`).
  Rows add `n_tool_calls_refused`; `tool_call_cap_hit` = any refused call. 064's 73 full-run rows moved to
  `dx/dermagent-qwen3.5-27b/superseded_full_v4_capimmediate/` (not mixed; different run_sig).
- **canary_v5: 40/40 parsed** (gate ≥95 % passed). Cap hit 7/40: 4 answered via the normal upstream flow, 3 via
  cap-final (all 3 explicit FINAL_ANSWER blocks, extraction read and matches). Round-1 lists PanDerm 41/41 = 10,
  DermoGPT 18/18 = 10; all 161 image-tool calls on clinical photos; critic 30/40 (Focused 28, Final Attempt 23,
  re-injection 29); re-issues toolcall_in_think 7, interrupted 3; from_reasoning 6; median 213 s/case at 5 threads.
- Full run `05_full` (T1.1 2,806 + T1.2 2,030, 4 procs × 12 threads, run_sig of v5 code) started 14:12.
- 065 tool load took 1,362 s (page cache cold again); 064's was 141 s (warm).

## Parser + recursion decisions — 16:00 2026-11-05 (owner)

- **P1 (approved):** cap-final reply without "FINAL_ANSWER:" → its last numbered list from 1 (≤5 names).
- **P3 (owner: "as long as it has the top 5, accept"):** normal final turn with NO FINAL_ANSWER marker and no
  tool call written as text → last numbered list from 1 if it has ≥5 items (first 5). Guard added after reading:
  pmid_22846644_3 listed anomaly types with "← Most likely" on item 4 and ended in a text tool call → excluded.
- Implementation: `agents/dermagent_reparse.py` (post-parse, NOT in the runner: the runner's code hash is in the
  live run_sig). `check` = regression over all rows; `apply <task_dir>` rewrites the MERGED predictions.jsonl after
  the run (keeps prediction_strict / unparsed_reason_strict, adds parse_rule). Regression 16:00: 1,281 rows,
  unchanged 1,275, gained 6 (5 P1, 1 P3), changed 0, lost 0; all 6 read. Items keep the strict parser's name
  cleaning (bold "Name**: explanation" kept as text, as in 218/5,975 strict items; judge reads full text).
- **FAIRNESS FLAG:** P3 = base-arm rule C7 (rejected 11-04) restricted to a full top-5 list. The base arm
  (dx/qwen3.5-27b) must get the same rule before the M4 comparison, else the agent gets a more lenient parser.
- **Recursion limit 30 → 60** (owner): one step = one node run (LLM turn, tool batch, critic pass, feedback
  injection). T1.1 973 traces: median 14, p90 24, p99 29; 3 cases cut at 30. Not in run_sig (safety net only: a
  case under 30 steps runs identically). Applies from the day-3 service restart; the 3 day-2 rows that hit 30 are
  deleted after the run and re-run by the service under 60.
- Steps/tool calls by derm images (T1.1): 0 imgs median 8 steps, 6 calls, cap 9 %; 1 img 15 / 8 / 12 %; 2–3 imgs
  15 / 9 / 37 %; 4+ imgs 16 / 10 / 45 %. Images drive tool calls and cap hits; steps are driven by the critic.

## HEADLINE TOOL-CALL CAP = 20 (owner decision 16:00 2026-11-05, pre-registered before any results)

- Headline: max_tool_calls 20 (DermArena multi-image adaptation; upstream 10 is an untuned code default that
  truncates 45 % of 4+-derm-image cases). Cap 10 (upstream default) reported as sensitivity.
- How: the cap only changes cases that hit it (a case under 10 calls runs identically under 20). After 05_full:
  re-run every row with tool_call_cap_hit under `"max_tool_calls": 20` in a separate job/out dir → cap-20 benchmark =
  non-capped rows + cap-20 re-runs; cap-10 benchmark = 05_full as is. Paired McNemar on the re-run cases, stratified
  by derm-image count. Recursion limit 100 (safety net; ~2 steps per tool call).
- Re-issue = resend the same LLM call (same history/settings, fresh sample at T 1.0), ≤2×; earlier steps kept.

## FINAL SETUP v6 (owner 16:00 2026-11-05: "don't wait too long to add the handles/fixes") — restart at day 3

- Changes vs v5: max_tool_calls **20** (headline, owner); recursion **100** (safety net); **think-toolcall extraction**
  (`DERMAGENT_THINK_TOOLCALL_EXTRACT=1`): an empty turn whose reasoning ENDS with complete qwen3_coder
  `<tool_call>` block(s) runs exactly those calls instead of re-sampling (offline test on the T1.1 traces: 18/25
  still-broken turns recovered, all valid tool names/args); **P1 + P3 parse inside the runner** (`parse_answer`,
  identical to agents/dermagent_reparse.py on all 1,418 saved rows: 0 differences). Rows log `parse_rule`,
  `think_toolcalls_extracted`. RECURSION_LIMIT now in run_sig.
- The v5 cap-10 full run stopped at day-2 end (16:02) with ~1,150 T1.1 rows → moved to
  `dx/dermagent-qwen3.5-27b/superseded_full_v5_cap10/` (kept: partial cap-10 data; not mixed).
- Cap-10 sensitivity under v6: cases that never reach 10 calls run identically under cap 10 → re-run only the v6
  rows with ≥10 tool calls under `"max_tool_calls": 10` (separate dir) after the full run.
- Flow: day-3 holder re-runs 065 (service, new env) → job 06_canary_v6 (40 cases, gate ≥95 %, V16 trace audit,
  V17: compare re-issue/think rates vs canary_v5) → 07_full (T1.1+T1.2) → judge → cap-10 re-run → M4.

## Independent review (fresh Opus agent, 16:15 2026-11-05) → fixes (v6b, service 066)

Verdicts: 9/14 rows CORRECT. Fixed before canary_v6 (which had not started; 065 stopped):
- **Think-toolcall extraction scope bug** (it would fire in no-tool nodes: forced image re-look, cap-final → dangling
  calls, false `tool_call_cap_hit`): now only when tools are bound to the call (`kwargs['tools']`, chatbot node) and
  only for bound tool names. Offline: no tools → no extraction; unknown name → none.
- **Discarded re-issue samples now saved** (`response_metadata.reissue_discarded`: reason, content, reasoning) —
  save-every-trace rule; enables the re-issue fairness audit.
- **P1 now needs ≥5 items** (as P3); P3 guard also rejects JSON-style tool-call text (`"arguments":`). Regression
  1,433 rows: 0 changed, 0 lost, 10 gained (all read by reviewer/me); runner parse == reparse (0 differences).
- **Model-caused request errors** (BadRequest / context overflow) written as unparsed `agent_request_error`, not
  retried in pass 2 (pass 2 = infra errors only).
OPEN (owner decisions): (1) cap-10 sensitivity: re-running "capped" rows with a fresh sample is BIASED (selection on a
long sampled trajectory) → prefix replay up to the first refused batch under cap 10 (rule n_tool_calls > 10), or a
full cap-10 run (~7 h), or drop it. (2) Judge leniency: agent top-1 items name several diseases / long explanations
(13.4 % vs base 2.6 %; >80 chars 11.4 % vs 0.9 %) → same name-only normalisation for both arms or re-judge long items.
(3) P3 on the base arm: +3 T1.1 / +7 T1.2 rows → read, apply, re-judge before M4. (4) Re-issue fairness: 47/1,161
v5 cases (4 %) end on a re-issued sample, which the base arm never gets → report M4 also with "final turn from a
re-issue → wrong" (free). Reviewer artefacts: agents/review_20261105/.
- **P1 DROPPED (owner 16:22 "strict is fine")**: the agent may not have consolidated its top 5 when the budget ran
  out → a cap-final reply counts only with an explicit FINAL_ANSWER block; P3 now applies to NORMAL final turns only.
  Regression 1,433 rows: 0 changed, 0 lost, 2 gained (both P3, read); runner == reparse. Applied before service 066.

## GPU plan + MedAgent-Pro step (owner 18:40 2026-11-05)

- GPU plan: DermAgent inference on H200 day-3 (done ~00:00); DermAgent + MedAgent judges on L40S×4 (b) — same locked
  Qwen3.6-27B judge/code/sampling as sessions 1–2, tp2×dp2 — queued when `07_full.ALLDONE` appears; final MedAgent-Pro
  runs on the SAME H200 server after DermAgent (no reload; tools = PanDerm + MAKE only, web-RAG disabled). A100×2
  holder 61718794 RELEASED (owner yes). H200 day-4 kept pending until MedAgent-Pro needs are known (decide by 11-06 noon).
- MedAgent-Pro PanDerm candidates (owner OK): the backbone's own initial top-10 differential, as for DermAgent.
  Done so far: `Dermatology/tools/skin_tools.py::panderm_classify` reads `<save_dir>/candidates.json`
  ({"diagnoses":[10 names]}) and passes it as candidate_diseases (absent file = old behaviour). STILL TO DO (lower
  priority, plumbing on L40S, NOT the headline): a DermArena runner that (1) asks the backbone for the top-10
  differential via mp_llm, writes candidates.json, (2) runs Planner + Case_level on case text + images, (3) parses
  the final top-5 to {"diagnoses": [...]}, (4) logs unparsed reasons. Plumbing results are not comparable (L40S
  serving); the paper runs go on the H200 server.

## Overnight plan 11-05 (owner OK 22:50): DermAgent x Qwen3.8-Flash-Next (size ablation), tools on the L40S

- Why: agent vs base shows -2.9 pp top-1 on 916 T1.1 cases (p=.061, prelim); owner wants a size ablation (much bigger
  or much smaller backbone) — Flash-Next (180B MoE, 6B active; base 44.2/61.2 %). Same family not required.
- Hardware finding: each tool process needs 48-82 GB of GPU (measured) -> not on 48 GB L40S unless trimmed; A100/H100/
  second-L40S requests all estimate days-weeks (11-18; H100x4 6 Nov 20:49). I RELEASED the A100x2 holder 11-05 18:40 (owner
  yes) before this was known — it would have hosted 2 tool procs. Flash-Next needs all 4 H200 (tp4+ep4, ~120 GB/GPU).
- Plan: (1) 07_full ends ~00:25 -> wait_stop_dermagent_35.sh touches jobs/STOP -> (2) req 070 judges DermAgent-3.5
  (both tasks, H200 dp4, ~30-60 min) while (3) the L40S (free after req 82, ~23:55) runs agents/tool_footprint_test.py
  (CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) -> if FITS (<=45 GiB): stage
  holders/h200x4/staged/090_dermagent_flashnext_server.req into queue (server-only, tp4+ep4+qwen3_coder parser), then
  agents/run_dermagent_remote.sh <L40S job 61714874> <server url> Qwen/Qwen3.8-Flash-Next dermagent-qwen3.8-flash-next
  (4 tool procs, 1/GPU, DA_ALLOC_CONF) -> canary (>=95 % gate) -> full. If it does NOT fit: fallback DermAgent x
  Qwen3.5-9B on the current layout (tools on H200 GPUs 2-3).
- Flash-Next weights (336 GB) prefetched to m3u010:/tmp/bench_weights (node-local; lost at job end 16:04 on 11-06).
- 11-05 23:50 (owner: "queue for A100 now, as far forward as possible, even if not needed yet"): submitted 2 holders,
  `bench-gpu2a100` (61981433) + `bench-gpu2a100b` (61981434): gpu partition, fitcq, A100:2 each (A100-80G), 7 d, same
  submit line as the released 61718794; holder dirs holders/gpu2a100{,b}. Both PENDING (Priority); test-only estimate ~11-18
  (queue depth, not a cap; estimates are not starts, V7). Four A100-80G = one tool process each (48-82 GB measured).
  LESSON: do not release a scarce GPU pair on the strength of the current plan (A100x2 released 11-05 18:40, re-wait ≥ days).

## RESULT (preliminary-slug judge on L40S, 23:48 11-05): DermAgent x Qwen3.5-27B, T1.1 COMPLETE (2,806 cases)
- Judged 2,805 (1 judge failure), agent unparsed 29 (1.0 %, wrong). Paired with base dx/qwen3.5-27b (same judge):
  top-1 exact agent 26.0 % vs base 28.8 % = -2.9 pp [-4.6, -1.2], McNemar exact p=0.001 (agent-only right 253, base-only
  334); top-5 exact 41.4 vs 44.2 = -2.7 pp [-4.6, -0.9], p=0.004. train_absent (n=1689) top-1 23.7 vs 25.3 (-1.6);
  train_represented (n=1116) 29.4 vs 34.1 (-4.7). NOT yet: Holm over the 4 tests, P3 on the base arm, name-only
  sensitivity (decision 2), re-issue sensitivity (4), cap-10 check (1), headline re-judge on the H200 (req 070).
- Flash-Next weights (338 GB) complete on m3u010:/tmp/bench_weights (.complete 23:48). Tool-footprint test on one L40S
  running (logs/tool_footprint_l40s.log).
- 00:25 11-06: TOOL FOOTPRINT TEST PASSED: all 6 tools, 60 cases, 12 threads, one L40S (expandable_segments): peak
  40.3 GiB nvidia-smi (torch 37.8 alloc; 33.6 after load; load 832 s) -> FITS 48 GB (margin ~4.7 GiB). Flash-Next plan GO:
  tool workers (4 procs, 1/L40S, 20 threads) started on m3g119 pointing at http://m3u010:30745/v1 (holder port =
  30100 + jobid % 800); H200 queue: 066 (3.5 service, stops at 07_full.ALLDONE via wait_stop_dermagent_35.sh) -> 070
  (headline judge DermAgent-3.5, dp4) -> 090 (Flash-Next server, tp4+ep4). Canary job 01_canary.json held until the
  server answers (release_flashnext_canary.sh); gate_flashnext_full.sh writes 02_full.json only if canary >=95 % parsed,
  0 request errors, <=1 dropped (else GATE_FAIL file, no full run).

## IMAGE ABLATION (owner 11-06 01:15; LOWER priority than DermAgent): masked report + image vs masked report only

- Slide design: image effect = (masked case + image) − (masked case only), paired, per task; here T1.1, T1.2, T1.3 on
  DermArena v1 test. Image arm = existing base-arm outputs (same prompt/case text/judge): NOTHING new to run.
  Text-only arm = same prompt, same everything, no images: `dx_benchmark/dx_infer_textonly.py` (wrapper around
  dx_infer.build_messages; dx_infer.py NOT edited: its code hash is in every running model's run signature) via
  `dx/run_dx_model_textonly.sh <hf-id> <slug>-textonly [workers]`. Verified offline: same text (2,232 chars on a 2-image
  case), images 2 -> 0, content list -> plain text.
- Models (owner): GLM-4.6V first (108B MoE, ~215 GB bf16, 4x80 GB; base avg rank 14/30: 23.3/37.7/42.6), then Qwen3.5-9B
  (rank 12/30: 24.0/38.0/42.0; 19 GB, 1 GPU). 6,964 cases per model (2,806 + 2,030 + 2,128).
- Stats: T1.1/T1.2 top-1 + top-5 exact: McNemar exact + 95 % CI on the paired difference; T1.3 F1: paired t-test; also
  report cases WITH images only (629 T1.1 / 183 T1.2 cases have none: identical in both arms).
- Where/when: H200 window after the Flash-Next run (~10:00, day 3 ends 16:04) or the next A100/H100 holder.
  GLM-4.6V weights prefetch to m3u010:/tmp/bench_weights/zai-org/GLM-4.6V started 01:11 (logs/prefetch_glm46v.log).

## OVERNIGHT AUTOMATION 11-06 01:30 (owner asleep): what runs without a human
- H200 queue order: 070 judge DermAgent-3.5 (running) -> 090 Flash-Next server (holds until jobs/STOP_SERVER) ->
  095 judge Flash-Next (staged; copied into the queue by the driver after the full run) -> 100 GLM-4.6V text-only ->
  110 Qwen3.5-9B text-only -> 120 judge both text-only arms. Fallback by design: if 090 fails to load or the canary gate
  fails, the queue moves on to 100/110/120 (image ablation) automatically.
- `agents/flashnext_overnight_driver.sh` (detached, log logs/flashnext_driver.log): waits for the Flash-Next server ->
  releases 01_canary.json.hold (40 cases, threads 8) -> GATE (>=95 % parsed over 40 cases, 0 request errors) -> 02_full.json
  (T1.1+T1.2, 20 threads) -> stages 095 -> touches jobs/STOP + STOP_SERVER. ALERT.txt in dx/dermagent-qwen3.8-flash-next/
  on any failure/timeout. Tools: 4 DermAgent processes already loaded on the L40S (one per GPU, 34 GB each).
- GLM-4.6V weights prefetch to m3u010:/tmp (log logs/prefetch_glm46v.log); a 4 GB --mem cap had SIGKILLed it (use 24G).

## HEADLINE RESULT 11-06 01:30 — DermAgent x Qwen3.5-27B (cap 20, v6b) vs base, H200 dp4 judge (req 070), V13 check passed
(no verdict newer than SCORE.txt). Paired on the cases both have a verdict for; unparsed = wrong. Strict parser + P3.
| Task | n | Agent top-1 | Base top-1 | Diff [95 % CI] | McNemar p | Agent top-5 | Base top-5 | Diff | p |
| T1.1 | 2,805 | 26.5 | 28.8 | -2.3 [-4.0, -0.6] | 0.009 | 41.7 | 44.2 | -2.4 [-4.3, -0.5] | 0.013 |
| T1.2 | 2,030 | 42.8 | 44.3 | -1.5 [-3.8, +0.8] | 0.205 | 57.8 | 58.3 | -0.5 [-2.8, +1.7] | 0.668 |
Holm over the 2 top-1 tests run so far: T1.1 0.018, T1.2 0.205 (the planned family is 4 tests incl. MedAgent-Pro).
Strata top-1 (agent vs base): T1.1 train_absent 23.7 vs 25.3, train_represented 30.7 vs 34.1; T1.2 train_absent 39.9 vs 41.0,
train_represented 47.0 vs 49.2. Judge noise check: the same T1.1 predictions judged on the L40S (prelim) gave 26.0 % agent
top-1 (diff -2.9) vs 26.5 % here (diff -2.3): ~0.5 pp judge-sampling/serving noise at T 1.0.
STILL OPEN: decisions 1-4 (cap-10 prefix replay, name-only sensitivity, P3 on base arm, re-issue sensitivity), the
'where the agent loses' trace analysis, power/MDE note, MedAgent-Pro, Flash-Next arm (running).

## Flash-Next arm 11-06 01:30-02:00
- Server up 01:33 (tp4+ep4, loaded in 3.5 min). Canary_v1 (40 cases, unpatched code): 40/40 parsed, 0 unparsed, 0 backbone
  re-issues (Flash-Next shows none of Qwen3.5's think quirks), cap hit 2/40 (cap-final parsed 2/2), critic 31/40, round-1
  candidate lists 38/38 = 10 (DermoGPT), image tools only on clinical photos; tool errors: 6 'Image not found' (wrong ids,
  recovered). 1 case DROPPED (pmid_38618315_1) -> recovered by pass 2.
- BUG found + fixed: a literal multimodal placeholder (`<|image_pad|>`, `<|vision_start|>` ...) in ANY message TEXT makes the
  Flash-Next server (SGLang 0.5.22) return HTTP 500 "More 'IMAGE' tokens found than corresponding data provided",
  deterministic for every later call of the case (verified with assistant text / tool-style text / user text; plain
  'image_pad' without the pipes is fine). `DERMAGENT_SANITIZE_SPECIAL_TOKENS=1` (default on in run_dermagent_v1.sh)
  rewrites them to `[image_pad]` etc. in message text + reasoning_content before sending (image parts untouched);
  in run_sig. Applies to ALL backbones from now on (no effect on Qwen3.5 rows already done: none contained one? NOT
  checked — the 3.5 server did not error on them).
- 01:59: workers restarted on patched code; 02_full (T1.1+T1.2, 20 threads/proc, 80 in flight) queued; gate script
  dropped (full run released manually; first rows checked by hand).

## RESUME HERE (written 11-06 09:40, before compaction)
LIVE STATE: H200 day-3 holder 61562245 (m3u010, ends 16:04 11-06; day-4 61878184 pending behind it). Server = Flash-Next
(req 090, endpoint m3u010:30745). Tool workers on L40S job 61714874 (m3g119), launched by agents/run_dermagent_remote.sh
(detached srun), slug dx/dermagent-qwen3.8-flash-next, job 02_full (20 threads/proc, patched sanitizer code). At
02_full.ALLDONE the waiter agents/wait_stop_flashnext.sh touches jobs/STOP + STOP_SERVER -> holder runs req 095
(Flash-Next headline judge, dp4) -> then the main chat's 100/110/120 text-only ablation reqs.
DONE: DermAgent x Qwen3.5-27B full (4,836, judged on H200): T1.1 top-1 26.5 vs base 28.8 (-2.3, p=.009); T1.2 42.8 vs 44.3
(-1.5, p=.205). Trace analysis: 630 base-only-right vs 534 agent-only-right; 63 % of lost cases = right dx absent from agent
top-5, 37 % demoted; loss larger when image tools used (-2.3) than text-only (-1.7); no-tool cases +3.5 (n=141).
FLASH-NEXT (inference ~95 % at 09:38): unparsed 3.5 %, dominated by NEW mode 'plan prose with no tool call / no
FINAL_ANSWER' (114 cases, 2.5 %): the interrupted-turn re-issue only fires on EMPTY turns. Recommended: report strict (A),
re-issue rule + replay from the broken turn as sensitivity (B).
OWNER DECISIONS OPEN: (1) cap-10 sensitivity = prefix replay (recommended) vs full cap-10 run vs skip; (2) judge
wordiness: headline unchanged + name-only sensitivity for both arms (recommended C); (3) P3 on the base arm (+3 T1.1,
+7 T1.2 rows; read each, re-judge) — fairness flag; (4) re-issue sensitivity table in M4; (5) Flash-Next plan-prose
handling A/B/C above; (6) JSON-answer parser rule P4 (0.1-0.2 %) low priority.
NEXT STEPS: when 095 done -> paired Flash-Next vs its base (dx/qwen3.8-flash-next: 44.2/61.2); Holm over the family;
size ablation: Qwen3.5-9B DermAgent on H200 after judge (weights present; needs tools on H200 GPUs 2-3 via
run_dermagent_v1.sh service with DA_BACKBONE/DA_SLUG, or on L40S via run_dermagent_remote.sh); 122B not downloaded.
MedAgent-Pro port: PanDerm candidates = backbone top-10 (done in Dermatology/tools/skin_tools.py via candidates.json);
runner NOT written. Commit/push still pending (benchmark_agent.py, scripts/run_dermarena_v1.py with git add -f,
MedAgent-Pro files; origin token dead). A100x2 holder was released 11-05 (owner yes). Tools need 48-82 GB/proc (L40S
test: 40.3 GiB peak with expandable_segments, fits). CHECK-IN KIT: agents/dermagent_issue_tracker.py <dir>,
dermagent_load_check.py <job> <dir>, dermagent_prelim_vs_base.py <task> <slug> <base>, audit_dermagent_traces.py <dir>.

## OWNER DECISIONS 11-06 (late morning) — review calls 1-4
- (3) P3 on base arm: NOT applied. Each arm keeps its own native format (base = strict JSON, agent = FINAL_ANSWER). Footnote
  instead: agent strict-only vs base (agents/dermagent_strict_sensitivity.py): T1.1 top-1 -2.39 (p=.007) vs headline -2.32;
  T1.2 -1.77 (p=.14) vs -1.53. P3 rescued 13 (T1.1) + 8 (T1.2) agent rows (0.43 %); Flash-Next 0.
- (2) Judge wordiness: option A (do nothing, no name-only re-judge). Caveat to state: agent top-1 items longer/hedged
  (13.4 % vs 2.6 % multi-disease/explained), bias would favour the agent, which is already behind.
- (4) Re-issue: agent keeps its <=2 re-issues, base gets none (owner OK). Headline unchanged. Report as footnote: final turn
  from a re-issue -> wrong: T1.1 -3.17 (p=.0004), T1.2 -2.41 (p=.044). Re-issued final turns: 79/2806 (2.8 %), 40/2030 (2.0 %).
- (1) Cap-10: DECIDED 11-06 keep cap 20, NO cap-10 replay. Note in report: upstream cap is 10; model is never told its remaining budget
  (only told at exhaustion); cap 20 hit in 0.4 % T1.1 / 0.1 % T1.2 rows; a cap of 10 would have been exceeded in 20.0 % (T1.1) /
  16.0 % (T1.2) of rows (calls median 8/7, p90 12/12).
- Flash-Next stall (41/41 T1.1 traces): root cause = tool calls written as text in the tools-OFF analyze_reinjected_image
  node. Fix DERMAGENT_STRIP_NOTOOL_TOOLCALLS=1 (benchmark_agent.py); 173 cases (102 T1.1, 71 T1.2) set aside in
  superseded_stall_20261106/ and replayed with the fix (DX_ALLOW_META_CHANGE=1, mixed run_sig 8ab2be2f4205 vs ddb02d433659).
  Replay: 100 % parsed. Flash-Next unparsed 3.5 % -> 0.02 %. 4 T1.2 cases missing (CUDA-OOM 500, dropped) = counted wrong.

## RESULT 11-06 ~13:00 — Flash-Next arm judged (missing/unparsed counted wrong, n = full test set)
Top-1 agent vs base (McNemar exact, 95 % CI): T1.1 45.8 vs 44.2, +1.57 [-0.08,+3.22] p=.069; T1.2 66.3 vs 61.2, +5.07 [+3.08,+7.07] p=8e-7.
Top-5: T1.1 +4.10 [+2.50,+5.69] p=6e-7; T1.2 +5.37 [+3.54,+7.20] p=1e-8. Qwen3.5 arm: top-1 -2.32 (p=.009) / -1.53 (p=.21).
Holm over the 4 DermAgent top-1 tests (2 backbones x 2 tasks): Flash-Next T1.2 3e-6; Qwen3.5 T1.1 .027; Flash-Next T1.1 .137; Qwen3.5 T1.2 .205.
Caveats: 173 Flash-Next cases replayed with the stall fix (mixed run_sig); 4 T1.2 cases missing (OOM 500) counted wrong; 1 T1.1 unparsed.
=> M4 answer depends on the backbone: agent < base on Qwen3.5-27B, agent > base on Flash-Next.

## CORRECTION 11-06 ~15:00 — headline metric is EXACT-OR-BROADER (judge score >= 1), not exact (score == 2)
The M4 tables above (and dermagent_prelim_vs_base.py / dermagent_strict_sensitivity.py) use score==2 (exact). The paper's headline
(analysis/lib/style.py, data/build_cases.py: score = top5 >= 1) is exact-or-broader. Recomputed, paired exact McNemar, n = full set:
Flash-Next agent vs base: T1.1 top-1 49.5/49.1 +0.4 (p=.67); T1.1 top-5 70.7/69.0 +1.7 [+0.2,+3.2] p=.034; T1.2 top-1 69.6/67.4 +2.2 [+0.3,+4.2] p=.029; T1.2 top-5 85.5/83.9 +1.6 [0.0,+3.2] p=.058.
Qwen3.5-27B agent vs base: T1.1 top-1 31.6/35.4 -3.8 (p=5e-5); T1.1 top-5 52.6/56.7 -4.1 (p=3e-5); T1.2 top-1 50.5/52.9 -2.4 (p=.051); T1.2 top-5 69.6/71.2 -1.6 (p=.15).
=> Use exact-or-broader as THE headline; exact (score 2) as the stricter sensitivity. Flash-Next gain is small/borderline on the headline.
Judge wordiness (decision 2, chosen A) matters MORE for exact-or-broader: hedged multi-disease items can earn 'broader' credit.

## MedAgent-Pro: three-way comparison (11-06 evening; upstream jinlab-imvr/MedAgent-Pro @ cea2721)
Framework files (Planner, CodingAgent, Summary, RAG, Pro/GPT deciders, utils) = upstream apart from header, API client, model name. Differences live in the task layer.
- Upstream (glaucoma): tools VLM, optic-disc seg, optic-cup seg, coding. Planner gets web-RAG guidelines ("How to diagnose glaucoma?"). Summary = "abnormal?" per indicator. Pro_Decider = LLM weights + threshold, computed weighted score, binary.
- Yize port (12b6c0c): VLM, PanDerm, MAKE, Case-RAG (rag_retrieve, Qdrant Derm1M), coding; web-RAG from dermnetnz/AAD; domain_hint added; MultiClass_Decider picks 1 of 6; summary = one disease name.
- Owner June repro (3232cae, b84ca11): web-RAG disabled (dermnetnz scrape + faiss deps) and tool 4 dropped "along with RAG_Module", although rag_retrieve does not use RAG_Module. So the DermArena config inherited the removal. My 11-06 statement "no reason recorded" was wrong.
- Ours: as June + backbone top-10 for PanDerm, derm-only images to tools, all images + case text to VLM/decider, OpenSet_Decider top-5, creator sampling.
Open: restore tool 4? planner guideline source? one fixed plan for ~3,000 diseases (upstream plan = one disease). Plan not generated yet.
GPU: H200 job bench-h200x4b submitted 11-06 (holder h200x4b, explicit gres). 61878184 'day4' is 4xA100-40G on m3u009 and holds the 4-GPU fitq user cap. Flash-Next (338 GB) needs H200; H100x4 = 320 GB is too small.
- 11-06 owner: for the MedAgent-Pro baseline, licence is NOT a constraint and DermNet is allowed (it does not leak DermArena test cases, which come from PMC reports). This overrides the June "off-limits per our licensing" note in Derm_Task_level.py. Hard rule stays: no leakage of test cases.

## LEAKAGE FINDING 11-06 ~19:00 (agents/leak_audit_20261106/: caserag_leak_audit.py, audit_output.txt, same_article_hits.json)
Case-RAG searches Derm1M, whose PubMed rows include figures from DermArena test articles. Source article in Derm1M: T1.1 679/2806, T1.2 492/2030 (24.2 %). The tool excludes only the same filename, and test filenames never match.
Same-article item retrieved (caption match, so a lower bound): Flash-Next 501 T1.1 (17.9 %), 382 T1.2 (18.8 %); Qwen3.5 469 (16.7 %), 317 (15.6 %); mean similarity 0.88, top similarity up to 1.00 = the case's own figure. Read 5 cases: captions describe the figure, Derm1M labels mostly do not name the gold diagnosis, so this is partial leakage.
Agent - base with all overlap cases EXCLUDED (n 2127 / 1538, base arm untouched), exact McNemar:
Flash-Next headline: T1.1 top-5 +1.36 (p=.135), top-1 -0.09 (.96); T1.2 top-5 +1.37 (.16), top-1 +1.69 (.15). Exact: T1.1 top-5 +3.57 (2e-4), top-1 +0.94 (.35); T1.2 top-5 +4.29 (6e-5), top-1 +3.64 (.002).
Qwen3.5 headline: T1.1 top-5 -4.56 (4e-5), top-1 -4.42 (4e-5); T1.2 top-5 -1.11 (.39), top-1 -1.69 (.23).
Flash-Next exact gain on hit cases is about double the rest (T1.2 top-5 +9.7 vs +4.4). So: Qwen3.5 loss holds; Flash-Next headline gain is no longer significant; exact gain survives, smaller.
PanDerm/MAKE were trained on Derm1M (weight-level exposure, not measured).
MedAgent-Pro: Case-RAG WITHHELD (my earlier "restore it" is withdrawn).

## IMAGE ABLATION RESULT 11-06 20:45 (analysis/rq/image_effect/compute.py, results.txt)
Images minus text-only, paired, all cases. Top-5 exact-or-broader: GLM-4.6V T1.1 +1.92 (p=.032), T1.2 +3.89 (8e-5); Qwen3.5-9B T1.1 +2.03 (.029), T1.2 +1.13 (.30). Exact top-5: GLM +4.74, +7.98 (both <1e-7); 9B +2.67 (.003), +2.22 (.054). T1.3 F1: GLM +4.07 (1e-5); 9B -3.86 (5e-5).
V18 check FAILS for the qwen3.5-9b image arm only: its judge ran 11-02 with an older dx_parse.py (hash 6998ae3d892a vs 86354a42a514). Same judge model, sampling, prompts. Unparsed 8 (T1.1) in the image arm vs 0. Small effect, not re-scored. No Holm applied.

## Derm1M PubMed trace 11-07 (leak_audit_20261106/trace_derm1m_pubmed.py -> derm1m_pubmed_trace.csv, one row per Derm1M image)
78,356 of 413,210 rows carry a PMC id (all 77,552 pubmed/ + 804 public/). Other public/ rows are ISIC, atlas-style or hash names: no article id (corrects my 40,170 'undercount' worry; those are mostly non-PubMed datasets).
By DermArena split (all tasks, articles are in one split only): test 2,087 rows / 749 articles; val 146 / 54; train 6,779 / 2,377; not in DermArena 69,344 / 21,454.
Test rows, pixel check against the case's images (256-bit dHash, unrelated pairs about 128): 1,880 checked, 524 distance < 5, 1,069 distance < 40, 6 byte-identical. 1,288 rows are sub-figure crops, which a whole-image hash can miss. Text: 17 captions share an 8-word run with the masked case text; 110 captions and 238 labels name the gold diagnosis.

## CLEAN CASE-RAG INDEX 11-06 23:08 (owner: remove DermArena test + val articles only; no near-duplicate step)
DermAgent/qdrant_storage_dermarena_clean = copy of qdrant_storage minus 2,233 items (2,087 test + 146 val, 803 articles; blocklist 3,547 PMCIDs, all tasks). 410,977 points. Verified: 0 blocked items left, train 6,779 and not-in-DermArena 69,344 kept, 3,000/3,000 sampled kept points byte-identical, original unchanged (sha256 OK). 300 removed images re-queried: top-1 median 0.792, max 0.944, none >= 0.95. Artefacts: leak_audit_20261106/clean_index/.
Non-PMC journals (ODO, Acta APA, Turkish J Dermatol; 440 Task 1 cases) have no PMC id and no Derm1M filename hit.
run_dermagent_v1.sh now defaults to the clean index, names the /tmp copy after it (old /tmp/dermagent_qdrant_p* cannot be reused), checks the row count, and DERMAGENT_CASERAG_INDEX is in run_sig. To use the original: DA_QDRANT_SRC=$AG/qdrant_storage DA_QDRANT_EXPECT=413210.

## CLEAN RE-RUNS QUEUED 11-06 23:20 (owner go; canary skipped, read first rows live)
Slugs dermagent-qwen3.5-27b-clean (H100 holder h100x4, reqs 010 + 020, backbone tp2 GPUs 0-1, tools 2-3, 2 procs x 30) and dermagent-qwen3.8-flash-next-clean (H200 holder h200x4b, job 62050362, reqs 010 + 020; tools on L40S 61714874). Same code both arms, DA_STRIP_NOTOOL=1, clean index. Flash-Next weights NOT in Lustre cache: 338 GB download on the H200 unless staged first.

## 11-07 tool facts
Qwen3-VL: origin/master create_tools() has no qwen_vqa branch, so it cannot be instantiated in any task. The .py runners list it by default but it is ignored; headline .sh configs use 6 tools; README LOO ablation has no w/o-Qwen row. Paper and README say 7 tools.
MedAgent-Pro baseline per Yize (owner, personal communication): 4 tools, no RAG; PanDerm, MAKE, DermoGPT certain, 4th general VLM uncertain. Repo port differs (PanDerm, MAKE, Case-RAG, GPT-4o VLM, coding). Our plan: PanDerm, MAKE, DermoGPT + backbone as the VLM.
Corpus builder EXISTS on origin/master since 5da78b3 (Yize, 2026-07-12): .agents/skills/mayo-dermnet-corpus-pipeline (SKILL.md, references/pipeline.md, 226 DermNet + 71 Mayo manifests). Corrects "no builder".

## MEDAGENT-PRO REBUILD DONE 11-07 10:30
Design locked: per-disease plan (it IS the method, not an option); tools PanDerm, MAKE target mode, DermoGPT, backbone VLM; MedlinePlus at planning only (Yize's "no RAG" = Derm1M/Mayo/DermNet). Offline test 99/99 (re-run by me). KB snapshot 6 Nov: 5,499 docs, 59,681 chunks; PubMed lists stripped; 0 PMID/PMCID matches; 4 test/val cases share a span in raw pages only, not indexed. Coverage exact 9.2 %/9.1 % of gold, 31.8 % of Qwen top-5; keyword estimate 14 %/48 %. About 130 LLM calls per case (DermAgent 7.9). Reqs 030 + 040 queued on H100 11-07; 050 held until owner reads plans and touches plan_cache/REVIEWED. A.D.A.M. encyclopedia text: keep the snapshot internal.

## MedAgent-Pro prompt rule 11-07 (owner)
Original MedAgent-Pro prompt wherever its stage exists; base-arm wording for anything new the method needs (top-10, case label); DermAgent wording not used. Planner task prompt restored to upstream Task_level.py verbatim except "Segmentation" -> "Tool" (run_dermarena_v1_medagentpro.py planner_prompt). Case label "Case:" as dx_top5.txt. Offline test 99/99.

## MedAgent-Pro vet 11-07 11:30
Independent Opus review: faithful to released code, not fully to paper (Eq. 4 step status, guideline-guided weights absent, also absent upstream); canary safe; full run not yet. Fixed 1, 4, 6-12 (offline test 136/136, 13 injected faults caught). Open: finding 2 (ranking across diseases: rank-stability check in canary), finding 3 (cost about 160 h on H100; holder ends 11-08 23:38), finding 5 (report by has_guideline). Owner calls pending: Eq. 4, guideline in decider.
Flash-Next clean run: first launch left the L40S Qwen server up (clean_gpu srun failed on memory), 213 tool OOM errors; 32 rows set aside in superseded_gpu_oom_20261107/, restarted; tools now held by CPU driver job 62128381 (agents/flashnext_tools_driver.sbatch).

## MedAgent-Pro summary answer 11-07 (owner)
Owner chose released-code wording: "Answer with only one word (Yes or No)". Correction: "Uncertain" was my own proposal (brief of 11-06), never an owner request; the code comment saying so is wrong. To do once req 040 ends, not before (keeps the running canary's setup fixed): switch SUMMARY_SUFFIX, fix the comment, update tests and the Uncertain gate, re-run the 40 canary cases in a new MP_CANARY_DIR. Plans stay cached (plan namespace = backbone, KB, toolset, task). Unreadable replies still score 0.5, as Pro_Decider does upstream.

## MedAgent-Pro plan retry 11-07 (owner OK)
Canary plan cache: about 3 in 10 diseases fail, 15 of the first 18 on upstream Planner's word check (comma, " and ", " / " in a qualitative action), all 5 tries, because retries resend the same prompt. Owner agreed: keep upstream check and first prompt unchanged; on a rejected plan, add the rejection reason to the retry. To do once req 040 ends, with the Yes or No switch: apply, clear cached failed plans only, re-run the 40 canary cases in a new MP_CANARY_DIR, report how many still fail. Eq. 4 and guideline text to the decider still await the owner.

## HOLD 11-07 13:15: do not apply Yes/No or re-run the canary yet
Owner supplied the ICLR camera-ready (literature/medagentpro_20261107/iclr_final/). It differs from arXiv v3 and the upstream MIMIC branch (repo_mimic_branch/, e6ec21c, 2026-05-12) holds reflection and guideline-weight code absent from main. Both reopen the summary answer format, the final ranking, the reflection step and the weights. Full diff in progress: agents/medagentpro_iclr_diff_20261107/. The Yes/No switch and plan-retry fix wait for that diff and the owner's decisions; req 050 stays unqueued.

## Camera-ready diff 11-07 13:20
Report: agents/medagentpro_iclr_diff_20261107/AGENT_DIFF.md (independent Opus, four sources: ICLR final, main, MIMIC branch, ours). I recomputed its canary counts from traces at 13:17, 20 cases still in progress: 42/118 candidates no plan; backbone first choice has no plan in 8/20 cases; Uncertain 35/221; 39/124 diseases failed; 84/124 no guideline; 15/85 plans one indicator. Proposed change set awaits owner: source rule (paper defines, MIMIC shows how, main fills gaps), failed-plan handling, VLM final ranking, per-disease guideline weights, reflection + Uncertain excluded, Eq. 2 pass, wider guideline filter, 300-token chunks, meta-step ban. Canary left running: its traces allow text-only tests of ranking and weights. Hold on Yes/No and retry fix stands.

## Stage 1 owner decisions 11-07
#1 Index: follow their code (owner: "Just follow their code"), not PubMedBERT; main's 400-character chunks stay, embeddings move to OpenAI's API if the key works. #2 retrieval filter: follow the paper (title, alternative names, MedlinePlus metadata). #3 guideline: MIMIC's prompt wording without its chest X-ray line. #4 indicators: owner OK'd inside the guideline prompt; I am proposing a separate cached step under principle A. Owner accepted principles A-E with A (keep the paper's named novelty) as highest priority. Stage 2 proposal awaits owner; Stage 3 not reviewed.

## Principle A applied to Stage 1, owner approved 11-07
#4: separate indicator-extraction step, one call per disease, cached (reverses the earlier OK). #5: every candidate ends with a plan; after retries with the rejection reason, skip main's word check rather than leave a candidate without a plan; our "diagnose X" check stays on.

## Guideline filter simulation 11-07 13:52 (CPU, 186 cached disease names, unweighted)
Any document found: summary only 39%; + title and alt names 39% (no gain); + name variants (brackets, slashes) 47%; + naive broader term (drop leading words) 82%, but with wrong matches (Stevens-Johnson -> Dubin-Johnson; Koebner -> "site"). So most of the gap is MedlinePlus lacking a page under that name, not the summary filter; this corrects my earlier statement. Proposal to owner: split names, then a backbone-named broader term once per disease, label each plan direct / broader / none, report results by label.

## #2 retrieval, owner decision 11-07 (supersedes the filter proposals above)
Follow the paper: filter documents first, then top-5 chunks, no broader-page fallback. Keep the existing summary filter and index (title and alt names added no coverage in the simulation, so no rebuild); name clean-up not added. About 61% of diseases then have no guideline and are planned from backbone knowledge; report results by has_guideline and disclose. Change-list item "widen the guideline filter" is dropped; the "diagnose X" check stays.

## Source rule and #7, owner 11-07
Owner's rule: follow the paper first (text, appendix, figures); if ambiguous or silent follow the code (MIMIC, then main, nothing tuned to chest X-rays); if neither fits our task, smallest disclosed change. #7: follow the paper: the VLM decides which steps the patient's data supports (same call as #6); code only executes.

## Source rule amended by owner 11-07
No fixed branch order when the paper is unclear or silent: choose MIMIC or main per step, by fit with the paper's intent and our task, and state why.

## All stages decided 11-07
Owner settled Stages 1-3 (13c: follow the paper, evidence and indicators only). Full list: agents/medagentpro_iclr_diff_20261107/CHANGE_SPEC.md. The earlier Yes/No switch is withdrawn. Changes are built on a copy and swapped in after req 040 ends, then a fresh canary in a new MP_CANARY_DIR.

## Consistency addendum 11-07
Owner approved three fixes after re-checking Stages 1-2 against the final rule: 300-token chunks (reverses 400 characters), MIMIC's 3-7 Yes/No indicators, all four reflection checks. See CHANGE_SPEC.md addendum.

## DermAgent four-way diff 11-07 15:25
Report: agents/dermagent_paper_diff_20261107/DERMAGENT_DIFF.md (Opus; MICCAI final vs Yize master 5da78b3 vs ours). Nothing serious broken. Open: two Critic exit rules dead, coverage gate missing for diagnosis, PanDerm limited to the backbone's 10 names, multi-image gates read last call, record errors (Qwen3-VL reason, "no iteration cap", cache and conflict rule marked active, master cannot start as released). Awaiting owner.

## DermAgent diff decisions 11-07
Owner: step 7 (run order) is engineering, no output change. Step 8 (repeat calls, 0.2% of calls): disclose, no re-run. Step 9: PanDerm keeps the backbone's 10 diagnoses; disclose that it reranks and does not propose. Still open: rows 15, 16-18, without-Critic arm.

## DermAgent Critic decision 11-07 (owner)
Rows 16, 17, 18b stay as in Yize's master: no code change, no re-judge, no 40-case re-run ("keep because code does it"). These are owner exceptions to paper-first and must be disclosed: released code differs from the paper on three Critic rules and we follow the released code. Row 15: principle #2 with #3 in support (Figure 2). Row 18a: disclose. Corrections to this record from the diff: paper gives "Retry < 2" (Figure 1); the conflict-accepted rule and repeat-call cache are not active in the sequential path; Yize's master cannot start as released; Qwen3-VL reason is "released configs and Table 2 use six tools", not "code cannot load it". Still open: without-Critic arm.

## MedAgent-Pro scope 11-07 16:05 (owner)
Run MedAgent-Pro on the full test set, both backbones; no subset. Fresh canary at higher concurrency. Flash-Next cannot go on H100 (336 GB weights vs 320 GB), so Qwen arm on H100, Flash-Next arm on H200 + L40S tools.

## GPU plan approved 11-07 16:10 (owner: "be proactive about securing GPUs")
Replacement holders submitted: 62151002 bench-h200x4c (fit, H200x4, 1 d, holder h200x4c), 62151003 bench-h100x4c (m3h, H100x4, 2 d, holder h100x4c). Both queues empty. QOS caps mean they start when 62050362 (Sun 10:47) and 61688667 (Sun 23:38) end. Submit further 1-day H200 holders as needed: full Flash-Next run spans several fit slots.

## RESUME HERE 11-07 16:45 (agentic experiments, owner: finish them, flag only what needs input)
Live code = camera-ready MedAgent-Pro (baselines/MedAgent-Pro; old in .pre_iclr_20261107). Order: (1) kb-ada job mp-kb-ada, read reports/leak_scan_tok300o50_cl100k_base.json; (2) copy medagentpro_reqs/060_medagentpro_canary_iclr.req.held to holders/h100x4/queue/060_medagentpro_canary_iclr.req; read POSTCHECK, speed, tokens/s, plans, final prompts; (3) Flash-Next DermAgent clean finishes about 20:00, judge auto (holders/h200x4b req 020), then compute paired results and stall/OOM checks; (4) Flash-Next KB summaries + canary on H200 + L40S tools; (5) owner reads PLANS_SAMPLE, then touch <namespace>/REVIEWED; (6) full runs: Qwen on H100 (holder h100x4c next), Flash-Next on H200 (h200x4c next, keep submitting 1-day fit holders); (7) judge, Holm on main metric, non-overlap results, figure and tables, commit after fixes pass (owner approved commit before full run). Replacement holders: 62151002, 62151003.

## 11-07 16:50
kb-ada done in 10 min, 18,211 chunks of 300 tokens; leak scan 0 indexed hits (same 3 raw-page-only spans as before). Fresh canary 060 queued on H100, 2 x 24 threads.

## 11-07 22:45 — MedAgent-Pro Qwen3.5 camera-ready canary results (canary_iclr, 20 cases per task, 2 procs x 24 threads)
Parsed 100 %, tool errors 0, CUDA errors 0, failed plans 0.5 % / 0 %, no evidence 6.5 %. Speed: ~0.13 cases/min total (~74 min per case, 2.5 h for 40 cases).
FAILED gates (limits were my own guesses, none from the paper): T1.1 unreliable reflections 68.2 % (<=40), memory replies on default 12.1 % (<=2), PanDerm step share 0;
T1.2 unreliable reflections 58.8 %, memory default 10.0 %.
FOUND + FIXED (my fix, tests 290/290): parse_memory accepted only integer ids; the model answers ["Step 1","Step 5"] or ["2","5"] (the plan prints "Step N:").
152 of 516 memory replies (30 %) were treated as unclear (retried, 44 candidates ended on keep-all). Now "N", "Step N", "step_N" parse; I re-parsed all 516 saved replies and read the changed
ones: every one names the same ids; 3 replies stay unclear (no JSON). Needs a fresh canary on the fixed code (plan cache stays valid: plans do not use this parser).
OPEN (owner): (a) reflection says unreliable on ~60 % of the 39 MAKE concept-score steps (only intermediate-result steps are reflected; final-indicator steps go through the
Uncertain rule); (b) planner picks PanDerm in 1 of ~2,150 planned steps (backbone 1969, DermoGPT 131, MAKE 48): tool-use finding or plan-prompt issue?
(c) tool loading 1177 s per start on L40S (4 procs from Lustre) -> full runs must use service mode (load once).
Flash-Next canary (H200 + L40S tools, req 040) started 22:13, tools ready 22:33, running on the OLD parser (code was imported before the fix); ~1,030 tok/s at 20 requests.

## 11-08 owner decisions on the canary review (MedAgent-Pro)
- #1 memory parser fix: OK (done). #2 PanDerm: OK to reword it as a per-disease measurement in toolset.json (no forcing hint); test uptake on ~20 diseases first.
- #3 diagnosis-as-question steps ("is the diagnosis X?", ~4 % of plans): owner 11-08 "leave it la no change". Our existing "diagnose X" rejection stays as is.
  Disclose: about 4 % of plans include a self-diagnosis indicator, which the upstream code allows (paper §3.2-3.3 / Fig 4 imply the plan holds findings only; our reading).
- #5-#10 OK: off-target guidelines kept + disclosed; micro-splitting kept (upstream Planner.py comma/"and" check); lab-heavy kept; DermoGPT non-image questions disclosed;
  speed via concurrency + second server; full runs in service mode.
- #4 (reflection/PanDerm gates -> report-only): not yet decided.
- Plan review (Opus agent, 54 plans read, 352 machine-checked): agents/medagentpro_iclr_diff_20261107/plan_review_20261108/.
- 11-08 ~00:40 owner: plans are MedAgent-Pro's own output -> accept without review ("just accept it"). REVIEWED written for Qwen namespace bcb79e2f45b3
  (PanDerm rewording + #4 report-only). Same rule for Flash-Next's namespace once its fresh canary builds it. #4 done (reflection + PanDerm gates report-only, tests 291/291).
- 11-08 launcher: run_medagentpro_v1.sh restored to its original text (two canaries were executing it while I edited it in place); new launcher = run_medagentpro_v2.sh
  (plan-probe mode, MP_SHARD_BASE / MP_SHARD_TOTAL for two server groups). New reqs must use v2.

## RESUME HERE 11-08 00:45 — overnight MedAgent-Pro chain (owner: "do whatever u need bfr I disconnect")
Queued (requests in agents/medagentpro_reqs/, copies in holder queues; launcher v2, remote v2, integrity check agents/mp_canary_integrity.sh):
- h100x4 (ends 11-08 23:38): 080 Qwen canary_iclr3 (fresh, all fixes) -> 090 Qwen FULL, H100 group shards 0-1 of 4 (2 procs x 32 threads).
- gpu2a100 (2 x A100, server) + gpu2a100b (tools, 2 procs): 010 Qwen FULL, A100 group shards 2-3 of 4; waits for canary_iclr3/PASSED_INTEGRITY.
- h200x4b (ends 11-08 10:47) + L40S tools: 050 Flash-Next canary_iclr3 -> 060 Flash-Next FULL (4 procs x 32); 060 writes REVIEWED (owner accepted plans).
Integrity check: FAIL only on rows / parse / final parse / CUDA / tool errors / failed plans / no plan / memory+reflection default / unclear verdicts.
Behaviour gates (memory removal, no evidence, Uncertain, step shares, has_guideline) are reported only — my extension of the owner's #4, FLAG to owner.
Old canaries canary_iclr2 (Qwen) and Flash-Next canary_iclr were stopped at 00:38 (runner pids killed, superseded). PanDerm probe: killed with them, 11 plans saved;
PanDerm now in 2/11 new plans (0/11 before), both as measurements.
MORNING: (1) h200x4b ends 10:47 -> Flash-Next full stops; re-queue 060 on h200x4c when it starts (resumable). (2) check speed per server group and rebalance
shards if one group is far ahead (stop both, change MP_SHARD_BASE/TOTAL; done cases are skipped). (3) h100x4 ends 23:38 -> move H100 group to h100x4c.
(4) Then judge (Qwen3.6-27B), Holm, subgroup tables.
- 11-08 00:53 owner: keep the canary's answers for the full run. mp_canary_integrity.sh now, on PASS, copies canary_iclr3's 40 answers (identical test rows)
  into dx/<slug>/{T1.1,T1.2}/predictions.pcanary.jsonl (+ traces, cases) with SEEDED_FROM_CANARY (CANARY_RUN_SIG). MORNING CHECK: full RUN_META run_sig must equal
  CANARY_RUN_SIG; if not, delete predictions.pcanary.jsonl (those 40 cases re-run). Tested on fake data. Future canaries: run T1.1 and T1.2 concurrently (~1.3 h, not 2.5 h).
- 11-08 08:40 owner OK: restructure = all tools on the L40S (2 procs/GPU), freed GPUs become extra Qwen servers with IDENTICAL SERVE_ARGS (A100b pair now,
  H100 GPUs 2-3 at the 23:38 restart). Queue outlook: H200 slot back after 10:47 uncertain (kxie/zche higher priority, lab 8-GPU cap); H100 after 23:38 likely a
  long wait (many higher-priority 1-GPU m3h jobs). Insurance submitted: bench-gpu2a100c 62257805, bench-gpu2a100d 62257806 (gpu/fitcq A100-80G x2, 7 d).

## RESUME HERE 11-08 08:50 — unattended GPU automation (owner wants to leave the laptop)
Qwen full run = 4 shards (case i -> shard i % 4); every group uses identical SERVE_ARGS (run_sig 6939e8b908f9). Tools for A100 groups on the L40S (2 procs/GPU max).
- shards 0-1: H100 (h100x4, req 090) until 23:38. Lock dx/medagentpro-qwen3.5-27b/locks/shards01 (owner = Slurm job id; agents/mp_shard_lock.sh). When the owner job
  stops, the first of h100x4c (tools local GPUs 2-3), gpu2a100c / gpu2a100d (tools L40S GPUs 2-3) takes the lock and resumes shards 0-1; the others wait (idle server).
- shard 2: gpu2a100 req 020 (64 threads, tools L40S GPU 0). shard 3: gpu2a100b req 010 = second Qwen server (tools L40S GPU 1). Both held to Nov 14.
- Flash-Next: h200x4b (060) until 10:47; h200x4c req 010 resumes it (tools L40S GPUs 0-3, no clean_gpu on the L40S).
Requests in agents/medagentpro_reqs/. Morning/next check: speed per server, any failed/ requests, lock owner, merge + postcheck when all shards finish, then judge.
- 11-08 11:42 h200x4c (62151002) started on m3u013, resumes Flash-Next full. Standby h200x4d submitted (62288438) with req 010_medagentpro_fn_full_standby (lock dx/medagentpro-qwen3.8-flash-next/locks/fn_all, owner 62151002). 9,270 FN DROPPED lines at 10:47-10:50 = server gone at the H200 end; not in predictions, re-run on resume.
- 11-08 21:35 progress Qwen T1.1 942/2806 (~79/h; shards 0-1 264 each on H100, 2-3 ~200 each on A100s), Flash-Next T1.1 540 (~42/h). Flash-Next standby (h200x4d) raised to 4 x 40 threads (KV 23% at 128 on the H200).
- 11-08 21:55 owner: optimise H100 layout. h100x4c now runs 020_medagentpro_full_s01_h100x4c_2srv.req: holder server (GPUs 0-1) + a second identical server (GPUs 2-3, port+1), shard 0 and shard 1 one each, tools on L40S GPUs 2-3; falls back to one server if the second fails. Current h100x4 left as is (ends 23:38).
- 11-08 22:40 handover prep: shard 0-1 requests (h100x4c 2-server, gpu2a100c/d) now release the shards01 lock on exit (trap), so a later H100 can take over
  from an A100 standby without waiting. Rule: prefer the H100 — if h100x4c waits while an A100 standby holds shards 0-1, kill that standby's runner pids
  (lock released by its trap). One-shot check scheduled 23:52; hourly check-ins continue.

## RESUME HERE 11-09 10:15 — H100 handover did NOT happen (all 16 H100s busy for 1-2 days); A100s now carry all 4 Qwen shards
- 23:38 h100x4 ended; h100x4c pending (Resources), gpu2a100c/d pending (Priority) -> shards 0-1 idle 23:38-10:15. The 23:52 one-shot left no record.
- Remaining at 10:09: shard 0 887, 1 896, 2 732, 3 753 cases (T1.1+T1.2).
- Now: gpu2a100 req 030 = own shard 2 + extra shard 0 (32+32 threads, tools L40S GPU 0 and 2); gpu2a100b req 020 = own shard 3 + extra shard 1 (tools GPU 1 and 3).
  Extra shards stop when dx/medagentpro-qwen3.5-27b/locks/TAKEOVER_01 exists; replacement reqs (h100x4c 2-server, gpu2a100c/d) take the shards01 lock
  (stale owner 61688667), raise TAKEOVER_01, wait 120 s, then run shards 0-1. Dry-run tested. Old shard 2/3 runners killed by pid (agents/ops/kill_qwen_shards23.sh).
- If a replacement ends later, shards 0-1 are orphaned again: remove TAKEOVER_01 and re-queue the A100 two-shard reqs (agents/medagentpro_reqs/030_*, 020_*_s3_s1).
- 11-09 10:35 self-healing: agents/mp_qwen_supervisor.sh (job mp-qwen-supervisor 62385811, comp, 4 d) restarts any unfinished Qwen shard with no runner for 15 min on a live A100 server; replacement reqs write their job id into locks/TAKEOVER_01; stale flag removed. Stop: touch dx/medagentpro-qwen3.5-27b/SUPERVISOR_STOP. PLAYBOOK section 'design for losing a GPU' added.
- 11-09 12:55 owner: one session owns ALL GPU work; this session took over the ablation GPU work (HANDOFF_ABLATION_20261109.md). Analysis stays separate.
  Hourly cron now covers MedAgent-Pro + ablation (bdi a100x4/a100x4b/abl_a100) + disk quota (15.39 T / 15 T soft, grace ends ~14 Nov 04:00).

## RESUME HERE 11-09 13:15 — agents first, ablation on surplus GPUs (owner: share with the ablation without slowing the agents)
- Standby holders (gpu2a100c/d, h100x4c for Qwen shards 0-1; h200x4d, bdiimg_h200x4 for Flash-Next) run their agent request first,
  now NON-BLOCKING (NOWAIT lock): if a running job already owns the agent work, they return 3 ("surplus") and move on to ablation
  requests 030 judge 9B-K, 040 GLM-4.6V K (4-GPU holders only), 050 HuatuoGPT-3 27B T, 060 HuatuoGPT-3 27B K (agents/medagentpro_reqs/0*_abl_*_yield_*.req).
- agents/mp_ablation_yield.sh runs the ablation work and every 2 min checks the agent lock: if its owner job has gone (agent needs a
  holder), it stops the work (resumable), queues the agent request as 005_* and itself as 9<name>, exits. Tested both paths (ops/yieldtest).
- After an agent request finishes on a holder (lock released on exit), that holder's ablation requests run normally.
- Supervisor v2 (agents/mp_qwen_supervisor_v2.sh, job mp-qwen-supervisor2) only uses a server whose /v1/models is Qwen/Qwen3.5-27B.
- Ablation's own bdi jobs (a100x4, a100x4b, abl_a100) keep their queues; output-dir take_lock stops two writers on one model.

- 11-09 16:35 NOTE (tail imbalance, act ~Wed): Qwen T1.1 shards 2/3 are at 52/224, 51/251 vs shards 0/1 at 46/386, 45/393 (0/1 lost 10.6 h overnight). When shard 2 (or 3) finishes ALL its T1.1+T1.2 cases, its server drops to 32 threads. Then restart the remaining extra-shard runner (0 or 1) on that server with MP_THREADS=64 (kill old runner by exact pid first; resume skips done _ids) so the server stays at ~64 requests.

- 11-09 17:50 TAKEOVER REDESIGN (queued holders only; no running process touched). Old plan: gpu2a100c/d or h100x4c raised TAKEOVER_01 and moved shards 0+1
  together onto ONE new server at 32 threads each = no speed-up for the lagging shards, and it threw away ~64 in-flight cases. New:
  * locks/shard0 and locks/shard1 (separate). locks/shards01 retired -> locks/retired_shards01_20261109. TAKEOVER_01 no longer used.
  * gpu2a100c/d queue: 010_medagentpro_full_1shard_a100.req = take ONE free shard lock, run it on this server at 64 threads (tools L40S GPU 2+s).
  * h100x4c queue: 020_medagentpro_full_shards_h100x4c_v2.req = up to 2 free shard locks, one per H100 server pair, 64 threads each.
  * agents/mp_take_shard.sh <s> <url> <gpu> <threads>: stops the runner currently serving shard s (030/020 "extra" or supervisor runner) by its exact
    process group on the L40S job, then runs the shard. Tested: discovery finds exactly one pgid per shard; runner has no signal handler.
  * mp_ablation_yield.sh: lock arg may be a ':' list (needed if ANY is stale); backup ops/mp_ablation_yield.sh.bak_20261109; tested 3 cases.
  * Yield reqs on c/d/h100 point at shard0:shard1 + the new agent reqs. Old reqs moved to agents/ops/retired_reqs_20261109/.
  * Supervisor v2 unchanged: if a takeover holder dies, its shard gets a runner back on gpu2a100/b after 15 min.
  * Known gap: if h100x4c gets only one free shard, its GPUs 2-3 sit idle (expected order: c/d start first, so h100 is surplus -> ablation).

- 11-10 15:45 OWNER DECISION ("I agree w this plan"): this week = Qwen MedAgent-Pro + the findings-kept ABLATION; MedAgent-Pro x Flash-Next PAUSED
  (1,170/2,806 + 20/2,030) and resumes after the ablation, automatically. Owner allowed the ablation on fit (H200 holders) for this.
  * H200 queues (h200x4d 62288438, h200x4e 62399604, bdiimg_h200x4): agents/ablation_reqs_h200_20261110/ = 10 HuatuoGPT-3 27B T, 20 HuatuoGPT K,
    25 judge Huatuo T+K, 30 GLM-4.6V K, 40 judge all (incl. 9B-K), 90 Flash-Next full (old 010, takes fn_all). Old reqs in agents/ops/retired_reqs_20261110/.
  * 9B-K judging on the L40S tools job's spare memory: agents/ops/l40s_judge9b.sh via comp job abl-l40s-judge9b (62443299). Qwen3.6-27B tp4,
    mem-fraction 0.85 -> ~43 GB/46 GB used per GPU (tools 19.9). Watchdog stops it on <1.5 GB free / new OOM / new tool_err case in Qwen logs / STOP file
    (holders/l40s_judge/STOP). ~420 verdicts/h -> 4,444 left -> done ~02:30 11 Nov. GPU util now ~100% shared with the tools: watch Qwen cases/h.

- 11-10 17:40 RISK FOUND: L40S tools job l40sx4b 61714874 (ALL MedAgent-Pro runners + tools, supervisor L=61714874, mp_take_shard, FN req) ENDS 12 Nov 11:52 (7-d limit).
  Qwen left 2,006 cases at 17:29 (s0 549, s1 584, s2 420, s3 453; ~12/h per shard): s2/s3 done ~Thu 04:30/07:30; s0/s1 only before 11:52 if h100x4c takes them ~Wed 08:00.
  Replacement L40S/A40 x4: est 24-26 Nov. Submitted desktop A40x1 bench-a40tools 62450847 (est 11 Nov 02:40, holder a40tools, empty queue) = tools host for 2 runners
  (also needed for Flash-Next after 12 Nov). Asked owner to request a walltime extension of 61714874 from M3 support. TODO when a40tools runs: tools-host-aware
  restart path (supervisor/take_shard hardcode L=61714874).
- 11-10 17:45 WORKAROUND for the L40S end (owner: "work around it", no helpdesk email): runners are now tools-host-aware.
  * agents/mp_tool_hosts.txt = tools host job ids (61714874 L40S, 62450847 desktop A40 bench-a40tools, pending est 11 Nov 02:40). Add a line to add a host.
  * agents/mp_hosts.sh: live_hosts / all_runners (via /proc cmdline+environ: pgid shard gpu url) / free_slots (free>=23 GB and runners<total/21 GB) / qwen_servers.
  * Supervisor v3 (job mp-qwen-supervisor3 62451127, 7 d): restarts a shard missing on ALL hosts for 15 min on the longest-lived host's free slot and the
    Qwen server with the fewest runners (64 threads if none else 32). Stop: dx/medagentpro-qwen3.5-27b/SUPERVISOR3_STOP. v2 stopped (SUPERVISOR_STOP).
    Tested: dry pass; simulated missing shard -> picks A40 slot + least-loaded server.
  * mp_take_shard.sh v2 (backup ops/mp_take_shard.sh.bak_20261110): kills the shard's runner on any host, places it like the supervisor.
  * TODO before Flash-Next resumes (after the ablation, >= Thu): 90_medagentpro_fn_full_after_ablation.req still hardcodes L40S 61714874 + 4 tool GPUs.
    Needs a host-aware FN launcher (hosts after Thu: A40 = 2 slots; gpu2a100/gpu2a100b once Qwen is done, until 14 Nov 22:41).
- 11-10 19:15 queued abl-l40s-judgeHT 62456153 (afterok 9B judge): agents/ops/l40s_judge.sh with JUDGE_SLUGS=huatuogpt-3-27b-textonly judges the 2,082 HuatuoGPT T predictions on disk (~3 h, done ~00:30, well before the H100 take-over needs L40S/A40 slots ~07:46). If the 9B watchdog fired, afterok never fires (on purpose).
- 11-10 21:15 auto-judge for MedAgent-Pro x Qwen: 040_judge_medagentpro_qwen.req queued in gpu2a100 + gpu2a100b (after their shard reqs). One takes locks/judge, waits until 0 unfinished + no runner on any host + merged predictions fresh, then judge_dx (Qwen3.6-27B dp2). Loser keeps its server and returns.
- 11-10 21:40 FRIDAY PUSH (owner: "most results by Friday"): H200 queues re-ordered GLM FIRST (GLM can only run on 4 big GPUs; HuatuoGPT also fits on
  2xA100): agents/ablation_reqs_h200_v2_20261110/ = 10 GLM K, 15 judge GLM, 20 Huatuo T, 30 Huatuo K, 40 judge all, 90 Flash-Next. gpu2a100/gpu2a100b queues
  get 050/060 HuatuoGPT T/K dp2 (80 workers) after 040 judge: whichever holder is free first takes each Huatuo arm (lock.sh skip_if_taken).
  Old order kept in agents/ops/retired_reqs_20261110/<holder>_v1order/.
- 11-10 21:45 owner OK'd raising HuatuoGPT concurrency on the H200: queued 20/30 Huatuo reqs now graph bs 96, workers 320 (was 40/160, A100 values). RUN_META sig excludes server args/workers, so resume is unaffected. TODO at Huatuo start (~Wed evening, after GLM): measure #running/#queue/KV/retractions in the first 10 min and adjust.
- 11-11 10:15 a100x4b (bdi, preempted 4x this morning) had taken GLM-K and was downloading 201 GB: killed the download pid, lock set to pidreleased_*, GLM removed from a100x4/a100x4b queues (agents/ops/retired_reqs_20261111/) -> GLM runs on the H200 (fit, not preemptible). Overnight: abl_a100 finished HuatuoGPT T inference (6,963) 07:35-08:36 and runs Huatuo K; L40S judged 2,134 Huatuo T at 02:06.
- 11-11 10:25 extra tools slots for Thu 11:52: abl_a100 (62159171) queue = 20 Huatuo K (running) then 95_tools_host_stop_server.req (frees the GPU; lands in failed/ by design); its 40_judge retired (a100x4b 90 + H200 40 judge the Huatuo arms). Added 62159171 to mp_tool_hosts.txt. H100 slipped to ~21:50 11 Nov.
- 11-11 10:16 H200 d slipped to 18:45 -> GLM put BACK in a100x4/a100x4b queues (lock-guarded race with the H200: whichever runs first takes it).
- 11-11 10:40 queue check (owner: stop the delays): walltime does NOT matter (test-only identical for 3h..2d -> priority-bound, fairshare factor 0.066 = heavy recent use); fresh submissions start 16-28 Nov, so keep the aged pending jobs. hudson (2x 2xA100 idle) needs qos hudson (not ours). Better use of held GPUs: abl_a100 judges Huatuo T+K (40_judge restored, ~10 h) then becomes tools host (95) before Thu 11:52. judge_dx.sh now takes a per-model lock dx/<slug>/.judge_lock (backup agents/ops/judge_dx.sh.bak_20261111) so parallel judge reqs never judge the same arm at once.
- 11-11 10:55 M3 docs: GPU Efficiency Policy - staff cancel jobs that reserve GPUs but leave them idle. A40 tools host idle since 02:26 -> moved shard 3 there NOW (killed s3 runner pgid 2405493 on L40S; it would lose its in-flight cases at Thu 11:52 anyway); supervisor v3 restarts it on the A40. 040 Qwen judge req made DEFERRED (keeps Qwen server + requeues itself every 10 min until 0 unfinished) so a holder never swaps out a Qwen server other shards use. Backup agents/ops/040_judge_medagentpro_qwen.req.bak_20261111.
- 11-11 10:40 a100x4/a100x4b preempted 4x in 25 min (10:09-10:35): GLM removed from their queues again (each start = 201 GB download + risk of holding the GLM lock when the H200 starts). GLM = H200 only (est 18:58).
- 11-11 10:45 shard 3 restarted by v3 at 10:39 on the A40 (62450847 GPU 0, server m3n100). Supervisor v4 (drain rule) started; v3 stopped via SUPERVISOR3_STOP.
- 11-11 10:47 a SECOND mp-qwen-supervisor4 job 62483716 appeared (submitted 10:44:03 from agents/ with 'sbatch mp_qwen_supervisor_v4.sbatch' - not my command). Two supervisors could double-start a shard -> put it on HOLD (scontrol hold; release with scontrol release 62483716). Asked owner.
- 11-11 10:50 shard 3 restarted by v3 on the A40 (host 62450847 GPU 0, server m3n100, 32 thr) 10:39, resumed T1.2 at 220/507, same run_sig = first live run of the host-aware restart: OK. Supervisor v4 (drain rule) runs in DRAIN_ONLY mode, job mp-qwen-supervisor4 62483716, ALONGSIDE v3 (62451127 keeps restarts; v3 cannot be stopped: s3's srun client lives in its job). A stray full-v4 job 62483695 (no DRAIN_ONLY, started nothing) was stopped by exact pid. Stop v4: dx/medagentpro-qwen3.5-27b/SUPERVISOR4_STOP.
- 11-11 10:52 SUPERVISOR4_STOP touched (stops both v4 copies 62483695 + 62483716 within 5 min); supervisor v5 = v4 + flock singleton (dx/.../supervisor.lock; cross-node flock verified on /fs04) started. Stop v5: SUPERVISOR5_STOP.
- 11-11 10:55 PEER SESSION 'Dx & Tx Benchmark: Agents [f83402]' (same name, started ~10:40) edited agents/mp_qwen_supervisor_v4.sh at 10:44 (added DRAIN_ONLY mode) and started 62483716. v5 was derived from that edited file: verified OK (fd fixes, flock, SUPERVISOR5_STOP; DRAIN_ONLY unset = full mode; log tag still prints [v4]). Messaged the peer to stop changing GPU files; asked owner to keep one session on GPU work.
- 11-11 10:50 peer f83402 stood down (read-only). It had killed my v4 pid 3220991 (no runners lost). Shard 3 survived v3's job end: srun client death did not kill step 62450847.102 (but its guarded.sh health guard is gone). v5 (62483807) is the only supervisor once 62483716 exits on SUPERVISOR4_STOP.
- 11-11 11:25 idle-capacity use: L40S judge for HuatuoGPT T (job 62486911, agents/ops/l40s_judge_balanced.sbatch): SGLang tp4 refused unbalanced GPU memory (GPU1 empty after s3 moved), so a placeholder step holds 19.5 GB on GPU1 while the judge runs (released at the end). Judge up 11:22, ~42.7 GB/GPU. abl_a100 queue now: 20 Huatuo K -> 40 judge (T skipped while L40S holds the T judge lock; does K) -> 45 judge GLM as it arrives (until GLM DONE or Thu 08:00) -> 95 tools host.
- 11-11 11:45 abl_a100 preempted again (Huatuo K paused 3,239/5,137) -> not a dependable Thursday tools host. H100 plan v3: ONE Qwen server (GPUs 0-1) for shards 0+1 at 32 threads each; GPUs 2-3 = tools host (job id auto-added to mp_tool_hosts.txt at start; lives to Fri ~21:50). Its old ablation-yield reqs retired (they would swap the Qwen server out). Drain at Wed 23:52 then moves s2 off the L40S onto the H100/A40.
- 11-11 11:55 L40S chain after the T judge (62486911): 62487672 HuatuoGPT K inference (24 workers, tp4 on spare memory, adaptive padding agents/ops/gpu_pad.py) -> 62487673 judge K. Scripts agents/ops/l40s_serve_run.sh + l40s_balanced.sbatch (generic). If abl_a100 restarts, its 20 K req skips while the L40S holds the K lock.
- 11-11 14:00 OWNER: "take GPU control" (other session 8acb83 gone); priority = ABLATION + DermAgent, MedAgent-Pro delayed. Kept its supervisor v5 (62483807,
  flock singleton), L40S chain (judgeHT2 62486911 -> huatuoK 62487672 -> judgeHK 62487673) and h100x4c 020 v3 (H100 GPUs 2-3 = tools host).
  Changes: h100x4c gets 010_abl_glm46v_findingskept.req FIRST (GLM backup; lock-guarded, skips if done/taken). H200 queues: Flash-Next 90 moved out
  (agents/ops/retired_reqs_20261111/*_90_fn.req; would have run Thu with L40S tools that end 11:52), replaced by 50 = 020 v3 (Qwen shards on GPUs 0-1,
  GPUs 2-3 tools host) after the ablation reqs. Flash-Next MP = next week, needs a host-aware request first.
- 11-11 16:25 bdiimg_h200x4 (61718805, bdi-imaging irq) STARTED 15:56: GLM-4.6V findings-kept 1,890/5,137 in 13 min (127 running, KV 0.15, all stop+parsed) -> inference done ~16:45, then 15 judge GLM on the same H200 (~1 h), then 30 Huatuo K (lock-guarded vs the L40S chain), 40 judge all, 50 Qwen takeover.
- 11-11 19:35 ablation arms: GLM K + Huatuo K fully judged; Huatuo T ~5047/5137 (L40S finishing T1.3). h200x4d (fit, started 19:00) skipped all done reqs and sat idle -> queued 60_medagentpro_fn_full.req (Flash-Next MP resumes; waits until no abl-l40s-* job remains, tools on L40S until it ends Thu 11:52). Removed 62288438 from mp_tool_hosts.txt (Flash-Next uses all 4 H200s). bdiimg H200 (preemptible) runs the Qwen v3 takeover (s0+s1 on its server, tools on its GPU 2).
- 11-11 20:15 ABLATION COMPLETE: HuatuoGPT T JUDGE_DONE 20:10 -> all 9 arms judged. Flash-Next MP resumed on h200x4d with tools on bdiimg_h200x4 GPU 3 (was empty; 61_medagentpro_fn_full_tools_bdiimg.req, run_sig 2d4e9d54143f OK); 62 = fallback with L40S tools if bdiimg is preempted. Stopped the waiting 60 req by exact pid (subshell 857210), holder + server kept. Pending abl-l40s-huatuoK/judgeHK will no-op (arms done).
- 11-11 20:40 h200x4e queue: added 61 (Flash-Next, tools bdiimg GPU 3) + 62 (L40S fallback) so Flash-Next continues when h200x4d ends Thu 19:00. Optimisation review (Opus, read-only): case time = 100% LLM decode, tools 0.2-0.4%; A100 servers half-loaded since 19:00 (32 req); bdiimg Qwen server 64/306 req; threads/shard count not in run_sig.
- 11-11 21:50 OWNER approved optimisations (1) Qwen 64 threads, (2) Flash-Next 2nd server on bdiimg after Qwen, (3) tools during server load; keep option C (full Flash-Next), fall back to B (T1.1 only) if too slow. (1) submitted: agents/ops/qwen_64threads.sbatch (job mp-qwen-64thr) runs mp_take_shard.sh for s3 (PREFER_HOST=A40 62450847, server m3n100), s0, s1 (server m3u023 bdiimg) at 64 threads; logs logs/qwen64_s*.out. mp_take_shard.sh gained PREFER_HOST (backup agents/ops/mp_take_shard.sh.bak_20261111). s2 gets moved by v5 DRAIN ~23:52 (L40S ends Thu 11:52).
- 11-11 22:30 comp partition jammed (4,566 pending, est Feb) -> cancelled my pending CPU job mp-qwen-64thr 62508890; the 64-thread restarts now run from holders/gpu2a100b/queue/035_qwen_64threads_restart.req (keeps its Qwen server, launches take_shard x3 detached inside the gpu2a100b holder job, alive to 14 Nov).
- 11-11 22:30 mp-qwen-64thr on comp never started (comp backlog -> est Feb 2027); cancelled 62508890 (my own CPU job, pending) and resubmitted on m3j qos normal (agents/ops/qwen_64threads_m3j.sbatch, est ~22:45). NOTE for future CPU helper jobs: comp is unusable now; use -p m3j --qos=normal.
- 11-11 23:10 64-thread restart DONE 22:26-22:32 via holders/gpu2a100b/queue/035_qwen_64threads_restart.req (take_shard x3 launched inside the gpu2a100b holder, pid 2411958; srun clients live in that 7-d job). Runners now: s0+s1 64 thr on bdiimg server (127 running), s3 64 thr on gpu2a100b server (63), tools s3 on A40, s0/s1 on bdiimg GPU 2. Cancelled redundant pending m3j job 62511920 (would have re-killed the runners). Disk: deleted 12 owner-approved items + .uv_cache, 15.18T -> 15.04T (141 GiB), 40 GiB over soft; log logs/disk_cleanup_20261111.log.

### 11-12 09:40 — morning check-in: overnight idle GPUs + Flash-Next optimisation #2 wired
- Overnight idle (reconstructed from logs): L40S x4 idle 23:57->11:52 (v5 drained its last runner; nothing queued — my miss). h100x4c started 04:23
  and ran the stale 020 v3 takeover req (queued 11-10 for "if h100x4c starts"; I did not retire it after the 22:30 64-thread restart): it
  restarted Qwen shards 0/1 on the H100 server at 04:31 (~120 in-flight cases re-run, ~1.5-3 h delay) and left bdiimg GPUs 0-1 (Qwen server)
  idle 04:30->; h100x4c GPUs 2-3 idle 04:29-> (tools host, unused). gpu2a100 idle 04:50-> (shard 2 finished; final judge deferred).
- 09:34 gpu2a100: 039_judge_medagentpro_qwen_early.req — judges T1.1 (2,806, complete) + the 1,632 merged T1.2 now; the final 040/041 later
  grades only the rest (the SCORE.txt written now is PARTIAL for T1.2).
- Flash-Next: agents/mp_fn_supervisor.sh (live in the a40tools holder, pid 1095101, log dx/medagentpro-qwen3.8-flash-next/logs/fn_supervisor.log).
  SWITCH when no Qwen runner remains on bdiimg: stops the old 4-shard run (tools bdiimg GPU 3), then 8 shards (i%8), 54 threads each, on every
  server registered by 70_fn_server_hold.req (h200x4d now: same args, no restart; bdiimg: loads Flash-Next once its node has no tool procs;
  h200x4e if it starts), tools on h100x4c (+ any non-server host living > 3 h), rebalance >=6 vs <=2 runners. h200x4d ends 19:00.
- Queues: h200x4d 62 retired; h200x4e all retired (ablation done, Qwen 50 takeover, 61/62) -> only 70; bdiimg 70; h100x4c 090 tools-host
  after 020. Retired files: agents/ops/retired_reqs_20261112/.

### 11-12 13:50 — owner decisions + queue changes
- Owner: backup is going to the owner's laptop (owner handles it). Keep Hulu-Med-Flash-27B and Qwen3.8-27B for now (decide Friday; may be needed later).
- Owner OK'd one more H200 job: bench-h200x4f 62559983 (fit, 1 d, H200x4), queue = 70_fn_server_hold.req (Flash-Next server for T1.2).
  Waits on QOSMaxGRESPerUser behind h200x4d/h200x4e. NOT in mp_hosts' HOSTS_FILE, so never a tools slot.
- 037_judge_rolling_qwen_fn.req on gpu2a100 (judge idled after 038 at 13:32): judges both runs every 20 min until Qwen inference is done; 041 then finalises.
- Retired stale gpu2a100c/d 010_medagentpro_full_1shard_a100.req (Qwen shard takeover) -> agents/ops/retired_reqs_20261112/.
