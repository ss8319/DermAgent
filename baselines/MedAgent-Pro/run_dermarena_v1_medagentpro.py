#!/usr/bin/env python3
"""MedAgent-Pro on DermArena v1 T1.1 / T1.2 test, open-QA top-5: ICLR 2026 camera-ready design, one plan per DISEASE, then per-patient
reasoning, then one integrating call per case. Agreed changes: benchmark_runs/agents/medagentpro_iclr_diff_20261107/CHANGE_SPEC.md.

Conventions from DermAgent/scripts/run_dermarena_v1.py (helpers IMPORTED, not copied): shard k/n, threads, resume by _id,
predictions.p<k>.jsonl, dropped.p<k>.jsonl, RUN_META.json + run_sig (a resume with other settings is refused), per-case
trace JSON in traces/, --merge, --service <jobdir> persistent worker.

Per case:
  1. candidates: the backbone's top-10 differential (CAND_PROMPT; base-arm layout: images first, then text).
  2. stage 1, once per disease, cached by normalised name in <plan-cache>/<namespace>/ (atomic, first writer wins):
     MedlinePlus RAG (kb_rag: keyword filter over summaries, top-5 300-token chunks, ada-002 embeddings) -> guideline G (backbone)
     -> indicators I from G, or from the disease name when there is no G -> Planner gets G and I -> plan_postcheck -> weights and
     threshold from G and the plan's indicators, no patient data. A rejected plan is re-asked with the reason; after the normal
     tries the word check is skipped and a step that is the diagnosis itself is removed. A plan that still fails, or any entry
     hit by a fault, is used for this case only and never stored (audit copy in <namespace>/_failed/).
  3. stage 2, per candidate, in cases/<id>/cand_NN/:
     memory (Eq. 2): one backbone call sees the case text, the images and the plan and names the steps this patient's data supports.
       Other steps, and steps that depend on them, do not run.
     action: every remaining step is routed by its tool id, in plan order, at most MP_STEP_CAP (10) executed steps; later steps are
       excluded. Quantitative: PanDerm (all candidates, all scores), MAKE (target mode). Qualitative: backbone VLM (all images +
       case text) or DermoGPT (derm images only). A step that cannot run (no derm image) is dropped and recorded.
     reflection (Eq. 3, E.4): after each intermediate step, one backbone call sees the step's input images and its result and judges
       reliability. Unreliable = Terminate: that step and every step that depends on it are dropped; other indicators continue.
     indicators: each 'final indicator' is summarised to Yes / No / Uncertain (a blank or unclear verdict is asked once more). No extra
       reflection call: an Uncertain indicator is unreliable, dropped, and its dependants with it (Fig 4, E.4(4)).
  4. stage 3: comparison arm, no call: the cached weights rescaled over the surviving indicators, Yes 1, No 0, score - threshold.
     final answer: ONE backbone call per case gets each candidate's indicators and evidence, no case text, no images, and returns the
     ranked top 5 in the base arm's format (candidates without evidence stay listed as 'no evidence'). This call is the output.
     Unparsed after the retries = wrong, never guessed, never replaced by the comparison arm.
Errors: server / connection / timeout / 5xx anywhere -> case not written (dropped.p<k>.jsonl), retried in pass 2. A 400 in a step =
  that indicator is dropped and recorded; in memory or reflection = counted as an unclear reply; in the final call = unparsed.
An unclear memory or reflection reply is asked once more, then takes a stated default (keep the step; treat the result as reliable),
  recorded in the trace and counted in the record.

Usage:
  run_dermarena_v1_medagentpro.py --kb <kb> --plan-cache <dir> --tasks T1.1,T1.2 --out-tmpl <.../{task}> --shard 0/2 --threads 12
  run_dermarena_v1_medagentpro.py --kb <kb> --plan-cache <dir> --service <jobdir> --shard 0/2
  run_dermarena_v1_medagentpro.py --merge --tasks T1.1,T1.2 --out-tmpl <.../{task}>
  run_dermarena_v1_medagentpro.py --print-plans 10 --plan-cache <dir>        (CPU only: sampled plans for the owner to read)
  run_dermarena_v1_medagentpro.py --rank-stability 3 --tasks T1.1,T1.2 --out-tmpl <canary/{task}>   (text only: final call re-runs)
Env (set by benchmark_runs/agents/run_medagentpro_v1.sh): OPENAI_BASE_URL, MP_MODEL, MP_SAMPLING_JSON, MP_MAX_TOKENS,
  MP_EOS_RETRIES, MP_REASONING_FALLBACK, MP_TIMEOUT, MP_CAND_ATTEMPTS, MP_PLAN_ATTEMPTS, MP_PLAN_RELAXED_ATTEMPTS, MP_INDICATOR_ATTEMPTS,
  MP_WEIGHTS_ATTEMPTS, MP_FINAL_ATTEMPTS, MP_STEP_CAP, MP_CALL_BUDGET, MP_SANITIZE_SPECIAL_TOKENS, MP_EMBEDDER, MP_CHUNKS,
  OPENROUTER_API_KEY (real key, embedder only), MP_SERVER_ARGS (SGLang launch args; part of run_sig).
"""
import argparse
import difflib
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # DermAgent/baselines/MedAgent-Pro
AG = HERE.parents[1]                                   # DermAgent
DATA_ROOT = Path(os.environ.get("MP_DATA_ROOT") or HERE / "Dermatology" / "dermarena_dx")   # task.json / toolset.json
MAKE_TERMS = AG / "MAKE" / "concept_annotation" / "term_lists" / "ConceptTerms.json"

# Read the endpoint BEFORE importing modules that call load_dotenv() (Derm_Task_level): DermAgent/.env holds an
# OPENAI_BASE_URL / OPENAI_API_KEY for a remote API; it must never be picked up by a benchmark run.
_ENV0 = set(os.environ)
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY") or "local"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AG / "scripts"))
import run_dermarena_v1 as v1  # noqa: E402  (read-only reuse: rgb_path, done_ids, merge, lock_tool_classes, now, sha_file)
from medlineplus_kb import CHUNK_PROFILES, chunk_profile_name, normalise_name, sha_path  # noqa: E402

DC, RELEASE, TASK_DIR, DERM_MODALITIES = v1.DC, v1.RELEASE, v1.TASK_DIR, v1.DERM_MODALITIES
sys.path.insert(0, str(DC / "dx_benchmark"))
os.environ.setdefault("DX_IMAGE_ROOT", str(RELEASE))

# ----------------------------------------------------------------------------------------------- prompts (in run_sig)
# Top-10 differential = base-arm dx_top5.txt with 5 -> 10 (same message layout: images first, then text).
CAND_PROMPT = (
    "Case:\n{case}\n\nWhat are the 10 most likely diagnoses for this patient? List them in order, most likely first."
    '\n\nRespond with JSON only: {"diagnoses": ["<most likely>", "<2nd>", "<3rd>", "<4th>", "<5th>", "<6th>", '
    '"<7th>", "<8th>", "<9th>", "<10th>"]}'
)
# Raw input of a qualitative step: upstream passes the image only; DermArena's raw input is the case report + images, so
# the case text is the first context item (utils.build_qual_prompt layout unchanged).
QUAL_CASE_CONTEXT = "Case:\n{case}"   # base arm dx_top5.txt label
# Upstream Case_level.py summary_prompt, verbatim.
SUMMARY_PROMPT = ("Based on the above text, please provide a brief summary. The task is: {ques}. "
                  "Does this patient have the abnormal?")
# Upstream Summary_Module's commented-out line, re-enabled. Basis: the paper (Fig 4, E.4(4)), where an uncertain finding is unreliable and
# not considered; the reply must be able to say so. Uncertain is dropped, never scored.
SUMMARY_SUFFIX = "Answer with only one word (Yes, No or Uncertain)"
# New stage (paper p.4: 'summarizes disease-specific clinical indicators I from the guideline G'). 3-7 indicators and the Yes/No question
# form are the MIMIC branch's planner rules (Task_level.py), not chest X-ray specific.
INDICATOR_ASK = ("Give 3-7 indicators. Phrase each one as a diagnostic question that can be answered with Yes or No. "
                 "Do not ask whether the patient has the disease itself.")
INDICATOR_JSON = '\n\nRespond with JSON only: {"indicators": ["<question 1>", "<question 2>", "<question 3>"]}'
INDICATOR_PROMPT = ("Disease: {disease}\n\nClinical guideline:\n{guideline}\n\nSummarize the disease-specific clinical indicators from "
                    "this guideline. " + INDICATOR_ASK + INDICATOR_JSON)
INDICATOR_PROMPT_NO_G = "Disease: {disease}\n\nSummarize the disease-specific clinical indicators of this disease. " + INDICATOR_ASK + INDICATOR_JSON
# Planner input after G. The planner prompt itself stays as upstream's.
PLANNER_INDICATORS = ("Clinical indicators:\n{indicators}\n\nThe plan should have 3-7 final indicators (steps with output_type='final "
                      "indicator'). Each final indicator's action MUST be phrased as a diagnostic question that can be answered with Yes or No.")
PLAN_RETRY_NOTE = "\n\nYour previous plan was rejected: {reason}\nReturn a corrected plan."
# MIMIC generate_weights.py, minus the chest X-ray text and the roles (nothing here uses them); threshold range [0, 1] = main.
WEIGHTS_SYSTEM = ("You are a clinical decision assistant. Assign clinical importance weights to indicators for diagnosis.\n"
                  "Weights must sum to 1.0 and reflect how directly each indicator contributes to diagnosing the target disease.")
WEIGHTS_GUIDELINE = "Clinical guideline:\n{guideline}\n\n"
WEIGHTS_PROMPT = ("Disease: {disease}\n\n{guideline_block}Diagnostic indicators:\n{indicators}\n\n"
                  "For each indicator, assign:\n  - weight (float): clinical importance, all weights sum to 1.0\n\n"
                  "Also propose a decision threshold in [0, 1].\n  score = \u03a3(weight_i \u00d7 value_i), value_i in [0,1].\n"
                  "  Positive if score \u2265 threshold.\n\nIndicator keys (use EXACTLY): {keys}\n\n"
                  'Respond with ONLY a JSON object:\n{\n  "indicators": {\n    "<indicator_key>": {"weight": <float>},\n    ...\n  },\n'
                  '  "threshold": <float>\n}')
# New stage (paper Eq. 2; no code branch has a VLM version). Plain style of the base-arm prompts.
MEMORY_PROMPT = ("Case:\n{case}\n\n{images}\n\nDiagnostic plan:\n{plan}\n\nWhich steps does this patient's data support? Keep a step if "
                 "the input it needs is in the case text or the images. Leave out a step that needs data this patient does not have, "
                 'such as a kind of image or a test result that is missing.\n\nRespond with JSON only: {"supported_steps": [<step id>, <step id>]}')
# Upstream Examiner.py (dead code in main), text-result branch: its system line, then prompt + function + newline + result. Its word
# Examiner.py's wording, with its 'confidential' changed to 'reliable' (paper: phi 'evaluates the reliability'). The four checks are paper E.4 (1)-(4); (2) and (3) are reworded for scores instead of masks.
REFLECT_FUNCTION = "Determine if the result is reliable and accurate? Answer with only one word (Yes or No)"
REFLECT_SYSTEM = "You are a helpful assistant. Please determine if the result is reliable and accurate? Answer with only one word (Yes or No)"
REFLECT_PROMPT = (
    "{case}{guideline}{images}\n\nStep: {action}\n\nCheck the result of this step:\n"
    "1. Input quality: is the input image blurry, occluded, cropped, low-resolution, or missing the relevant anatomical region? "
    "Can this step be performed on it?\n"
    "2. Tool output: is the output plausible for what the image shows, and consistent with the clinical guideline if one is given?\n"
    "3. Scores: does each value lie in a valid range, and is it consistent with what the tool reports?\n"
    "4. Text: is each statement made only when clear visual or textual clues are present? If the result expresses uncertainty, "
    "it is unreliable.\n\n" + REFLECT_FUNCTION + "\n{result}")
# New stage (paper E.1: 'lets the VLM integrate the resulting evidence and indicators to choose the most plausible answer').
# Question and JSON line = base-arm dx_top5.txt. Each candidate's evidence, indicators and risk result (the per-disease analysis of
# paper 3.3): no case text, no images.
FINAL_PROMPT = ("Each candidate disease below was analysed separately for this patient.\n\n{evidence}\n\nBased on this evidence, "
                "these indicators and the risk results, what are the 5 most likely diagnoses for this patient? Choose from the candidates. List them in order, "
                'most likely first.\n\nRespond with JSON only: {"diagnoses": ["<most likely>", "<2nd>", "<3rd>", "<4th>", "<5th>"]}')
NO_EVIDENCE = "no evidence"
# Upstream Task_level.py planner_prompt, verbatim, except "Segmentation" -> "Tool" (no segmentation tools here). Owner 11-07: original prompt
# wherever a MedAgent-Pro stage exists; Yize's shortened build_planner_prompt is no longer used.
def planner_prompt(input_desc, disease_goal):
    return (
        "Plan a step-by-step, executable workflow using ONLY the available tools.\n"
        f"Input: {input_desc}\n"
        f"Goal: {disease_goal}\n"
        "Output format (STRICT): an array of objects with fields "
        "[id, tool, action_type, action, input_type, output_type, output_path].\n"
        "- id starts from 1 and increases by 1\n"
        "- tool is an ARRAY of integers (tool ids from the toolset)\n"
        "- action_type is a STRING: 'qualitative' or 'quantitative'\n"
        "- input_type is an ARRAY of integers; use 0 for raw/original inputs, or a prior step's id if the input is that step's output\n"
        "- The field output_type MUST be EXACTLY one of: 'intermediate result' or 'final indicator'\n"
        "- For any non-image output, set output_path EXACTLY to 'diagnosis.json'; only images may use distinct file paths\n"
        "- Use a VLM tool to observe potential qualitative indicators; list EACH indicator as a SEPARATE step (do not bundle multiple indicators).\n"
        "- Qualitative observation/judgement steps MUST set output_type='final indicator'\n"
        "- Tool/measurement/computation steps MUST set output_type='intermediate result' and be followed by a qualitative VLM judgement step referencing that metric step id to produce the final indicator\n"
        "- Steps must follow strict logical order with no forward references; each dependency must be produced before it is used.\n"
        "Return ONLY the JSON array."
    )


POSTCHECK_VERSION = "3"
PLAN_SCHEMA = "iclr-1"     # entry layout: guideline, indicators, plan, weights; a change gets a fresh cache
MAKE_EXCLUDE = {"gel", "orange sticker", "purple pen", "dermoscope border"}   # image artefacts, not patient findings
MAKE_FUZZY = 0.88
SKIN_MISSING = "no_derm_image"


def fill(tmpl, **kw):
    """str.replace, not str.format: case texts contain braces."""
    for k, v in kw.items():
        tmpl = tmpl.replace("{" + k + "}", str(v))
    return tmpl


class InfraError(Exception):
    """Server / connection / timeout / 5xx: the case is dropped and retried in pass 2 (= dx_infer policy)."""


class PlanRejected(ValueError):
    """plan_postcheck refused a plan the upstream validator accepted."""


def exc_kind(e):
    import openai
    if isinstance(e, InfraError):
        return "infra"
    if isinstance(e, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)):
        return "infra"
    if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", 0) and e.status_code >= 500:
        return "infra"
    if isinstance(e, openai.BadRequestError):
        return "request"
    return "other"


def guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa: BLE001
        if exc_kind(e) == "infra":
            raise InfraError(f"{type(e).__name__}: {e}") from e
        raise


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- lazy MedAgent-Pro imports
_M = {}


def mods():
    """Import the MedAgent-Pro modules once; drop anything load_dotenv() added (never use DermAgent/.env)."""
    if _M:
        return _M
    # Two different top-level `utils` modules: MedAgent-Pro's and DermArena's eval/utils (imported by dx_infer). Import the
    # MedAgent-Pro side first, then take its `utils` out of sys.modules so dx_infer binds its own.
    sys.path.insert(0, str(HERE))
    if "utils" in sys.modules and not str(getattr(sys.modules["utils"], "__file__", "")).startswith(str(HERE)):
        raise RuntimeError(f"foreign 'utils' already imported: {sys.modules['utils']}")
    import kb_rag
    import mp_llm
    from Planner import Planner
    from Summary_Module import Summary_Module
    from Decider import GPT_Decider, Pro_Decider
    import Derm_Task_level
    import utils as mp_utils
    from Dermatology.tools import skin_tools
    assert mp_utils.__file__.startswith(str(HERE)), mp_utils.__file__
    del sys.modules["utils"]
    import dx_infer
    import dx_parse
    assert sys.modules["utils"].__file__.startswith(str(DC)) and hasattr(dx_infer, "tables_block")
    for k in set(os.environ) - _ENV0:
        if k.startswith("OPENAI_"):
            del os.environ[k]
    _M.update(mp_llm=mp_llm, dx_infer=dx_infer, dx_parse=dx_parse, Planner=Planner, Summary_Module=Summary_Module,
              GPT_Decider=GPT_Decider, Pro_Decider=Pro_Decider, Task=Derm_Task_level, U=mp_utils, skin_tools=skin_tools,
              kb_rag=kb_rag)
    return _M


def make_client():
    from openai import OpenAI
    if not BASE_URL:
        raise SystemExit("[mp] OPENAI_BASE_URL is not set (local SGLang endpoint) — refusing to run")
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=float(os.environ.get("MP_TIMEOUT", "3600")),
                  max_retries=int(os.environ.get("MP_HTTP_RETRIES", "2")))


def check_backbone_env(backbone):
    if os.environ.get("MP_MODEL") != backbone:
        raise SystemExit(f"[mp] MP_MODEL={os.environ.get('MP_MODEL')!r} != --backbone {backbone!r}")
    if not os.environ.get("MP_SAMPLING_JSON"):
        raise SystemExit("[mp] MP_SAMPLING_JSON not set (creator sampling, = base arm) — refusing to run")


def load_task_cfg():
    return json.loads((DATA_ROOT / "task.json").read_text()), json.loads((DATA_ROOT / "toolset.json").read_text())


# ----------------------------------------------------------------------------------------------- names
def plan_token(name):
    """The disease as the planner sees it. Upstream's validator reads ',', ' and ', ' / ' in a qualitative action as
    bundled indicators, and plans repeat the disease name; this form has none of the three. Output keeps the original."""
    s = re.sub(r"\s*,\s*", " ", str(name))
    s = re.sub(r"\s+and\s+", " & ", s, flags=re.I)
    s = re.sub(r"\s+/\s+", "/", s)
    return " ".join(s.split())


def panderm_name(name):
    """PanDerm's own interface is a comma-separated string, and DermAgent passes names the same way: no comma in a name."""
    return " ".join(str(name).replace(",", " ").split())


# ----------------------------------------------------------------------------------------------- answers
_WORDS = {"yes": "Yes", "no": "No", "uncertain": "Uncertain"}


def normalise_answer(reply):
    """Summary reply -> (Yes|No|Uncertain, how). 'Yes.' must not fall through to Pro_Decider's 0.5 default. Anything that
    is not plainly one of the three comes back 'blank' / 'unclear_reply' (the caller asks once more); never turned into Yes or No."""
    t = (reply or "").split("</think>")[-1].strip()
    if not t:
        return "Uncertain", "blank"
    w = re.sub(r"[^a-z ]+", " ", t.lower()).split()
    if len(w) == 1 and w[0] in _WORDS:
        return _WORDS[w[0]], "exact"
    m = re.match(r"\W*(yes|no|uncertain)\W*[,.:;\u2013\u2014-]", t.lower())     # 'Yes, it is present', not 'No evidence to assess'
    if len(t) <= 200 and m:
        return _WORDS[m.group(1)], "first_word"
    return "Uncertain", "unclear_reply"


UNCLEAR = ("blank", "unclear_reply")


# ----------------------------------------------------------------------------------------------- MAKE concept map
class ConceptMap:
    """Plan text -> MAKE concept keys (ConceptTerms.json): whole-phrase match on any term, else the nearest term by string
    similarity >= MAKE_FUZZY, else none (the step then goes to the backbone VLM). No model involved."""

    def __init__(self, path=MAKE_TERMS):
        raw = json.loads(Path(path).read_text())
        self.terms = {}
        for key, syns in raw.items():
            if key in MAKE_EXCLUDE:
                continue
            for t in {key, *syns}:
                n = normalise_name(t)
                if n:
                    self.terms.setdefault(n, key)

    def map(self, action, disease=""):
        text = normalise_name(action)
        d = normalise_name(disease)
        if d:
            text = " ".join(text.replace(d, " ").split())
        padded = f" {text} "
        hits = []
        for term, key in sorted(self.terms.items(), key=lambda x: -len(x[0])):
            if f" {term} " in padded and key not in hits:
                hits.append(key)
        if hits:
            return hits, "phrase"
        words = text.split()
        best, score = None, 0.0
        for term, key in self.terms.items():
            k = len(term.split())
            if len(term) < 5:
                continue
            for i in range(len(words) - k + 1):
                r = difflib.SequenceMatcher(None, term, " ".join(words[i:i + k])).ratio()
                if r > score:
                    best, score = key, r
        return ([best], f"nearest:{score:.2f}") if score >= MAKE_FUZZY else ([], "none")


# ----------------------------------------------------------------------------------------------- plan post-check
def tool_kind(tool_def):
    cmd = str(tool_def.get("command") or "").split("(", 1)[0].strip()
    return {"": "backbone", "panderm_classify": "panderm", "make_annotate": "make", "dermogpt_vqa": "dermogpt"}[cmd]


# Words that name no finding. What is left of an action after removing them must be the finding.
GENERIC_WORDS = set("""classify classification classified annotate annotation annotated assess assessment assessing observe
observation judge judgement judgment determine determination evaluate evaluation examine examination analyze analyse analysis
does do did was were can could will would should has had which what how question questions
identify check review describe description score scores scoring run use apply perform detect detection look find compute measure
quantify estimate calculate derive extract obtain get generate
image images photo photos photograph photographs picture pictures lesion lesions skin patient patients case report text input
inputs raw disease diseases diagnosis diagnoses candidate candidates differential concept concepts feature features finding
findings indicator indicators sign signs tool tools model models output outputs result results probability probabilities
presence present absence absent whether step steps zero shot clinical dermoscopic dermatological dermatology visual medical
support supports consistent suggest suggests based on using from with for of the a an to in is are and or by this that its any
all if has have be there""".split())


def names_finding(action, token, keep_disease):
    """Does the action name something to look for? The disease itself counts only where the tool's job is about the disease:
    PanDerm's score, or a judgement of a tool result."""
    text = normalise_name(action)
    d = normalise_name(token)
    if d and not keep_disease:
        text = text.replace(d, " ")
    return any(w not in GENERIC_WORDS and not w.isdigit() for w in text.split())


_SYNTHESIS = re.compile(r"\b(synthesi[sz]e|synthesi[sz]ing|integrate|conclude|finali[sz]e|formulate)\b")
_ALL_INDICATORS = re.compile(r"\b(all|prior|observed)( prior| observed)? indicators\b")
_TOOL_WORDS = re.compile(r"\b(scores?|probabilit\w*|classif\w*|concepts?|annotations?)\b")
_VERB = r"(?:determine|assess|confirm|judge|evaluate|establish|classify|make)"
_LEAD = re.compile(rf"^{_VERB} (?:the |a )?(?:potential |final |primary |overall )*(?:diagnos[ie]s|diagnostic likelihood)\b")
_TAIL = re.compile(rf"^{_VERB} (?:(?!\b(?:for|of|if|whether|from|based|in|with|by|to|on|supports?)\b).)*\b(?:diagnos[ie]s|likelihood)$")


def is_diagnosis_step(action):
    """Is this step the diagnosis itself rather than an indicator ('diagnose X', 'synthesize ... final diagnosis', 'assess X
    diagnosis')? A judgement of a tool's scores is an indicator: 'assess X diagnosis based on probability scores' stays, since the
    tool's output needs one. Checked by reading every qualitative action in the canary's plan cache (BUILD_REPORT)."""
    t = " ".join(re.sub(r"[^a-z0-9\s-]+", " ", str(action).lower().replace("_", " ")).split())
    if _SYNTHESIS.search(t) or re.search(r"\bdiagnose\b", t) or re.search(r"\bfinal\b.*\bdiagnos", t) or _ALL_INDICATORS.search(t):
        return True
    if _TOOL_WORDS.search(t):
        return False
    return bool(_LEAD.search(t) or _TAIL.search(t))


def strip_diagnosis_steps(plan, notes):
    """After the normal tries: remove steps that are the diagnosis itself, what depends on them, and any tool step left without a
    judgement. Ids are kept, so the trace still matches the planner's reply."""
    gone = {s["id"] for s in plan if s["action_type"] == "qualitative" and is_diagnosis_step(s["action"])}
    for s in plan:
        if set(s["input_type"]) & gone:
            gone.add(s["id"])
    left = [s for s in plan if s["id"] not in gone]
    used = {d for s in left for d in s["input_type"]}
    orphans = {s["id"] for s in left if s["action_type"] == "quantitative" and s["id"] not in used}
    if gone:
        notes.append(f"removed steps {sorted(gone | orphans)}: diagnosis steps {sorted(gone)}, tool steps left unused {sorted(orphans)}")
    return [s for s in left if s["id"] not in orphans]


def plan_postcheck(plan, tool_by_id, concepts, token, relaxed=False):
    """Runs after the upstream validator (which is what prompts and rejects first). Covers what it cannot see: its metric rule
    looks for the verbs compute / measure / ..., so 'classify' and 'annotate' tool steps slip through; it accepts a bare verb as
    a step ('classify', 'annotate', 'assess'), which gives nothing to weigh; and it accepts the diagnosis itself as an indicator.
    relaxed (after the normal tries): the diagnosis check stays on, but removes such steps instead of rejecting the plan, as long
    as a real indicator remains. Returns (plan, notes). Raises PlanRejected: the Planner is then asked again, with the reason."""
    notes = []
    out = json.loads(json.dumps(plan))
    for s in out:
        if s["action_type"] == "quantitative" and s["output_type"] != "intermediate result":
            notes.append(f"step {s['id']}: output_type -> intermediate result")
            s["output_type"] = "intermediate result"
    bad = [s for s in out if s["action_type"] == "qualitative" and is_diagnosis_step(s["action"])]
    if bad and not relaxed:
        raise PlanRejected(f"step {bad[0]['id']} ('{bad[0]['action']}') is the diagnosis itself, not an indicator: "
                           "ask about one finding instead, and leave the final decision out of the plan")
    if bad:
        out = strip_diagnosis_steps(out, notes)
    tool_steps = {s["id"] for s in out if s["action_type"] == "quantitative"}
    for s in out:
        kinds = {tool_kind(tool_by_id[int(t)]) for t in s["tool"]}
        if not kinds:
            raise PlanRejected(f"step {s['id']} names no tool")
        keep = "panderm" in kinds or (s["action_type"] == "qualitative" and bool(tool_steps & set(s["input_type"])))
        if not names_finding(s["action"], token, keep):
            raise PlanRejected(f"step {s['id']} is vacuous: '{s['action']}' names no finding")
        if "make" in kinds and not concepts.map(s["action"], token)[0]:
            raise PlanRejected(f"MAKE step {s['id']} ('{s['action']}') names no MAKE concept")
        if s["action_type"] == "quantitative":
            if not kinds <= {"panderm", "make"}:
                raise PlanRejected(f"quantitative step {s['id']} must use a tool, not a VLM: {sorted(kinds)}")
            if not any(q["id"] > s["id"] and q["action_type"] == "qualitative" and s["id"] in q["input_type"]
                       and q["output_type"] == "final indicator" for q in out):
                raise PlanRejected(f"tool step {s['id']} ('{s['action']}') must be followed by a qualitative VLM judgement "
                                   "step that references it and outputs a final indicator")
        elif not kinds <= {"backbone", "dermogpt"}:
            raise PlanRejected(f"qualitative step {s['id']} must use a VLM: {sorted(kinds)}")
    if not any(s["output_type"] == "final indicator" for s in out):
        raise PlanRejected("plan has no final indicator: every indicator was the diagnosis itself" if bad else "plan has no final indicator")
    return out, notes


# ----------------------------------------------------------------------------------------------- weights, score
def tool_step_counts(plan, tool_by_id):
    """{tool kind: number of steps}: the per-tool share of a plan."""
    c = {}
    for s in plan or []:
        for t in s["tool"]:
            k = tool_kind(tool_by_id[int(t)])
            c[k] = c.get(k, 0) + 1
    return c


def make_decider(base_cls):
    class DermArenaProDecider(base_cls):
        """Upstream's parsing and weight lookup, used on the per-disease weights reply; its per-patient decide() is not called. Calling-side
        fixes only: (1) a weight whose name was paraphrased is put back on its indicator before upstream's lookup, which would
        otherwise give it 0; (2) reasoning before </think> is cut."""
        _tl = threading.local()

        def safe_json_parse(self, text):
            return super().safe_json_parse((text or "").split("</think>")[-1])

        def weights_from_model(self, obj, indicators):
            names = [str(it["indicator_name"]).strip() for it in indicators]
            ws = obj.get("weights", [])
            if isinstance(ws, dict):
                ws = [{"indicator_name": k, "weight": v} for k, v in ws.items()]
            ws = [w for w in ws if isinstance(w, dict)]
            taken = {str(w.get("indicator_name", "")).strip() for w in ws} & set(names)
            info = {"remapped": [], "unmatched": [], "threshold_missing": "threshold" not in obj}
            norm = {normalise_name(n): n for n in names}
            fixed = []
            for pos, w in enumerate(ws):
                name = str(w.get("indicator_name", "")).strip()
                if name not in names:
                    free = [n for n in names if n not in taken]
                    hit = norm.get(normalise_name(name)) if norm.get(normalise_name(name)) in free else None
                    if hit is None:
                        m = difflib.get_close_matches(normalise_name(name), [normalise_name(n) for n in free], 1, 0.6)
                        hit = norm[m[0]] if m else None
                    if hit is None and len(ws) == len(names) and names[pos] in free:
                        hit = names[pos]          # same count, same order as listed
                    if hit is None:
                        info["unmatched"].append(name)
                        continue
                    info["remapped"].append({"from": name, "to": hit})
                    taken.add(hit)
                    name = hit
                fixed.append({"indicator_name": name, "weight": w.get("weight")})
            info["no_weight"] = [n for n in names if n not in {w["indicator_name"] for w in fixed}]
            self._tl.info = info
            return super().weights_from_model({"weights": fixed}, indicators)

        def last_info(self):
            return getattr(self._tl, "info", {})

    return DermArenaProDecider


def action_to_key(action):
    """= MIMIC generate_weights.py."""
    s = re.sub(r"[^0-9a-zA-Z]+", " ", (action or "").strip()).strip().lower()
    return "_".join(s.split()) or "unknown"


def plan_indicators(plan):
    """[(step id, key, action)] of the final indicators. A repeated action gets the step id in its key."""
    rows = [s for s in plan if s["output_type"] == "final indicator"]
    keys = [action_to_key(s["action"]) for s in rows]
    return [(int(s["id"]), k if keys.count(k) == 1 else f"{k}_step_{s['id']}", s["action"]) for s, k in zip(rows, keys)]


def weights_prompt(disease, guideline, items):
    desc = "\n".join(f'{i + 1}. key="{k}"\n   action: {a}' for i, (_, k, a) in enumerate(items))
    block = fill(WEIGHTS_GUIDELINE, guideline=guideline) if guideline else ""
    return fill(WEIGHTS_PROMPT, keys=json.dumps([k for _, k, _ in items]), indicators=desc, guideline_block=block, disease=disease)


def parse_weights(decider, raw, finish, items):
    """Weights reply -> {'by_step', 'threshold', 'info'}. Raises ValueError when any weight or the threshold is unusable: no default is
    invented for a value that is cached for every patient."""
    if finish == "length":
        raise ValueError("reply hit max_tokens")
    obj = decider.safe_json_parse(raw)
    ents = obj.get("indicators") if isinstance(obj, dict) else None
    if not isinstance(ents, dict) or not ents:
        raise ValueError("no 'indicators' object")
    if isinstance(obj.get("threshold"), bool) or not isinstance(obj.get("threshold"), (int, float)):
        raise ValueError("no numeric 'threshold'")
    ws = [{"indicator_name": k, "weight": v.get("weight") if isinstance(v, dict) else v} for k, v in ents.items()]
    keyed = [{"indicator_name": k} for _, k, _ in items]
    w_map = decider.weights_from_model({"weights": ws}, keyed)
    info = dict(decider.last_info())
    if info.get("no_weight") or info.get("unmatched"):
        raise ValueError(f"indicators without a weight: {info.get('no_weight')}; unknown keys: {info.get('unmatched')}")
    if not all(isinstance(w["weight"], (int, float)) and not isinstance(w["weight"], bool) and w["weight"] >= 0 for w in ws) or \
            sum(w["weight"] for w in ws) <= 0:
        raise ValueError("weights must be non-negative numbers with a positive sum")
    th = float(obj["threshold"])
    info["threshold_clamped"] = not 0.0 <= th <= 1.0
    return {"by_step": {str(sid): w_map[k] for sid, k, _ in items}, "threshold": max(0.0, min(1.0, th)), "info": info}


def risk_score(decider, indicators, weights):
    """Paper Eq. 4 with the disease's fixed weights, rescaled over the indicators that survived for this patient. Yes 1, No 0 (upstream's
    norm_yesno); Uncertain never reaches this point. Positive if score >= threshold, as in Pro_Decider."""
    live = [(x, float(weights["by_step"].get(str(x["step"]), 0.0))) for x in indicators]
    total = sum(w for _, w in live)
    if total <= 0:
        return None
    score = sum(w / total * decider.norm_yesno(x["if_abnormal"]) for x, w in live)
    th = weights["threshold"]
    return {"score": score, "threshold": th, "weights": {x["indicator_name"]: w / total for x, w in live},
            "diagnosis": "Positive" if score >= th else "Negative"}


# ----------------------------------------------------------------------------------------------- level 1: plan cache
class PlanCache:
    """One JSON file per normalised disease name. First writer wins (hard link of a complete temp file), so a file is
    never torn and every reader sees the same plan; a losing duplicate is discarded."""

    def __init__(self, root, namespace):
        self.dir = Path(root) / namespace
        self.tmp = self.dir / "_tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._locks, self._guard = {}, threading.Lock()

    def path(self, key):
        slug = re.sub(r"[^a-z0-9]+", "_", key)[:60].strip("_")
        return self.dir / f"{slug}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"

    def read(self, key):
        p = self.path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def lock(self, key):
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def write(self, key, entry):
        """-> (entry now in the cache, True if ours)."""
        p = self.path(key)
        t = self.tmp / f"{p.stem}.{os.getpid()}.{threading.get_ident()}.json"
        with t.open("w") as f:
            json.dump(entry, f, ensure_ascii=False, indent=1, default=str)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(t, p)
            return entry, True
        except FileExistsError:
            return json.loads(p.read_text()), False
        finally:
            t.unlink(missing_ok=True)

    def note_failure(self, key, entry):
        """Audit copy of an entry that must not be final. _failed/ is never read as a plan; the disease is tried again on its next case."""
        d = self.dir / "_failed"
        d.mkdir(exist_ok=True)
        p = d / f"{self.path(key).stem}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.json"
        t = p.with_suffix(".tmp")
        t.write_text(json.dumps(entry, ensure_ascii=False, indent=1, default=str))
        os.replace(t, p)


def clean_names(M, v, limit):
    """A parsed 'diagnoses' list -> names as written: non-empty strings, no template placeholders, duplicates after normalisation dropped."""
    raw = [x.strip() for x in v if isinstance(x, str) and x.strip() and not M["dx_parse"].is_placeholder(x)] if isinstance(v, list) else []
    out, seen = [], set()
    for x in raw:
        k = normalise_name(x)
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return raw, out[:limit]


def make_indicators(ctx, disease, guideline):
    """Indicators I from G (paper p.4), or from the disease name alone when there is no G. One call per disease."""
    M = mods()
    prompt = fill(INDICATOR_PROMPT if guideline else INDICATOR_PROMPT_NO_G, guideline=guideline, disease=disease)
    out = {"prompt": prompt, "indicators": [], "status": "unparsed", "attempts": []}
    for _ in range(int(os.environ.get("MP_INDICATOR_ATTEMPTS", "2"))):
        try:
            comp = guard(M["mp_llm"].mp_create, ctx["client"], [{"role": "user", "content": prompt}], module="indicators")
        except InfraError:
            raise
        except Exception as e:  # noqa: BLE001  (400 etc.): says nothing about the disease
            out["attempts"].append({"error": f"{type(e).__name__}: {e}"})
            out["status"] = "request_error"
            return out
        ch = comp.choices[0]
        v, why = M["dx_parse"]._last_field(ch.message.content or "", ch.finish_reason, "indicators")
        _, items = clean_names(M, v, 100)
        out["attempts"].append({"reply": ch.message.content, "finish_reason": ch.finish_reason, "parse": why or "ok", "n_items": len(items)})
        if items:
            out.update(indicators=items, status="ok")
            break
    return out


def make_weights(ctx, disease, guideline, plan):
    """Weights and threshold, once per disease, from G and the plan's indicators, no patient data (MIMIC generate_weights.py)."""
    M = mods()
    items = plan_indicators(plan)
    prompt = weights_prompt(disease, guideline, items)
    out = {"system": WEIGHTS_SYSTEM, "prompt": prompt, "keys": {str(sid): k for sid, k, _ in items}, "status": "unparsed", "attempts": [],
           "by_step": None, "threshold": None}
    msgs = [{"role": "system", "content": WEIGHTS_SYSTEM}, {"role": "user", "content": prompt}]
    for _ in range(int(os.environ.get("MP_WEIGHTS_ATTEMPTS", "2"))):
        try:
            comp = guard(M["mp_llm"].mp_create, ctx["client"], msgs, module="weights")
        except InfraError:
            raise
        except Exception as e:  # noqa: BLE001
            out["attempts"].append({"error": f"{type(e).__name__}: {e}"})
            out["status"] = "request_error"
            return out
        ch = comp.choices[0]
        try:
            w = parse_weights(ctx["decider"], ch.message.content, ch.finish_reason, items)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            out["attempts"].append({"reply": ch.message.content, "finish_reason": ch.finish_reason, "error": f"{type(e).__name__}: {e}"})
            continue
        out["attempts"].append({"reply": ch.message.content, "finish_reason": ch.finish_reason})
        out.update(w, status="ok")
        break
    return out


def planner_rag_text(guideline, indicators):
    """What the Planner reads as its RAG text: G, then I. With no G, I alone."""
    if not indicators:
        return guideline
    block = fill(PLANNER_INDICATORS, indicators="\n".join(f"- {x}" for x in indicators))
    return f"{guideline}\n\n{block}" if guideline else block


def make_entry(ctx, disease, key, calls):
    """Stage 1 for one disease. Validator / post-check rejections retry with the reason; infrastructure errors propagate. An entry is
    stored only when guideline, indicators, plan and weights all came out of a clean run."""
    M = mods()
    start = len(calls)
    token = plan_token(disease)
    ret = guard(ctx["kb"].retrieve, disease)
    guideline, g_status, transient = "", "no_document", False
    if ret["chunks"]:
        try:
            comp = guard(M["mp_llm"].mp_create, ctx["client"], M["kb_rag"].guideline_messages(disease, ret["chunks"]),
                         module="guideline")
            text = (comp.choices[0].message.content or "").split("</think>")[-1].strip()
            if text:
                guideline, g_status = ("", "no_relevant_data") if M["kb_rag"].is_no_data(text) else (text, "ok")
            else:                       # a blank reply is a fault, not 'no relevant data': never stored
                g_status, transient = f"blank_reply: finish {comp.choices[0].finish_reason}", True
        except InfraError:
            raise
        except Exception as e:  # noqa: BLE001
            g_status, transient = f"request_error: {type(e).__name__}: {e}", True
    ind = make_indicators(ctx, disease, guideline)
    transient = transient or ind["status"] == "request_error"
    goal = fill(ctx["task_cfg"]["disease"], disease=token)
    prompt = planner_prompt(str(ctx["task_cfg"]["input"]).strip(), goal)
    rag_text = planner_rag_text(guideline, ind["indicators"])
    attempts, plan, notes, reason = [], None, [], None
    work = ctx["cache"].tmp / f"work.{os.getpid()}.{threading.get_ident()}"
    n_try = int(os.environ.get("MP_PLAN_ATTEMPTS", "5"))
    n_relaxed = int(os.environ.get("MP_PLAN_RELAXED_ATTEMPTS", "2"))
    for i in range(n_try + n_relaxed):
        relaxed = i >= n_try        # main's word check off; a step that is the diagnosis itself is removed, not rejected
        ask = prompt + fill(PLAN_RETRY_NOTE, reason=reason) if reason else prompt
        rec = {"attempt": i + 1, "relaxed": relaxed, "retry_reason": reason}
        try:
            raw = guard(ctx["planner"].plan, str(work), ask, rag_text, filename="plan.json",
                        model=ctx["backbone"], toolset=ctx["toolset"], check_bundling=not relaxed)
            plan, notes = plan_postcheck(raw, ctx["tool_by_id"], ctx["concepts"], token, relaxed=relaxed)
            attempts.append(dict(rec, ok=True))
            break
        except InfraError:
            raise
        except PlanRejected as e:
            attempts.append(dict(rec, ok=False, stage="postcheck", error=f"{type(e).__name__}: {e}"))
            reason = str(e)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as e:   # the validator and JSON parsing raise these
            attempts.append(dict(rec, ok=False, stage="validator", error=f"{type(e).__name__}: {e}"))
            reason = "the reply was not valid JSON" if isinstance(e, json.JSONDecodeError) else str(e)
        except Exception as e:  # noqa: BLE001  (400, disk ...): says nothing about the disease
            attempts.append(dict(rec, ok=False, stage="request", error=f"{type(e).__name__}: {e}"))
            transient = True
            break
    shutil.rmtree(work, ignore_errors=True)
    weights = make_weights(ctx, disease, guideline, plan) if plan else None
    weights_ok = bool(weights) and weights["status"] == "ok"
    return {"key": key, "disease": disease, "plan_token": token, "schema": PLAN_SCHEMA,
            "cacheable": bool(plan) and weights_ok and ind["status"] == "ok" and not transient,
            "status": "ok" if plan else "transient" if transient else "failed",
            "tool_steps": tool_step_counts(plan, ctx["tool_by_id"]), "has_guideline": bool(guideline), "guideline_status": g_status,
            "guideline": guideline, "retrieval": ret, "indicators": ind["indicators"], "indicator_status": ind["status"],
            "indicator_step": ind, "planner_prompt": prompt, "planner_rag_text": rag_text, "plan": plan, "plan_notes": notes,
            "n_final_indicators": len(plan_indicators(plan)) if plan else 0, "relaxed": bool(plan) and attempts[-1]["relaxed"],
            "attempts": attempts, "plan_sha": sha_text(json.dumps(plan, sort_keys=True)) if plan else None,
            "weights": weights, "weights_status": weights["status"] if weights else None, "backbone": ctx["backbone"],
            "kb_sha": ctx["kb_sha"], "created": v1.now(), "llm_calls": list(calls[start:])}


def get_plan(ctx, disease, calls):
    """-> (entry, 'cached' | 'new' | 'uncached'). A thread waits for its neighbour; another process may duplicate, and the first
    wins. A failed plan, or an entry hit by any fault, is used for this case only: never stored as final, tried again on the next case."""
    key = normalise_name(disease)
    cache = ctx["cache"]
    e = cache.read(key)
    if e is not None:
        return e, "cached"
    with cache.lock(key):
        e = cache.read(key)
        if e is not None:
            return e, "cached"
        e = make_entry(ctx, disease, key, calls)
        if not e["cacheable"]:
            print(f"[mp] PLAN NOT CACHED {disease!r}: {e['status']} | guideline {e['guideline_status']} | indicators "
                  f"{e['indicator_status']} | weights {e['weights_status']} | {e['attempts'][-1:]}", flush=True)
            cache.note_failure(key, e)
            return e, "uncached"
        e, ours = cache.write(key, e)
        return e, "new" if ours else "cached"


# ----------------------------------------------------------------------------------------------- tools
def smoke_image():
    """First clinical photo in the T1.1 test file: a real image for the tool load gate."""
    for l in (RELEASE / "tasks" / TASK_DIR["T1.1"] / "test.jsonl").open():
        r = json.loads(l)
        for im in r.get("images") or []:
            if im.get("modality") in DERM_MODALITIES and (RELEASE / im["image_path"]).exists():
                return v1.rgb_path(RELEASE / im["image_path"])
    raise SystemExit("[mp] no smoke image found")


def load_tools():
    """Preload once + hard gate (a 'loaded successfully' message can lie, PLAYBOOK V14): real calls must work, and the
    full-softmax PanDerm bridge must agree with what PanDermTool._run prints."""
    st = mods()["skin_tools"]
    objs = {"panderm": st._get_panderm(), "make": st._get_make(), "dermogpt": st._get_dermogpt()}
    for name, t in objs.items():
        if hasattr(t, "preload"):
            t.preload()
        if getattr(t, "_load_failed", False) or (hasattr(t, "is_initialized") and not t.is_initialized):
            raise SystemExit(f"[mp] TOOL FAILED TO LOAD: {name}")
    img = smoke_image()
    v1.lock_tool_classes(list(objs.values()))  # one lock per tool class; LLM calls overlap
    names = ["psoriasis", "melanoma", "eczema"]
    printed = objs["panderm"]._run(image_path=img, candidate_diseases=", ".join(names))
    scores = st.panderm_scores(img, names)
    rows = {m.group(1): float(m.group(2)) / 100 for m in re.finditer(r"^\d+\. (.+?): ([0-9.]+)%", printed, re.M)}
    if set(rows) != set(names) or any(abs(rows[k] - scores[k]) > 1e-3 for k in names):
        raise SystemExit(f"[mp] PanDerm full scores disagree with the tool's print: {scores} vs {printed!r}")
    mk = st.make_concepts(img, ["scale", "erythema"])
    dg = st.dermogpt_vqa(img, "Describe this skin lesion.")
    if len(dg) < 20:
        raise SystemExit(f"[mp] DermoGPT smoke call too short: {dg!r}")
    print(f"[mp] tool gate OK: panderm {scores} | make {mk} | dermogpt {dg[:100]!r}", flush=True)
    return objs


def render_tool(v):
    """A tool step's JSON as context text: every image, every score. Not cut (a cut would drop candidates or images)."""
    if v["tool"] == "multi":
        return "\n".join(render_tool(x) for x in v["parts"])
    head = {"panderm": "PanDerm zero-shot probabilities over the patient's candidate diseases",
            "make": "MAKE concept presence scores (above 0.5 = detected)"}[v["tool"]]
    lines = [f"{head}:"]
    target = v.get("target_candidate") if v["tool"] == "panderm" else None
    for im in v["images"]:
        s = sorted(im["scores"].items(), key=lambda x: -x[1])
        lines.append(f"{im['image']}: " + "; ".join(f"{k} {p:.4f}" for k, p in s))
        # toolset.json (owner 11-08, #2): PanDerm is described as a measurement of the goal disease -> state that value and its rank
        names = [k for k, _ in s]
        if target is not None and target in names:       # run_panderm maps the scores back to the candidate names
            r = names.index(target)
            lines.append(f"  {target}: probability {s[r][1]:.4f}, rank {r + 1} of {len(s)}")
    return "\n".join(lines)


def render_value(v):
    return render_tool(v) if isinstance(v, dict) else str(v).split("</think>")[-1].strip()


def plan_text(plan, tool_by_id):
    """The plan as the memory step reads it: what each step does and what it needs."""
    lines = []
    for s in plan:
        need = [str(tool_by_id[int(t)].get("input", "")).strip() for t in s["tool"]]
        need += [f"output of step {d}" for d in s["input_type"] if int(d) != 0]
        lines.append(f"Step {s['id']}: {s['action']}\n  Needs: {'; '.join(x for x in need if x)}")
    return "\n".join(lines)


def parse_memory(content, finish, ids):
    """Memory reply -> (set of supported step ids, None) or (None, why). Any id outside the plan makes the reply unclear."""
    v, why = mods()["dx_parse"]._last_field(content or "", finish, "supported_steps")
    if why:
        return None, why
    if not isinstance(v, list):
        return None, "not_a_list_of_ids"
    # the plan lists "Step 3: ...", so the examiner answers 3, "3" or "Step 3": all name the same step (canary 11-07: 44 clear replies
    # such as ["Step 1","Step 5"] were treated as unclear and kept every step). Anything else stays unclear.
    got = []
    for x in v:
        if isinstance(x, bool):
            return None, "not_a_list_of_ids"
        if isinstance(x, str):
            m = re.fullmatch(r"\s*(?:step[\s_-]*)?(\d+)\s*", x, re.I)
            x = int(m.group(1)) if m else x
        if not isinstance(x, int):
            return None, "not_a_list_of_ids"
        got.append(x)
    v = got
    if set(v) - ids:
        return None, f"unknown_steps_{sorted(set(v) - ids)}"
    return set(v), None


def parse_reflection(reply):
    """Examiner reply -> 'reliable' | 'unreliable' | None (unclear). 'Uncertain' is not one of its two answers."""
    ans, how = normalise_answer(reply)
    if how in UNCLEAR or ans == "Uncertain":
        return None
    return "reliable" if ans == "Yes" else "unreliable"


def evidence_items(srecs, indicators):
    """What the final call may see for one candidate: the surviving indicators and the steps they rest on. A step that was removed,
    terminated, capped or dropped is not evidence (paper: 'only clinically valid medical data')."""
    by = {s["id"]: s for s in srecs}
    verdict = {x["step"]: x["if_abnormal"] for x in indicators}
    need = set(verdict)
    todo = list(verdict)
    while todo:
        for d in by[todo.pop()]["input_type"]:
            if d in by and d not in need and by[d].get("output") is not None and by[d]["status"] == "ok":
                need.add(d)
                todo.append(d)
    return [{"step": i, "action": by[i]["action"], "abnormal": verdict.get(i), "text": render_value(by[i]["output"])} for i in sorted(need)]


def evidence_block(results):
    """Per candidate, in the backbone's order. Not cut: a cut would drop evidence the paper's step is there to integrate."""
    out = []
    for r in results:
        lines = [f"Candidate: {r['name']}"]
        if r.get("score") is not None:      # paper Eq. 4 and theta: the candidate's risk-based decision, an output of its analysis
            lines.append(f"Risk score: {r['score']:.2f} (threshold {r['threshold']:.2f}): {r['verdict']}")
        if not r["evidence"]:
            lines.append(NO_EVIDENCE)
        for e in r["evidence"]:
            if e["abnormal"] is None:
                lines.append(f"- Tool result, {e['action']}:\n{e['text']}")
            else:
                lines.append(f"- Indicator: {e['action']}\n  Abnormal: {e['abnormal']}\n  Evidence: {e['text']}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def ask_final(client, prompt):
    """The integrating call, text only. Parser and retry rules of the top-10 step: last JSON after </think>, asked up to
    MP_FINAL_ATTEMPTS times; no usable list = unparsed = wrong. Also what --rank-stability re-runs."""
    M = mods()
    info = {"prompt": prompt, "attempts": [], "finish_reason": None, "diagnoses": [], "answer_source": None}
    for _ in range(int(os.environ.get("MP_FINAL_ATTEMPTS", "2"))):
        try:
            comp = guard(M["mp_llm"].mp_create, client, [{"role": "user", "content": prompt}], module="final")
        except InfraError:
            raise
        except Exception as e:  # noqa: BLE001  (400, e.g. context length): no answer for this case
            info["attempts"].append({"error": f"{type(e).__name__}: {e}"})
            info["finish_reason"] = "agent_request_error"
            break
        ch = comp.choices[0]
        info["finish_reason"] = ch.finish_reason
        v, why = M["dx_parse"]._last_field(ch.message.content or "", ch.finish_reason, "diagnoses")
        raw, items = clean_names(M, v, 5)
        info["attempts"].append({"reply": ch.message.content, "finish_reason": ch.finish_reason, "parse": why or "ok", "raw_list": raw,
                                 "n_items": len(items)})
        if items:
            info["diagnoses"] = items
            trace = M["mp_llm"]._TRACE.get()
            info["answer_source"] = trace[-1].get("answer_source") if trace else None
            break
    info["status"] = "ok" if info["diagnoses"] else "unparsed"
    return info


# ----------------------------------------------------------------------------------------------- one case
class Case:
    def __init__(self, ctx, row, task, out_dir):
        self.ctx, self.row, self.task = ctx, row, task
        self.cid = row["_id"]
        self.save_dir = out_dir / "cases" / self.cid
        self.trace_path = out_dir / "traces" / f"{self.cid}.json"
        self.trace = {}
        self.memo = {}
        self.tool_errors = []
        self.n_tool_calls = 0
        self.tools_used = set()
        self.n_context_cut = 0
        self.encoded = {}

    def flush(self):
        tmp = self.trace_path.with_suffix(f".tmp{threading.get_ident()}")
        tmp.write_text(json.dumps(self.trace, ensure_ascii=False, indent=1, default=str))
        os.replace(tmp, self.trace_path)

    # -- 1. candidates
    def candidates(self):
        M, c = mods(), self.ctx
        parts, _ = M["dx_infer"].image_parts(self.row)
        text = fill(CAND_PROMPT, case=self.case_text)
        msgs = [{"role": "user", "content": parts + [{"type": "text", "text": text}] if parts else text}]
        info = {"attempts": [], "finish_reason": None}
        cands = []
        for _ in range(int(os.environ.get("MP_CAND_ATTEMPTS", "2"))):
            try:
                comp = guard(M["mp_llm"].mp_create, c["client"], msgs, module="candidates")
            except InfraError:
                raise
            except Exception as e:  # noqa: BLE001  (400 etc.): no list for this case
                info["attempts"].append({"error": f"{type(e).__name__}: {e}"})
                info["finish_reason"] = "agent_request_error"
                break
            ch = comp.choices[0]
            info["finish_reason"] = ch.finish_reason
            v, why = M["dx_parse"]._last_field(ch.message.content or "", ch.finish_reason, "diagnoses")
            raw, cands = clean_names(M, v, 10)      # original names kept; duplicates after normalisation dropped
            info["attempts"].append({"finish_reason": ch.finish_reason, "parse": why or "ok", "raw_list": raw,
                                     "n_items": len(cands)})
            if cands:
                break
        info["diagnoses"] = cands
        info["status"] = "ok" if cands else "unparsed"
        return info

    # -- image parts of the memory and reflection calls: same encoding as the step calls, in case order, no modality label
    def image_parts(self, images):
        enc = self.ctx["analyzer"].encode_image
        for im in images:
            if im["path"] not in self.encoded:
                self.encoded[im["path"]] = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{enc(im['path'])}"}}
        return [self.encoded[im["path"]] for im in images]

    @staticmethod
    def image_note(images):
        if not images:
            return "This case has no images."
        return "The images above are, in order: " + ", ".join(im["label"] for im in images) + "."

    def ask(self, messages, module):
        """One backbone call -> (content, finish_reason, answer_source). Infrastructure errors become InfraError."""
        comp = guard(mods()["mp_llm"].mp_create, self.ctx["client"], messages, module=module)
        ch = comp.choices[0]
        return ch.message.content or "", ch.finish_reason, self.calls[-1].get("answer_source")

    # -- memory initialisation (Eq. 2): this call is the data match; the code only executes its answer
    def memory_step(self, entry):
        c = self.ctx
        plan = entry["plan"]
        ids = {int(s["id"]) for s in plan}
        prompt = fill(MEMORY_PROMPT, plan=plan_text(plan, c["tool_by_id"]), images=self.image_note(self.images), case=self.case_text)
        msgs = [{"role": "user", "content": self.image_parts(self.images) + [{"type": "text", "text": prompt}]}]
        rec = {"prompt": prompt, "replies": [], "supported": None, "how": None}
        for attempt in (1, 2):
            try:
                content, fin, src = self.ask(msgs, "memory")
            except InfraError:
                raise
            except Exception as e:  # noqa: BLE001  (400 etc.): counted like an unclear reply
                rec["replies"].append({"error": f"{type(e).__name__}: {e}"})
                continue
            got, why = parse_memory(content, fin, ids)
            rec["replies"].append({"reply": content, "finish_reason": fin, "answer_source": src, "parse": why or "ok"})
            if got is not None:
                rec.update(supported=sorted(got), how="parsed" if attempt == 1 else "parsed_reasked")
                return rec
        rec.update(supported=sorted(ids), how="default_keep_all")      # stated default: nothing is removed on an unclear reply
        return rec

    # -- reflection (Eq. 3, E.4): input quality and plausibility of an intermediate result
    def reflect(self, step, kinds, value, entry):
        backbone = any(k == "backbone" for _, k in kinds)
        images = self.images if backbone else self.derm_images       # the step's own input images
        g = entry.get("guideline") or ""
        prompt = fill(REFLECT_PROMPT, action=step["action"], images=self.image_note(images),
                      guideline=f"Clinical guideline:\n{g}\n\n" if g else "", result=render_value(value),
                      case=f"{fill(QUAL_CASE_CONTEXT, case=self.case_text)}\n\n" if backbone else "")
        msgs = [{"role": "system", "content": REFLECT_SYSTEM},
                {"role": "user", "content": self.image_parts(images) + [{"type": "text", "text": prompt}]}]
        rec = {"step": int(step["id"]), "action": step["action"], "images": [im["label"] for im in images], "prompt": prompt,
               "replies": [], "verdict": None, "how": None}
        for attempt in (1, 2):
            try:
                content, fin, src = self.ask(msgs, "reflection")
            except InfraError:
                raise
            except Exception as e:  # noqa: BLE001  (400 etc.): counted like an unclear reply
                rec["replies"].append({"error": f"{type(e).__name__}: {e}"})
                continue
            v = parse_reflection(content)
            rec["replies"].append({"reply": content, "finish_reason": fin, "answer_source": src, "verdict": v})
            if v:
                rec.update(verdict=v, how="exact" if attempt == 1 else "reasked")
                return rec
        rec.update(verdict="reliable", how="default_reliable")         # stated default: an unclear reply does not terminate a branch
        return rec

    # -- tools, once per case and image: the same image and question must not run again for another candidate
    def tool_once(self, key, fn, tool, cand, step_id, image):
        if key in self.memo:
            return self.memo[key]
        t0 = time.time()
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001  tool trouble: recorded; that evidence is absent, the case goes on
            err = f"{type(e).__name__}: {str(e)[:500]}"
            cuda = bool(re.search(r"cuda|out of memory|cublas|cudnn", err, re.I))
            self.tool_errors.append({"candidate": cand, "step": step_id, "tool": tool, "image": image, "cuda": cuda, "error": err})
            print(f"[mp] TOOL ERROR {self.cid} {tool} {image}{' CUDA' if cuda else ''}: {err[:200]}", flush=True)
            return None
        self.n_tool_calls += 1
        self.tools_used.add(tool)
        self.memo[key] = out
        self.tool_secs[tool] = round(self.tool_secs.get(tool, 0) + time.time() - t0, 2)
        return out

    def run_panderm(self, cand, step_id):
        st = mods()["skin_tools"]
        names = [panderm_name(x) for x in self.cands]
        back = dict(zip(names, self.cands))
        images = []
        for im in self.derm_images:
            s = self.tool_once(("panderm", im["path"]), lambda p=im["path"]: st.panderm_scores(p, names), "panderm", cand,
                               step_id, im["label"])
            if s is not None:
                images.append({"image": im["label"], "scores": {back[k]: v for k, v in s.items()}})
        return {"tool": "panderm", "target_candidate": cand, "images": images} if images else None

    def run_make(self, cand, step_id, concepts):
        st = mods()["skin_tools"]
        images = []
        for im in self.derm_images:
            s = self.tool_once(("make", im["path"], tuple(concepts)), lambda p=im["path"]: st.make_concepts(p, concepts),
                               "make", cand, step_id, im["label"])
            if s is not None:
                images.append({"image": im["label"], "scores": s})
        return {"tool": "make", "concepts": concepts, "images": images} if images else None

    def run_dermogpt(self, cand, step_id, query):
        st = mods()["skin_tools"]
        out = []
        for im in self.derm_images:
            s = self.tool_once(("dermogpt", im["path"], query), lambda p=im["path"]: st.dermogpt_vqa(p, query), "dermogpt",
                               cand, step_id, im["label"])
            if s is not None:
                out.append(f"{im['label']}: {s}")
        return "\n\n".join(out) or None

    def run_backbone(self, cc, step, ques, field):
        """All images + case text on every backbone step: a step fed only by a prior step would otherwise see no patient."""
        c = self.ctx
        U = mods()["U"]
        text_input = [fill(QUAL_CASE_CONTEXT, case=self.case_text)] + cc["context"](step)
        prompt = U.build_qual_prompt(ques, text_input)
        out_file = str(cc["dir"] / "diagnosis.json")
        data = guard(c["analyzer"].decide, output_file=out_file, prompt=prompt,
                     image_paths=[im["path"] for im in self.images], field=field)
        return data.get(field), prompt

    # -- level 2 for one candidate
    def run_candidate(self, idx, cand, entry):
        M, c = mods(), self.ctx
        st, U = M["skin_tools"], M["U"]
        cdir = self.save_dir / f"cand_{idx:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        plan = entry["plan"]
        status, srecs, indicators, dropped = {}, [], [], []
        token = entry["plan_token"]
        memory = self.memory_step(entry)
        supported = set(memory["supported"])
        cap = int(os.environ.get("MP_STEP_CAP", "10"))
        blocked = {}            # step id -> 'memory' | 'terminated' | 'capped': it did not run or was dropped, and its dependants follow
        removed, terminated, capped, uncertain, reflections = [], [], [], [], []
        executed = 0

        def read(dep):
            p = cdir / "diagnosis.json"
            return (json.loads(p.read_text()) if p.exists() else {}).get(f"step_{dep}")

        def context(step):
            out = []
            for dep in step.get("input_type") or []:
                if int(dep) == 0:
                    continue
                v = read(int(dep))
                if isinstance(v, dict):
                    out.append(render_tool(v))
                elif v:
                    if len(v.strip()) > 2000:      # upstream's cut, kept for free text only
                        self.n_context_cut += 1
                    out.append(U.json_to_text(v))
                else:
                    out.append(f"[step {dep} produced no output]")
            return out

        cc = {"dir": cdir, "context": context}
        names = {}
        for step in plan:
            if step["output_type"] == "final indicator":
                names[step["action"]] = names.get(step["action"], 0) + 1
        for step in plan:
            sid = int(step["id"])
            field = f"step_{sid}"
            kinds = [(int(t), tool_kind(c["tool_by_id"][int(t)])) for t in step["tool"]]
            deps = [int(d) for d in step.get("input_type") or []]
            srec = {"id": sid, "action_type": step["action_type"], "action": step["action"], "tool": step["tool"],
                    "kinds": [k for _, k in kinds], "input_type": deps, "output_type": step["output_type"],
                    "status": "ok", "t_start": v1.now()}
            row = {"step": sid, "action": step["action"], "output_type": step["output_type"]}
            on = next((d for d in deps if d in blocked), None)
            if sid not in supported:
                blocked[sid] = "memory"
                srec["status"] = "memory_removed"
                removed.append(dict(row, reason="not_supported_by_patient_data"))
            elif on is not None:
                blocked[sid] = blocked[on]
                srec["status"] = {"memory": "memory_removed", "terminated": "branch_terminated", "capped": "capped"}[blocked[on]]
                {"memory": removed, "terminated": terminated, "capped": capped}[blocked[on]].append(dict(row, reason=f"depends_on_step_{on}"))
            elif executed >= cap:
                blocked[sid] = "capped"
                srec["status"] = "capped"
                capped.append(dict(row, reason=f"over_cap_{cap}"))
            if sid in blocked:
                srecs.append(srec)
                status[sid] = srec["status"]
                continue
            t0 = time.time()
            value = None
            if step["action_type"] == "quantitative":
                value = self.quantitative(cc, step, srec, kinds, cand, token, field)
            else:
                value = self.qualitative(cc, step, srec, kinds, deps, status, cand)
            executed += srec["status"] != SKIN_MISSING      # a step with no image to run on did not execute
            if value is not None:
                st._merge_json(str(cdir), "diagnosis.json", field, value)
                srec["output"] = value
            status[sid] = srec["status"]
            if step["output_type"] == "final indicator":
                name = step["action"] if names[step["action"]] == 1 else f"{step['action']} (step {sid})"
                if value is None:           # paper Eq. 3: only steps whose input was available go on
                    dropped.append({"step": sid, "action": step["action"], "reason": srec["status"]})
                else:
                    ans, why = self.summarise(cdir, step, field, srec)
                    srec.update(indicator=ans, indicator_how=why)
                    if ans == "Uncertain":  # Fig 4, E.4(4): unreliable, not considered, and the analysis built on it stops
                        srec["status"] = "dropped_uncertain"
                        blocked[sid] = "terminated"
                        uncertain.append(dict(row, indicator_name=name, how=why))
                    else:
                        indicators.append({"indicator_name": name, "if_abnormal": ans, "step": sid, "how": why,
                                           "tools": [k for _, k in kinds]})
            elif value is not None:
                rv = self.reflect(step, kinds, value, entry)
                reflections.append(rv)
                srec["reflection"] = rv
                if rv["verdict"] == "unreliable":       # s_i = Terminate: this branch stops, other indicators continue
                    srec["status"] = "terminated"
                    blocked[sid] = "terminated"
                    terminated.append(dict(row, reason="reflection_unreliable"))
            srec["secs"] = round(time.time() - t0, 1)
            srecs.append(srec)
        return {"steps": srecs, "indicators": indicators, "dropped": dropped, "dir": cdir, "memory": memory, "memory_removed": removed,
                "reflection": reflections, "terminated": terminated, "capped_steps": capped, "dropped_uncertain": uncertain,
                "n_executed": executed, "evidence": evidence_items(srecs, indicators)}

    def quantitative(self, cc, step, srec, kinds, cand, token, field):
        """Every tool of the step runs and every result is kept (a step may name PanDerm and MAKE together)."""
        sid = int(step["id"])
        outs = []
        if not self.derm_images:
            srec["status"] = SKIN_MISSING
            return None
        for _, kind in kinds:
            if kind == "panderm":
                v = self.run_panderm(cand, sid)
            else:
                concepts, how = self.ctx["concepts"].map(step["action"], token)   # plan_postcheck guarantees a concept
                srec["make_map"] = {"concepts": concepts, "how": how}
                v = self.run_make(cand, sid, concepts)
            if v:
                outs.append(v)
        if not outs:
            srec["status"] = "tool_error"
            return None
        return outs[0] if len(outs) == 1 else {"tool": "multi", "parts": outs}

    def qualitative(self, cc, step, srec, kinds, deps, status, cand):
        sid, ques = int(step["id"]), step["action"]
        nonraw = [d for d in deps if d != 0]
        # An image-tool judgement with no image to judge: the step was planned on tool output only, and the tool could not run.
        if 0 not in deps and nonraw and all(status.get(d) == SKIN_MISSING for d in nonraw):
            srec["status"] = SKIN_MISSING
            return None
        parts = []
        for tid, kind in kinds:
            if kind == "backbone":
                try:
                    v, srec["prompt"] = self.run_backbone(cc, step, ques, f"_raw_step_{sid}_tool_{tid}")
                except InfraError:
                    raise
                except Exception as e:  # noqa: BLE001  (400: context length etc.; upstream: print + continue)
                    srec.update(status="vlm_error", error=f"{type(e).__name__}: {e}")
                    continue
                if v and str(v).strip():
                    parts.append(("Backbone VLM", str(v)))
            elif not self.derm_images:
                if len(kinds) == 1:
                    srec["status"] = SKIN_MISSING
            else:
                q = mods()["U"].build_qual_prompt(ques, cc["context"](step))
                srec["dermogpt_query"] = q
                v = self.run_dermogpt(cand, sid, q)
                if v:
                    parts.append(("DermoGPT", v))
                elif len(kinds) == 1:
                    srec["status"] = "tool_error"
        if not parts:
            if srec["status"] == "ok":
                srec["status"] = "blank"
            return None
        return parts[0][1] if len(parts) == 1 else "\n\n".join(f"{a}:\n{b}" for a, b in parts)

    def summarise(self, cdir, step, field, srec):
        """Summary_Module with upstream's question. A blank or unclear verdict is asked once more, so that thinking copied in by
        the reasoning fallback does not pass as uncertainty; if it stays unclear it is Uncertain and counted, never hidden."""
        c = self.ctx
        replies = srec.setdefault("summary_replies", [])
        for attempt in (1, 2):
            try:
                reply = guard(c["summarizer"].summarize, input_file=str(cdir / "diagnosis.json"),
                              output_file=str(cdir / "brief_diagnosis.json"), prompt=fill(SUMMARY_PROMPT, ques=step["action"]),
                              field=field, answer_suffix=SUMMARY_SUFFIX)
            except InfraError:
                raise
            except Exception as e:  # noqa: BLE001
                srec["summary_error"] = f"{type(e).__name__}: {e}"
                return "Uncertain", "summary_error"
            ans, how = normalise_answer(reply)
            source = self.calls[-1].get("answer_source") if self.calls else None
            replies.append({"reply": reply, "how": how, "answer_source": source})
            if how not in UNCLEAR:
                return ans, how if attempt == 1 else f"{how}_reasked"
        return "Uncertain", "unclear_after_reask"

    # -- whole case
    def run(self):
        M, c = mods(), self.ctx
        t1 = time.time()
        if self.save_dir.exists():  # a retried (pass-2) case starts clean
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True)
        calls = self.calls = M["mp_llm"].start_case_trace()
        self.case_text = M["dx_infer"].case_text(self.row, self.task)
        self.images, missing = [], 0
        for im in self.row.get("images") or []:
            fp = RELEASE / im["image_path"]
            if not fp.exists():
                missing += 1
                continue
            k = len(self.images) + 1
            self.images.append({"path": v1.rgb_path(fp), "src": im["image_path"], "modality": im.get("modality"),
                                "tool_usable": im.get("modality") in DERM_MODALITIES,
                                "label": f"image {k}"})
        self.derm_images = [im for im in self.images if im["tool_usable"]]
        self.tool_secs = {}
        self.trace = {"_id": self.cid, "task": self.task, "run_sig": c["run_sig"], "case_text": self.case_text,
                      "images": self.images, "n_images_missing": missing, "candidates": None, "per_candidate": [],
                      "final": None, "ranking": None, "margin_ranking": None, "record": None, "llm_calls": calls}
        self.flush()
        results, n_new, n_cached, n_uncached, over = [], 0, 0, 0, False
        tool_steps = {}
        final = None
        budget = int(os.environ.get("MP_CALL_BUDGET", "0"))
        try:
            info = self.candidates()
            self.trace["candidates"] = info
            self.flush()
            self.cands = info["diagnoses"]
            for i, cand in enumerate(self.cands):
                t0 = time.time()
                entry, how = get_plan(c, cand, calls)
                n_new += how == "new"
                n_cached += how == "cached"
                n_uncached += how == "uncached"
                r = {"name": cand, "backbone_rank": i + 1, "plan_source": how, "plan_status": entry["status"],
                     "plan_sha": entry.get("plan_sha"), "has_guideline": entry["has_guideline"],
                     "n_planned_steps": len(entry["plan"] or []), "n_planned_indicators": entry.get("n_final_indicators", 0),
                     "score": None, "threshold": None, "margin": None, "weights": None, "verdict": None, "indicators": [],
                     "dropped_steps": [], "memory": None, "memory_removed": [], "reflection": [], "terminated": [],
                     "capped_steps": [], "dropped_uncertain": [], "n_executed": 0, "evidence": [], "no_evidence_reason": "no_plan"}
                steps = []
                if entry["status"] == "ok":
                    run = self.run_candidate(i, cand, entry)
                    steps, r["indicators"], r["dropped_steps"] = run["steps"], run["indicators"], run["dropped"]
                    r.update({k: run[k] for k in ("memory", "memory_removed", "reflection", "terminated", "capped_steps",
                                                  "dropped_uncertain", "n_executed", "evidence")})
                    r["no_evidence_reason"] = None if run["evidence"] else "no_indicator_left"
                    for k_, v_ in tool_step_counts(entry["plan"], c["tool_by_id"]).items():
                        tool_steps[k_] = tool_steps.get(k_, 0) + v_
                    w = entry.get("weights") or {}
                    res = risk_score(c["decider"], run["indicators"], w) if run["indicators"] and w.get("by_step") else None
                    if res is not None:
                        r.update(score=res["score"], threshold=res["threshold"], weights=res["weights"],
                                 margin=round(res["score"] - res["threshold"], 9), verdict=res["diagnosis"])
                r["secs"] = round(time.time() - t0, 1)
                results.append(r)
                self.trace["per_candidate"].append(dict(r, steps=steps))
                self.flush()
                if budget and len(calls) > budget and not over:
                    over = True     # logged, never acted on: all candidates still run
                    print(f"[mp] CALL BUDGET {self.cid}: {len(calls)} calls > {budget} after candidate {i + 1}/"
                          f"{len(self.cands)}; continuing", flush=True)
            if self.cands:
                final = ask_final(c["client"], fill(FINAL_PROMPT, evidence=evidence_block(results)))
                self.trace["final"] = final
                self.flush()
        except Exception as e:  # noqa: BLE001  infra error (or a crash): keep the partial trace, case retried in pass 2
            self.trace["dropped"] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}
            self.flush()
            os.replace(self.trace_path, self.trace_path.with_name(f"{self.cid}.dropped_{int(time.time())}.json"))
            raise
        margin_order = rank(results)      # comparison arm, no extra call: never the output
        for pos, r in enumerate(margin_order):
            r["margin_rank"] = pos + 1
        items = final["diagnoses"] if final else []
        by_name = {normalise_name(r["name"]): r for r in results}
        for pos, name in enumerate(items):
            if normalise_name(name) in by_name:
                by_name[normalise_name(name)]["final_rank"] = pos + 1
        top = by_name.get(normalise_name(items[0])) if items else None
        prediction = json.dumps({"diagnoses": items}, ensure_ascii=False) if items else ""
        fin = final["finish_reason"] if final else info.get("finish_reason")
        why = "final_call" if items else "final_unparsed" if final else "candidates_unparsed"
        every = [x for r in results for x in r["indicators"] + r["dropped_uncertain"]]
        refl = [x for r in results for x in r["reflection"]]
        mem = [r["memory"] for r in results if r["memory"]]
        rec = {
            "_id": self.cid, "task": self.task, "examinee": f"MedAgent-Pro/{c['backbone']}", "system": "medagent-pro",
            "prediction": prediction, "parsed": bool(items), "unparsed_reason": None if items else why,
            "n_items": len(items), "finish_reason": fin, "parse_rule": why if items else None,
            "answer_source": final["answer_source"] if items else None, "error": None,
            "agent_response": next((a.get("reply") for a in reversed(final["attempts"]) if a.get("reply")), prediction) if final else prediction,
            "final_called": bool(final), "final_parsed": bool(items), "final_attempts": len(final["attempts"]) if final else 0,
            "n_final_outside_candidates": sum(normalise_name(x) not in by_name for x in items),
            "final_top1_has_guideline": top["has_guideline"] if top else None,
            "margin_top5": [r["name"] for r in margin_order[:5]],
            "tools_used": sorted(self.tools_used), "n_tool_calls": self.n_tool_calls, "tool_secs": self.tool_secs,
            "tool_errors": self.tool_errors, "n_tool_errors": len(self.tool_errors), "tool_steps": tool_steps,
            "n_cuda_errors": sum(e["cuda"] for e in self.tool_errors), "n_llm_calls": len(calls),
            "call_budget": budget or None, "call_budget_exceeded": over, "n_candidates": len(self.cands),
            "n_plans_new": n_new, "n_plans_cached": n_cached, "n_plans_uncached": n_uncached,
            "n_plans_failed": sum(r["plan_status"] == "failed" for r in results),
            "n_no_plan": sum(r["plan_status"] != "ok" for r in results),
            "n_indicators": sum(len(r["indicators"]) for r in results),
            "n_uncertain": sum(len(r["dropped_uncertain"]) for r in results),
            "n_unclear_verdicts": sum(i["how"] in ("unclear_after_reask", "summary_error") for i in every),
            "n_verdicts_reasked": sum(i["how"].endswith("_reasked") or i["how"] == "unclear_after_reask" for i in every),
            "n_dropped_indicators": sum(len(r["dropped_steps"]) for r in results),
            "n_steps_planned": sum(r["n_planned_steps"] for r in results), "n_steps_executed": sum(r["n_executed"] for r in results),
            "n_memory_calls": len(mem), "n_memory_reasked": sum(m["how"] == "parsed_reasked" for m in mem),
            "n_memory_default": sum(m["how"] == "default_keep_all" for m in mem),
            "n_steps_memory_removed": sum(len(r["memory_removed"]) for r in results),
            "n_reflections": len(refl), "n_reflection_unreliable": sum(x["verdict"] == "unreliable" for x in refl),
            "n_reflection_reasked": sum(x["how"] == "reasked" for x in refl),
            "n_reflection_default": sum(x["how"] == "default_reliable" for x in refl),
            "n_steps_terminated": sum(len(r["terminated"]) for r in results),
            "n_steps_capped": sum(len(r["capped_steps"]) for r in results),
            "n_no_evidence": sum(not r["evidence"] for r in results),
            "n_no_guideline": sum(not r["has_guideline"] for r in results),
            "n_unscored": sum(r["score"] is None for r in results), "n_context_cut": self.n_context_cut,
            "candidates": self.cands, "candidates_status": info.get("status"),
            "candidate_results": [compact(r) for r in results],
            "completion_tokens": sum(((x.get("usage") or {}).get("completion_tokens") or 0) for x in calls),
            "max_tokens": c["max_tokens"], "call_attempts": 1,
            "backbone_reissues": {"immediate_eos": sum(int(x.get("eos_reissues") or 0) for x in calls)},
            "n_answer_from_reasoning": sum(1 for x in calls if x.get("answer_source") == "reasoning"),
            "n_images_sent": len(self.images), "n_images_case": len(self.images),
            "n_images_tool_usable": len(self.derm_images),
            "images_modality": [im["modality"] for im in self.images], "n_images_missing": missing,
            "corpus_status": self.row.get("corpus_status"), "image_status": self.row.get("image_status"),
            "wall_s": round(time.time() - t1, 1), "run_sig": c["run_sig"], "timestamp": v1.now(),
        }
        self.trace["ranking"] = [{"name": x, "final_rank": i + 1, "backbone_rank": (by_name.get(normalise_name(x)) or {}).get("backbone_rank")}
                                 for i, x in enumerate(items)]
        self.trace["margin_ranking"] = [{k: r.get(k) for k in ("name", "margin_rank", "final_rank", "backbone_rank", "score", "threshold", "margin")}
                                        for r in margin_order]
        self.trace["record"] = rec
        self.flush()
        return rec


def compact(r):
    """One candidate in the predictions row: the counts and verdicts, without the prompts and evidence text that the trace holds in full."""
    out = {k: v for k, v in r.items() if k not in ("memory", "reflection", "evidence")}
    m = r["memory"]
    out["memory"] = {"supported": m["supported"], "how": m["how"]} if m else None
    out["reflection"] = [{k: x[k] for k in ("step", "verdict", "how")} for x in r["reflection"]]
    out["no_evidence"] = not r["evidence"]
    return out


def rank(results):
    """Comparison arm: (score - threshold) descending; ties by the backbone's order; no score (no plan, no weights, no indicator) last."""
    return sorted(results, key=lambda r: (r["margin"] is None, -(r["margin"] or 0.0), r["backbone_rank"]))


# ----------------------------------------------------------------------------------------------- task loop
def run_task(ctx, task, out_dir, todo, k, threads):
    t0 = time.time()
    (out_dir / "traces").mkdir(exist_ok=True)
    (out_dir / "cases").mkdir(exist_ok=True)
    out_f = (out_dir / f"predictions.p{k}.jsonl").open("a")
    drop_f = (out_dir / f"dropped.p{k}.jsonl").open("a")
    wlock = threading.Lock()
    stats = {"done": 0, "parsed": 0, "dropped": 0, "llm_calls": 0, "plans_new": 0}
    stop_file = Path(os.environ.get("MP_STOP_FILE", "/nonexistent/STOP"))

    def one(row):
        if stop_file.exists():
            return
        try:
            rec = Case(ctx, row, task, out_dir).run()
        except Exception as e:  # noqa: BLE001  infra error or crash: not written, retried once in pass 2
            with wlock:
                drop_f.write(json.dumps({"_id": row["_id"], "error": f"{type(e).__name__}: {e}",
                                         "tb": traceback.format_exc()[-3000:], "timestamp": v1.now()}) + "\n")
                drop_f.flush()
                stats["dropped"] += 1
            print(f"[mp] {task} p{k} DROPPED {row['_id']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        with wlock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            stats["done"] += 1
            stats["parsed"] += rec["parsed"]
            stats["llm_calls"] += rec["n_llm_calls"]
            stats["plans_new"] += rec["n_plans_new"]
            el = time.time() - t0
            print(f"[mp] {task} p{k} [{stats['done']}/{len(todo)}] {rec['_id']} parsed={rec['parsed']} cands="
                  f"{rec['n_candidates']} new_plans={rec['n_plans_new']} tools={rec['n_tool_calls']} llm={rec['n_llm_calls']} "
                  f"tool_err={rec['n_tool_errors']} no_evidence={rec['n_no_evidence']} {rec['wall_s']}s | parse {stats['parsed']}/{stats['done']} | total llm {stats['llm_calls']} | "
                  f"{stats['done'] / max(el, 1) * 60:.2f} cases/min", flush=True)

    for p in (1, 2):
        if p == 2:
            have = v1.done_ids(out_dir, ctx["run_sig"])
            todo = [r for r in todo if r["_id"] not in have]
            if not todo:
                break
            print(f"[mp] {task} p{k} pass 2: retrying {len(todo)} dropped case(s)", flush=True)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for f in as_completed([ex.submit(one, r) for r in todo]):
                f.result()
    out_f.close()
    drop_f.close()
    print(f"[mp] DONE {task} shard {k}: {stats} in {(time.time() - t0) / 60:.1f} min", flush=True)


def code_shas():
    files = [HERE / f for f in ("run_dermarena_v1_medagentpro.py", "mp_llm.py", "Planner.py", "Derm_Task_level.py", "utils.py",
                                "Summary_Module.py", "kb_rag.py", "medlineplus_kb.py", "Decider/GPT_Decider.py",
                                "Decider/Pro_Decider.py", "Dermatology/tools/skin_tools.py")]
    files += [AG / "skin_agent/tools/skin_tools.py", AG / "scripts/run_dermarena_v1.py",
              DC / "dx_benchmark/dx_infer.py", DC / "dx_benchmark/dx_parse.py", DC / "dx_benchmark/prompts/dx_top5.txt",
              DC / "eval/utils.py", MAKE_TERMS]
    return {os.path.relpath(f, AG.parent): v1.sha_file(f) for f in files}


def namespace(backbone, kb_sha_, toolset_sha, task_sha):
    """Everything stored per disease depends on the backbone, the knowledge base (embedder and chunks included), the toolset, the
    planner and every stage-1 prompt and setting; a change gets a fresh cache, so plans from earlier designs are never read."""
    M = mods()
    planner = v1.sha_file(HERE / "Planner.py")
    prompt = planner_prompt("{input}", "{goal}")
    profile = chunk_profile_name()
    return sha_text(json.dumps([PLAN_SCHEMA, backbone, kb_sha_, toolset_sha, task_sha, planner, prompt, POSTCHECK_VERSION,
                                M["kb_rag"].GUIDELINE_PROMPT, M["kb_rag"].RAG_QUESTION, INDICATOR_PROMPT, INDICATOR_PROMPT_NO_G,
                                PLANNER_INDICATORS, PLAN_RETRY_NOTE, WEIGHTS_SYSTEM, WEIGHTS_GUIDELINE, WEIGHTS_PROMPT,
                                M["kb_rag"].embedder_name(), profile, CHUNK_PROFILES[profile],
                                os.environ.get("MP_SAMPLING_JSON"), os.environ.get("MP_PLAN_ATTEMPTS", "5"),
                                os.environ.get("MP_PLAN_RELAXED_ATTEMPTS", "2"), os.environ.get("MP_INDICATOR_ATTEMPTS", "2"),
                                os.environ.get("MP_WEIGHTS_ATTEMPTS", "2")]))[:12]


def print_plans(root, n, seed):
    """CPU only, no server: the owner's reading copy of n plans drawn at random from the newest cache namespace."""
    spaces = [d for d in Path(root).glob("*") if d.is_dir() and list(d.glob("*.json"))]
    if not spaces:
        print(f"[mp] no cached plans under {root}: plans are written during a canary or full run")
        return
    ns = max(spaces, key=lambda d: max(p.stat().st_mtime for p in d.glob("*.json")))
    entries = [json.loads(p.read_text()) for p in sorted(ns.glob("*.json"))]
    ok = [e for e in entries if e["status"] == "ok"]
    share = {}
    for e in ok:
        for k, v in (e.get("tool_steps") or {}).items():
            share[k] = share.get(k, 0) + v
    tot = sum(share.values()) or 1
    n_failed = len(list((ns / "_failed").glob("*.json"))) if (ns / "_failed").exists() else 0
    print(f"namespace {ns.name}{' (of ' + str(len(spaces)) + ')' if len(spaces) > 1 else ''} | {len(entries)} cached diseases | "
          f"{len(ok)} with a plan | {n_failed} failed attempts not stored (_failed/) | {sum(e['has_guideline'] for e in entries)} with a guideline | "
          f"{sum(bool(e.get('relaxed')) for e in entries)} needed the relaxed tries | "
          f"mean attempts {sum(len(e['attempts']) for e in entries) / len(entries):.1f} | seed {seed}")
    count = {}
    for e in ok:
        k = e.get("n_final_indicators", 0)
        count[k] = count.get(k, 0) + 1
    print("final indicators per plan (target 3-7, not enforced): " + ", ".join(f"{k}: {v}" for k, v in sorted(count.items())))
    print("tool step share: " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(share.items(), key=lambda x: -x[1])))
    print(f"after reading, approve this namespace for the full run: touch {ns / 'REVIEWED'}\n")
    toolset = {int(t["id"]): t["type"] for t in load_task_cfg()[1]}
    for e in random.Random(seed).sample(entries, min(n, len(entries))):
        print("=" * 100)
        print(f"{e['disease']}  [plan: {e['status']} | has_guideline: {e['has_guideline']} ({e['guideline_status']}) | "
              f"attempts {len(e['attempts'])}]")
        print("sources: " + ("; ".join(sorted({c['title'] for c in e['retrieval']['chunks']})) or "none"))
        print(f"guideline: {e['guideline'] or '(none: planned from backbone knowledge)'}")
        print("indicators: " + ("; ".join(e.get("indicators") or []) or "none"))
        w = e.get("weights") or {}
        print(f"threshold: {w.get('threshold')}")
        for s in e["plan"] or []:
            tools = ", ".join(toolset.get(int(t), str(t)) for t in s["tool"])
            wt = (w.get("by_step") or {}).get(str(s["id"]))
            print(f"  {s['id']:>2}. [{s['action_type']}, {s['output_type']}{'' if wt is None else f', weight {wt:.2f}'}] in={s['input_type']} "
                  f"tool={tools}\n      {s['action']}")
        if e["status"] != "ok":
            for a in e["attempts"]:
                print(f"  attempt {a['attempt']}: {a.get('stage')}: {a.get('error')}")
        print()


def rank_stability(dirs, repeats, threads, backbone):
    """Text only. The final call makes the output now, so this re-sends each case's saved final prompt `repeats` times and reports how far
    the top 5 moves with nothing but that call's sampling changed. Writes rank_stability.json and rank_stability_cases.jsonl per directory."""
    M = mods()
    client = make_client()
    for d in dirs:
        traces = [t for t in sorted((d / "traces").glob("*.json")) if ".dropped_" not in t.name]
        cases = []
        lock = threading.Lock()
        out_f = (d / "rank_stability_cases.jsonl").open("w")

        def one(tf):
            tr = json.loads(tf.read_text())
            fin = tr.get("final") or {}
            orig = fin.get("diagnoses") or []
            if not orig:
                return
            calls = M["mp_llm"].start_case_trace()
            runs = [ask_final(client, fin["prompt"]) for _ in range(repeats)]
            row = {"_id": tr["_id"], "original": orig, "repeats": [r["diagnoses"] for r in runs], "llm_calls": calls}
            with lock:
                cases.append(row)
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

        with ThreadPoolExecutor(max_workers=threads) as ex:
            for f in as_completed([ex.submit(one, t) for t in traces]):
                f.result()
        out_f.close()
        pairs = [(c["original"], r) for c in cases for r in c["repeats"]]
        n = max(1, len(pairs))
        rep = {"repeats": repeats, "n_cases": len(cases), "n_llm_calls": sum(len(c["llm_calls"]) for c in cases),
               "n_unparsed_repeats": sum(not b for _, b in pairs),
               "mean_top5_overlap_pct": round(100 * sum(len(set(a) & set(b)) / max(1, len(a)) for a, b in pairs) / n, 1),
               "same_top5_set_pct": round(100 * sum(set(a) == set(b) for a, b in pairs) / n, 1),
               "same_top5_order_pct": round(100 * sum(a == b for a, b in pairs) / n, 1),
               "same_top1_pct": round(100 * sum(a[:1] == b[:1] for a, b in pairs) / n, 1),
               "cases_with_any_top1_change_pct": round(
                   100 * sum(any(r[:1] != c["original"][:1] for r in c["repeats"]) for c in cases) / max(1, len(cases)), 1)}
        (d / "rank_stability.json").write_text(json.dumps(rep, indent=1))
        print(f"[mp] RANK STABILITY {d}: {json.dumps(rep)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="T1.1,T1.2")
    ap.add_argument("--input-tmpl", default=str(RELEASE / "tasks/{task_dir}/test.jsonl"))
    ap.add_argument("--out-tmpl", default=None, help="output dir per task, e.g. .../dx/<slug>/{task}")
    ap.add_argument("--kb", default="/fs04/scratch2/ub62/ssim0070/benchmark_runs/agents/medlineplus_kb")
    ap.add_argument("--plan-cache", default=None, help="disease-level plan cache root (one per backbone)")
    ap.add_argument("--print-plans", type=int, default=None, help="print N sampled cached plans and exit (CPU only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank-stability", type=int, default=None, help="re-run the final call N times on saved traces (text only) and exit")
    ap.add_argument("--require-reviewed", action="store_true", help="refuse unless <plan-cache>/<namespace>/REVIEWED exists")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backbone", default=os.environ.get("MP_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--slug", default=os.environ.get("MP_SLUG", "medagentpro-qwen3.5-27b"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--service", default=None)
    a = ap.parse_args()
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    if a.print_plans is not None:
        print_plans(a.plan_cache, a.print_plans, a.seed)
        return
    if a.merge:
        for t in tasks:
            v1.merge(Path(a.out_tmpl.format(task=t)))
        return
    check_backbone_env(a.backbone)
    if a.rank_stability is not None:
        rank_stability([Path(a.out_tmpl.format(task=t)) for t in tasks], a.rank_stability, a.threads, a.backbone)
        return
    if not os.environ.get("MP_SERVER_ARGS"):
        raise SystemExit("[mp] MP_SERVER_ARGS not set (SGLang launch args, part of run_sig) — refusing to run")
    if not a.plan_cache:
        raise SystemExit("[mp] --plan-cache is required")
    k, n = (int(x) for x in a.shard.split("/"))
    M = mods()
    task_cfg, toolset = load_task_cfg()
    kb_sha_ = M["kb_rag"].kb_sha(a.kb, a.backbone)
    toolset_sha, task_sha = sha_path(DATA_ROOT / "toolset.json")[:16], sha_path(DATA_ROOT / "task.json")[:16]
    ns = namespace(a.backbone, kb_sha_, toolset_sha, task_sha)
    if a.require_reviewed and not (Path(a.plan_cache) / ns / "REVIEWED").exists():
        raise SystemExit(f"[mp] {Path(a.plan_cache) / ns / 'REVIEWED'} missing: read the sampled plans of THIS namespace, then touch it")
    sampling = json.loads(os.environ["MP_SAMPLING_JSON"])
    base_meta = {
        "system": "MedAgent-Pro (ICLR 2026 camera-ready design: per-candidate disease-level plan, patient-level reasoning with memory and "
                  "reflection, one integrating call; baselines/MedAgent-Pro)",
        "slug": a.slug, "backbone": a.backbone, "base_url": BASE_URL, "sampling": sampling,
        "max_tokens_per_call": int(os.environ.get("MP_MAX_TOKENS", "0")) or None,
        "eos_retries": int(os.environ.get("MP_EOS_RETRIES", "0")),
        "reasoning_fallback": os.environ.get("MP_REASONING_FALLBACK") == "1",
        "cand_attempts": int(os.environ.get("MP_CAND_ATTEMPTS", "2")),
        "plan_attempts": int(os.environ.get("MP_PLAN_ATTEMPTS", "5")),
        "plan_relaxed_attempts": int(os.environ.get("MP_PLAN_RELAXED_ATTEMPTS", "2")),
        "indicator_attempts": int(os.environ.get("MP_INDICATOR_ATTEMPTS", "2")),
        "weights_attempts": int(os.environ.get("MP_WEIGHTS_ATTEMPTS", "2")),
        "final_attempts": int(os.environ.get("MP_FINAL_ATTEMPTS", "2")), "step_cap": int(os.environ.get("MP_STEP_CAP", "10")),
        "embed_model": M["kb_rag"].embedder_name(), "chunk_profile": chunk_profile_name(),
        "chunking": {k: v for k, v in CHUNK_PROFILES[chunk_profile_name()].items() if k != "file"},
        "call_budget": int(os.environ.get("MP_CALL_BUDGET", "0")) or None,
        "sanitize_special_tokens": os.environ.get("MP_SANITIZE_SPECIAL_TOKENS") == "1",
        "server_args": os.environ.get("MP_SERVER_ARGS"),
        "kb_dir": a.kb, "kb_manifest_sha": kb_sha_, "toolset_sha": toolset_sha, "task_sha": task_sha,
        "plan_cache": str(a.plan_cache), "plan_namespace": ns,
        "prompt_sha": sha_text(CAND_PROMPT + QUAL_CASE_CONTEXT + SUMMARY_PROMPT + SUMMARY_SUFFIX + MEMORY_PROMPT + REFLECT_SYSTEM +
                               REFLECT_PROMPT + FINAL_PROMPT + NO_EVIDENCE),
        "tool_images": "clinical_photo+dermoscopy", "code": code_shas(), "shards": n,
        "git_head": subprocess.run(["git", "-C", str(AG), "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
    }
    run_sig = hashlib.sha256(json.dumps({x: base_meta[x] for x in (
        "backbone", "sampling", "max_tokens_per_call", "eos_retries", "reasoning_fallback", "cand_attempts", "plan_attempts",
        "plan_relaxed_attempts", "indicator_attempts", "weights_attempts", "final_attempts", "step_cap", "embed_model", "chunk_profile",
        "chunking", "sanitize_special_tokens", "server_args", "kb_manifest_sha", "toolset_sha", "task_sha", "plan_namespace", "prompt_sha",
        "tool_images", "code")}, sort_keys=True).encode()).hexdigest()[:12]

    def build(tasks_, input_tmpl, out_tmpl, limit, threads):
        out = []
        for task in tasks_:
            inp = Path(input_tmpl.format(task=task, task_dir=TASK_DIR[task]))
            out_dir = Path(out_tmpl.format(task=task))
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = [json.loads(l) for l in inp.open() if l.strip()]
            mine = [r for i, r in enumerate(rows) if i % n == k]
            have = v1.done_ids(out_dir, run_sig)
            todo = [r for r in mine if r["_id"] not in have]
            if limit:
                todo = todo[:limit]
            meta = dict(base_meta, task=task, threads_per_process=threads,
                        input={"path": str(inp), "sha": v1.sha_file(inp)}, run_sig=run_sig)
            mp = out_dir / "RUN_META.json"
            try:
                prev = json.loads(mp.read_text()).get("run_sig") if mp.exists() else None
            except Exception:  # noqa: BLE001
                prev = None
            if prev not in (None, run_sig) and not os.environ.get("DX_ALLOW_META_CHANGE"):
                raise SystemExit(f"[mp] {mp} has a different run_sig ({prev} != {run_sig}) — refusing to mix settings")
            if k == 0:
                tmp = mp.with_name(f"RUN_META.json.tmp{os.getpid()}")
                tmp.write_text(json.dumps(meta, indent=1))
                os.replace(tmp, mp)
            print(f"[mp] {task} shard {k}/{n}: {len(mine)} cases, {len(mine) - len(todo)} done, {len(todo)} to run "
                  f"| threads {threads} | run_sig {run_sig}", flush=True)
            out.append((task, out_dir, todo))
        return out

    if not a.service:
        work = build(tasks, a.input_tmpl, a.out_tmpl, a.limit, a.threads)
        if not any(todo for _, _, todo in work):
            return

    t0 = time.time()
    client = make_client()
    analyzer, summarizer = M["GPT_Decider"](API_KEY), M["Summary_Module"](API_KEY)
    decider, planner = make_decider(M["Pro_Decider"])(API_KEY), M["Planner"](API_KEY)
    for o in (analyzer, summarizer, decider, planner):
        o.client = client  # explicit local endpoint + timeout (never the env/.env default)
    tools = load_tools()
    embedder = M["kb_rag"].make_embedder()
    kb = M["kb_rag"].KnowledgeBase(a.kb, a.backbone, embedder)
    probe = embedder.encode(["psoriasis"], query=True)
    if probe.shape[1] != kb.emb.shape[1]:
        raise SystemExit(f"[mp] embedder dim {probe.shape[1]} != index dim {kb.emb.shape[1]}")
    print(f"[mp] tools {sorted(tools)} + KB ({len(kb.docs)} docs, {len(kb.chunks)} chunks) loaded in {time.time() - t0:.0f}s "
          f"| plan cache {a.plan_cache}/{ns} | run_sig {run_sig}", flush=True)
    ctx = {"client": client, "analyzer": analyzer, "summarizer": summarizer, "decider": decider, "planner": planner,
           "kb": kb, "kb_sha": kb_sha_, "cache": PlanCache(a.plan_cache, ns), "concepts": ConceptMap(),
           "toolset": toolset, "tool_by_id": {int(t["id"]): t for t in toolset if "id" in t}, "task_cfg": task_cfg,
           "run_sig": run_sig, "backbone": a.backbone, "max_tokens": base_meta["max_tokens_per_call"]}
    print("[mp] per-case LLM calls: 2 + per candidate (1 memory + VLM steps + indicators + intermediate steps), plus 4+ per new disease; "
          "see call_budget in records", flush=True)

    if not a.service:
        for task, out_dir, todo in work:
            if todo:
                run_task(ctx, task, out_dir, todo, k, a.threads)
        return

    # persistent worker (= run_dermarena_v1 --service): job files <jobdir>/*.json
    # {"tasks", "input_tmpl", "out_tmpl", "limit", "threads"}; <job>.p<k>.done; shard 0 merges + postchecks ->
    # <job>.ALLDONE; <jobdir>/STOP ends the workers.
    jd = Path(a.service)
    jd.mkdir(parents=True, exist_ok=True)
    print(f"[mp] SERVICE ready (shard {k}/{n}) — polling {jd}", flush=True)
    while not (jd / "STOP").exists():
        jobs = [j for j in sorted(jd.glob("*.json")) if not (jd / f"{j.stem}.p{k}.done").exists()]
        if not jobs:
            time.sleep(30)
            continue
        jf = jobs[0]
        try:
            job = json.loads(jf.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[mp] SERVICE cannot read {jf.name}: {e}", flush=True)
            time.sleep(10)
            continue
        jt = [t.strip() for t in job.get("tasks", "T1.1,T1.2").split(",") if t.strip()]
        thr = int(job.get("threads", a.threads))
        itmpl = job.get("input_tmpl", str(RELEASE / "tasks/{task_dir}/test.jsonl"))
        work = build(jt, itmpl, job["out_tmpl"], job.get("limit"), thr)
        for task, out_dir, todo in work:
            if todo and not (jd / "STOP").exists():
                run_task(ctx, task, out_dir, todo, k, thr)
        (jd / f"{jf.stem}.p{k}.done").write_text(v1.now())
        if k == 0:
            while not all((jd / f"{jf.stem}.p{x}.done").exists() for x in range(n)) and not (jd / "STOP").exists():
                time.sleep(20)
            for task, out_dir, _ in work:
                v1.merge(out_dir)
                r = subprocess.run(["python3", str(DC / "dx_benchmark/dx_check.py"), "post", str(out_dir), task,
                                    itmpl.format(task=task, task_dir=TASK_DIR[task])], capture_output=True, text=True)
                r2 = subprocess.run([sys.executable, str(HERE / "mp_postcheck.py"), str(out_dir), task], capture_output=True, text=True)
                (out_dir / "POSTCHECK.txt").write_text(r.stdout + r.stderr + r2.stdout + r2.stderr)
                print(f"[mp] SERVICE postcheck {jf.stem} {task}:\n{r.stdout[-1500:]}", flush=True)
            (jd / f"{jf.stem}.ALLDONE").write_text(v1.now())
    print("[mp] SERVICE STOP seen — exiting", flush=True)


if __name__ == "__main__":
    main()
