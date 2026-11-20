# =============================================================================
# Adapted from MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
# Original work: Wang et al., "MedAgent-Pro: Towards Evidence-Based
#   Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow",
#   ICLR 2026.
# Upstream license: no license file. Used with attribution; see
#   baselines/MedAgent-Pro/ATTRIBUTION.md for details.
# Modifications by the DermAgent authors for dermatology benchmarks.
# =============================================================================

"""
Open-set final decision module for DermArena v1 T1.1 / T1.2 (2026-11-06).

= MultiClass_Decider (task-1 final step) with the minimal open-set edits, the same rule used for the task-4
planner config: "select exactly ONE diagnosis from the candidate list" -> "list the 5 most likely diagnoses,
ranked (open set)". The reply format is the benchmark's own JSON ({"diagnoses": [...]}, as dx_top5.txt) plus
MultiClass_Decider's "reasoning" key; "confidence" is dropped (one number for a ranked list is ill-defined).
Unlike MultiClass_Decider this module does NOT fuzzy-match or repair the answer: it stores the raw reply and the
runner applies the benchmark parser (dx_parse.parse); unparsed = wrong.
"""

import base64
import json
import os

from openai import OpenAI
from mp_llm import mp_create  # DermArena v1: env-configurable backbone + full trace

ANSWER_JSON = ('{"diagnoses": ["<most likely>", "<2nd>", "<3rd>", "<4th>", "<5th>"], '
               '"reasoning": "<brief>"}')

SYSTEM_MSG = (
    "You are a board-certified dermatologist making a final diagnosis.\n"
    "Given the diagnostic evidence from multiple analysis tools, list the 5 most likely "
    "diagnoses of the patient, ranked (most likely first). Open set: any disease, no candidate list.\n"
    "Return ONLY a JSON object with keys:\n"
    '  "diagnoses" (list of 5 disease names, most likely first),\n'
    '  "reasoning" (brief string).'
)


class OpenSet_Decider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def build_messages(self, prompt: str, indicators: list, image_paths=None):
        if image_paths and not isinstance(image_paths, list):
            image_paths = [image_paths]
        lines = []
        for it in indicators:
            name = str(it.get("indicator_name", "")).strip()
            ev = it.get("evidence", it.get("if_abnormal", ""))
            if isinstance(ev, (dict, list)):
                ev = json.dumps(ev, ensure_ascii=False)
            lines.append(f"- {name}: {ev}")
        ind_text = "Collected diagnostic evidence:\n" + "\n".join(lines)

        image_messages = []
        for path in image_paths or []:
            b64 = self.encode_image(path)
            image_messages.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        user_content = image_messages + [{
            "type": "text",
            "text": f"{prompt}\n\n{ind_text}\n\nReturn ONLY the JSON object: {ANSWER_JSON}",
        }]
        return [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user_content}]

    def decide(self, output_file: str, prompt: str, indicators: list, image_paths=None, field: str = "overall"):
        """Returns the raw completion (the runner parses it). Writes {field: {raw, reasoning_content, finish_reason}}."""
        messages = self.build_messages(prompt, indicators, image_paths)
        completion = mp_create(self.client, messages, model_default="gpt-4o", module="openset_decider")
        ch = completion.choices[0]
        result_obj = {
            "raw": ch.message.content,
            "reasoning_content": getattr(ch.message, "reasoning_content", None)
                                 or (getattr(ch.message, "model_extra", None) or {}).get("reasoning_content"),
            "finish_reason": ch.finish_reason,
        }
        existing = {}
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing[field] = result_obj
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)
        return completion
