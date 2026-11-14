# =============================================================================
# Adapted from MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
# Original work: Wang et al., "MedAgent-Pro: Towards Evidence-Based
#   Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow",
#   ICLR 2026.
# Upstream license: no license file. Used with attribution; see
#   baselines/MedAgent-Pro/ATTRIBUTION.md for details.
# Modifications by the DermAgent authors for dermatology benchmarks.
# =============================================================================

import os
import json
from openai import OpenAI
from mp_llm import mp_create  # DermArena v1: env-configurable backbone + full trace

class Summary_Module:
    def __init__(self, api_key):
        """
        Initialize the Summary object with the OpenAI API Key.

        Args:
            api_key (str): OpenAI API Key
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    # DermArena v1 (2026-11-06): answer_suffix=None keeps the original hard-coded Yes/No suffix (tasks 1-3 unchanged);
    # the DermArena runner passes SUMMARY_SUFFIX (the commented-out Yes, No or Uncertain line below).
    def summarize(self, input_file, output_file, prompt, field, answer_suffix=None):
        """
        Summarize the content of a specified field in a JSON file using OpenAI ChatCompletion.

        Args:
            input_file (str): input file path
            output_file (str): output file path
            field (str): field name to summarize

        Returns:
            str: summarized text
        """
        with open(input_file, "r", encoding="utf-8") as file:
            input_data = json.load(file)

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as file:
                output_data = json.load(file)
        else:
            output_data = {}

        if field not in input_data:
            print(f"field '{field}' not found in the input data.")
            return
        content = input_data[field]

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please help me summarize the information."},
            # {"role": "user", "content": f"{content}\n {prompt} \nAnswer with only one word (Yes, No or Uncertain)"}
            {"role": "user", "content": f"{content}\n {prompt} \nAnswer with only one word (Yes or No)"}
        ]
        if answer_suffix is not None:
            messages[1]["content"] = f"{content}\n {prompt} " + (f"\n{answer_suffix}" if answer_suffix else "")

        completion = mp_create(self.client, messages, model_default="gpt-4o", module="summary")
        summary_text = completion.choices[0].message.content

        output_data[field] = summary_text

        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(output_data, json_file, indent=4)

        return summary_text
