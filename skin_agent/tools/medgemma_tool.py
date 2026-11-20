"""MedGemma tool — general medical-image VLM, ported from Cerebra.

Fills DermAgent's gap: its vision tools (PanDerm/DermoGPT/MAKE) are trained on
clinical/dermoscopic SKIN photos, so DermArena's non-derm images (histopathology,
radiology, IHC, gross pathology, immunofluorescence) go unanalyzed. MedGemma-1.5-4b-it
is a general medical VLM that describes those. Loads from LOCAL weights (no internet).
"""
from __future__ import annotations

import os
from typing import Any, Type

import torch
from PIL import Image
from pydantic import BaseModel, Field

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # older transformers
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
from transformers import AutoProcessor

from .base import BaseSkinTool

MEDGEMMA_PATH = os.environ.get(
    "MEDGEMMA_PATH", "/fs04/scratch2/ub62/ssim0070/models/medgemma-1.5-4b-it"
)

SYSTEM_PROMPT = "You are an expert clinician analyzing medical images from a patient case report."

# Case-grounded user prompt. {patient_case} is injected by the caller:
#   RDS / DxTest → case_report ; RDC → case_report + examination_results.
PROMPT_TEMPLATE = (
    "Based on the patient case below, describe the key clinical findings from the image.\n"
    "If the image provided is a pathology image, please provide a pathology report that "
    "covers the description and the differentials that these findings support.\n\n"
    "Patient case:\n{patient_case}"
)


class MedGemmaInput(BaseModel):
    image_path: str = Field(description="Path to the medical image file")
    patient_case: str = Field(
        default="",
        description="The case narrative (RDS/DxTest) or narrative + exam results (RDC) "
                    "to ground the image reading.",
    )


class MedGemmaTool(BaseSkinTool):
    """General medical-image analyzer (MedGemma-1.5-4b-it) for non-derm modalities."""

    name: str = "medgemma_analyzer"
    description: str = (
        "Analyze a medical image with MedGemma, a general medical vision-language model. "
        "Use for images the dermatology tools can't handle — histopathology, radiology, "
        "immunohistochemistry, gross pathology, immunofluorescence, or unknown modality. "
        "Returns visual findings + a ranked differential."
    )
    args_schema: Type[BaseModel] = MedGemmaInput

    model_id: str = MEDGEMMA_PATH
    model: Any = None
    processor: Any = None
    max_new_tokens: int = 512

    def _load_model(self) -> None:
        if not os.path.isdir(self.model_id):
            raise FileNotFoundError(
                f"MedGemma weights not found at {self.model_id}; set $MEDGEMMA_PATH."
            )
        print(f"[MedGemmaTool] Loading model: {self.model_id}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print("[MedGemmaTool] Model loaded.")

    def _run(self, image_path: str, patient_case: str = "") -> str:
        try:
            from ..utils.image_utils import resolve_image_path
            from .base import validate_image_path
            img_path = validate_image_path(resolve_image_path(image_path))
            self._ensure_initialized()

            image = Image.open(img_path).convert("RGB")
            user_text = PROMPT_TEMPLATE.format(patient_case=(patient_case.strip() or "(not provided)"))
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image},
                ]},
            ]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                # MedGemma: greedy decoding per HF docs (no sampling params).
                gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            text = self.processor.decode(gen[0][input_len:], skip_special_tokens=True).strip()
            return text
        except Exception as e:  # noqa: BLE001
            return self._handle_error(e, "medgemma analysis")

    def unload(self) -> None:
        if self.processor is not None:
            del self.processor
            self.processor = None
        super().unload()
        print("[MedGemmaTool] Model unloaded.")
