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
Bridge layer: wraps DermAgent's LangChain tools into MedAgent-Pro's
fn(image_path, save_dir, save_name) calling convention.

Each wrapper:
  1. Lazily initialises the underlying tool (singleton per process).
  2. Calls tool._run(...) to get a result string.
  3. Merges the result into save_dir/save_name (JSON).
"""

import json
import os
import re
import sys
import threading
from pathlib import Path

# DermAgent project root: baselines/MedAgent-Pro/Dermatology/tools -> DermAgent_submission/
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3].parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_panderm_tool = None
_make_tool = None
_rag_tool = None
_dermogpt_tool = None

_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES_TOOL", "cuda")


def _get_panderm():
    global _panderm_tool
    if _panderm_tool is None:
        from skin_agent.tools.skin_tools import PanDermTool
        _panderm_tool = PanDermTool(device=_DEVICE)
    return _panderm_tool


def _get_make():
    global _make_tool
    if _make_tool is None:
        from skin_agent.tools.skin_tools import MAKETool
        _make_tool = MAKETool(device=_DEVICE)
    return _make_tool


def _get_rag():
    global _rag_tool
    if _rag_tool is None:
        from skin_agent.tools.skin_tools import RAGTool
        _rag_tool = RAGTool(device=_DEVICE)
    return _rag_tool


def _get_dermogpt():
    global _dermogpt_tool
    if _dermogpt_tool is None:
        from skin_agent.tools.skin_tools import DermoGPTTool
        _dermogpt_tool = DermoGPTTool(device=_DEVICE)
    return _dermogpt_tool


# ---------------------------------------------------------------------------
# JSON merge helper
# ---------------------------------------------------------------------------
def _merge_json(save_dir: str, save_name: str, key: str, value):
    """Read existing JSON at save_dir/save_name, add key=value, write back."""
    path = os.path.join(save_dir, save_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data[key] = value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public tool functions (MedAgent-Pro compatible signatures)
# ---------------------------------------------------------------------------

def panderm_classify(image_path: str, save_dir: str, save_name: str) -> str:
    """Zero-shot skin disease classification using PanDerm/DermLIP.

    Signature: fn(image_path, save_dir, save_name) as required by
    MedAgent-Pro's Case_level quantitative step dispatcher.
    """
    tool = _get_panderm()
    # DermArena v1 (owner 2026-11-05): PanDerm scores the BACKBONE's own top-10 differential (same as DermAgent), not
    # the HAM10000 default list. The runner writes <save_dir>/candidates.json = {"diagnoses": [10 names]} before the
    # plan runs; absent file = original behaviour (tool default list).
    cand = None
    cf = os.path.join(save_dir, "candidates.json")
    if os.path.exists(cf):
        try:
            cand = ", ".join(json.load(open(cf, encoding="utf-8"))["diagnoses"]) or None
        except Exception:  # noqa: BLE001
            cand = None
    result_str = tool._run(image_path=image_path, candidate_diseases=cand) if cand else tool._run(image_path=image_path)
    _merge_json(save_dir, save_name, "panderm_classification", result_str)
    return result_str


def make_annotate(image_path: str, save_dir: str, save_name: str) -> str:
    """Dermoscopic concept extraction using MAKE/OpenCLIP.

    Returns structured concept annotations with confidence scores.
    """
    tool = _get_make()
    result_str = tool._run(image_path=image_path)
    _merge_json(save_dir, save_name, "make_concepts", result_str)
    return result_str


def rag_retrieve(image_path: str, save_dir: str, save_name: str) -> str:
    """Retrieve similar dermatology cases from Derm1M via Qdrant.

    Uses image embedding similarity to find top-5 matching cases.
    """
    tool = _get_rag()
    result_str = tool._run(image_path=image_path, top_k=5)
    _merge_json(save_dir, save_name, "rag_similar_cases", result_str)
    return result_str


# ---------------------------------------------------------------------------
# DermArena v1 per-candidate bridges (2026-11-07). Each returns structured data and raises ToolError instead of passing
# on a tool's "Error ..." text; the runner stores the result under step_<id> (no tool-named keys, so a later step reads
# its own evidence and never the whole file). The DermAgent tool classes are not edited: a live experiment imports them.
# ---------------------------------------------------------------------------
class ToolError(RuntimeError):
    """The tool returned an error text or a placeholder. The runner records it; that evidence is simply absent."""


_PANDERM_LOCK = threading.Lock()   # _panderm_softmax bypasses PanDermTool._run, so the runner's class lock does not cover it
_MAKE_ROW = re.compile(r"^\s*\d+\.\s+(.+?):\s+([0-9.]+)\s*$")


def _check_text(out, tool):
    t = out if isinstance(out, str) else ""
    if not t.strip() or t.lstrip().lower().startswith("error") or "[Model not available]" in t:
        raise ToolError(f"{tool}: {t[:300]!r}")
    return t


def _panderm_softmax(tool, image_path, diseases):
    """Same computation as PanDermTool._run (which prints only its top 5), returning the whole softmax."""
    import numpy as np
    import torch
    from PIL import Image
    from skin_agent.tools.skin_tools import resolve_image_path, validate_image_path
    img_path = validate_image_path(resolve_image_path(image_path))
    tool._ensure_initialized()
    image_tensor = tool.preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(tool.device)
    text_features = tool._build_text_features(diseases)
    with torch.no_grad():
        img_features = tool.model.encode_image(image_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
    return [float(x) for x in np.asarray(probs)]


def panderm_scores(image_path: str, diseases: list) -> dict:
    """{disease: probability} for ALL candidates, comma-free names as given (fixes the 5-of-10 print limit)."""
    tool = _get_panderm()
    with _PANDERM_LOCK:
        probs = _panderm_softmax(tool, image_path, list(diseases))
    return dict(zip(diseases, probs))


def make_concepts(image_path: str, concepts: list) -> dict:
    """MAKE target mode: {concept: presence score}, parsed from the tool's numbered text."""
    tool = _get_make()
    text = _check_text(tool._run(image_path=image_path, target_concepts=", ".join(concepts)), "MAKE")
    scores = {}
    for line in text.splitlines():
        m = _MAKE_ROW.match(line)
        if m:
            scores[m.group(1)] = float(m.group(2))
    if set(scores) != set(concepts):
        raise ToolError(f"MAKE: expected {sorted(concepts)}, parsed {sorted(scores)}")
    return scores


def dermogpt_vqa(image_path: str, query: str) -> str:
    """DermoGPT free-form VQA on one image (the tool's VQA mode: no candidate list, no concept list)."""
    tool = _get_dermogpt()
    text = _check_text(tool._run(image_path=image_path, query=query), "DermoGPT")
    return text.split("DermoGPT VQA Response:", 1)[-1].strip()
