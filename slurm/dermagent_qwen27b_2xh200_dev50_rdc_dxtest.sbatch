#!/bin/bash
#SBATCH --job-name=dermagent-dev50-rdc
#SBATCH --partition=bdi-imaging
#SBATCH --qos=irq
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dermagent-dev50-rdc-%j.out
#SBATCH --error=logs/dermagent-dev50-rdc-%j.err

# DermAgent FULL 7-tool — Qwen3.5-27B — remaining dev50 task: MM_RDC.
# Chained (--dependency) after the MM_RDS job so both task-appropriate baselines
# complete in one continuous hold of the 2 H200 (no slot lost). Same config as the
# MM_RDS job; only the task + benchmark change. Resume-safe: adapter skips _ids
# already in the output JSONL.
#
# DxTest DROPPED (user decision 2026-07-04): the agent emits a top-5 diagnosis but
# the DxTest judge scores precision/recall over recommended diagnostic tests
# (diagnostic_test_gt) — output-type mismatch → meaningless score. RDS is the
# meaningful DermAgent-vs-Cerebra comparison; RDC at least asks for a diagnosis.

set -euo pipefail

SGLANG_REPO="/fs04/scratch2/ub62/ssim0070/sglang"
AGENT_ROOT="/fs04/scratch2/ub62/ssim0070/DermAgent"
BENCH_ROOT="/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2/dev50"
HF_WEIGHTS_CACHE="/fs04/scratch2/ub62/ssim0070/sglang/hf_cache"
PY="/fs04/scratch2/ub62/ssim0070/dermagent/bin/python3"

source "${SGLANG_REPO}/slurm/env.sh"

MODEL="Qwen/Qwen3.5-27B"
PORT=30014
MEM_FRACTION=0.92
ENABLED_TOOLS="panderm,make,dermogpt_vqa,qwen_vqa,rag,text_rag,ontology"
OUT_DIR="${BENCH_ROOT}/eval_results/DermAgent_Qwen3.5-27B_all7tools"
mkdir -p "${OUT_DIR}" "${SGLANG_REPO}/logs"

echo "Job ${SLURM_JOB_ID} on $(hostname) | port ${PORT} | tasks: MM_RDC + DxTest | start $(date)"
nvidia-smi || true

# ── SGLang backbone on GPU 0 ───────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv \
    --home "${CONTAINER_HOME}:${CONTAINER_HOME}" \
    --bind "${SGLANG_REPO}:${SGLANG_REPO}" \
    --env "HF_HOME=${HF_WEIGHTS_CACHE}" \
    --env "HUGGINGFACE_HUB_CACHE=${HF_WEIGHTS_CACHE}/hub" \
    --env "TRANSFORMERS_CACHE=${HF_WEIGHTS_CACHE}/hub" \
    --env "HF_HUB_ENABLE_HF_TRANSFER=1" \
    --env "HF_HUB_OFFLINE=1" \
    --env "TRANSFORMERS_OFFLINE=1" \
    --env "FLASHINFER_WORKSPACE_DIR=${CONTAINER_HOME}/.cache/flashinfer" \
    --env "TRITON_CACHE_DIR=${CONTAINER_HOME}/.triton" \
    --env "TORCH_HOME=${CONTAINER_HOME}/.cache/torch" \
    "${SGLANG_CONTAINER}" \
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" --host 0.0.0.0 --port "${PORT}" --tp 1 \
        --context-length 32768 --mem-fraction-static "${MEM_FRACTION}" \
        --reasoning-parser qwen3 --tool-call-parser qwen3_coder --trust-remote-code \
        > "${SGLANG_REPO}/logs/dermagent-dev50-rdcdxt-sglang-${SLURM_JOB_ID}.log" 2>&1 &
SGLANG_PID=$!
trap "kill ${SGLANG_PID} 2>/dev/null || true" EXIT

NODE=$(hostname)
HEALTH_URL="http://${NODE}:${PORT}/health"
for i in $(seq 1 180); do
    curl -sf "${HEALTH_URL}" >/dev/null 2>&1 && { echo "[init] SGLang ready (attempt ${i})"; break; }
    kill -0 "${SGLANG_PID}" 2>/dev/null || { echo "[ERROR] SGLang died"; tail -40 "${SGLANG_REPO}/logs/dermagent-dev50-rdcdxt-sglang-${SLURM_JOB_ID}.log"; exit 1; }
    sleep 5
done
curl -sf "${HEALTH_URL}" >/dev/null 2>&1 || { echo "[ERROR] SGLang not ready"; exit 1; }

# ── Agent env ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1
export OPENAI_API_KEY="local-sglang"
export OPENAI_BASE_URL="http://${NODE}:${PORT}/v1"
export HF_HOME="${HF_WEIGHTS_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_WEIGHTS_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_WEIGHTS_CACHE}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QDRANT_PATH="${AGENT_ROOT}/qdrant_storage"
export QDRANT_TEXT_PATH="${AGENT_ROOT}/qdrant_storage_text"
export DERMARENA_IMG_ROOT="${BENCH_ROOT}/.."     # = dermarena_dx_v2 (v2 crops)
export TEST_MODE=true
cd "${AGENT_ROOT}"

# ── Task 2: MM_RDC dev50 ───────────────────────────────────────────────────
echo "[task] MM_RDC dev50 starting $(date)"
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/MM_RDC_benchmark_dev50.jsonl" \
    --task rdc --backbone "qwen/qwen3.5-27b" --base-url "${OPENAI_BASE_URL}" \
    --enabled-tools "${ENABLED_TOOLS}" --device cuda \
    --max-tool-calls 15 --recursion-limit 60 \
    --output "${OUT_DIR}/mm_rdc_dev50_predictions.jsonl"
echo "[task] MM_RDC done $(date)"
echo "[job] Complete (RDS+RDC baseline; DxTest dropped). Preds in ${OUT_DIR}. End: $(date)"
