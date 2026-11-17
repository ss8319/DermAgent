#!/bin/bash
#SBATCH --job-name=dermagent-dev50-rerun128k
#SBATCH --partition=bdi-imaging
#SBATCH --qos=irq
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/dermagent-dev50-rerun128k-%j.out
#SBATCH --error=logs/dermagent-dev50-rerun128k-%j.err

# Re-run ONLY the empty-prediction cases (root cause: 32768 context exhaustion —
# multimodal backbone + base64 images + tool accumulation + unbounded qwen3
# reasoning slammed the 32k ceiling → truncated FINAL_ANSWER → empty).
# FIX #1: --context-length 131072 (native cap is 262144, rope_type=default).
# The 45 RDS / 37 RDC good predictions were stripped-and-backed-up, so the
# adapter's resume logic re-runs only the 5 RDS + 13 RDC missing _ids.
# One allocation, backbone+tools loaded once, both tasks sequential.

set -euo pipefail
SGLANG_REPO="/fs04/scratch2/ub62/ssim0070/sglang"
AGENT_ROOT="/fs04/scratch2/ub62/ssim0070/DermAgent"
BENCH_ROOT="/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2/dev50"
HF_WEIGHTS_CACHE="/fs04/scratch2/ub62/ssim0070/sglang/hf_cache"
PY="/fs04/scratch2/ub62/ssim0070/dermagent/bin/python3"
source "${SGLANG_REPO}/slurm/env.sh"

MODEL="Qwen/Qwen3.5-27B"
PORT=30015
MEM_FRACTION=0.92
CONTEXT_LEN=131072                 # FIX #1: was 32768
ENABLED_TOOLS="panderm,make,dermogpt_vqa,qwen_vqa,rag,text_rag,ontology"
OUT_DIR="${BENCH_ROOT}/eval_results/DermAgent_Qwen3.5-27B_all7tools"
mkdir -p "${OUT_DIR}" "${SGLANG_REPO}/logs"

echo "Job ${SLURM_JOB_ID} on $(hostname) | ctx=${CONTEXT_LEN} | re-run empties (RDS+RDC) | $(date)"
nvidia-smi || true

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
        --context-length "${CONTEXT_LEN}" --mem-fraction-static "${MEM_FRACTION}" \
        --reasoning-parser qwen3 --tool-call-parser qwen3_coder --trust-remote-code \
        > "${SGLANG_REPO}/logs/dermagent-dev50-rerun128k-sglang-${SLURM_JOB_ID}.log" 2>&1 &
SGLANG_PID=$!
trap "kill ${SGLANG_PID} 2>/dev/null || true" EXIT

NODE=$(hostname)
HEALTH_URL="http://${NODE}:${PORT}/health"
for i in $(seq 1 240); do
    curl -sf "${HEALTH_URL}" >/dev/null 2>&1 && { echo "[init] SGLang ready (attempt ${i})"; break; }
    kill -0 "${SGLANG_PID}" 2>/dev/null || { echo "[ERROR] SGLang died"; tail -40 "${SGLANG_REPO}/logs/dermagent-dev50-rerun128k-sglang-${SLURM_JOB_ID}.log"; exit 1; }
    sleep 5
done
curl -sf "${HEALTH_URL}" >/dev/null 2>&1 || { echo "[ERROR] SGLang not ready"; exit 1; }

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
export DERMARENA_IMG_ROOT="${BENCH_ROOT}/.."
export TEST_MODE=true
cd "${AGENT_ROOT}"

echo "[task] RDS empties re-run $(date)"
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/MM_RDS_benchmark_dev50.jsonl" \
    --task rds --backbone "qwen/qwen3.5-27b" --base-url "${OPENAI_BASE_URL}" \
    --enabled-tools "${ENABLED_TOOLS}" --device cuda \
    --max-tool-calls 15 --recursion-limit 60 \
    --output "${OUT_DIR}/mm_rds_dev50_predictions.jsonl"
echo "[task] RDS empties done $(date)"

echo "[task] RDC empties re-run $(date)"
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/MM_RDC_benchmark_dev50.jsonl" \
    --task rdc --backbone "qwen/qwen3.5-27b" --base-url "${OPENAI_BASE_URL}" \
    --enabled-tools "${ENABLED_TOOLS}" --device cuda \
    --max-tool-calls 15 --recursion-limit 60 \
    --output "${OUT_DIR}/mm_rdc_dev50_predictions.jsonl"
echo "[task] RDC empties done $(date)"
echo "[job] Complete (128k re-run). End: $(date)"
