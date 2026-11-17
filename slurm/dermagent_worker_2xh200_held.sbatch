#!/bin/bash
#SBATCH --job-name=dermagent-held-worker
#SBATCH --partition=bdi-imaging
#SBATCH --qos=irq
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/dermagent-held-worker-%j.out
#SBATCH --error=logs/dermagent-held-worker-%j.err

# HELD DermAgent allocation (workspace CLAUDE.md GPU rule): holds 2x H200 at MAX
# walltime, loads backbone + 7 tools ONCE, then services run requests via a file
# queue with NO reload. Drive it by dropping request JSONs into $DERMAGENT_REQ_DIR:
#   {"task":"rds","benchmark":".../MM_RDS_benchmark_dev50.jsonl",
#    "output":".../mm_rds_dev50_predictions.jsonl","limit":null}
# Stop with: touch $DERMAGENT_REQ_DIR/STOP  (do NOT scancel without instruction).
#
# GPU 0: SGLang Qwen3.5-27B backbone @128k context (port 30016)
# GPU 1: 7 DermAgent tools resident, LangGraph agent

set -euo pipefail
SGLANG_REPO="/fs04/scratch2/ub62/ssim0070/sglang"
AGENT_ROOT="/fs04/scratch2/ub62/ssim0070/DermAgent"
HF_WEIGHTS_CACHE="/fs04/scratch2/ub62/ssim0070/sglang/hf_cache"
PY="/fs04/scratch2/ub62/ssim0070/dermagent/bin/python3"
source "${SGLANG_REPO}/slurm/env.sh"

MODEL="Qwen/Qwen3.5-27B"
PORT=30016
CONTEXT_LEN=131072
MEM_FRACTION=0.92
export DERMAGENT_REQ_DIR="${AGENT_ROOT}/results/warm_langgraph/requests"
mkdir -p "${DERMAGENT_REQ_DIR}" "${SGLANG_REPO}/logs" "${AGENT_ROOT}/logs"

echo "Job ${SLURM_JOB_ID} on $(hostname) | HELD worker | ctx=${CONTEXT_LEN} | queue=${DERMAGENT_REQ_DIR} | $(date)"
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
        > "${SGLANG_REPO}/logs/dermagent-held-sglang-${SLURM_JOB_ID}.log" 2>&1 &
SGLANG_PID=$!
trap "echo '[exit] killing SGLang'; kill ${SGLANG_PID} 2>/dev/null || true" EXIT

NODE=$(hostname)
for i in $(seq 1 240); do
    curl -sf "http://${NODE}:${PORT}/health" >/dev/null 2>&1 && { echo "[init] SGLang ready (attempt ${i})"; break; }
    kill -0 "${SGLANG_PID}" 2>/dev/null || { echo "[ERROR] SGLang died"; tail -40 "${SGLANG_REPO}/logs/dermagent-held-sglang-${SLURM_JOB_ID}.log"; exit 1; }
    sleep 5
done
curl -sf "http://${NODE}:${PORT}/health" >/dev/null 2>&1 || { echo "[ERROR] SGLang not ready"; exit 1; }

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
export DERMARENA_IMG_ROOT="/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2"
export DERMAGENT_BACKBONE="qwen/qwen3.5-27b"
export ENABLED_TOOLS="panderm,make,dermogpt_vqa,qwen_vqa,rag,text_rag,ontology"

echo "[worker] backbone ready @ ${OPENAI_BASE_URL} — starting held LangGraph worker"
cd "${AGENT_ROOT}"
exec "${PY}" skin_agent/dermagent_langgraph_worker.py
