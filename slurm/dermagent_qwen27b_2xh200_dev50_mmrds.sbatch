#!/bin/bash
#SBATCH --job-name=dermagent-dev50-2xh200
#SBATCH --partition=bdi-imaging
#SBATCH --qos=irq
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dermagent-dev50-2xh200-%j.out
#SBATCH --error=logs/dermagent-dev50-2xh200-%j.err

# DermAgent FULL 7-tool baseline — Qwen3.5-27B backbone — on MM_RDS dev50 (n=50).
# reproduction-fixes branch. Gives the DermAgent baseline number to sit beside the
# Cerebra agent on the SAME dev50 image-bound subset.
#
# Lands on m3u023 (only bdi-imaging node): 2 of its 4 H200 are already mine
# (judge 58104561 + Qwen3.5-27B Cerebra backbone 58104828). This job claims the
# remaining 2 → total 4 = my bdi-imagingq gres/gpu cap. Dedicated backbone (does
# NOT share the Cerebra server → no contention).
#
# GPU layout:
#   GPU 0: SGLang Qwen3.5-27B backbone (mem_fraction=0.92, port 30013)
#   GPU 1: all 7 DermAgent tools resident (panderm, make, dermogpt_vqa, qwen_vqa,
#          rag, text_rag, ontology)
#
# TEST_MODE=true → profiler attaches per-tool + per-LLM-call timing to traces
# (feed the traces to the dermagent-postcheck skill after the run).
#
# Resume: resubmit if walltime hits before 50 cases — adapter skips _ids already
# in the output JSONL.

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
SGLANG_REPO="/fs04/scratch2/ub62/ssim0070/sglang"
AGENT_ROOT="/fs04/scratch2/ub62/ssim0070/DermAgent"
BENCH_ROOT="/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2/dev50"
HF_WEIGHTS_CACHE="/fs04/scratch2/ub62/ssim0070/sglang/hf_cache"
PY="/fs04/scratch2/ub62/ssim0070/dermagent/bin/python3"

source "${SGLANG_REPO}/slurm/env.sh"

MODEL="Qwen/Qwen3.5-27B"
PORT=30013                  # distinct from 30010/30011/30012 (other DermAgent jobs)
MEM_FRACTION=0.92
ENABLED_TOOLS="panderm,make,dermogpt_vqa,qwen_vqa,rag,text_rag,ontology"

OUT_DIR="${BENCH_ROOT}/eval_results/DermAgent_Qwen3.5-27B_all7tools"
mkdir -p "${OUT_DIR}" "${SGLANG_REPO}/logs"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Visible GPUs:  ${CUDA_VISIBLE_DEVICES:-unset}  (expect 2)"
echo "Model:         ${MODEL}  port=${PORT}  mem_frac=${MEM_FRACTION}"
echo "Tools:         ${ENABLED_TOOLS}  (all 7)"
echo "Task:          MM_RDS only — dev50 (n=50)"
echo "Profiling:     TEST_MODE=true"
echo "Out dir:       ${OUT_DIR}"
echo "Start:         $(date)"
echo "=========================================="
nvidia-smi || true
echo "=========================================="

# ── 1. Start SGLang backbone on GPU 0 ──────────────────────────────────────
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
        --model-path "${MODEL}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --tp 1 \
        --context-length 32768 \
        --mem-fraction-static "${MEM_FRACTION}" \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --trust-remote-code \
        > "${SGLANG_REPO}/logs/dermagent-dev50-sglang-${SLURM_JOB_ID}.log" 2>&1 &

SGLANG_PID=$!
echo "[init] SGLang PID=${SGLANG_PID} on GPU 0"
trap "echo '[exit] killing SGLang PID=${SGLANG_PID}'; kill ${SGLANG_PID} 2>/dev/null || true" EXIT

# ── 2. Wait for SGLang ready ───────────────────────────────────────────────
NODE=$(hostname)
HEALTH_URL="http://${NODE}:${PORT}/health"
echo "[init] waiting for SGLang at ${HEALTH_URL} ..."
for i in $(seq 1 180); do
    if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "[init] SGLang ready (attempt ${i})"
        break
    fi
    if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "[ERROR] SGLang process died. Tail:"
        tail -40 "${SGLANG_REPO}/logs/dermagent-dev50-sglang-${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    sleep 5
done

if ! curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
    echo "[ERROR] SGLang not ready after 900 s. Aborting."
    exit 1
fi

# ── 3. DermAgent environment ───────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1
export OPENAI_API_KEY="local-sglang"
export OPENAI_BASE_URL="http://${NODE}:${PORT}/v1"
export HF_HOME="${HF_WEIGHTS_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_WEIGHTS_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_WEIGHTS_CACHE}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QDRANT_PATH="${AGENT_ROOT}/qdrant_storage"            # derm1m (image RAG)
export QDRANT_TEXT_PATH="${AGENT_ROOT}/qdrant_storage_text"  # derm_rag (text RAG)
export DERMARENA_IMG_ROOT="${BENCH_ROOT}/.."                 # = dermarena_dx_v2 (v2 crops; resolves 104/104)
export TEST_MODE=true                                         # ← enables profiler

echo "[agent] backbone=${MODEL} at ${OPENAI_BASE_URL}"
echo "[agent] tool GPU = ${CUDA_VISIBLE_DEVICES}"
echo "[agent] HF_HOME=${HF_HOME}"
echo "[agent] DERMARENA_IMG_ROOT=${DERMARENA_IMG_ROOT}"
echo "[agent] QDRANT_PATH=${QDRANT_PATH}"
echo "[agent] QDRANT_TEXT_PATH=${QDRANT_TEXT_PATH}"
echo "[agent] TEST_MODE=${TEST_MODE} → profiler enabled"
echo "[agent] all 7 tools: ${ENABLED_TOOLS}"

cd "${AGENT_ROOT}"

# ── 4. MM_RDS dev50 — full 50 cases, NO --limit ────────────────────────────
echo "==========================================" && echo "[task] MM_RDS dev50 starting $(date)" && echo "=========================================="
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/MM_RDS_benchmark_dev50.jsonl" \
    --task rds \
    --backbone "qwen/qwen3.5-27b" \
    --base-url "${OPENAI_BASE_URL}" \
    --enabled-tools "${ENABLED_TOOLS}" \
    --device cuda \
    --max-tool-calls 15 \
    --recursion-limit 60 \
    --output "${OUT_DIR}/mm_rds_dev50_predictions.jsonl"

echo "[task] MM_RDS done $(date)"
echo "=========================================="
echo "[job] Complete. Predictions: ${OUT_DIR}/mm_rds_dev50_predictions.jsonl"
echo "[job] Traces (per-case profiling): ${OUT_DIR}/traces/"
echo "[job] End: $(date)"
