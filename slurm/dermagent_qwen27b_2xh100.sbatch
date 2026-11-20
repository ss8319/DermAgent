#!/bin/bash
#SBATCH --job-name=dermagent-27b-2xh100
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dermagent-27b-2xh100-%j.out
#SBATCH --error=logs/dermagent-27b-2xh100-%j.err

# DermAgent on 2× H100 80 GB — split layout for the FULL agent.
#
# Splits the GPUs:
#   GPU 0: SGLang Qwen3.5-27B backbone (mem_fraction=0.92 → ~74 GB; full memory)
#   GPU 1: DermAgent driver with all heavyweight tools resident (~38 GB)
#
# Differs from the 1× H100 sibling (dermagent_qwen27b_h100.sbatch):
#   • 1× H100 path uses LRU + 4 tools (no DermoGPT-RL / Qwen3-VL / Text RAG)
#     because 80 GB can't hold 27B + the heavy VQAs at the same time.
#   • This 2× H100 path runs the full multimodal agent (6 tools — see caveat
#     below) without LRU thrashing.
#
# Tool set this submits with — 6 of 7:
#   panderm, make, dermogpt_vqa, qwen_vqa, rag, ontology
#
#   text_rag is SKIPPED: the embedded-Qdrant path-mode lock on
#   ./qdrant_storage collides with `rag` (see Finding #2 guard added in
#   create_tools()). Splitting collections into separate dirs is a follow-up;
#   for now 6 tools is the smoke baseline.
#
# Critical SGLang flags (same as 1× H100 path — discovered during DermArena
# v4 eval, Jun 2026):
#   --tool-call-parser qwen3_coder  → required for Qwen3.5 XML tool-call format
#   HUGGINGFACE_HUB_CACHE / TRANSFORMERS_CACHE override after env.sh source
#     → so PanDerm's open_clip + HFTextEncoder find weights at HF_WEIGHTS_CACHE
#
# Smoke vs full: SMOKE_LIMIT env var caps each task at N cases. Default 10.
# Unset (empty) for a full dev1000 sweep.

set -euo pipefail

# ── User-configurable paths ────────────────────────────────────────────────
SGLANG_REPO="/fs04/scratch2/ub62/ssim0070/sglang"
AGENT_ROOT="/fs04/scratch2/ub62/ssim0070/DermAgent"
BENCH_ROOT="/fs04/scratch2/ub62/ssim0070/dermarena_dx/stratified"
HF_WEIGHTS_CACHE="/fs04/scratch2/ub62/ssim0070/sglang/hf_cache"
PY="/fs04/scratch2/ub62/ssim0070/dermagent/bin/python3"
# ──────────────────────────────────────────────────────────────────────────
# Smoke mode: cap each task at N cases. Empty = full dev1000 sweep.
SMOKE_LIMIT="${SMOKE_LIMIT:-10}"

source "${SGLANG_REPO}/slurm/env.sh"

MODEL="Qwen/Qwen3.5-27B"
PORT=30011                  # 30010 is the 1× H100 path's port — distinct here
MEM_FRACTION=0.92           # GPU 0 dedicated to SGLang → can use most of it
ENABLED_TOOLS="panderm,make,dermogpt_vqa,qwen_vqa,rag,text_rag,ontology"

OUT_DIR="${BENCH_ROOT}/eval_results/DermAgent_Qwen3.5-27B_2xH100"
mkdir -p "${OUT_DIR}" "${SGLANG_REPO}/logs"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Visible GPUs:  ${CUDA_VISIBLE_DEVICES:-unset}  (expect 2)"
echo "Model:         ${MODEL}  port=${PORT}  mem_frac=${MEM_FRACTION}"
echo "Tools:         ${ENABLED_TOOLS}"
echo "Smoke limit:   ${SMOKE_LIMIT:-<full>}"
echo "Out dir:       ${OUT_DIR}"
echo "=========================================="
nvidia-smi || true
echo "=========================================="

# ── 1. Start SGLang server on GPU 0 (background) ───────────────────────────
# CUDA_VISIBLE_DEVICES=0 binds SGLang to the first GPU only. The DermAgent
# Python process below sets CUDA_VISIBLE_DEVICES=1 to use the second.
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
        > "${SGLANG_REPO}/logs/dermagent-2xh100-sglang-${SLURM_JOB_ID}.log" 2>&1 &

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
        echo "[ERROR] SGLang process died. Tail of log:"
        tail -40 "${SGLANG_REPO}/logs/dermagent-2xh100-sglang-${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    sleep 5
done

if ! curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
    echo "[ERROR] SGLang not ready after 900 s. Aborting."
    exit 1
fi

# ── 3. DermAgent environment (GPU 1 for all tools) ─────────────────────────
export CUDA_VISIBLE_DEVICES=1
export OPENAI_API_KEY="local-sglang"
export OPENAI_BASE_URL="http://${NODE}:${PORT}/v1"
export HF_HOME="${HF_WEIGHTS_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_WEIGHTS_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_WEIGHTS_CACHE}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QDRANT_PATH="${AGENT_ROOT}/qdrant_storage"            # derm1m (image RAG)
export QDRANT_TEXT_PATH="${AGENT_ROOT}/qdrant_storage_text"  # derm_rag (text RAG) — split to avoid lock collision
export DERMARENA_IMG_ROOT="${BENCH_ROOT}/.."

echo "[agent] backbone=${MODEL} at ${OPENAI_BASE_URL}"
echo "[agent] tool GPU = ${CUDA_VISIBLE_DEVICES}"
echo "[agent] HF_HOME=${HF_HOME}"
echo "[agent] QDRANT_PATH=${QDRANT_PATH}"

cd "${AGENT_ROOT}"

LIMIT_FLAG=""
if [[ -n "${SMOKE_LIMIT}" ]]; then
    LIMIT_FLAG="--limit ${SMOKE_LIMIT}"
fi

# Common flags for all three tasks. NOTE: no --enable-lru / --max-loaded-models
# because GPU 1 has plenty of room for the full tool set resident.
COMMON_FLAGS="--backbone qwen/qwen3.5-27b \
              --base-url ${OPENAI_BASE_URL} \
              --enabled-tools ${ENABLED_TOOLS} \
              --device cuda \
              --max-tool-calls 15 \
              --recursion-limit 60"

# ── 4. MM_RDS ──────────────────────────────────────────────────────────────
echo "==========================================" && echo "[task] MM_RDS starting $(date)" && echo "=========================================="
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/MM_RDS_benchmark_dev1000.jsonl" \
    --task rds \
    ${COMMON_FLAGS} \
    ${LIMIT_FLAG} \
    --output "${OUT_DIR}/mm_rds_smoke_predictions.jsonl"
echo "[task] MM_RDS done $(date)"

# ── 5. MM_RDC ──────────────────────────────────────────────────────────────
echo "==========================================" && echo "[task] MM_RDC starting $(date)" && echo "=========================================="
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/MM_RDC_benchmark_dev1000.jsonl" \
    --task rdc \
    ${COMMON_FLAGS} \
    ${LIMIT_FLAG} \
    --output "${OUT_DIR}/mm_rdc_smoke_predictions.jsonl"
echo "[task] MM_RDC done $(date)"

# ── 6. DxTest ──────────────────────────────────────────────────────────────
echo "==========================================" && echo "[task] DxTest starting $(date)" && echo "=========================================="
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/DxTest_benchmark_dev1000.jsonl" \
    --task dxtest \
    ${COMMON_FLAGS} \
    ${LIMIT_FLAG} \
    --output "${OUT_DIR}/dxtest_smoke_predictions.jsonl"
echo "[task] DxTest done $(date)"

echo "==========================================" && echo "[job] All tasks complete. $(date)" && echo "=========================================="
