#!/bin/bash
#SBATCH --job-name=dermagent-27b-h100
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dermagent-27b-%j.out
#SBATCH --error=logs/dermagent-27b-%j.err

# DermAgent (dermarena-open-set-agent branch) on 1× H100 80GB.
#
# Single-job design: SGLang serves Qwen3.5-27B (mem_fraction=0.75 → ~60 GB KV cache),
# DermAgent tool models (panderm, make, rag, ontology, ~4 GB combined) share the
# remaining ~20 GB on the same GPU via --enable-lru --max-loaded-models 2.
#
# Resume-by-_id: resubmit this job to continue from a checkpoint — run_dermarena_agent.py
# skips cases whose _id already exists in the output JSONL.
#
# ── Critical SGLang flags (discovered during DermArena v4 eval, Jun 2026) ────────────
# 1. --tool-call-parser qwen3_coder
#    Qwen3.5's chat_template.jinja emits <tool_call>\n<function=name>\n<parameter=k>v...
#    XML format, NOT the JSON {"name":"...","arguments":{...}} that qwen/qwen25 parsers
#    expect. Without this flag SGLang returns raw XML as text → LangGraph never sees
#    structured tool_calls → tools never execute → no FINAL_ANSWER.
#
# 2. HUGGINGFACE_HUB_CACHE must be re-exported after sourcing env.sh
#    PanDermTool's open_clip calls hf_hub_download(cache_dir=None), which reads
#    HUGGINGFACE_HUB_CACHE from env. env.sh sets this to the SGLang repo cache;
#    the agent must override it to point at the actual model weights cache.
#
# 3. TRANSFORMERS_CACHE must also be re-exported after sourcing env.sh
#    HFTextEncoder.__init__ calls AutoConfig.from_pretrained (PanDerm text tower:
#    neuml/pubmedbert-base-embeddings). This reads TRANSFORMERS_CACHE. Same issue
#    as #2 — env.sh sets the wrong dir; must override after sourcing.
# ─────────────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── User-configurable paths ────────────────────────────────────────────────
# Edit these four variables for your cluster layout:
SGLANG_REPO="/fs04/scratch2/ub62/ssim0070/sglang"
AGENT_ROOT="/fs04/scratch2/ub62/ssim0070/DermAgent"
BENCH_ROOT="/fs04/scratch2/ub62/ssim0070/dermarena_dx/stratified"
HF_WEIGHTS_CACHE="/fs04/scratch2/ub62/ssim0070/sglang/hf_cache"
PY="/fs04/scratch2/ub62/ssim0070/dermagent/bin/python3"
# ──────────────────────────────────────────────────────────────────────────
# Smoke mode: cap each task at N cases. Unset/empty = full run.
SMOKE_LIMIT="${SMOKE_LIMIT:-10}"

source "${SGLANG_REPO}/slurm/env.sh"

MODEL="Qwen/Qwen3.5-27B"
PORT=30010
MEM_FRACTION=0.75

OUT_DIR="${BENCH_ROOT}/eval_results/DermAgent_Qwen3.5-27B"
mkdir -p "${OUT_DIR}"
mkdir -p "${SGLANG_REPO}/logs"

echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU(s):    ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:     ${MODEL}  port=${PORT}"
echo "Out dir:   ${OUT_DIR}"
echo "=========================================="
nvidia-smi || true
echo "=========================================="

# ── 1. Start SGLang server in background ───────────────────────────────────
apptainer exec --nv \
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
        > "${SGLANG_REPO}/logs/dermagent-sglang-${SLURM_JOB_ID}.log" 2>&1 &

SGLANG_PID=$!
echo "[init] SGLang PID=${SGLANG_PID}"
trap "echo '[exit] killing SGLang PID=${SGLANG_PID}'; kill ${SGLANG_PID} 2>/dev/null || true" EXIT

# ── 2. Wait for SGLang ready ───────────────────────────────────────────────
NODE=$(hostname)
HEALTH_URL="http://${NODE}:${PORT}/health"
echo "[init] waiting for SGLang at ${HEALTH_URL} ..."
for i in $(seq 1 120); do
    if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "[init] SGLang ready (attempt ${i})"
        break
    fi
    if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "[ERROR] SGLang process died. Check ${SGLANG_REPO}/logs/dermagent-sglang-${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
done

if ! curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
    echo "[ERROR] SGLang not ready after 600 s. Aborting."
    exit 1
fi

# ── 3. DermAgent environment ────────────────────────────────────────────────
# Override HF cache vars — env.sh may set these to the SGLang repo cache (wrong).
# PanDerm's open_clip and HFTextEncoder both read these directly from env.
export OPENAI_API_KEY="local-sglang"
export OPENAI_BASE_URL="http://${NODE}:${PORT}/v1"
export HF_HOME="${HF_WEIGHTS_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_WEIGHTS_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_WEIGHTS_CACHE}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QDRANT_PATH="${AGENT_ROOT}/qdrant_storage"
export DERMARENA_IMG_ROOT="${BENCH_ROOT}/.."

echo "[agent] backbone=${MODEL} at ${OPENAI_BASE_URL}"
echo "[agent] HF_HOME=${HF_HOME}"
echo "[agent] QDRANT_PATH=${QDRANT_PATH}"

# ── 4. Run MM_RDS ──────────────────────────────────────────────────────────
echo "==========================================" && echo "[task] MM_RDS starting $(date)" && echo "=========================================="
cd "${AGENT_ROOT}"
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/MM_RDS_benchmark_dev1000.jsonl" \
    --task rds \
    --backbone "qwen/qwen3.5-27b" \
    --base-url "${OPENAI_BASE_URL}" \
    --enabled-tools panderm,make,rag,ontology \
    --device cuda \
    --enable-lru \
    --max-loaded-models 2 \
    --max-tool-calls 15 \
    --recursion-limit 60 \
    --limit ${SMOKE_LIMIT} --output "${OUT_DIR}/mm_rds_smoke_predictions.jsonl"
echo "[task] MM_RDS done $(date)"

# ── 5. Run MM_RDC ──────────────────────────────────────────────────────────
echo "==========================================" && echo "[task] MM_RDC starting $(date)" && echo "=========================================="
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/MM_RDC_benchmark_dev1000.jsonl" \
    --task rdc \
    --backbone "qwen/qwen3.5-27b" \
    --base-url "${OPENAI_BASE_URL}" \
    --enabled-tools panderm,make,rag,ontology \
    --device cuda \
    --enable-lru \
    --max-loaded-models 2 \
    --max-tool-calls 15 \
    --recursion-limit 60 \
    --limit ${SMOKE_LIMIT} --output "${OUT_DIR}/mm_rdc_smoke_predictions.jsonl"
echo "[task] MM_RDC done $(date)"

# ── 6. Run DxTest ──────────────────────────────────────────────────────────
echo "==========================================" && echo "[task] DxTest starting $(date)" && echo "=========================================="
"${PY}" scripts/run_dermarena_agent.py \
    --benchmark "${BENCH_ROOT}/DxTest_benchmark_dev1000.jsonl" \
    --task dxtest \
    --backbone "qwen/qwen3.5-27b" \
    --base-url "${OPENAI_BASE_URL}" \
    --enabled-tools panderm,make,rag,ontology \
    --device cuda \
    --enable-lru \
    --max-loaded-models 2 \
    --max-tool-calls 15 \
    --recursion-limit 60 \
    --limit ${SMOKE_LIMIT} --output "${OUT_DIR}/dxtest_smoke_predictions.jsonl"
echo "[task] DxTest done $(date)"
echo "==========================================" && echo "[job] All tasks complete. $(date)" && echo "=========================================="
