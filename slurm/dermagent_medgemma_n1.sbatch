#!/bin/bash
#SBATCH --job-name=dermagent-mg-n1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/DermAgent/slurm/logs/dermagent-mg-n1-%j.out
#SBATCH --error=/fs04/scratch2/ub62/ssim0070/DermAgent/slurm/logs/dermagent-mg-n1-%j.err

# n=1-per-task (RDS + RDC) run of DermAgent + MedGemma, Qwen3.5-27B backbone.
# Same first case per task as the Cerebra n=1 (dev500 stratified) → directly comparable.
# DermAgent skin tools on skin images; MedGemma (ported from Cerebra) on non-derm images.

set -uo pipefail   # NOT -e: one task failing shouldn't abort the other + grading

DERMAGENT=/fs04/scratch2/ub62/ssim0070/DermAgent
EVAL=/fs04/scratch2/ub62/ssim0070/DermArena/dataset_collection/eval
DATA=/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2
STRAT=$DATA/stratified
HYP=/fs04/scratch2/ub62/ssim0070/DermArena/dataset_collection/data/benchmark/icd11_hypernym.json
PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
OUT=$DERMAGENT/results/dermagent_medgemma_n1
mkdir -p "$OUT" "$DERMAGENT/slurm/logs"

# local weights (no downloads)
export DERMAGENT_ROOT="$DERMAGENT"
# CRITICAL: _resolve() joins image_path to this (default /mnt/hdd doesn't exist on M3).
export DERMARENA_REPO_ROOT=/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2
export MEDGEMMA_PATH=/fs04/scratch2/ub62/ssim0070/models/medgemma-1.5-4b-it
export HF_HOME=/fs04/scratch2/ub62/ssim0070/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# backbone (Qwen3.5-27B) via OpenRouter, OpenAI-compatible; judge ledger path
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
_ORKEY=$(grep -m1 '^OPENROUTER_API_KEY=' /fs04/scratch2/ub62/ssim0070/DermArena_data/.env | cut -d= -f2)
export OPENAI_API_KEY="$_ORKEY"        # backbone (Qwen) via OpenRouter
export OPENROUTER_API_KEY="$_ORKEY"    # score_dx judge reads THIS name
export LLM_LEDGER_PATH="$OUT/llm_ledger.jsonl"

MODEL="qwen/qwen3.5-27b"
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

declare -A BENCH=( [rds]="$STRAT/MM_RDS_benchmark_dev500.jsonl" [rdc]="$STRAT/MM_RDC_benchmark_dev500.jsonl" )

for TASK in rds rdc; do
  echo "=================== INFER $TASK (dermagent + medgemma, Qwen3.5-27B) ==================="
  # Qwen runs in THINKING mode (ideal thinking sampling params, reasoning enabled).
  # --max-tokens 8000: generous budget so reasoning + the final top-5 both fit.
  # --max-tool-images 3: MedGemma covers up to 3 non-derm images per case.
  $PY "$EVAL/infer_dx_agent.py" --task "$TASK" --benchmark "${BENCH[$TASK]}" \
      --baseline dermagent --medgemma --model "$MODEL" --limit 1 \
      --max-tokens 8000 --max-tool-images 3 --output "$OUT/${TASK}_pred.jsonl" \
      || echo "[warn] infer $TASK failed"
  head -1 "${BENCH[$TASK]}" > "$OUT/${TASK}_bench1.jsonl"
  echo "=================== GRADE $TASK (score_dx, gemini-2.5-flash judge) ==================="
  $PY "$EVAL/score_dx.py" --benchmark "$OUT/${TASK}_bench1.jsonl" \
      --predictions "$OUT/${TASK}_pred.jsonl" --output-dir "$OUT/${TASK}_grade" \
      --hypernyms "$HYP" --model google/gemini-2.5-flash || echo "[warn] grade $TASK failed"
done

echo "=================== DONE — under $OUT ==================="
