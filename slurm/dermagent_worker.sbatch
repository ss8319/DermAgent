#!/bin/bash
#SBATCH --job-name=dermagent-warm
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/DermAgent/slurm/logs/dermagent-warm-%j.out
#SBATCH --error=/fs04/scratch2/ub62/ssim0070/DermAgent/slurm/logs/dermagent-warm-%j.err

# Persistent warm worker: loads PanDerm+MAKE+DermoGPT+MedGemma ONCE and stays
# resident (up to 4h) servicing requests, so iterations don't reload the models.
# Submit an iteration by dropping a request JSON into $DERMAGENT_REQ_DIR.
# Stop it by: touch $DERMAGENT_REQ_DIR/STOP   (or scancel the job).

set -uo pipefail
DERMAGENT=/fs04/scratch2/ub62/ssim0070/DermAgent
PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
export DERMAGENT_REQ_DIR="$DERMAGENT/results/warm/requests"
mkdir -p "$DERMAGENT_REQ_DIR" "$DERMAGENT/slurm/logs"

export DERMAGENT_ROOT="$DERMAGENT"
# CRITICAL: _resolve() joins image_path to this; default /mnt/hdd doesn't exist on M3,
# so without it NO images resolve → tools never run → text-only Qwen.
export DERMARENA_REPO_ROOT=/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2
export MEDGEMMA_PATH=/fs04/scratch2/ub62/ssim0070/models/medgemma-1.5-4b-it
export HF_HOME=/fs04/scratch2/ub62/ssim0070/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
_ORKEY=$(grep -m1 '^OPENROUTER_API_KEY=' /fs04/scratch2/ub62/ssim0070/DermArena_data/.env | cut -d= -f2)
export OPENAI_API_KEY="$_ORKEY"
export OPENROUTER_API_KEY="$_ORKEY"

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
cd "$DERMAGENT"
$PY skin_agent/dermagent_worker.py
