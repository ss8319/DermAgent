# A1 masking ablation (handoff benchmark_runs/HANDOFF_ABLATION_A1_20261116.md; owner GO 2026-11-16). Sourced by the 0x_a1_*.req files.
# a1_done_guard <marker regex>: if the marker is already in any holder's runs.log, make this request a no-op (no model load).
B0=/fs04/scratch2/ub62/ssim0070/benchmark_runs
A1_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
a1_done_guard() {
  if grep -qsE "$1" $B0/holders/*/out/runs.log; then
    echo "[a1] already done ($1): skip $(date)"; MODEL=__tools_host__; CTX=1024; SERVE_ARGS=""; run() { :; }; return 0
  fi; return 1
}
