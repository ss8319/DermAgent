#!/bin/bash
# GENERIC serve-and-run on the L40S spare memory (11-11): serves $L40S_MODEL (default Qwen/Qwen3.6-27B) and runs $L40S_CMD (default: judge $JUDGE_SLUGS). 11-10 (owner agreed: "ablation this week" plan) — judge the Qwen3.5-9B findings-kept ablation arm with Qwen3.6-27B on the spare memory of
# the L40S tools job (61714874, 4x L40S 46 GB; the Qwen MedAgent-Pro tools use ~19.9 GB per GPU). Runs INSIDE that job:
#   srun --jobid=61714874 --overlap -n1 -c8 --mem=100G --gres=gpu:4 bash agents/ops/l40s_judge9b.sh
# Agents come first: a watchdog stops the judge (resumable) at the first sign of harm to the Qwen runners:
#   any GPU with < MINFREE MiB free, a new "out of memory" line or a new tool_err>0 case in the Qwen logs, or the STOP file.
# Memory: tp4 -> weights ~13.8 GB/GPU; --mem-fraction-static 0.85 keeps 15% of 46 GB (~6.9 GB/GPU) out of SGLang's static pool,
# so after weights the KV pool gets ~5 GB/GPU and ~5 GB/GPU stays free for activations and the tools.
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; R=/fs04/scratch2/ub62/ssim0070/sglang; source $R/slurm/env.sh
W=$B/holders/l40s_judge; mkdir -p $W; STOP=$W/STOP; MINFREE=${MINFREE:-1500}; PORT=30987
QL=$B/dx/medagentpro-qwen3.5-27b/logs; MP="$QL/full_p0.log $QL/full_p1.log $QL/full_p2.log $QL/full_p3.log"
log() { echo "[l40s-judge] $(date '+%m-%d %H:%M:%S') $*"; }
oom0=$(cat $MP | grep -ci "out of memory"); te0=$(cat $MP | grep -c "tool_err=[1-9]")
source $B/local_weights.sh; MODEL=${L40S_MODEL:-Qwen/Qwen3.6-27B}; MID=$MODEL; local_weights "$MODEL" || { log "weights failed"; exit 1; }
SA="--tp-size 4 --reasoning-parser qwen3 --mem-fraction-static 0.85 --cuda-graph-max-bs 32 --disable-custom-all-reduce"
[[ "$MODEL" == /tmp/* ]] && SA="$SA --served-model-name $MID"
log "starting server: $MODEL $SA (baseline oom=$oom0 tool_err_cases=$te0)"
setsid apptainer exec --nv --home "${CONTAINER_HOME}:${CONTAINER_HOME}" --bind "${R}:${R}" \
  --env "HF_HOME=${HF_HOME}" --env "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}" \
  --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" --env "HF_HUB_OFFLINE=1" --env "TRANSFORMERS_OFFLINE=1" \
  --env "FLASHINFER_WORKSPACE_DIR=${CONTAINER_HOME}/.cache/flashinfer" \
  --env "TRITON_CACHE_DIR=${CONTAINER_HOME}/.triton" --env "TORCH_HOME=${CONTAINER_HOME}/.cache/torch" \
  "${SGLANG_CONTAINER}" python3 -m sglang.launch_server --model-path "$MODEL" --host 0.0.0.0 --port $PORT \
    --context-length ${L40S_CTX:-32768} --trust-remote-code $SA >> $W/server.log 2>&1 &
SP=$!; JP=""
cleanup() { [ -n "$JP" ] && kill -TERM -- -$JP 2>/dev/null; kill -TERM -- -$SP 2>/dev/null; sleep 15; kill -KILL -- -$SP 2>/dev/null; rm -f $W/ENDPOINT; log "stopped: $1"; }
trap 'cleanup signal; exit 1' TERM INT
harm() {  # prints a reason if the agents might be harmed
  local f
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  [ -n "$f" ] && [ "$f" -lt $MINFREE ] && { echo "GPU free memory ${f} MiB < $MINFREE"; return; }
  [ "$(cat $MP | grep -ci 'out of memory')" -gt "$oom0" ] && { echo "new out-of-memory line in the Qwen MedAgent-Pro logs"; return; }
  [ "$(cat $MP | grep -c 'tool_err=[1-9]')" -gt "$te0" ] && { echo "new case with tool errors in the Qwen MedAgent-Pro logs"; return; }
  [ -f $STOP ] && echo "STOP file"
}
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break
  r=$(harm); [ -n "$r" ] && { cleanup "$r (while loading)"; exit 3; }; sleep 10
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { log "server failed to start"; tail -30 $W/server.log; cleanup "no server"; exit 2; }
echo "http://$(hostname):$PORT/v1 $MID l40s-job since $(date -Is)" > $W/ENDPOINT
log "server ready; GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
export OPENROUTER_BASE_URL=http://127.0.0.1:$PORT/v1 OPENROUTER_API_KEY=local LLM_TRACE=1
if [ -n "${L40S_CMD:-}" ]; then setsid bash -c "$L40S_CMD" >> $W/job.log 2>&1 &
else setsid env DX_JUDGE_PASSES=3 DX_JUDGE_WORKERS=${WORKERS:-12} bash $B/dx/judge_dx.sh ${JUDGE_SLUGS:-qwen3.5-9b-findingskept} >> $W/judge.log 2>&1 &
fi
JP=$!; log "judge started (pid $JP, ${WORKERS:-12} workers per task)"
while kill -0 $JP 2>/dev/null; do
  r=$(harm); [ -n "$r" ] && { cleanup "$r"; exit 3; }
  kill -0 $SP 2>/dev/null || { cleanup "server died"; exit 2; }
  sleep 30
done
wait $JP; rc=$?; JP=""; cleanup "judge finished rc=$rc"; exit $rc
