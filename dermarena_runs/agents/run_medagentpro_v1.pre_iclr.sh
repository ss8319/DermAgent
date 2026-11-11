#!/bin/bash
# MedAgent-Pro (paper design: per-candidate disease-level plan + patient-level reasoning) on DermArena v1 T1.1 + T1.2 test,
# inside a GPU holder. NOT YET RUN ON GPU (rebuilt 2026-11-07; offline test: baselines/MedAgent-Pro/test_run_dermarena_v1_medagentpro.py).
# Backbone = the holder's SGLang server (OPENROUTER_BASE_URL); tools = PanDerm + MAKE + DermoGPT (no Case-RAG, no Qdrant, no
# ontology, no Qwen3-VL, no coding agent) + Qwen3-Embedding-8B for retrieval; MP_NPROC processes on MP_TOOL_GPUS.
# Usage (from a .req run(), or by hand on the GPU node):  run_medagentpro_v1.sh kb-index|canary|plans|rank-stability|full|service
#   kb-index : one-sentence summaries from this backbone + chunk embeddings + index manifest (resumable; run again after a stop)
#   canary   : canary20 sets -> $B/dx/$SLUG/canary/{T1.1,T1.2}; fills the plan cache; exits 3 if a canary gate FAILs
#   plans    : print 10 sampled cached plans (CPU only, no server) and the exact REVIEWED path of their cache namespace
#   rank-stability : text only: re-run the decider 3x on the canary traces, report how far the top 5 moves
#   full     : release test sets -> $B/dx/$SLUG/{T1.1,T1.2}      (refuses until <namespace>/REVIEWED exists)
#   MP_SERVER_ARGS (the .req's SERVE_ARGS + context length) is required for canary / full / service: it is part of run_sig.
#   service  : persistent workers polling $B/dx/$SLUG/jobs/*.json (= run_dermagent_v1.sh service)
set -uo pipefail
MODE=${1:-canary}
NPROC=${MP_NPROC:-2}; THREADS=${MP_THREADS:-12}; read -r -a GPUS <<< "${MP_TOOL_GPUS:-2 3}"
SLUG=${MP_SLUG:-medagentpro-qwen3.5-27b}; BACKBONE=${MP_BACKBONE:-Qwen/Qwen3.5-27B}
AG=/fs04/scratch2/ub62/ssim0070/DermAgent; MPD=$AG/baselines/MedAgent-Pro; PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python3
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; R=/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1
DC=/fs04/scratch2/ub62/ssim0070/DermArena/dataset_collection; DX=$DC/dx_benchmark; HC=/fs04/scratch2/ub62/ssim0070/sglang/hf_cache
KB=${MP_KB:-$B/agents/medlineplus_kb}
declare -A F=([T1.1]=T1.1_disease_screening [T1.2]=T1.2_disease_confirmation)
CACHE=$B/dx/$SLUG/plan_cache; JD=$B/dx/$SLUG/jobs; LOG=$B/dx/$SLUG/logs; mkdir -p $LOG $CACHE
if [ "$MODE" = plans ]; then
  exec $PY $MPD/run_dermarena_v1_medagentpro.py --print-plans ${MP_N_PLANS:-10} --seed ${MP_PLAN_SEED:-0} --plan-cache $CACHE
fi
case $MODE in
  canary)   IN="$B/dx/canary20_{task}.jsonl"; OUT="$B/dx/$SLUG/${MP_CANARY_DIR:-canary}/{task}"; THREADS=${MP_CANARY_THREADS:-5} ;;
  full)     IN="$R/tasks/{task_dir}/test.jsonl"; OUT="$B/dx/$SLUG/{task}" ;;
  rank-stability) IN=""; OUT="$B/dx/$SLUG/${MP_CANARY_DIR:-canary}/{task}" ;;
  service|kb-index) IN=""; OUT="" ;;
  *) echo "usage: $0 kb-index|canary|plans|rank-stability|full|service"; exit 2 ;;
esac
export MP_STOP_FILE=$B/dx/$SLUG/STOP_$MODE; rm -f $MP_STOP_FILE   # touch it -> workers skip remaining cases and exit

export OPENAI_BASE_URL=${OPENROUTER_BASE_URL:?no server} OPENAI_API_KEY=local
export HF_HOME=$HC HUGGINGFACE_HUB_CACHE=$HC/hub TRANSFORMERS_CACHE=$HC/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MP_MODEL=$BACKBONE MP_SLUG=$SLUG MP_MAX_TOKENS=${MP_MAX_TOKENS:-32768}   # = base arm budget per call
export MP_EOS_RETRIES=${MP_EOS_RETRIES:-2}             # immediate-EOS blank re-issued <=2x (= base arm dx_infer)
export MP_REASONING_FALLBACK=${MP_REASONING_FALLBACK:-1}   # blank content + stop + reasoning>20 chars -> reasoning (= base arm)
export MP_CAND_ATTEMPTS=${MP_CAND_ATTEMPTS:-2}         # top-10 differential: asked <=2x; none -> unparsed = wrong
export MP_PLAN_ATTEMPTS=${MP_PLAN_ATTEMPTS:-5}         # planner re-asked when the validator or post-check rejects the plan
export MP_DECIDER_ATTEMPTS=${MP_DECIDER_ATTEMPTS:-2}   # Pro_Decider re-asked on unusable JSON; then that candidate has no score
export MP_CALL_BUDGET=${MP_CALL_BUDGET:-150}           # per-case LLM calls: exceeding it is logged and flagged, nothing is cut
export MP_SANITIZE_SPECIAL_TOKENS=${MP_SANITIZE_SPECIAL_TOKENS:-1}   # literal <|image_pad|> in text = HTTP 500 on the server
export MP_TIMEOUT=${MP_TIMEOUT:-3600} MP_HTTP_RETRIES=${MP_HTTP_RETRIES:-2}
export DX_IMAGE_ROOT=$R DERMAGENT_IMG_CACHE=/tmp/dermagent_rgb_cache TOKENIZERS_PARALLELISM=false
[ -n "${MP_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF=$MP_ALLOC_CONF
# Creator sampling from the benchmark's own registry -> identical to the base-model arm (owner: backbone creator settings).
export MP_SAMPLING_JSON=$(cd $DC && python3 - "$BACKBONE" <<'PY'
import sys, json; sys.path.insert(0, "eval"); from sampling_profiles import resolve
p = resolve(sys.argv[1]); print(json.dumps({k: v for k, v in p.items() if not k.startswith("_")}))
PY
)
[ -n "$MP_SAMPLING_JSON" ] || { echo "no sampling profile"; exit 2; }
echo "[medagentpro] $MODE $(date) | backbone $BACKBONE @ $OPENAI_BASE_URL | sampling $MP_SAMPLING_JSON | max_tokens $MP_MAX_TOKENS"

if [ "$MODE" = kb-index ]; then
  # summaries retries inside the step; a document still missing does not stop the embedding, and gets one more round after it
  $PY $MPD/kb_index_build.py summaries --kb $KB --backbone "$BACKBONE" --workers ${MP_KB_WORKERS:-36} 2>&1 | tee -a $LOG/kb_index.log
  src=${PIPESTATUS[0]}
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY $MPD/kb_index_build.py embed --kb $KB --batch ${MP_EMBED_BATCH:-32} 2>&1 | tee -a $LOG/kb_index.log
  [ "$src" = 0 ] || $PY $MPD/kb_index_build.py summaries --kb $KB --backbone "$BACKBONE" --workers ${MP_KB_WORKERS:-36} 2>&1 | tee -a $LOG/kb_index.log
  $PY $MPD/kb_index_build.py finalize --kb $KB --backbone "$BACKBONE" 2>&1 | tee -a $LOG/kb_index.log
  rc=${PIPESTATUS[0]}; echo "MEDAGENTPRO_KB_INDEX_DONE rc=$rc $SLUG $(date)"; exit $rc
fi
RUN="$PY $MPD/run_dermarena_v1_medagentpro.py --kb $KB --plan-cache $CACHE --backbone $BACKBONE --slug $SLUG"
if [ "$MODE" = rank-stability ]; then
  $RUN --rank-stability ${MP_RS_REPEATS:-3} --tasks T1.1,T1.2 --out-tmpl "$OUT" --threads ${MP_RS_THREADS:-32} 2>&1 | tee -a $LOG/rank_stability.log
  rc=${PIPESTATUS[0]}; echo "MEDAGENTPRO_RANK_STABILITY_DONE rc=$rc $SLUG $(date)"; exit $rc
fi
GATE=""; [ "$MODE" = canary ] && GATE=--gate
REV=""; [ "$MODE" != canary ] && REV=--require-reviewed     # the runner knows the namespace: <cache>/<namespace>/REVIEWED

pids=()
for ((p=0; p<NPROC; p++)); do
  if [ "$MODE" = service ]; then TARGS=(--service "$JD"); else TARGS=(--tasks T1.1,T1.2 --input-tmpl "$IN" --out-tmpl "$OUT"); fi
  CUDA_VISIBLE_DEVICES=${GPUS[$p]} $RUN $REV "${TARGS[@]}" --shard $p/$NPROC --threads $THREADS >> $LOG/${MODE}_p$p.log 2>&1 &
  pids+=($!)
  sleep 10   # stagger tool loading
done
rc=0; for x in "${pids[@]}"; do wait "$x" || rc=$?; done
echo "[medagentpro] workers exited rc=$rc $(date)"
if [ "$MODE" = service ]; then echo "MEDAGENTPRO_SERVICE_DONE $SLUG $(date)"; exit $rc; fi
$RUN --merge --tasks T1.1,T1.2 --out-tmpl "$OUT"
for t in T1.1 T1.2; do
  d=${OUT//\{task\}/$t}; inp=${IN//\{task\}/$t}; inp=${inp//\{task_dir\}/${F[$t]}}
  python3 $DX/dx_check.py post "$d" $t "$inp" | tee $d/POSTCHECK.txt
  $PY $MPD/mp_postcheck.py "$d" $t --n-input $(wc -l < "$inp") $GATE | tee -a $d/POSTCHECK.txt
  g=${PIPESTATUS[0]}; [ "$g" = 0 ] || rc=$g
done
grep -h 'Traceback\|Error' $LOG/${MODE}_p*.log | sort | uniq -c | sort -rn | head -15
echo "MEDAGENTPRO_${MODE^^}_DONE $SLUG $(date)"
exit $rc
