#!/bin/bash
# DermAgent (open-set LangGraph agent, 7 local tools) on DermArena v1 T1.1 + T1.2 test, inside an H200 bench holder.
# The backbone is the holder's SGLang server (OPENROUTER_BASE_URL; .req serves it on GPUs 0..DP-1); the tools run in
# DA_NPROC processes on the other GPUs (DA_TOOL_GPUS), each with DA_THREADS cases in flight and its own copy of the
# embedded Qdrant stores on node-local /tmp (embedded Qdrant takes an exclusive dir lock).
# Usage (from a .req run()):  run_dermagent_v1.sh canary|full
# Outputs: benchmark_runs/dx/$DA_SLUG/{canary/,}T1.x/{predictions.jsonl,predictions.p<k>.jsonl,traces/,RUN_META.json}
set -uo pipefail
MODE=${1:-full}
NPROC=${DA_NPROC:-4}; THREADS=${DA_THREADS:-12}; read -r -a GPUS <<< "${DA_TOOL_GPUS:-2 2 3 3}"
SLUG=${DA_SLUG:-dermagent-qwen3.5-27b}; BACKBONE=${DA_BACKBONE:-Qwen/Qwen3.5-27B}
AG=/fs04/scratch2/ub62/ssim0070/DermAgent; PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python3
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; R=/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1
DC=/fs04/scratch2/ub62/ssim0070/DermArena/dataset_collection; DX=$DC/dx_benchmark; HC=/fs04/scratch2/ub62/ssim0070/sglang/hf_cache
declare -A F=([T1.1]=T1.1_disease_screening [T1.2]=T1.2_disease_confirmation)
JD=$B/dx/$SLUG/jobs   # service mode: job files (*.json) + STOP live here
if [ "$MODE" = service ]; then
  IN=""; OUT=""
elif [ "$MODE" = canary ]; then
  IN="$B/dx/canary20_{task}.jsonl"; OUT="$B/dx/$SLUG/${DA_CANARY_DIR:-canary}/{task}"; THREADS=${DA_CANARY_THREADS:-5}
else
  IN="$R/tasks/{task_dir}/test.jsonl"; OUT="$B/dx/$SLUG/{task}"
fi
LOG=$B/dx/$SLUG/logs; mkdir -p $LOG
# Operator stop (no SSH needed): touch $B/dx/$SLUG/STOP_$MODE -> workers skip remaining cases and exit; holder moves on.
export DERMAGENT_STOP_FILE=$B/dx/$SLUG/STOP_$MODE; rm -f $DERMAGENT_STOP_FILE

export OPENAI_BASE_URL=${OPENROUTER_BASE_URL:?no server} OPENAI_API_KEY=local
export HF_HOME=$HC HUGGINGFACE_HUB_CACHE=$HC/hub TRANSFORMERS_CACHE=$HC/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DERMAGENT_BACKBONE=$BACKBONE DERMAGENT_SLUG=$SLUG DERMAGENT_MAX_TOKENS=${DA_MAX_TOKENS:-32768}
# Upstream headline Dx configuration (YizeezLiu/DermAgent scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh):
# 6 tools (no qwen_vqa), Critic on + image re-injection, 2 critic retries, LLM summary off, agent defaults
# max_tool_calls 10 / recursion_limit 30; prompt = upstream DERMOGPT_FULL with only the closed-set lines made open-set.
export ENABLED_TOOLS=${DA_TOOLS:-panderm,make,dermogpt_vqa,rag,text_rag,ontology}
export DERMAGENT_OPEN_PROMPT=${DA_OPEN_PROMPT:-upstream_full} DERMAGENT_ENABLE_CRITIC=${DA_CRITIC:-1}
export DERMAGENT_CRITIC_IMAGE_REINJECT=${DA_REINJECT:-1} DERMAGENT_CRITIC_LLM_SUMMARY=0 DERMAGENT_MAX_CRITIC_RETRIES=2
export MAX_TOOL_CALLS=${DA_MAX_TOOL_CALLS:-20} RECURSION_LIMIT=${DA_RECURSION:-100}   # owner 11-05: 30 cut 3/973 cases (p99 29 steps); 100 leaves room for cap-20 jobs (~2 steps per tool call); only a safety net now that the 10-call cap + cap-final bound the loop
# Immediate-EOS blanks from the backbone are re-issued (= base-arm dx_infer policy), max 2 per call.
export DERMAGENT_EOS_RETRIES=${DA_EOS_RETRIES:-2}
# Base-arm answer rule (llm_client._answer_from): blank content + stop + reasoning > 20 chars -> reasoning is the answer.
export DERMAGENT_REASONING_FALLBACK=1
# Qwen3.5 creator agentic setup: pass each tool-loop step's reasoning back (the template keeps it within the loop).
export DERMAGENT_PASS_REASONING=${DA_PASS_REASONING:-1}
# Re-issue (<=2, same budget as above) a turn that stopped inside unclosed thinking with no tool call and no FINAL_ANSWER.
export DERMAGENT_REISSUE_INTERRUPTED=${DA_REISSUE_INTERRUPTED:-1}
export DERMAGENT_CAP_FINAL=${DA_CAP_FINAL:-1}   # owner 11-05 option A: at the 10-call cap, one no-tool FINAL_ANSWER turn
export DERMAGENT_SANITIZE_SPECIAL_TOKENS=${DA_SANITIZE_MM:-1}   # 11-06: literal <|image_pad|> in text 500s the Flash-Next server
export DERMAGENT_STRIP_NOTOOL_TOOLCALLS=${DA_STRIP_NOTOOL:-0}   # 11-06: strip <tool_call> text from the no-tools analysis node (Flash-Next stall fix)
export DERMAGENT_THINK_TOOLCALL_EXTRACT=${DA_THINK_TC:-1}   # owner 11-05: run a complete tool call found at the end of the thinking instead of re-sampling
[ -n "${DA_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF=$DA_ALLOC_CONF   # tools on 48 GB GPUs: e.g. expandable_segments:True
# Owner 2026-11-05: backbone input = the base arm's (ALL images first, then case + tables); upstream run_diagnosis
# user-message template (closed-set lines made open-set). (Upstream itself: include_image_in_message=False.)
export DERMAGENT_ATTACH_IMAGES=${DA_ATTACH_IMAGES:-1} DERMAGENT_USER_MSG=${DA_USER_MSG:-upstream}
# Qwen3.5 template rejects non-leading system messages; the Critic's feedback (a mid-conversation SystemMessage
# upstream) is sent as a user message with identical content.
export DERMAGENT_MIDCONV_SYSTEM_AS_USER=1
# Owner 2026-11-05: image tools (PanDerm, MAKE, DermoGPT, Case-RAG) take clinical photos / dermoscopy only: only those
# get a tool id; every image line shows its type (release 'modality'), e.g. "image 2: IMG_x (dermoscopy)",
# "image 3: histopathology (not available to the image tools)".
export DERMAGENT_IMAGE_MODALITY_TAGS=${DA_MODALITY_TAGS:-1} DERMAGENT_TOOL_IMAGES=${DA_TOOL_IMAGES:-derm}
# Owner 2026-11-05 critic adaptations: open-set "Focused Re-diagnosis" shortlist (tool outputs, pooled over all images)
# and re-showing ALL clinical/dermoscopy images on retry ("Re-analyze this case and these images ...").
export DERMAGENT_OPEN_SHORTLIST=${DA_OPEN_SHORTLIST:-1} DERMAGENT_REINJECT_ALL_DERM=${DA_REINJECT_ALL_DERM:-1}
export DX_IMAGE_ROOT=$R DERMAGENT_IMG_CACHE=/tmp/dermagent_rgb_cache TOKENIZERS_PARALLELISM=false
# Creator sampling, read from the benchmark's own registry -> identical to the base-model arm.
export DERMAGENT_SAMPLING_JSON=$(cd $DC && python3 - "$BACKBONE" <<'PY'
import sys, json; sys.path.insert(0, "eval"); from sampling_profiles import resolve
p = resolve(sys.argv[1]); print(json.dumps({k: v for k, v in p.items() if not k.startswith("_")}))
PY
)
echo "[dermagent] $MODE $(date) | backbone $BACKBONE @ $OPENAI_BASE_URL | sampling $DERMAGENT_SAMPLING_JSON | max_tokens $DERMAGENT_MAX_TOKENS"
echo "[dermagent] $NPROC procs x $THREADS threads on GPUs ${GPUS[*]} | tools $ENABLED_TOOLS | prompt $DERMAGENT_OPEN_PROMPT | critic $DERMAGENT_ENABLE_CRITIC reinject $DERMAGENT_CRITIC_IMAGE_REINJECT | max_tool_calls $MAX_TOOL_CALLS recursion $RECURSION_LIMIT"
[ -n "$DERMAGENT_SAMPLING_JSON" ] || { echo "no sampling profile"; exit 2; }

# Warm the node page cache with the tool weights ONCE (parallel reads), so the NPROC workers load from RAM instead of
# each pulling ~50 GB from Lustre (061: 4 workers in lockstep took ~25 min). No code/path change.
t0=$(date +%s)
for d in $AG/model-weights/DermoGPT-RL/ $AG/model-weights/models--Qwen--Qwen3-Embedding-8B/ \
         $AG/model-weights/models--Qwen--Qwen3-Reranker-0.6B/ $HC/hub/models--redlessone--DermLIP_PanDerm-base-w-PubMed-256/ \
         $HC/hub/models--xieji-x--MAKE/ $HC/hub/models--neuml--pubmedbert-base-embeddings/; do
  find -L "$d" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.json' \) -print0 2>/dev/null \
    | xargs -0 -P 8 -I{} sh -c 'cat "{}" > /dev/null'
done
echo "[dermagent] page cache warmed in $(( $(date +%s) - t0 ))s"

QSRC=${DA_QDRANT_SRC:-$AG/qdrant_storage_dermarena_clean}; QEXPECT=${DA_QDRANT_EXPECT:-410977}; QIDX=$(basename $QSRC)
[ -d "$QSRC" ] || { echo "no Case-RAG index at $QSRC"; exit 2; }
export DERMAGENT_CASERAG_INDEX=$QIDX:$QEXPECT
echo "[dermagent] Case-RAG index $QIDX ($QEXPECT points)"
pids=()
for ((p=0; p<NPROC; p++)); do
  # 11-07 owner: Case-RAG index = Derm1M minus DermArena test+val articles (2,233 items). The /tmp copy is named after the index,
  # so an older cached copy (/tmp/dermagent_qdrant_p*) can never be picked up; the row count is checked before every start.
  Q=/tmp/dermagent_qdrant_${QIDX}_p$p
  if [ ! -d $Q/img/collection ]; then
    mkdir -p $Q && cp -r $QSRC $Q/img && cp -r $AG/qdrant_storage_text $Q/text || { echo "qdrant copy failed"; exit 2; }
  fi
  rm -f $Q/img/.lock $Q/text/.lock
  QN=$($PY -I -c "import sqlite3,sys;print(sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True).execute('select count(*) from points').fetchone()[0])" $Q/img/collection/derm1m/storage.sqlite)
  [ "$QN" = "$QEXPECT" ] || { echo "CASE-RAG INDEX CHECK FAILED: $Q has $QN points, expected $QEXPECT"; exit 2; }
  if [ "$MODE" = service ]; then TARGS=(--service "$JD"); else TARGS=(--tasks T1.1,T1.2 --input-tmpl "$IN" --out-tmpl "$OUT"); fi
  CUDA_VISIBLE_DEVICES=${GPUS[$p]} QDRANT_PATH=$Q/img QDRANT_TEXT_PATH=$Q/text \
    $PY $AG/scripts/run_dermarena_v1.py "${TARGS[@]}" \
      --shard $p/$NPROC --threads $THREADS --qdrant-path $Q/img --backbone $BACKBONE --slug $SLUG --enabled-tools $ENABLED_TOOLS \
      >> $LOG/${MODE}_p$p.log 2>&1 &
  pids+=($!)
  sleep 20   # stagger tool loading (Lustre reads, CPU RAM peak)
done
rc=0; for x in "${pids[@]}"; do wait "$x" || rc=$?; done
echo "[dermagent] workers exited rc=$rc $(date)"
if [ "$MODE" = service ]; then echo "DERMAGENT_SERVICE_DONE $SLUG $(date)"; exit $rc; fi   # worker 0 merged + postchecked each job
$PY $AG/scripts/run_dermarena_v1.py --merge --tasks T1.1,T1.2 --out-tmpl "$OUT"
for t in T1.1 T1.2; do
  d=${OUT//\{task\}/$t}; inp=${IN//\{task\}/$t}; inp=${inp//\{task_dir\}/${F[$t]}}
  python3 $DX/dx_check.py post "$d" $t "$inp"
done
grep -h 'Traceback\|Error' $LOG/${MODE}_p*.log | sort | uniq -c | sort -rn | head -15
echo "DERMAGENT_${MODE^^}_DONE $SLUG $(date)"
exit $rc
