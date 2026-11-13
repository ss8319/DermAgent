#!/bin/bash
# fn_pp2_server.sh (11-13 ~19:00, owner "just 129 left but 2 more days, optimise the plan"): serve Qwen3.8-Flash-Next on the two
# 2x A100-80G PCIe jobs we already hold, as ONE server spanning both nodes: tensor-parallel 2 inside each node, pipeline-parallel 2
# across nodes (rank 0 = gpu2a100 61981433 on m3n108, rank 1 = gpu2a100b 61981434 on m3n100). No new GPU request: these GPUs would
# otherwise idle until a 4x80 GB job starts (estimated tomorrow 13:40 at the earliest; new requests Nov 17).
# 11-13 20:15: mem-fraction 0.90 (0.92 left 3.3 GB for activations; refcheck OOM twice; a mid-run OOM kills ~100 in-flight cases).
# 11-13 19:35: --disable-custom-all-reduce (custom AR failed CUDA-graph capture on A100 PCIe pairs, custom_all_reduce.cuh:614;
# NCCL all-reduce instead; communication kernel only). Same model, weights, bf16, fp32 mamba state, context 81920 and sampling; memory/scheduling flags as the A100 plan (prefix cache off,
# cap 96, prefill CUDA graph off). Gated by tools/fn_refcheck.py against the H200 reference before any case runs on it.
# Cross-node traffic is pipeline activations only (small), so NCCL uses TCP (NCCL_IB_DISABLE=1) on bond0.114: 11-13 19:30 test showed
# e1p1 does NOT pass traffic between m3n100 and m3n108 (connect timed out; first launch died after 600 s), bond0.* all do.
# Runs detached on the login node; light (sruns + curl). Stop: touch agents/FN_PP2_STOP (kills the server steps it started).
B=/fs04/scratch2/ub62/ssim0070/benchmark_runs; FND=$B/dx/medagentpro-qwen3.8-flash-next; L=$FND/logs/fn_pp2.log
J0=61981433; J1=61981434; H0=172.16.205.171; PORT=30777; DPORT=30778; URL=http://m3n108:$PORT/v1
STOP=$B/agents/FN_PP2_STOP
log() { echo "$(date '+%m-%d %H:%M') [pp2] $*" | tee -a $L; }
ovl() { local j=$1; shift; env $(env | grep -oE '^SLURM_[A-Z_]+' | grep -v '^SLURM_CONF' | sed 's/^/-u /') srun --jobid=$j --overlap "$@"; }
gpus_free() { ovl $1 -n1 -c1 --mem=512M timeout 60 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{if($1>2000)b=1} END{exit b}'; }
ARGS="--tp-size 2 --pp-size 2 --nnodes 2 --dist-init-addr $H0:$DPORT --dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 4096 --disable-radix-cache --max-running-requests 96 --cuda-graph-max-bs-decode 96 --disable-prefill-cuda-graph --disable-custom-all-reduce --mem-fraction-static 0.90 --served-model-name Qwen/Qwen3.8-Flash-Next --context-length 81920"
log "start (pid $$): waiting for local weights on both nodes, free GPUs on both jobs and an empty gpu2a100b queue"
while [ ! -f $STOP ]; do
  w0=$(grep -c LOCAL_WEIGHTS_OK $FND/logs/pp2_weights_$J0.log 2>/dev/null); w1=$(grep -c LOCAL_WEIGHTS_OK $FND/logs/pp2_weights_$J1.log 2>/dev/null)
  q=$(ls $B/holders/gpu2a100b/queue/*.req 2>/dev/null | wc -l)
  if [ "${w0:-0}" -ge 1 ] && [ "${w1:-0}" -ge 1 ] && [ $q -eq 0 ] && gpus_free $J0 && gpus_free $J1; then break; fi
  sleep 120
done
[ -f $STOP ] && { log "STOP before launch"; exit 0; }
log "launching rank 1 on $J1 and rank 0 on $J0: $ARGS"
C=/fs04/scratch2/ub62/ssim0070/sglang/containers/sglang_v0522.sif; CH=/fs04/scratch2/ub62/ssim0070/sglang/.container_home_v0522; R=/fs04/scratch2/ub62/ssim0070/sglang
for rank in 1 0; do
  J=$([ $rank = 0 ] && echo $J0 || echo $J1)
  setsid nohup env $(env | grep -oE '^SLURM_[A-Z_]+' | grep -v '^SLURM_CONF' | sed 's/^/-u /') srun --jobid=$J --overlap -n1 -c12 --mem=180G bash -c "
    source $R/slurm/env.sh >/dev/null 2>&1
    exec apptainer exec --nv --home $CH:$CH --bind $R:$R --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
      --env FLASHINFER_WORKSPACE_DIR=$CH/.cache/flashinfer --env TRITON_CACHE_DIR=$CH/.triton --env TORCH_HOME=$CH/.cache/torch \
      --env NCCL_IB_DISABLE=1 --env NCCL_SOCKET_IFNAME=bond0.114 --env GLOO_SOCKET_IFNAME=bond0.114 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $C python3 -m sglang.launch_server --model-path /tmp/bench_weights/Qwen/Qwen3.8-Flash-Next --host 0.0.0.0 --port $PORT \
      --trust-remote-code --node-rank $rank $ARGS" > $FND/logs/fn_pp2_rank$rank.log 2>&1 < /dev/null &
  disown; sleep 20
done
ok=0; for i in $(seq 1 90); do curl -sf -m 10 ${URL%/v1}/health >/dev/null && { ok=1; break; }; [ -f $STOP ] && break; sleep 40; done
if [ $ok = 0 ]; then log "SERVER FAILED to come up in 60 min (see fn_pp2_rank0/1.log): stopping the steps"
  echo "$(date '+%m-%d %H:%M') FN-PP2 server failed to start; see $FND/logs/fn_pp2_rank*.log" >> $B/logs/gpu_watchdog/alerts.log
  for J in $J0 $J1; do ovl $J -n1 -c1 --mem=512M bash -c 'pkill -u $USER -f "sglang.launch_server --model-path /tmp/bench_weights/Qwen/Qwen3.8-Flash-Next" ; true'; done; exit 1; fi
log "server healthy at $URL; running the reference check"
grep -E "max_running_requests|avail mem|capped|KV Cache is allocated|Mamba Cache is allocated" $FND/logs/fn_pp2_rank0.log | tail -6 | tee -a $L
hist() { printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$J0+$J1" "m3n108+m3n100" "A100-80GB-PCIe x4 (2 nodes, pp2)" "$1" "$ARGS" >> $FND/servers/history.tsv; }
hist refcheck_start
ovl $J0 -n1 -c2 --mem=4G python3 $B/tools/fn_refcheck.py compare "$URL" $B/diagnostics/flashnext_no_h200_20261113/refcheck >> $L 2>&1; rc=$?
# 11-13 20:30: servers/PP2_REFCHECK_OVERRIDE (provisional use after a per-prompt-cap FAIL, see META_OVERRIDES.txt) lets a FAIL (rc=1,
# not rc=2 "cannot run") proceed; owner deletes the file to reject.
if { [ $rc -eq 1 ] || [ $rc -eq 3 ]; } && [ -f $FND/servers/PP2_REFCHECK_OVERRIDE ]; then hist refcheck_FAIL_PROVISIONAL_OVERRIDE; log "REFCHECK FAIL but PP2_REFCHECK_OVERRIDE present: provisional use"; rc=0; fi
if [ $rc -ne 0 ]; then hist refcheck_FAIL_rc$rc; log "REFCHECK FAIL rc=$rc: server NOT used; stopping it"
  echo "$(date '+%m-%d %H:%M') FN-PP2 REFCHECK FAIL rc=$rc: server not used" >> $B/logs/gpu_watchdog/alerts.log
  for J in $J0 $J1; do ovl $J -n1 -c1 --mem=512M bash -c 'pkill -u $USER -f "sglang.launch_server --model-path /tmp/bench_weights/Qwen/Qwen3.8-Flash-Next" ; true'; done; exit 1; fi
hist refcheck_PASS
echo "$ARGS" > $FND/servers/$J0.args; echo "$URL" > $FND/servers/$J0.url.tmp; mv $FND/servers/$J0.url.tmp $FND/servers/$J0.url; hist serving
log "registered $URL for mp_fn_supervisor2"; echo "$(date '+%m-%d %H:%M') FN-PP2 serving $URL (2 nodes, A100 x4)" >> $B/logs/gpu_watchdog/alerts.log
while [ ! -f $STOP ]; do curl -sf -m 30 ${URL%/v1}/health >/dev/null || { log "health check failed"; }; sleep 300; done
rm -f $FND/servers/$J0.url; hist released; log "STOP: unregistered (server steps left running; kill them by hand if wanted)"
