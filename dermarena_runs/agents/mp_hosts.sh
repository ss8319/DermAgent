# mp_hosts.sh — shared helpers (source it) for MedAgent-Pro runners on ANY tools host (11-10).
# Why: every runner hosts its tools in-process on a GPU of a "tools host" job. The only host was the L40S job 61714874, which ends
# 12 Nov 11:52 (7-day limit). Hosts now come from agents/mp_tool_hosts.txt (one Slurm job id per line, read on every call, so a new
# host can be added while things run). A host is used only while RUNNING; the longest-lived host is preferred.
# Needs: SLUG (e.g. medagentpro-qwen3.5-27b). Every srun here is a tiny overlap step inside the host job.
HB=/fs04/scratch2/ub62/ssim0070/benchmark_runs
HOSTS_FILE=$HB/agents/mp_tool_hosts.txt
TOOL_MIB=${TOOL_MIB:-23000}   # free GPU memory one runner's tools need (they settle at ~19.9 GB)

live_hosts() {   # running host job ids, latest end time first
  local j; for j in $(grep -oE '^[0-9]+' $HOSTS_FILE 2>/dev/null); do squeue -h -j $j -t R -o "%e %i" 2>/dev/null; done | sort -r | awk '{print $2}'
}
runners_on() {   # $1 host job -> "pgid shard gpu url" for every $SLUG runner on it
  SL=$SLUG srun --jobid=$1 --overlap -n1 -c1 --mem=1G bash -c 'echo OK; for p in $(pgrep -f "[r]un_dermarena_v1_medagentpro.py"); do
      a=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null); echo "$a" | grep -qF "/$SL/" || continue
      s=$(echo "$a" | grep -oE -- "--shard [0-9]+/" | grep -oE "[0-9]+"); e=$(tr "\0" "\n" < /proc/$p/environ 2>/dev/null)
      echo "$(ps -o pgid= -p $p | tr -d " ") $s $(echo "$e" | grep "^CUDA_VISIBLE_DEVICES=" | cut -d= -f2) $(echo "$e" | grep "^OPENROUTER_BASE_URL=" | cut -d= -f2-)"
    done' 2>/dev/null
}
all_runners() {  # "host pgid shard gpu url" over all live hosts; returns 1 if any live host could not be inspected
  local j o; for j in $(live_hosts); do o=$(runners_on $j); echo "$o" | grep -q '^OK$' || return 1; echo "$o" | grep -v '^OK$' | grep . | sed "s/^/$j /"; done
}
free_slots() {   # $1 = current runner list ("host pgid shard gpu url"); prints "host gpu freeMiB" for GPUs that can take one more runner:
  # free >= TOOL_MIB AND fewer runners on that GPU than total/21000 (a runner still loading its tools uses less memory than it will)
  local j; for j in $(live_hosts); do
    srun --jobid=$j --overlap -n1 -c1 --mem=1G nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null \
      | tr -d ' ' | while IFS=, read g f t; do
          n=$(echo "$1" | awk -v j=$j -v g=$g '$1==j && $4==g' | wc -l)
          [ "$f" -ge $TOOL_MIB ] && [ $n -lt $((t / 21000)) ] && echo "$j $g $f"
        done | sort -k3,3nr
  done
}
qwen_servers() { # live Qwen3.5-27B backbone URLs (A100 holders, H100 holder's two servers)
  local h e u
  for h in gpu2a100 gpu2a100b gpu2a100c gpu2a100d h100x4c; do
    e=$(awk '{print $1}' $HB/holders/$h/ENDPOINT 2>/dev/null); [ -n "$e" ] || continue
    for u in $e $(echo $e | awk -F'[:/]' '{printf "http://%s:%d/v1", $4, $5+1}'); do
      curl -sf -m 15 "$u/models" 2>/dev/null | grep -q '"Qwen/Qwen3.5-27B"' && echo $u
    done
  done
}
