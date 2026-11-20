#!/bin/bash
# runner processes of Qwen shards 2/4 and 3/4 only (not Flash-Next, not anything else)
P=$(ps -eo pid,args | grep "[r]un_dermarena_v1_medagentpro.py" | grep "medagentpro-qwen3.5-27b" | grep -E -- "--shard (2|3)/4" | awk '{print $1}')
echo "killing: $P"; ps -o pid,etime,args -p $(echo $P | tr ' ' ',') | cut -c1-150
[ -n "$P" ] && kill $P
sleep 10; ps -eo pid,args | grep "[r]un_dermarena_v1_medagentpro.py" | grep -oE "medagentpro-qwen[0-9.a-z-]*/\{task\}|--shard [0-9]/4" | paste - - | sort | uniq -c
