# DermArena v1 agent runs — operating scripts

Copies of the live scripts under `benchmark_runs/agents/` on M3, which is not a git repo. They drive the agent code in this repo
(`baselines/MedAgent-Pro/run_dermarena_v1_medagentpro.py`, `scripts/run_dermarena_v1.py`) on M3 GPU holders.
Refresh with `bash benchmark_runs/tools/sync_agents_to_repo.sh` (a copy is in this folder), then commit.

- `agents/run_medagentpro_v2.sh`, `run_medagentpro_remote_v2.sh`: MedAgent-Pro runner (shards, tools on another job's GPU)
- `agents/run_dermagent_v1.sh`, `run_dermagent_remote.sh`: DermAgent runner
- `agents/mp_qwen_supervisor_v5.sh`, `mp_fn_supervisor.sh`, `mp_take_shard.sh`, `mp_hosts.sh`, `mp_shard_lock.sh`: restart / placement / drain
- `agents/medagentpro_reqs/`, `agents/ablation_reqs_h200_v2_20261110/`: holder requests (`bench_holder.sbatch` file queue)
- `agents/DERMAGENT_REPRO_SPEC.md`: repro spec + dated run log
