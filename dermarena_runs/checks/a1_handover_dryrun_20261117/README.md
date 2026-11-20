# A1 handover dry-run (2026-11-17)
Purpose: before the fit 4xA100 job (62728289) starts, run the REAL holder loop (sglang/slurm/bench_holder.sbatch, as stored in the
pending job) over the REAL request files of its queue, with only the GPU-facing calls faked. Usage: `bash sim.sh <scenario>` (A-E).
Faked: nvidia-smi / pgrep (bin/), apptainer, the server start (records model + args, fails for an empty or non-local model),
the benchmark and judge commands (`bash ...` inside run(): prints the command, writes the DONE marker the scenario asks for),
sleep and `date +%s` (a fake clock), squeue only where a scenario says so. Real: request files, lock.sh, local_weights.sh, the
queue loop, `set -uo pipefail`, done/ and failed/ moves, the A40 job's real lock (62749880).
The only edits to the copied requests: the path of _a1_common.sh, and the sandbox runs.log is added to the marker greps (a marker
written under benchmark_runs/holders/ would be read by the live jobs).
