# Brief: independent review of DermAgent PR #4 (for a fresh agent)

Written 2026-11-17 by the run session. Launch only after MedAgent-Pro x Qwen3.8-Flash-Next T1.2 has finished (all rows written and
judged): PR #4 changes the runner that T1.2 uses, so it must not reach a live run.

## What PR #4 is
- Repo: `/fs04/scratch2/ub62/ssim0070/DermAgent` (GitHub `ss8319/DermAgent`, remote name `fork`). PR: https://github.com/ss8319/DermAgent/pull/4
- Head `pr3-review-fixes` -> base `reproduction-fixes` (itself PR #3 -> `master`). Title: "Fix 4 review findings: merge/image-cache
  races, per-row run_sig, honest .done (merge after current runs)".
- 5 commits, 3 files, mainly `scripts/run_dermarena_v1.py` + new offline tests `scripts/test_run_dermarena_v1_helpers.py`:
  0c257ee merge() race-safe temp file; deba4af rgb_path() race-safe image cache; 9c76a2e resume/merge check run_sig per row;
  47a4228 service mode writes .done only when every planned case has a row; 2ec915c offline tests.
- The base has moved: `reproduction-fixes` has ~24 commits that PR #4 does not contain (as of 11-17).

## What to check (report findings with file:line; verify each before reporting)
1. Correctness of each of the 4 fixes: does it close the race / gap it claims, without a new failure mode (e.g. temp files left
   behind, rename across filesystems, cache key collisions)?
2. **Compatibility with results already on disk (most important).** After the merge, would resuming or merging the EXISTING
   outputs fail or silently change? Check every output folder the runner writes for the finished runs, at least
   `benchmark_runs/dx/medagentpro-*` and `benchmark_runs/dx/dermagent-*`: do all existing rows carry a `run_sig` that the new
   per-row check accepts? Count rows that would be rejected, per folder. A merge must not invalidate finished results.
3. Rebase onto the current `reproduction-fixes`: list conflicts (do it in a scratch worktree, e.g. `git worktree add`; never
   touch the working copies the live runs use, never push).
4. Run the offline tests (`scripts/test_run_dermarena_v1_helpers.py`). If anything imports torch or loads a model, run it
   through `srun` on a CPU allocation, not on the login node.
5. Anything in the diff that would change outputs (prompt, sampling, parsing, scoring): there should be none.

## Rules
- Read-only on everything outside your scratch worktree. Do not merge, do not push, do not comment on GitHub; the owner merges.
- Never edit files under `benchmark_runs/` or any script a running job uses; never touch GPU jobs (no scancel, no srun into them).
- No sub-agents or forks.
- Plain-language report: verdict (merge as is / merge after listed fixes / do not merge), then each finding with evidence,
  then the per-folder compatibility counts from check 2.
