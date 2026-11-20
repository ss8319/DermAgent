#!/bin/bash
# wait_job_end.sh <jobdir> <job-stem> [timeout_s=86400] [poll_s=30]: bounded wait for a service job's end marker
# (written by shard 0 in run_dermarena_v1.finish_job). Use this in waiters instead of `until [ -f ALLDONE ]` (PR #4 review
# F3: a job with an incomplete shard never gets ALLDONE, so such loops waited forever).
# rc 0 = <stem>.ALLDONE (every shard complete), 2 = <stem>.ENDED_INCOMPLETE (merged; see <stem>.p<k>.incomplete), 3 = timeout.
J=${1:?jobdir}; S=${2:?job stem}; T=${3:-86400}; P=${4:-30}; t0=$(date +%s)
while :; do
  [ -f "$J/$S.ALLDONE" ] && exit 0
  [ -f "$J/$S.ENDED_INCOMPLETE" ] && exit 2
  [ $(( $(date +%s) - t0 )) -ge "$T" ] && exit 3
  sleep "$P"
done
