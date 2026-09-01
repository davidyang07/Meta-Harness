# Holdout benchmark — generalisation protocol

The canonical [pass-rate experiment](../pass-rate/README.md) measures the
five tasks the proposer optimised against. That number answers "did the
search work"; it does not answer "does the evolved harness generalise".
This protocol answers the second question, and is reported separately so
the two are never conflated.

## The protocol

| | |
|---|---|
| Task set | `eval/holdout/` — 2 tasks the proposer never sees |
| Trials per task per arm | 20, independent |
| Arms | `baseline` (control) vs the same evolved candidate |
| Total | 2 × 20 × 2 = **80 task trials** |
| Model | identical to the canonical experiment's |
| Config | [`config.json`](config.json) |

Same runner (`experiment.run_two_arm_experiment`), same per-task trial
count, same model, same output shape. The only difference is the task
set, which is what makes the two deltas comparable.

## Why the holdout stays holdout

A holdout number means "generalisation" only while the proposer never
optimised against those tasks. Every channel by which a holdout task
could reach the proposer is closed and tested
(`backend/tests/test_holdout_isolation.py`):

- the search task set the outer loop benchmarks against is `eval/tasks/`;
- the branch artifact directory the proposer's prompt points at holds
  only search-set traces;
- the prompt and appended system prompt mention no holdout task.

`experiment.check_task_set_isolation` re-checks disjointness at run time,
and `meta-harness resume-experiment` refuses to start on overlap.

## Running it

```bash
# as part of the full pipeline (this is the default)
uv run meta-harness resume-experiment

# on its own, against an already-evolved candidate
uv run meta-harness experiment --config benchmarks/holdout/config.json \
    --candidate <name>
```

## Reading the result

**A smaller delta than the search-set experiment is the expected shape of
the result, not a failure.** The harness was evolved against the search
tasks; some of the gain there is fit to those five tasks. The holdout
delta is the part that transferred.

**Two tasks is two clusters.** That is the binding constraint on this
number, and it is worse here than on the search set. The 80 trials are 80
looks at two problems, so `cluster_bootstrap_ci` resamples exactly two
units and its interval is coarse by construction — it can take only a
handful of distinct values. `summary.json` reports `clusters: 2` and
`informative: false` for this reason.

Report the holdout delta with that interval and that cluster count, always.
A holdout delta quoted bare is not a generalisation claim; it is one number
from two tasks. No significance test is computed, because none would be
defensible at this cluster count.

## Output

Identical in shape to the canonical experiment — `config.json`,
`environment.json` (including per-task content hashes), both
`*-results.jsonl` files, `validation.json`, `summary.json` and
`REPORT.md`. See [`../results/README.md`](../results/README.md) for how
to verify a published one.
