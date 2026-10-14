# Pass-rate benchmark — canonical 200-trial protocol

This directory holds the committed, immutable protocol for the one
quantitative result this project reports about agent pass rate.
Published results live in `benchmarks/results/<experiment-id>/`.

## The protocol

| | |
|---|---|
| Task set | `eval/tasks/` — the 10 frozen search tasks |
| Trials per task per arm | 10, independent |
| Arms | `baseline` (control) vs an evolved candidate |
| Total | 10 × 10 × 2 = **200 task trials** |
| Model | `META_HARNESS_INNER_MODEL`, default `claude-haiku-4-5-20251001` |
| Config | [`config.json`](config.json) |

Both arms run the identical protocol — same tasks, same trial count,
same worker pool, same model — so the measured difference is
attributable to the harness and not to the setup.

## Running it

One command runs the whole sequence — evolve, select, measure, verify,
report — and is the supported way to produce a publishable result:

```bash
uv run meta-harness canonical-experiment --dry-run   # plan + cost estimate, spends nothing
uv run meta-harness canonical-experiment             # THIS COSTS MONEY
```

The stages, and why they are in this order:

1. **Evolve** on `eval/tasks/` with the real proposer.
2. **Select** the best candidate from the validation accuracy the outer
   loop measured *during* evolution. The canonical experiment has not run
   at this point, so its trials cannot influence which candidate they
   then measure. The decision and the table it came from are written to
   `runs/<run>/selection.json`.
3. **Measure** this protocol: fresh, independent trials for both arms.
4. **Generalise** on [`../holdout/`](../holdout/README.md).
5. **Verify** that a recorded trial replays exactly.
6. **Report** — derive the capability-evidence document from the artifacts.

The two halves can still be run separately:

```bash
uv run meta-harness loop --proposer claude --budget 5 --fresh
uv run meta-harness experiment --candidate <name>
```

Useful overrides while smoke-testing (they change the protocol, so the
result is no longer the canonical one — say so if you publish it):

```bash
uv run meta-harness experiment --candidate <name> --trials 2 --workers 2
```

**Cost.** 200 trials on Haiku 4.5 is real spend. Price it from measured
trials already on disk before committing to it:

```bash
uv run meta-harness report cost-estimate
```

That reports `null` rather than a number when there is nothing measured
to extrapolate from. To produce a basis, run one real trial:
`uv run meta-harness inner --task task-001-fix-typo --candidate baseline`.

## What gets written

```
benchmark-results/<experiment-id>/
├── config.json               # the resolved protocol actually run
├── environment.json          # provenance (see below)
├── baseline-results.jsonl    # one JSON object per baseline trial
├── candidate-results.jsonl   # one JSON object per candidate trial
├── validation.json           # methodology checks (see below)
├── summary.json              # derived mechanically from the two JSONL files
├── REPORT.md                 # human-readable, generated
├── traces/{arm}/{task}-trial-{n}/      # per-trial inner-loop traces
└── recordings/{arm}/{task}-trial-{n}/  # execution tapes (--record-trials N)
```

### Raw trial row

```json
{
  "task_id": "task-001-fix-typo",
  "trial": 4,
  "arm": "baseline",
  "passed": true,
  "score": 1.0,
  "llm_calls": 7,
  "input_tokens": 18432,
  "output_tokens": 2104,
  "total_tokens": 20536,
  "cost_usd": 0.0289,
  "wall_time_s": 23.71,
  "metrics_source": "measured",
  "inner_thread_id": "inner::<exp>::<exp>::baseline::task-001-fix-typo::trial-4"
}
```

`cost_usd` is `null` when the model has no configured price. It is never
`0.0` to mean "unknown" — see `META_HARNESS_PRICING` in
`app/meta_harness/metrics.py`.

### Provenance (`environment.json`)

Commit SHA and dirty flag, branch, timestamp, model id, pricing source,
Python version, platform, per-task content hashes (`task.json` plus every
workspace and test file), and the SHA-256 of each arm's harness source.

No environment variables are captured, so an API key cannot leak into a
published artifact.

### Summary

`summary.json` is produced by `app.meta_harness.experiment.summarize`,
whose only inputs are the raw trial rows:

- `baseline_passes` / `baseline_trials` / `baseline_accuracy`
- `candidate_passes` / `candidate_trials` / `candidate_accuracy`
- `absolute_percentage_point_delta`
- `total_trials` and `distinct_tasks` — trials per arm, and how many
  independent evaluation tasks those trials are spread over
- `difference_ci` — 95% Wald interval on the difference in proportions.
  Assumes independent trials, which this design does not have; kept for
  comparison with the naive reading
- `cluster_bootstrap_ci` — 95% percentile interval from a **task-cluster
  bootstrap**, which is the one to quote
- `per_task` breakdown

`summarize()` takes no target, expected value, or override — there is no
code path by which a desired number can reach the output, and a test
asserts that by signature inspection. The runner prints exactly what the
trials say:

```
Baseline  (baseline): NN/100 = NN.N%
Candidate (evolved):  NN/100 = NN.N%
Absolute improvement: +N.N percentage points
95% CI on the difference: [-N.N, +N.N] pp (wald-95; assumes independent trials)
95% CI, clustering by task: [-N.N, +N.N] pp (task-cluster-bootstrap-percentile,
                                             10000 resamples, seed 20260901)
Total trials: 200
Distinct evaluation tasks (clusters): 10
```

*(Shape of the output only — the values are deliberately left as `N`, so
that no concrete figure in this README can be mistaken for a measurement.
The published measurement lives in
`benchmarks/results/<experiment-id>/summary.json`.)*

### Methodology checks (`validation.json`)

A delta is attributable to the harness only if everything else was held
constant. These are derived from the raw rows and published beside them:

| Check | What it rules out |
|---|---|
| `same_task_set` | the arms ran different tasks |
| `same_trials_per_task` | one arm got more attempts |
| `same_model` | the "harness" delta is really a model delta |
| `single_metrics_source` / `measured_only` | a mock trial folded into a measured result |
| `baseline_complete` / `candidate_complete` | missing, duplicated, or malformed trials |

`trial_completeness` names the offending trials rather than reporting a
count. A row missing its outcome is an **unknown**, not a failure —
averaging over it would bias the result. If `identical_protocol` is
false, `REPORT.md` says so above the number and the CLI exits non-zero.

## Scope of a result

- **Trials are clustered within tasks.** The 10 trials of one task are 10
  looks at the same problem, so the independent unit is the task.
  `cluster_bootstrap_ci` resamples **tasks** with replacement, carrying
  every trial of a drawn task with it and drawing the same task into both
  arms (both arms ran the identical task set). It is deterministic under
  the published seed, so anyone can recompute it from the published rows,
  and it describes the tasks in hand.
- **Search-set, not holdout.** This measures the ten tasks the proposer
  optimised against, so whatever it reports is a search-set number and
  must be described as one. Generalisation to unseen tasks is a separate,
  separately-reported measurement:
  [`benchmarks/holdout/`](../holdout/README.md).
- **Selection is upstream of measurement, not downstream.** The candidate
  is chosen on validation numbers before this protocol runs. Choosing it
  on *these* trials would make the delta a selection artifact.
- **Task identity.** A result is only comparable to another result with
  the same task hashes. Changing a task after publishing invalidates the
  comparison; publish a new experiment id instead. Those are *byte*
  hashes, so `.gitattributes` pins the eval trees to LF — without it a
  Windows checkout hashes every frozen task differently from a Linux one
  and the comparison quietly stops meaning anything.
- **Model drift.** The provider's model behind a given id can change.
  `environment.json` records the id and the date; it cannot pin weights.

## Publishing a result

Copy the run directory into `benchmarks/results/<experiment-id>/` and
commit it. Keep `config.json`, `environment.json`, both `*-results.jsonl`
files, `summary.json` and `REPORT.md` — that is everything needed to
independently recompute the headline number. Do **not** commit
`traces/`; it is large and not needed to verify the summary.
