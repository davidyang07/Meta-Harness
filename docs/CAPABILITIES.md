# Capability reference

What this system does, the code that implements it, and the command that
validates it. Where a capability is implemented but not yet measured,
this document says so rather than the wording being softened elsewhere.

Status values used below:

- **Implemented and tested** — shipped and covered by an automated test.
- **Implemented — precise wording required** — shipped, but the obvious
  short description would overclaim; the accurate wording is given.
- **Implemented — not yet measured** — the machinery exists, the
  measurement does not.

---

## 1. Self-improving harness

**Status: implemented and tested**

### Implementation

| Piece | File |
|---|---|
| Proposer (spawns the `claude` CLI with SKILL.md appended) | `backend/app/meta_harness/proposer.py` |
| Outer loop that drives propose → validate → benchmark → update_frontier | `backend/app/meta_harness/outer.py` |
| The 11 override points that form the search space | `backend/app/meta_harness/harness.py` |
| Candidate source snapshotting and isolated loading | `backend/app/meta_harness/candidates.py` |
| The proposer's workflow contract | `skills/meta-harness-coding-agent/SKILL.md` |

The proposer reads the branch's own `evolution_summary.jsonl`,
`frontier_val.json` and per-trial traces, then writes a new
`agents/<name>.py` subclassing `CodingAgentHarness`. The 6 inner-loop
tools are a fixed contract it may not change.

### Validation

```bash
# offline: the loop runs, snapshots candidate source, evolves state
cd backend && uv run pytest tests/test_outer.py tests/test_proposer.py -q

# live: the real claude-CLI proposer writes an importable candidate
bash scripts/live_smoke.sh          # section 3
```

Key tests: `test_candidate_source_is_snapshotted_per_branch`,
`test_mock_outer_loop_produces_all_files`,
`test_claude_propose_fails_loudly_when_the_branch_handoff_is_missing`.

### Limitations

The search space is bounded to the 11 override points; a candidate
cannot change the tool contract, and validate-time enforcement rejects
one that tries.

---

## 2. Dual LangGraph execution

**Status: implemented and tested**

### Implementation

**Outer** — `backend/app/meta_harness/outer.py`, `OuterLoopRunner.build()`

```
START → propose → validate → benchmark → update_frontier ─┐
                                  ▲                        │ budget > 0
                                  └────────────────────────┘
                                            └→ END
```

**Inner** — `backend/app/meta_harness/inner.py`, `build_inner_graph()`

```
START → orient → plan → act → verify ─┬→ submit → END
                          ▲           │
                          └───────────┘  tests failed, retries remain
```

Both compile with a checkpointer; the outer loop passes its
`AsyncPostgresSaver` down into every inner trial.

### Validation

```bash
cd backend && uv run pytest tests/test_inner.py tests/test_outer.py \
                            tests/test_inner_checkpoints.py -q
```

`test_inner_graph_compiles_with_and_without_a_checkpointer` and
`test_benchmark_core_threads_the_checkpointer_into_every_trial` cover
the two-machine structure directly.

### Limitations

Both graphs compile without a checkpointer as well, for tests and
offline runs; in that mode nothing is persisted and neither branching
nor recovery is available.

---

## 3. Harness evolution across iterations

**Status: implemented and tested**

### Implementation

Each iteration appends one row to the branch's
`evolution_summary.jsonl` with the candidate, its parent, its measured
scores, and its delta against the measured prior best. Iteration 0 is
the benchmarked baseline, so the first candidate's delta is a real
comparison rather than a comparison against zero.

### Validation

```bash
cd backend && uv run pytest tests/test_outer.py -q
uv run meta-harness loop --proposer mock --mock-bench --budget 2 --fresh --run-name demo
cat runs/demo/threads/demo/evolution_summary.jsonl
```

`test_baseline_is_benchmarked_before_any_candidate` asserts the first
candidate's delta equals `candidate_accuracy - baseline_accuracy` and
that `status.json` records what it was compared against.

### Limitations

Deltas produced under `--mock-bench` are synthesized, not measured.
Every row carries `metrics_source`, and mock rows never aggregate with
measured ones.

---

## 4. PostgreSQL checkpoint history

**Status: implemented and tested**

### Implementation

`backend/app/meta_harness/persistence.py` (`AsyncPostgresSaver` on a
sized pool), `branches.py` (`get_state_history`,
`get_checkpoint_state`), `backend/app/event_loop.py` (keeps the server
on a psycopg-compatible event loop).

Both graphs are checkpointed. Inner-loop threads are namespaced
`inner::{run}::{branch}::{candidate}::{task}::trial-{n}`, so any inner
checkpoint is attributable to the exact trial that produced it.

### Validation

```bash
docker compose -f infra/docker-compose.yml up -d postgres
cd backend && uv run pytest tests/test_persistence.py tests/test_inner_checkpoints.py -q
uv run meta-harness checkpoints <run-name>
```

`test_checkpoints_persist_in_postgres`,
`test_inner_graph_transitions_persist_in_postgres`,
`test_two_branches_do_not_share_inner_checkpoint_threads`.

### Limitations

Postgres must be reachable. `GET /health` reports `persistence` and
`persistence_error`; a server started without a usable connection
degrades to in-memory checkpointing, and branching and recovery stop
working with it.

---

## 5. Historical branching

**Status: implemented and tested**

### Implementation

`backend/app/meta_harness/branches.py` (`worktree_add` →
`aupdate_state` → `ainvoke(None, fork_config)` in an `asyncio.Task`),
`backend/app/api/forks.py`, thread-scoped artifacts in
`backend/app/meta_harness/runs.py`.

The part that makes this a real search tree rather than a demo: every
artifact a branch writes is scoped to its LangGraph thread. Two branches
forked from the same checkpoint reach the same iteration number and
still keep separate pending-eval handoffs, frontiers, evolution logs,
proposer sessions, candidate directories and candidate source snapshots.

### Validation

```bash
cd backend && uv run pytest tests/test_branch_isolation.py \
                            tests/test_branch_persistence.py -q
API_URL=http://127.0.0.1:8000 uv run python scripts/acceptance_api_flow.py
cd frontend/dashboard && npx playwright test --project=live-backend
```

`test_same_iteration_branches_keep_separate_artifacts` reproduces the
exact race that used to corrupt a run and asserts it cannot happen.
`test_trajectory_survives_registry_reset` proves the branch tree is
reconstructable after a backend restart.

### Limitations

A running branch's asyncio task does not survive a backend restart. The
branch tree is reconstructed from durable metadata, and an interrupted
branch reports as `interrupted` rather than silently resuming.

---

## 6. Checkpoint recovery and recorded-execution replay

**Status: implemented — precise wording required**

### Accurate wording

> Checkpoint recovery and branching from historical states, with
> deterministic replay of recorded execution.

Do **not** describe this as "exact replay" or "deterministic replay"
without qualification. Those phrases suggest that re-running the agent
from a restored checkpoint reproduces the same output. It does not: LLM
inference is stochastic, and the provider's model behind a given id can
change.

### Implementation

`backend/app/meta_harness/replay.py`. What is guaranteed:

1. Restoring a stored checkpoint returns byte-identical state, provable
   via SHA-256 over a canonical JSON encoding.
2. Walking a thread's recorded transitions is deterministic and issues
   no model calls.
3. Forking from a restored checkpoint starts from exactly that state.

### Validation

```bash
cd backend && uv run pytest tests/test_replay.py -q
uv run meta-harness replay <run-name> --checkpoint <checkpoint-id>
```

`test_restored_checkpoint_state_is_identical` asserts
`hash(saved) == hash(restored)`; `test_replay_does_not_advance_the_thread`
asserts replay is read-only.

### Limitations

Byte-identical regeneration of model output is not guaranteed when the
graph is re-executed from a restored checkpoint.

---

## 7. Pass-rate benchmarking

**Status: implemented — not yet measured**

### Implementation

The experiment runner (`backend/app/meta_harness/experiment.py`), the
committed protocol (`benchmarks/pass-rate/config.json`), the raw-row
schema, the provenance capture and the summary derivation are all
implemented and tested.

**No canonical 200-trial experiment has been executed**, because this
environment has no `ANTHROPIC_API_KEY` — the inner loop cannot issue a
single model call, so there is nothing to measure. There is no
placeholder pass-rate value anywhere in this repository.

To produce the measurement:

```bash
# 1. credentials
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env

# 2. evolve a candidate (writes agents/<name>.py)
uv run meta-harness loop --proposer claude --budget 5 --fresh

# 3. run the canonical protocol: 5 tasks x 20 trials x 2 arms = 200 trials
uv run meta-harness experiment --candidate <name>
```

The runner prints, and `benchmark-results/<id>/summary.json` records:

```
Baseline  (baseline): <passes>/100 = <accuracy>
Candidate (<name>):   <passes>/100 = <accuracy>
Absolute improvement: <delta> percentage points
95% CI on the difference: [<lower>, <upper>] pp (wald-95)
Total trials: 200
```

The reported metric is the sentence the runner prints —
`experiment.reported_metric_sentence(summary)` — with whatever delta the
trials produced. Any delta is an acceptable outcome; the protocol is
never tuned to reach a particular number.

**Protocol**: [`benchmarks/pass-rate/README.md`](../benchmarks/pass-rate/README.md)
**Published results**: `benchmarks/results/` (currently empty)

### Validation

```bash
cd backend && uv run pytest tests/test_experiment.py -q
```

`test_summarize_accepts_no_target_or_expected_value` asserts by
signature inspection that `summarize()` takes only raw rows and labels —
no target, no expected value, no override.
`test_summary_recomputes_identically_from_the_written_files` proves any
reader can re-derive the published number from the committed raw rows,
and CI re-derives every published summary on each push.

### Limitations

Trials are clustered within tasks, so the Wald interval understates
uncertainty. The protocol measures the five search tasks the proposer
optimised against; generalisation is a separate holdout measurement
(`meta-harness benchmark --candidate <name> --holdout`).

---

## 8. Token and cost accounting

**Status: implemented and tested — no published totals**

### Implementation

`backend/app/meta_harness/metrics.py` records the Anthropic `usage`
block for every inner-loop call and aggregates to trial, candidate and
run level. Cost uses configurable pricing (`META_HARNESS_PRICING`); a
model with no configured price yields `cost_usd: null` and
`cost_complete: false` rather than `$0.00`. Mock and measured results
never mix: `aggregate_trials` refuses to fold rows whose
`metrics_source` disagrees.

### Validation

```bash
cd backend && uv run pytest tests/test_metrics.py -q
```

### Limitations

No aggregate token or dollar total is published anywhere in this
repository, because none has been measured.

---

## 9. Cross-run memory

**Status: implemented and tested**

### Implementation

`backend/app/meta_harness/memory.py` (`AsyncPostgresStore`) stores
accepted-candidate patterns and injects them into a later run's proposer
prior.

### Validation

```bash
cd backend && uv run pytest tests/test_memory.py tests/test_memory_e2e.py -q
uv run meta-harness memory list
```

### Limitations

Memory lives in Postgres alongside the checkpoints; without a database
connection a run starts with an empty prior.

---

## 10. Trial isolation

**Status: implemented — precise wording required**

### Accurate wording

> Each trial runs in a fresh temp-directory workspace with wall-clock
> timeouts and, on Unix, CPU/address-space rlimits.

Do **not** describe this as "sandboxed", "container-isolated" or
"network-isolated".

### Implementation

`backend/app/meta_harness/sandbox.py` provides process isolation only.

### Validation

```bash
cd backend && uv run pytest tests/test_sandbox.py tests/test_tools.py -q
```

### Limitations

No container, no network restriction, no binary allowlist. Task
`test_command`s run with `shell=True` and are trusted repository content
(`eval/tasks/*/task.json`), not user input.

---

## Design FAQ

| Question | Answer |
|---|---|
| What is the contribution? | The substrate: the Stanford meta-harness loop mapped onto two LangGraph state machines, which makes checkpointing, forking and concurrent search fall out of the framework rather than being bolted on. |
| What was the hardest bug? | Concurrent branches shared run-level artifacts, so a fork's proposer overwrote the root's `pending_eval.json` and the root then benchmarked a candidate it never proposed. Fixed by scoping every execution artifact to its LangGraph thread and snapshotting candidate source per branch. |
| How is branch isolation established? | `test_same_iteration_branches_keep_separate_artifacts` forks two branches from one checkpoint, runs both to the same iteration concurrently against a shared `AsyncPostgresSaver`, and asserts nothing is shared. |
| Why is the baseline benchmarked? | Otherwise iteration 1's delta is measured against zero and every first candidate looks like a huge win. The baseline runs the identical task/trial protocol and becomes the search-tree root. |
| Can a run be replayed? | Any checkpoint's exact state can be restored and proven with a SHA-256, and recorded transitions replay without calling a model. Stochastic model output is not reproducible, and is not claimed to be. |
| What is the measured improvement? | The experiment runner is built and tested; no number is published, because the work was done without API credentials. The command is in `benchmarks/pass-rate/README.md`, and the summary is derived from raw trial rows so it cannot be hand-entered. |
