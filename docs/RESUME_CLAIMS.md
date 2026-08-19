# Resume claims → implementation → evidence

Every claim this project makes about itself, the code that implements it,
and the command that proves it. If a claim is not measured yet, it says
so here rather than being softened elsewhere.

Verdicts used below:

- **VERIFIED** — implemented and covered by an automated test.
- **VERIFIED WITH PRECISE WORDING** — implemented, but the obvious short
  phrasing would overclaim; the exact defensible sentence is given.
- **NOT YET VERIFIED** — the machinery exists, the measurement does not.

---

## 1. A self-improving coding agent that rewrites its own harness from execution traces

**Verdict: VERIFIED**

**Implementation**

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

**Evidence**

```bash
# offline: the loop runs, snapshots candidate source, evolves state
cd backend && uv run pytest tests/test_outer.py tests/test_proposer.py -q

# live: the real claude-CLI proposer writes an importable candidate
bash scripts/live_smoke.sh          # section 3
```

Key tests: `test_candidate_source_is_snapshotted_per_branch`,
`test_mock_outer_loop_produces_all_files`,
`test_claude_propose_fails_loudly_when_the_branch_handoff_is_missing`.

---

## 2. Dual LangGraph state machines

**Verdict: VERIFIED**

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

**Evidence**

```bash
cd backend && uv run pytest tests/test_inner.py tests/test_outer.py \
                            tests/test_inner_checkpoints.py -q
```

`test_inner_graph_compiles_with_and_without_a_checkpointer` and
`test_benchmark_core_threads_the_checkpointer_into_every_trial` cover
the two-machine structure directly.

---

## 3. Harness evolution across iterations

**Verdict: VERIFIED**

Each iteration appends one row to the branch's
`evolution_summary.jsonl` with the candidate, its parent, its measured
scores, and its delta against the measured prior best. Iteration 0 is
the benchmarked baseline, so the first candidate's delta is a real
comparison rather than a comparison against zero.

**Evidence**

```bash
cd backend && uv run pytest tests/test_outer.py -q
uv run meta-harness loop --proposer mock --mock-bench --budget 2 --fresh --run-name demo
cat runs/demo/threads/demo/evolution_summary.jsonl
```

`test_baseline_is_benchmarked_before_any_candidate` asserts the first
candidate's delta equals `candidate_accuracy - baseline_accuracy` and
that `status.json` records what it was compared against.

---

## 4. PostgreSQL-backed checkpoint and version history

**Verdict: VERIFIED**

**Implementation**: `backend/app/meta_harness/persistence.py`
(`AsyncPostgresSaver` on a sized pool), `branches.py`
(`get_state_history`, `get_checkpoint_state`),
`backend/app/event_loop.py` (keeps the server on a psycopg-compatible
event loop).

Both graphs are checkpointed. Inner-loop threads are namespaced
`inner::{run}::{branch}::{candidate}::{task}::trial-{n}`, so any inner
checkpoint is attributable to the exact trial that produced it.

**Evidence**

```bash
docker compose -f infra/docker-compose.yml up -d postgres
cd backend && uv run pytest tests/test_persistence.py tests/test_inner_checkpoints.py -q
uv run meta-harness checkpoints <run-name>
```

`test_checkpoints_persist_in_postgres`,
`test_inner_graph_transitions_persist_in_postgres`,
`test_two_branches_do_not_share_inner_checkpoint_threads`.

---

## 5. Branching from historical checkpoints, with concurrent branches

**Verdict: VERIFIED**

**Implementation**: `backend/app/meta_harness/branches.py`
(`worktree_add` → `aupdate_state` → `ainvoke(None, fork_config)` in an
`asyncio.Task`), `backend/app/api/forks.py`, thread-scoped artifacts in
`backend/app/meta_harness/runs.py`.

The part that makes this a real search tree rather than a demo: every
artifact a branch writes is scoped to its LangGraph thread. Two branches
forked from the same checkpoint reach the same iteration number and
still keep separate pending-eval handoffs, frontiers, evolution logs,
proposer sessions, candidate directories and candidate source snapshots.

**Evidence**

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

---

## 6. Replay

**Verdict: VERIFIED WITH PRECISE WORDING**

**Say this:**

> Checkpoint recovery and branching from historical states, with
> deterministic replay of recorded execution.

**Do not say** "exact replay" or "deterministic replay" without
qualification. Those phrases suggest that re-running the agent from a
restored checkpoint reproduces the same output. It does not: LLM
inference is stochastic, and the provider's model behind a given id can
change.

**What is guaranteed** (`backend/app/meta_harness/replay.py`):

1. Restoring a stored checkpoint returns byte-identical state, provable
   via SHA-256 over a canonical JSON encoding.
2. Walking a thread's recorded transitions is deterministic and issues
   no model calls.
3. Forking from a restored checkpoint starts from exactly that state.

**What is not guaranteed:** byte-identical regeneration of model output
when the graph is re-executed.

**Evidence**

```bash
cd backend && uv run pytest tests/test_replay.py -q
uv run meta-harness replay <run-name> --checkpoint <checkpoint-id>
```

`test_restored_checkpoint_state_is_identical` asserts
`hash(saved) == hash(restored)`; `test_replay_does_not_advance_the_thread`
asserts replay is read-only.

---

## 7. Measured improvement in agent pass rate

**Verdict: NOT YET VERIFIED — PENDING MEASURED BENCHMARK**

**Status:** the experiment runner, the committed protocol, the raw-row
schema, the provenance capture and the summary derivation are all
implemented and tested. **No canonical 200-trial experiment has been
executed**, because this environment has no `ANTHROPIC_API_KEY` — the
inner loop cannot issue a single model call, so there is nothing to
measure.

**Do not put a number on a resume until this has been run.** There is no
placeholder value anywhere in this repository to copy.

**To produce the number:**

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

**Then, and only then**, the resume sentence is the one the runner
prints — `experiment.reported_metric_sentence(summary)` — with whatever
delta the trials produced. `+7`, `+9` and `+15` are all acceptable
answers. The protocol is never tuned to reach a particular number.

**Protocol**: [`benchmarks/pass-rate/README.md`](../benchmarks/pass-rate/README.md)
**Published results**: `benchmarks/results/` (currently empty)

**Evidence that the number cannot be fabricated**

```bash
cd backend && uv run pytest tests/test_experiment.py -q
```

`test_summarize_accepts_no_target_or_expected_value` asserts by
signature inspection that `summarize()` takes only raw rows and labels —
no target, no expected value, no override.
`test_summary_recomputes_identically_from_the_written_files` proves any
reader can re-derive the published number from the committed raw rows,
and CI re-derives every published summary on each push.

**Stated limitations**: trials are clustered within tasks, so the Wald
interval understates uncertainty; the protocol measures the five search
tasks the proposer optimised against, and generalisation is a separate
holdout measurement (`meta-harness benchmark --candidate <name> --holdout`).

---

## 8. Real token / cost accounting

**Verdict: VERIFIED (mechanism) — no published totals**

`backend/app/meta_harness/metrics.py` records the Anthropic `usage`
block for every inner-loop call and aggregates to trial, candidate and
run level. Cost uses configurable pricing (`META_HARNESS_PRICING`); a
model with no configured price yields `cost_usd: null` and
`cost_complete: false` rather than `$0.00`. Mock and measured results
never mix: `aggregate_trials` refuses to fold rows whose
`metrics_source` disagrees.

No aggregate token or dollar total is published anywhere in this
repository, because none has been measured.

**Evidence**

```bash
cd backend && uv run pytest tests/test_metrics.py -q
```

---

## 9. Cross-run memory

**Verdict: VERIFIED**

`backend/app/meta_harness/memory.py` (`AsyncPostgresStore`) stores
accepted-candidate patterns and injects them into a later run's proposer
prior.

**Evidence**

```bash
cd backend && uv run pytest tests/test_memory.py tests/test_memory_e2e.py -q
uv run meta-harness memory list
```

---

## 10. Sandbox isolation

**Verdict: VERIFIED WITH PRECISE WORDING**

**Say this:**

> Each trial runs in a fresh temp-directory workspace with wall-clock
> timeouts and, on Unix, CPU/address-space rlimits.

**Do not say** "sandboxed", "container-isolated" or "network-isolated".
`backend/app/meta_harness/sandbox.py` provides process isolation only:
no container, no network restriction, no binary allowlist. Task
`test_command`s run with `shell=True` and are trusted repository content
(`eval/tasks/*/task.json`), not user input.

**Evidence**

```bash
cd backend && uv run pytest tests/test_sandbox.py tests/test_tools.py -q
```

---

## Interview crib

| Question | One-sentence answer |
|---|---|
| What is the contribution? | The substrate: the Stanford meta-harness loop mapped onto two LangGraph state machines, which makes checkpointing, forking and concurrent search fall out of the framework rather than being bolted on. |
| Hardest bug you fixed? | Concurrent branches shared run-level artifacts, so a fork's proposer overwrote the root's `pending_eval.json` and the root then benchmarked a candidate it never proposed. Fixed by scoping every execution artifact to its LangGraph thread and snapshotting candidate source per branch. |
| How do you know the branches are really isolated? | `test_same_iteration_branches_keep_separate_artifacts` forks two branches from one checkpoint, runs both to the same iteration concurrently against a shared `AsyncPostgresSaver`, and asserts nothing is shared. |
| Why is the baseline benchmarked? | Otherwise iteration 1's delta is measured against zero and every first candidate looks like a huge win. The baseline runs the identical task/trial protocol and becomes the search-tree root. |
| Can you replay a run? | I can restore any checkpoint's exact state and prove it with a SHA-256, and replay the recorded transitions without calling a model. I can't reproduce stochastic model output, and I don't claim to. |
| What's the measured improvement? | The experiment runner is built and tested; I haven't published a number yet because I ran the work without API credentials. The exact command is in `benchmarks/pass-rate/README.md`, and the summary is derived from raw trial rows so it can't be fudged. |
