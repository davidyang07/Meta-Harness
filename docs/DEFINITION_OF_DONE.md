# DEFINITION_OF_DONE.md — the original acceptance script

> **Superseded as the acceptance gate, and partly retracted.** The
> authoritative gate is now `make verify` (everything provable without
> credentials, described in the README) and `bash scripts/live_smoke.sh`
> for the credentialed half. This document is kept because its
> *structural* checklist is still a good list of things that must work,
> and because the demo script below records what the system was built to
> show.
>
> **What was removed, and why.** Two of the original criteria required
> the system to produce specific accuracy values — a "score arc landing
> within ±5% of expected: 0.62 → 0.70 → 0.66 → 0.74 → 0.80", and a forked
> branch reaching "≥ 0.83". An acceptance contract that names the numbers
> the measurement must produce is not an acceptance contract; it is a
> target, and the natural way to satisfy it is to adjust the eval set
> until it is met — which is what happened, and has since been undone
> (see `PROJECT_KNOWLEDGE_BASE.md` §27). Those two bullets are struck
> below. The illustrative scores in the demo script are left in place as
> a record of the intended narrative, and are **not** measurements: no
> pass-rate number has been measured in this repository at all.

*Structural criteria below remain binary and are worth keeping green.*

---

## The literal demo command

```bash
# One-time setup
docker compose -f infra/docker-compose.yml up -d postgres
uv sync
(cd frontend/dashboard && npm install && npm run build)

# Prove the system is demo-ready before the demo (no API key needed)
bash scripts/demo_acceptance.sh

# Three terminals at demo time
# Terminal 1 — backend (NOT a bare `uvicorn app.main:app`; see the note below)
cd backend && uv run meta-harness serve --port 8000

# Terminal 2 — frontend
cd frontend/dashboard && npm run dev

# Terminal 3 — kick off the run
uv run meta-harness loop \
  --domain coding-agent \
  --proposer claude \
  --budget 5 \
  --fresh \
  --holdout \
  --run-name demo

# Browser: http://localhost:3000/runs/demo
```

`ANTHROPIC_API_KEY` is loaded from `.env`; Claude Code (`claude` CLI)
must be on `PATH` for the real proposer. Without credentials, swap in
`--proposer mock --mock-bench` and everything below still holds except
the model-dependent parts.

**Use `meta-harness serve`.** uvicorn selects Windows'
`ProactorEventLoop`, which psycopg's async driver cannot use, so a bare
`uvicorn app.main:app` starts a backend with checkpointing silently
degraded to in-memory — no checkpoint history, no forking, no branch
recovery. `GET /health` reports `persistence` and `persistence_error`;
check it before demoing.

---

## Expected output structure

> **The tables below are illustrative, not measured.** They describe the
> *shape* a run should take so a reader knows what "working" looks like.
> No benchmark has been executed in this repository, so no accuracy,
> token or cost figure here is a measurement. See
> [`docs/CAPABILITIES.md`](CAPABILITIES.md) for what is actually
> verified and [`benchmarks/pass-rate/README.md`](../benchmarks/pass-rate/README.md)
> for the protocol that would produce real numbers.

### Linear branch — illustrative shape

| Iter | Candidate | Outcome |
|---|---|---|
| 0 | `baseline` | **benchmarked first**; the measured search root |
| 1 | retry on schema_drift errors | keep, if it beats the measured baseline |
| 2 | stricter tool-description hashing | reject on regression |
| 3 | early-exit on auth failures | keep |
| 4 | more specific tool descriptions | keep, new best |

The delta on every row is computed against the *measured* accuracy of
the prior best, and `status.json` records `compared_against` and
`compared_against_accuracy` so the comparison is auditable.

### Forked branch — illustrative shape

Right-click a checkpoint in the trajectory → **Fork from here** → edit
`proposer_prior` → **Create Fork**. The branch runs concurrently with
the root branch (`asyncio.create_task` over a shared
`AsyncPostgresSaver`).

| Iter | Candidate | Outcome |
|---|---|---|
| 2′ | rewrite tool descriptions w/ examples | keep |
| 3′ | add few-shot demos to descriptions | keep, branch best |

What must be **verifiably true** during this, and is covered by
`backend/tests/test_branch_isolation.py`:

- both branches reach the same iteration number concurrently;
- neither can read or overwrite the other's `pending_eval.json`,
  `frontier_val.json`, `evolution_summary.jsonl`, proposer session,
  candidate directory or candidate source snapshot;
- every evolution row is attributable to its originating `thread_id`;
- no candidate name is claimed by two branches.

### Pareto frontier at end of run

Each branch keeps its own `frontier_val.json`. A candidate is
non-dominated when no other has `accuracy >=` and `avg_tokens <=` with
at least one strict. A candidate whose tokens were never measured
carries `avg_tokens: null` and is compared on accuracy alone — it can be
dominated, but it can never dominate on a cost it never paid.

The frontier payload carries `metrics_source`, and reports `"mixed"`
when mock and measured candidates appear together, so no consumer can
present a synthetic frontier as a measurement.

### Holdout

`--holdout` re-evaluates the best candidate against the two unseen tasks
in `eval/holdout/`. The result is written to the branch's
`holdout-result.json` tagged `task_set: "holdout"`, kept separate from
search-set numbers. The gap between the two is the honest signal about
overfitting; the proposer never sees holdout tasks during search.

### Cost & runtime

Not asserted here, because nothing has been measured. Every real run
records what it actually spent:

- per trial: `llm_calls`, `input_tokens`, `output_tokens`,
  `total_tokens`, `cost_usd`, `wall_time_s` in the trace's
  `metrics.json`;
- per candidate: totals, mean and median tokens per trial, and
  `cost_complete` in `eval-result.json`;
- proposer cost separately, in each `proposer-sessions/iter-N/session.json`.

`cost_usd` is `null` when the model has no configured price
(`META_HARNESS_PRICING`). It is never `0.0` to mean "unknown".

To estimate before committing to a big run:

```bash
uv run meta-harness inner --task task-001-fix-typo --candidate baseline
```

---

## The three things a judge sees on screen

1. **Outer state graph (ReactFlow).** Top-left panel. Nodes
   `propose → validate → benchmark → update_frontier` light up in
   sequence per outer-loop iteration. The proposer node displays the
   live `claude` subprocess transcript while it runs.

2. **Candidate trajectory tree (D3).** Top-right panel. One node per
   candidate. Edges follow `parent_candidate_name`. **On fork, the
   tree visibly branches**, with both branches growing in real time
   (Appendix A — concurrent `asyncio.Task`s, not sequential rewind).

3. **Code diff viewer (Monaco unified-diff).** Right side. Shows the
   live `agents/<n>.py` diff vs parent for the currently-selected
   candidate. Selecting a different node in the tree swaps the diff
   instantly.

Plus three supporting views: a score chart with Pareto frontier
(lower-left), a memory panel sidebar showing cross-run patterns
(persisted via `PostgresStore`), and a fork modal triggered by
right-click → "Fork from here" on any checkpoint.

---

## The 90-second demo script (validation walkthrough)

> **The scores in this script are placeholders for narration.** They are
> not results, and they must not be read aloud as measurements. When
> demoing, read the numbers off the screen — a mock-bench run is
> labelled "Mock data (synthetic)" in the status bar, and a live run
> reports whatever it actually measured.

```
[0:00–0:08] HOOK
"Stanford published Meta-Harness four weeks ago — Lee, Khattab, Finn.
Their proposer agent reads execution traces and rewrites the harness,
beating ACE by 7.7 points. But their loop is linear. We mapped it
onto LangGraph and made it a tree."

[0:08–0:23] ACT 1 — Local launch (no install needed beyond `claude` CLI)
[Browser at localhost:3000. Click "New run" → "Coding agent template"
 → "Start". Run dashboard renders.]
"30 seconds, no cloud. Five-task eval. Baseline: 62%."

[0:23–0:53] ACT 2 — Linear loop
[State graph populates. Iterations stream in via SSE.]
  Iter 1: retry on schema_drift errors        → 0.70 (+0.08) ✓
  Iter 2: stricter tool-description hashing   → 0.66 (−0.04) ✗
  Iter 3: early-exit on auth failures         → 0.74 (+0.04) ✓
  Iter 4: more specific tool descriptions     → 0.80 (+0.06) ✓ NEW BEST
"Stanford's regime — exactly. But here's where it gets interesting."

[0:53–1:20] ACT 3 — Time-travel + memory
[Right-click iter-2 in the trajectory tree → "Fork from here" →
 modal opens → edit proposer_prior → Resume. Tree visibly branches.]
"Rewinding to iteration 2. Forking with a different prior."
[Both branches grow concurrently. Compare view side-by-side.]
  Iter 2′: rewrite tool descriptions w/ examples → 0.78 (+0.16) ✓
  Iter 3′: add few-shot demos to descriptions    → 0.85 (+0.07) ✓ GLOBAL BEST
"Two branches. Original 0.80. Fork 0.85. The meta-harness loop is
no longer a sequence — it's a search tree."
[Click memory panel.]
"And LangGraph's cross-thread memory means the next run starts smarter."

[1:20–1:30] CLOSE
"Time-travel for Meta-Harness. Built on LangGraph state machines.
Secure, consistent, reversible — by construction. Open source.
That's Meta-Harness. One spark."
```

---

## Acceptance criteria (binary checklist)

The run is "done" iff every box ticks:

- [ ] Every BUILD_ORDER.md DoD command (steps 1–13) exits 0.
- ~~Linear score arc lands within ±5% of expected at every iteration:
      0.62 → 0.70 → 0.66 (rejected) → 0.74 → 0.80.~~ **Struck.** A
      required score arc is a target; see the note at the top of this
      file. What replaces it: the run completes, every artifact below is
      well-formed, and whatever accuracies result are reported as they
      are.
- ~~Forked branch reaches **≥ 0.83** by iter 3′ (target 0.85).~~
      **Struck**, same reason. What replaces it: the fork runs
      concurrently, in isolation, and produces its own frontier — which
      is the property the fork feature actually claims.
- [ ] All **11** SSE event types from `INTERFACES.md` §7.2 fire at
      least once during the run.
- [ ] The candidate trajectory tree visibly branches when the fork is
      created; both branches grow concurrently (not serially).
- [ ] The Monaco diff viewer renders `agents/<n>.py` diffs vs parent
      live during runs; switching tree nodes updates the diff.
- [ ] The memory panel shows ≥ 1 entry from a prior run before
      iteration 1's proposer fires.
- [ ] `pending_eval.json`, `frontier_val.json` (with
      `dominated_by_names`), and `evolution_summary.jsonl` (with
      `parent_candidate_name`) are well-formed at end of run.
- [ ] `proposer-sessions/iter-N/` exists for every N ∈ {1..budget}
      with `session.json` schema-compatible with Stanford's reference.
- [ ] Total wall time < 8 minutes; total cost < $5 USD.
- [ ] Holdout result file `runs/demo/holdout-result.json` exists and
      reports a distinct (search vs holdout) score pair.
- [ ] Process restart resilience: kill mid-iteration, then
      `meta-harness resume demo` completes the run without duplicating
      iterations.
- [ ] Two concurrent branches share `AsyncPostgresSaver` without
      deadlock (verified by step (9) test still green at demo time).
- [ ] `POST /runs` returns **201 Created** with `Location` header
      (verified by step (10)'s `scripts/smoke_api.py`).
- [ ] SSE channel registry rejects unregistered event types with a
      500-class error (verified by step (10) test).

If any box is unticked, the run is not "done" — fix it and re-run the
acceptance test before claiming completion.
