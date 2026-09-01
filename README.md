# Meta-Harness

> *Stanford's Meta-Harness paper had a linear loop. We mapped it onto LangGraph and made it a tree.*
<img src="https://raw.githubusercontent.com/browser-use/media/main/browser-harness/banner-ink.svg" alt="Browser Harness" width="100%" />

LangGraph-native substrate for **self-improving agent harnesses**. Applies
the Stanford Meta-Harness paradigm
([arXiv:2603.28052](https://arxiv.org/abs/2603.28052),
[yoonholee.com/meta-harness](https://yoonholee.com/meta-harness/)) to a
coding-agent domain — expressed as **two LangGraph state machines** with
Postgres-backed checkpointing, time-travel forking, and cross-run memory.

A creative reinterpretation of the work at
[yoonholee.com/meta-harness](https://yoonholee.com/meta-harness/).

---

## The insight

The Stanford paper showed that an outer-loop agent reading raw execution
traces and rewriting an inner-loop harness beats every prior text optimizer
— **+7.7 points over ACE with 4× fewer context tokens, top-2 on
TerminalBench-2**.

But their loop is **linear**: `iter 1 → 2 → 3 → 4`. Real harness optimization
needs branching: rewind to iter 2, try a different proposer prior, fork,
compare. By mapping the loop onto LangGraph state machines, three properties
fall out **by construction**:

| Property | Mechanism | What that does *not* mean |
|---|---|---|
| **Isolated** | Each candidate runs in a fresh temp workspace, and every artifact a branch writes is scoped to its LangGraph thread — two branches cannot overwrite each other | Not container or network isolation; see [Security posture](#security-posture) |
| **Recoverable** | Every state transition is checkpointed via `AsyncPostgresSaver`; any checkpoint restores to byte-identical state, provable by SHA-256 | Not that *re-running* from a checkpoint reproduces the same model output — that is `resume`, and it is a fresh stochastic execution |
| **Reversible** | Time-travel via `get_state_history` + `update_state` + `ainvoke(None, ckpt_id)`, with branch history persisted to disk so it survives a restart | Not that a running branch's asyncio task survives a restart — it doesn't, and it reports as `interrupted` |
| **Replayable** | A **recorded** run re-executes from any of its stored checkpoints against its tape: same transitions, same final state byte-for-byte, zero model calls | Not that an *unrecorded* run can be replayed exactly — recording is opt-in (`--record`) |

The substrate IS the contribution. Every property above maps to a test in
[`docs/CAPABILITIES.md`](docs/CAPABILITIES.md).

---

## Architecture

```
   OUTER STATE MACHINE  (4 nodes, checkpointed via AsyncPostgresSaver)
   ──────────────────────────────────────────────────────────────────
   propose ──► validate ──► benchmark ──► update_frontier
      │                          │                │
      │                          │                └─ loop while budget > 0
      ▼                          ▼
   spawns `claude` CLI        spawns inner
   subprocess + SKILL.md      subgraph per
   (proposer writes a         candidate
   new agents/<name>.py)
                                  │
                                  ▼
   INNER STATE MACHINE  (5 nodes, sandboxed subgraph per candidate)
   ────────────────────────────────────────────────────────────────
   orient ─► plan ─► act ─► verify ─► submit
      │
      │  ▸ 6 fixed tools (read_file, apply_patch, write_file,
      │       run_bash, grep_search, task_complete) — the contract
      │  ▸ 11 override points (system prompt, plan template, turn
      │       budget, retry policy, tool-result formatting, ...)
      │       — the search space
                                  │
                                  ▼  traces, scores, file diffs streamed via SSE
   DASHBOARD  (Next.js 16)
   ───────────────────────
   ▸ outer state graph (ReactFlow) — live nodes lighting up per iteration
   ▸ candidate trajectory tree (D3) — branches when you fork a checkpoint
   ▸ code diff viewer (Monaco) — agents/<n>.py vs parent, live
   ▸ score chart + Pareto frontier — accuracy × context tokens
   ▸ cross-run memory panel — patterns learned by prior runs
   ▸ right-click any checkpoint → fork modal → resume on a new branch
```

---

## The demo arc

> **These numbers are illustrative, not measured.** They show the shape of
> a run — a linear trajectory, a fork, two Pareto-optimal branches — not
> results this repository has produced. No benchmark has been published
> yet; see [Measured results](#measured-results).

```text
Illustrative shape of a run (baseline harness, 5 tasks x 5 trials):

Iter 0:   baseline (benchmarked first, the search root)   ->  b
Iter 1:   retry on schema_drift errors                    ->  b + d1
Iter 2:   stricter tool-description hashing               ->  regression
Iter 3:   early-exit on auth failures                     ->  new best
Iter 4:   more specific tool descriptions                 ->  new best

      +- right-click iter 2 -> "Fork from here" -> edit proposer prior -+
      |                                                                 |
      v                                                                 v
Iter 2':  rewrite tool descriptions w/ examples          ->  branch best
Iter 3':  add few-shot demos to descriptions             ->  global best

Two branches, running concurrently, each with its own artifacts and its
own Pareto frontier. The meta-harness loop is no longer a sequence --
it is a search tree.
```

---

## Quickstart

**Prerequisites**

- Python 3.11+ and [uv](https://github.com/astral-sh/uv)
- Docker (for local Postgres)
- Node.js 20+ + npm (for the dashboard)
- For live model runs only: an `ANTHROPIC_API_KEY`, and the
  [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code/overview)
  (`claude`) for the real proposer

Everything in "Deterministic demo" below runs with **no API key**.

```bash
git clone https://github.com/davidyang07/Meta-Harness.git
cd Meta-Harness
cp .env.example .env                                          # ANTHROPIC_API_KEY optional
uv sync
docker compose -f infra/docker-compose.yml up -d postgres

cd backend && uv run pytest tests/ -q                         # backend suite
```

### Or run the whole stack in Docker

The backend has a production-shaped image (`infra/Dockerfile`) and is a
compose service beside Postgres, so a checkout with Docker and nothing
else installed can bring up a working API:

```bash
docker compose -f infra/docker-compose.yml up -d --build       # postgres + backend
curl localhost:8000/health
# {"status":"ok","persistence":"postgres","persistence_error":null,...}

docker compose -f infra/docker-compose.yml logs -f backend
docker compose -f infra/docker-compose.yml down                # stop
docker compose -f infra/docker-compose.yml down -v             # stop and wipe data
```

`up -d postgres` on its own still gives you just the database, which is
what you want when running the CLI and the test suite on the host.

A few properties worth knowing, because they are enforced rather than
aspirational — `bash scripts/docker_smoke.sh` checks every one of them,
and so does the `docker` job in CI:

- **Reproducible dependencies.** The image installs from `uv.lock` with
  `uv sync --frozen`, so a drifted lockfile fails the build instead of
  shipping an image whose dependencies differ from the tested ones.
- **Non-root.** The process runs as uid 10001, not root.
- **No credentials in the image.** `POSTGRES_DSN`, `ANTHROPIC_API_KEY`
  and the `META_HARNESS_*` knobs are read from the environment at run
  time; `.env` is excluded from the build context entirely.
- **An honest healthcheck.** The container reports *unhealthy* when the
  API is up but Postgres is not. A backend in that state still answers
  requests, with checkpointing silently degraded to in-memory — no
  checkpoint history, no forking, no branch recovery — and that is
  precisely the failure worth surfacing rather than hiding. Set
  `META_HARNESS_HEALTHCHECK_REQUIRE_POSTGRES=0` to run without a database
  on purpose.

No API key is needed to bring the stack up; nothing it starts spends
anything.

---

## Demo modes

Three distinct things, deliberately never blended.

### 1. Deterministic demo — no API key, no cost

Mock proposer, mock benchmark. Synthetic scores, always labelled
`metrics_source: "mock"` in artifacts, the API, and the dashboard status
bar. Proves the substrate: two state machines, checkpointing, forking,
branch isolation, the full REST/SSE surface, the dashboard.

```bash
# outer loop, completes in seconds
uv run meta-harness loop --proposer mock --mock-bench --budget 2 --fresh

# backend + dashboard
uv run meta-harness serve --port 8000                       # terminal 1
cd frontend/dashboard && npm install && npm run dev         # terminal 2

# the whole acceptance ladder, end to end
bash scripts/demo_acceptance.sh
```

> Start the backend with `meta-harness serve`, not a bare
> `uvicorn app.main:app`. uvicorn selects Windows' `ProactorEventLoop`,
> which psycopg cannot use — the server would come up with checkpointing
> degraded to in-memory. `/health` reports `persistence` and
> `persistence_error` so a degraded backend is never silent.

### 2. Live model demo — needs credentials, small cost

Real Haiku 4.5 inner loop, real `claude` CLI proposer, measured token
and cost metrics.

```bash
uv run meta-harness inner --task task-001-fix-typo --candidate baseline
uv run meta-harness loop --proposer claude --budget 1 --fresh --mock-bench
bash scripts/live_smoke.sh        # prints SKIPPED without credentials
```

### 3. Published benchmark experiment — needs credentials, real cost

One command runs the whole measurement pipeline: evolve candidates,
select one on validation results only, run the canonical 200-trial
experiment, run the 80-trial holdout experiment, verify that a recorded
trial replays exactly, and regenerate the evidence document.

```bash
uv run meta-harness resume-experiment --dry-run   # plan + cost estimate, spends nothing
uv run meta-harness resume-experiment             # THIS COSTS MONEY
```

See [`benchmarks/pass-rate/README.md`](benchmarks/pass-rate/README.md) and
[`benchmarks/holdout/README.md`](benchmarks/holdout/README.md).

---

## Measured results

**Status: no benchmark has been published yet.**

The measurement pipeline — runner, both committed protocols, raw-trial
schema, provenance capture, methodology checks, summary derivation and
evidence generation — is implemented and tested. The benchmark itself has
not been executed, because this environment has no API credentials. There
is no measured pass-rate number in this repository and none is claimed
anywhere in it.

[`docs/RESUME_EVIDENCE.md`](docs/RESUME_EVIDENCE.md) is generated from
committed artifacts and reports every claim as `PASS`, `FAIL` or
`UNSUPPORTED`. The pass-rate rows are currently `UNSUPPORTED`. That is the
correct state, not a placeholder: a claim with no supporting artifact is
reported as unsupported rather than softened.

- Protocol: [`benchmarks/pass-rate/README.md`](benchmarks/pass-rate/README.md)
- Generalisation protocol: [`benchmarks/holdout/README.md`](benchmarks/holdout/README.md)
- Published results (currently empty): [`benchmarks/results/`](benchmarks/results/)
- Capability status: [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md)

### What keeps a future number honest

- **Selection cannot see the test.** The candidate is chosen from the
  validation accuracy measured *during* evolution; the canonical
  experiment runs afterwards with fresh independent trials. Choosing on
  the final trials would make the delta a selection artifact.
- **Both arms run one protocol.** Identical tasks, trial counts, worker
  pool and model, checked mechanically and published as `validation.json`.
- **The summary cannot be authored.** `summarize()` accepts raw rows and
  labels — no target, no expected value — and a test asserts that by
  signature inspection.
- **Incomplete arms are detected.** Missing, duplicated and malformed
  trials are named. A row with no outcome is unknown, not a failure.
- **CI re-derives everything.** Every published summary is recomputed
  from its committed rows, and `docs/RESUME_EVIDENCE.md` is regenerated
  and diffed, on every push.

Numbers you will **not** find presented as measurements: average context
tokens, total run cost, or a pass-rate delta. Those require a run that
has not happened.

---

## Other useful commands

```bash
# resume an interrupted run from its last Postgres checkpoint
# NOTE: this issues FRESH model calls. It is not a replay.
uv run meta-harness resume <run-name>

# inspect checkpoint history, restore one checkpoint's exact state
uv run meta-harness checkpoints <run-name>
uv run meta-harness replay <run-name> --checkpoint <checkpoint-id>

# EXACT REPLAY of a recorded execution: re-runs the graph against the
# recorded tape and fails loudly unless it reproduces it exactly
uv run meta-harness inner --task task-001-fix-typo --candidate baseline --record
uv run meta-harness replay <run-name> --checkpoint <id> --verify
uv run meta-harness verify-replay runs/<run-name>

# fork a run from a checkpoint into a concurrent branch
uv run meta-harness fork <run-name> --checkpoint <id> --mod proposer_prior="try X"

# post-evaluate on the unseen holdout tasks
uv run meta-harness benchmark --candidate <name> --holdout

# evidence reports, all derived from committed artifacts
uv run meta-harness report resume-evidence
uv run meta-harness report version-graph <run-name>
uv run meta-harness report cost-estimate
uv run meta-harness report wandb-check

# optional experiment tracking (off by default, works offline)
uv sync --extra wandb
WANDB_MODE=offline uv run meta-harness loop --wandb --proposer mock --mock-bench --budget 2

# cross-run memory
uv run meta-harness memory list
```

### Three things called "replay"

| Command | Executes? | Model calls | Guarantee |
|---|---|---|---|
| `replay <run>` | no | none | the recorded transitions, in order |
| `replay <run> --checkpoint <id>` | no | none | that checkpoint's exact state, provable by SHA-256 |
| `replay <run> --checkpoint <id> --verify` | **yes** | **none** | same nodes, same per-step state hashes, same final state hash, tape consumed exactly |
| `resume <run>` | yes | **fresh** | a new stochastic execution from an old state |

Only the third row supports the unqualified phrase *exact replay*, and it
applies only to a run that was recorded.

---

## Security posture

Stated plainly, because the honest version is short:

- **Sandboxing is process isolation only.** Each trial gets a fresh
  temp-directory workspace, a wall-clock timeout, and — on Unix —
  `RLIMIT_CPU` / `RLIMIT_AS`. There is **no container, no network
  restriction, and no binary allowlist**. On Windows the rlimits do not
  apply at all.
- **Eval task commands are trusted repository content.** `test_command`
  from `eval/tasks/*/task.json` runs with `shell=True`. That is a
  deliberate trust boundary: task definitions are committed source, not
  user input.
- **The proposer runs with `--dangerously-skip-permissions`** in the
  repository working directory. Treat a run as "this repo executes model-
  written code", because it does.
- **Path inputs are validated.** Run ids, candidate names, branch names
  and thread ids from HTTP are validated against a strict name pattern
  and containment-checked before being joined to a path.

Do not run this against untrusted task definitions or on a host you care
about without adding real isolation.

---

## What's distinctive about this implementation

1. **Two LangGraph state machines, not one.** The outer machine evolves
   the inner machine's source code. Both are checkpointed via
   `AsyncPostgresSaver` — the outer loop threads its saver down into
   every inner trial — and the outer graph supports time-travel forking.
2. **The "meta-harness tool" is a SKILL.md, not a framework feature.**
   ~150 lines of Markdown injected via `--append-system-prompt` when
   the proposer's `claude` subprocess is spawned. Anti-overfitting and
   anti-parameter-tuning rules live there; they're load-bearing per the
   paper's Section 5 ablations.
3. **The inner loop has a fixed contract and an evolvable shape.**
   Six tools (`read_file`, `apply_patch`, `write_file`, `run_bash`,
   `grep_search`, `task_complete`) are the contract with the evaluator
   and **cannot be modified** by candidates. Eleven override points
   define the search space.
4. **`apply_patch` returns `context_echo` on mismatch.** When a unified
   diff fails to apply, the tool surfaces the file's actual current
   content at the failed range so the model fixes the patch without
   re-reading the file.
5. **Forks are concurrent, and genuinely isolated.** `asyncio.create_task`
   over `graph.ainvoke` calls share one `AsyncPostgresSaver`, so both
   branches grow on the dashboard at once. The part that makes that a
   real search tree: **every artifact a branch writes is scoped to its
   LangGraph thread** — `runs/<run>/threads/<thread_id>/` — and each
   branch snapshots its own candidate source. Two branches that reach
   the same iteration cannot overwrite each other's pending evaluation,
   frontier, evolution log, proposer session or traces.
6. **The baseline is measured, not assumed.** Every run benchmarks
   `agents/baseline.py` under the identical task/trial protocol before
   the first propose, so iteration 1's delta is a real comparison rather
   than a comparison against zero.
7. **Unknown is not zero.** A model with no configured price yields
   `cost_usd: null`, not `$0.00`; an unmeasured candidate cannot dominate
   a measured one on the Pareto cost axis; and mock results are tagged
   `metrics_source: "mock"` and refuse to be aggregated with measured
   ones.
8. **Cross-run memory persists across runs.** A pattern learned in
   run A flows into run B's proposer system prompt, so each new run
   starts smarter than cold.

---

## Repository layout

```
meta-harness/
├── backend/                                   # FastAPI + LangGraph
│   ├── app/
│   │   ├── cli.py                             # `meta-harness` CLI (typer)
│   │   ├── main.py                            # FastAPI app factory
│   │   ├── event_loop.py                      # psycopg-compatible loop for uvicorn
│   │   ├── streaming.py                       # closed-set SSE event registry
│   │   ├── api/                               # FastAPI routers (runs, checkpoints, forks, branches, memory, events)
│   │   └── meta_harness/                      # internal namespace
│   │       ├── outer.py                       # outer 4-node StateGraph
│   │       ├── inner.py                       # inner 5-phase StateGraph
│   │       ├── state.py                       # MetaHarnessState + CodingAgentState
│   │       ├── harness.py                     # CodingAgentHarness (11 override points)
│   │       ├── proposer.py                    # claude_propose + mock_propose
│   │       ├── candidates.py                  # per-branch source snapshot + isolated import
│   │       ├── benchmark.py                   # shared (tasks × trials) measured core
│   │       ├── metrics.py                     # per-call/trial/candidate token + cost
│   │       ├── experiment.py                  # canonical 200-trial pass-rate experiment
│   │       ├── replay.py                      # checkpoint restore, event replay, exact replay
│   │       ├── recording.py                   # the execution tape + integrity
│   │       ├── effects.py                     # live / recording / replaying boundary
│   │       ├── versioning.py                  # the checkpoint DAG as a version graph
│   │       ├── tracking.py                    # optional W&B adapter (the only wandb import)
│   │       ├── pipeline.py                    # evolve → select → measure → verify → report
│   │       ├── evidence.py                    # derives docs/RESUME_EVIDENCE.md
│   │       ├── tools.py                       # 6 fixed inner-loop tools
│   │       ├── sandbox.py                     # <temp>/meta-harness-task-{uuid}/
│   │       ├── frontier.py                    # Pareto on (accuracy × measured tokens)
│   │       ├── persistence.py                 # AsyncPostgresSaver
│   │       ├── runs.py                        # thread-scoped artifact lifecycle
│   │       ├── memory.py                      # cross-run patterns (AsyncPostgresStore)
│   │       └── branches.py                    # forks + durable branch metadata
│   └── tests/                                 # backend pytest suite
├── frontend/dashboard/                        # Next.js 16 dashboard
│   └── e2e/                                   # playwright: mock + live-backend projects
├── benchmarks/
│   ├── pass-rate/                             # committed 200-trial protocol
│   ├── holdout/                               # committed 80-trial generalisation protocol
│   └── results/                               # published, immutable evidence
├── scripts/
│   ├── demo_acceptance.sh                     # LEVEL 1 acceptance (no API key)
│   ├── live_smoke.sh                          # LEVEL 2 acceptance (credentialed)
│   └── acceptance_api_flow.py                 # shared API assertions
├── sdk/meta_harness/                          # public Python library
├── skills/meta-harness-coding-agent/SKILL.md  # the proposer's workflow
├── eval/
│   ├── tasks/                                 # 5 frozen calibration tasks
│   ├── holdout/                               # 2 unseen holdout test tasks
│   └── score.py                               # multi-task pytest scorer
├── agents/
│   ├── baseline.py                            # immutable starting harness
│   └── (...)                                  # proposer-generated candidates (gitignored)
├── infra/
│   ├── docker-compose.yml                     # postgres:16 + backend services
│   ├── Dockerfile                             # backend image (multi-stage, non-root)
│   └── healthcheck.py                         # container health: API up AND Postgres reached
└── docs/                                      # contracts + capability evidence
```

### Run artifact layout

Execution state is **thread-scoped**, which is what makes concurrent
branches safe:

```
runs/<run_id>/
├── manifest.json                      # run config + metrics_source
├── branches.json                      # durable branch metadata (survives restart)
└── threads/<thread_id>/
    ├── pending_eval.json              # this branch's proposer -> benchmark handoff
    ├── frontier_val.json              # this branch's Pareto frontier
    ├── evolution_summary.jsonl        # this branch's candidates, tagged by thread_id
    ├── agents/<candidate>.py          # what this branch actually benchmarked
    ├── candidates/<candidate>/        # eval-result.json, status.json, traces/
    └── proposer-sessions/iter-N/      # transcript, events, session.json
```

---

## Documentation

The reference docs are layered so a new contributor can read in order
and end up oriented:

| Doc | When to read |
|---|---|
| [`ARCHITECTURE_SECTION_1.md`](ARCHITECTURE_SECTION_1.md) | First — the locked architecture |
| [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md) | First — repo tree + naming rules |
| [`docs/INTERFACES.md`](docs/INTERFACES.md) | **Always** — every cross-component contract |
| [`docs/BUILD_ORDER.md`](docs/BUILD_ORDER.md) | When picking a step — DoD per step |
| [`docs/DEFINITION_OF_DONE.md`](docs/DEFINITION_OF_DONE.md) | Before demo — the formal acceptance test |
| [`docs/TEAM_HANDOFF.md`](docs/TEAM_HANDOFF.md) | When joining the build — 4-person coordination |
| [`relay_metaharness_v7.md`](relay_metaharness_v7.md) | For the *why* — canonical design doc |
| [`relay_v7_appendix_a_worktrees.md`](relay_v7_appendix_a_worktrees.md) | For step 9 — concurrent branches via asyncio |
| [`relay_v7_appendix_b_metaharness_internals.md`](relay_v7_appendix_b_metaharness_internals.md) | For step 6+ — Stanford repo deep-dive |
| [`relay_v7_appendix_c_inner_loop.md`](relay_v7_appendix_c_inner_loop.md) | For inner-loop work — 5-phase agent design |
| [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) | **Before quoting any capability** — capability → code → the command that validates it |
| [`benchmarks/pass-rate/README.md`](benchmarks/pass-rate/README.md) | Before running or citing a benchmark |
| [`skills/meta-harness-coding-agent/SKILL.md`](skills/meta-harness-coding-agent/SKILL.md) | When debugging the proposer — what it actually reads |

The single most important rule: **`docs/INTERFACES.md` is the contract.**
Every change touching a state schema, JSON shape, REST endpoint, SSE
event, tool I/O, override point, or SKILL.md section updates that doc
in the same commit.

---

## Tech stack

| Component | Choice |
|---|---|
| State machines | LangGraph 0.2+ |
| Checkpointer | `AsyncPostgresSaver` (langgraph-checkpoint-postgres) |
| Database | Postgres 16 (Docker; `infra/docker-compose.yml`) |
| Containers | Docker + Compose: backend image (`infra/Dockerfile`, multi-stage, non-root, healthchecked) and Postgres, built and smoke-tested in CI |
| Backend API | FastAPI 0.115+ + Uvicorn |
| Inner-loop LLM | Claude Haiku 4.5 (default; rate-limit-friendly + ~10× cheaper than Sonnet) |
| Proposer LLM | Claude Code CLI subprocess (subscription auth) |
| CLI | Typer + python-dotenv |
| Frontend | Next.js 16, Tailwind 4, ReactFlow, D3, Monaco |
| Workspace tooling | uv (workspace mode: `sdk/` + `backend/`) |
| Testing | pytest, pytest-asyncio (`asyncio_mode = "auto"`), Playwright |

`META_HARNESS_INNER_MODEL` env var overrides the inner-loop model if a
higher API tier is available (e.g. `claude-sonnet-4-6`).

---

## Acknowledgments

Built on, and grateful for, the work of:

- Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab,
  and Chelsea Finn — *Meta-Harness: End-to-End Optimization of Model
  Harnesses*, [arXiv:2603.28052](https://arxiv.org/abs/2603.28052),
  [project page](https://yoonholee.com/meta-harness/), and the
  reference framework at
  [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness).
- The LangChain team for LangGraph's time-travel primitives, which
  make the linear-to-tree mapping possible without a bespoke
  orchestration layer.
- Anthropic for the Claude Code CLI's `--append-system-prompt` and
  stream-json output format, which let us reproduce the paper's
  filesystem-mediated proposer pattern verbatim.

---

## License

MIT — see [LICENSE](LICENSE).

---

> *Time-travel for Meta-Harness. Built on LangGraph state machines.
> Secure, consistent, reversible — by construction. Open source. One spark.*
