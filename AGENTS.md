# Repository Guidelines

## Project Structure & Module Organization
This repository is a `uv` workspace with two Python packages: `sdk/` and `backend/`. The installable SDK lives in `sdk/meta_harness/`; backend orchestration code lives in `backend/app/meta_harness/`. Keep those namespaces distinct: `meta_harness` is the public library, while `app.meta_harness` is backend-internal. CLI entrypoints live in `backend/app/cli.py`. Tests live under `backend/tests/`. Frozen evaluation tasks live in `eval/tasks/<task-id>/` with a `task.json` and `workspace/`. `agents/` contains the committed baseline harness; generated candidates and `runs/` outputs are artifacts and should not be checked in.

## Build, Test, and Development Commands
Use `uv` for all Python workflows.

- `uv sync`: install the workspace packages and dependencies.
- `docker compose -f infra/docker-compose.yml up -d postgres`: start the local Postgres instance used by persistence features.
- `uv run python -m eval.score --task task-001-fix-typo`: run the baseline eval scorer on one task.
- `cd backend && uv run pytest tests -q`: run the backend test suite.
- `uv run meta-harness benchmark --candidate baseline --trials 5`: benchmark a candidate across the eval set.
- `uv run meta-harness loop --proposer mock --mock-bench --budget 2 --fresh`: exercise the outer loop without live LLM calls.
- `uv run meta-harness serve --port 8000`: serve the API. Use this rather than a bare `uvicorn app.main:app` — uvicorn picks Windows' ProactorEventLoop, which psycopg cannot use, and the backend would come up with checkpointing silently degraded to in-memory.
- `bash scripts/demo_acceptance.sh`: LEVEL 1 acceptance, no API key required.
- `bash scripts/live_smoke.sh`: LEVEL 2 acceptance, needs credentials; prints SKIPPED without them.
- `uv run meta-harness experiment --candidate <name>`: the canonical 200-trial pass-rate experiment (real cost).
- `uv run meta-harness resume-experiment --dry-run`: print the measurement plan and a cost estimate; spends nothing. Drop `--dry-run` to evolve, select, measure, verify replay and regenerate the evidence document (real cost).
- `uv run meta-harness verify-replay <recordings-dir>`: re-execute recorded trials against their tapes and fail loudly on any divergence. No model calls.
- `uv run meta-harness report resume-evidence --check`: what CI runs; non-zero if `docs/RESUME_EVIDENCE.md` disagrees with the artifacts it is derived from.

## Invariants That Must Not Regress

These were bugs. Each is covered by a test; if you find yourself
weakening one of those tests, fix the root cause instead.

1. **Execution artifacts are thread-scoped.** Everything a running branch
   writes lives under `runs/<run>/threads/<thread_id>/`. Never reintroduce
   a run-level `pending_eval.json`, `frontier_val.json` or
   `evolution_summary.jsonl` — two branches forked from one checkpoint
   reach the same iteration and will overwrite each other.
   (`tests/test_branch_isolation.py`)
2. **A branch benchmarks its own source snapshot**, not the shared
   `agents/<label>.py` a concurrent branch may have rewritten.
   (`tests/test_candidates.py`)
3. **The baseline is benchmarked before the first propose**, so deltas
   compare against a measurement rather than zero.
   (`tests/test_outer.py`)
4. **Unknown is not zero.** An unpriced model yields `cost_usd: None`;
   an unmeasured candidate carries `avg_tokens: None` and cannot dominate
   on the Pareto cost axis. (`tests/test_metrics.py`, `tests/test_frontier.py`)
5. **Mock never mixes with measured.** Every payload carries
   `metrics_source`, and aggregation rejects mismatched rows.
6. **Node side effects are idempotent.** An interrupted LangGraph node is
   re-executed on resume; writes must key on identity rather than append.
   (`tests/test_runs_artifacts.py`)
7. **Holdout tasks never reach the proposer.**
   (`tests/test_holdout_isolation.py`)
8. **The benchmark summary is derived from raw trial rows.** `summarize()`
   takes no target or expected value, and must not gain one.
   (`tests/test_experiment.py`)
9. **Selection cannot see the test.** `pipeline.select_candidate` takes
   the outer loop's terminal state and nothing else, so the final
   experiment's trials cannot influence which candidate they measure.
   (`tests/test_pipeline.py`)
10. **Every crossing into the world goes through `effects`.** A new
    nondeterministic call in an inner-loop node that skips
    `effects.observe` silently breaks exact replay.
    (`tests/test_exact_replay.py`)
11. **Message identity is deterministic and `act` replaces.** Without
    stable ids `add_messages` mints a random UUID per message and no two
    executions produce the same state; without the clear, a trimmed
    trajectory leaves its tail behind. (`tests/test_inner_messages.py`)
12. **Tracking never leaks into core logic.** `tracking.py` is the only
    module that may name `wandb`, tracking is off by default, and it
    never raises into a caller. (`tests/test_tracking.py`)
13. **Task hashes are byte hashes.** `.gitattributes` pins `eval/**`,
    `benchmarks/**` and `agents/**` to LF; without it a Windows checkout
    hashes every frozen task differently from a Linux one.
    (`tests/test_experiment.py`)
14. **Resume evidence is derived, never written.** A claim with no
    supporting artifact reports `UNSUPPORTED`; no pass rate or delta is
    hard-coded. (`tests/test_resume_evidence.py`)

## Capabilities and evidence

`docs/CAPABILITIES.md` maps every capability to the code that implements
it and the command that validates it. If you change behaviour a
capability entry depends on, update that document in the same change. Do
not report a quantitative result without a committed artifact under
`benchmarks/results/` that reproduces it.

`docs/RESUME_EVIDENCE.md` is **generated** by
`uv run meta-harness report resume-evidence` and must never be edited by
hand: CI regenerates it and fails on any disagreement. Its inputs are the
machine-written artifacts in `docs/evidence/` — also never hand-edited.

Say which "replay" you mean. `replay_recorded_execution` re-runs a
recorded run against its tape and reproduces it exactly with zero model
calls; `meta-harness resume` and forking issue *fresh* model calls and
are not reproducible. Do not describe the second as a replay.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints, `Path`-based filesystem code, and concise module docstrings. Use `snake_case` for modules, functions, and package directories; use `PascalCase` for classes such as `BaselineHarness`. Keep CLI and backend imports explicit. No formatter or linter config is checked in today, so match the surrounding file style closely and avoid introducing new tooling conventions inside a single change.

## Testing Guidelines
Write tests in `backend/tests/` using `test_<feature>.py` and `test_<behavior>()` naming. Prefer deterministic unit tests first; reserve live end-to-end tests for flows that require `ANTHROPIC_API_KEY`. When changing eval, sandbox, or outer-loop behavior, add or update pytest coverage and run the narrowest relevant command before the full suite.

## Commit & Pull Request Guidelines
Recent history follows step-oriented subjects like `step 6: real proposer ...`; use that format for milestone work and `fixup:` only for small follow-ups. PRs should state the affected area, reference the relevant build step or interface contract when applicable, and list the exact verification commands you ran. Include artifact paths or screenshots only when the change produces user-visible output.

## Commit Authorship Policy (mandatory)

Every commit in this repository is authored **and** committed by the repository
owner's configured Git identity. Automated assistants operating in this repo
must use the existing local identity and must never rewrite it:

```bash
git config user.name     # David
git config user.email    # the owner's configured address
```

Commits must **not** contain AI attribution of any kind. Specifically, none of
the following may appear in a commit message, trailer, author field, or
committer field:

- `Co-Authored-By: Claude ...` / `Co-Authored-By: Anthropic ...` / any other
  `Co-Authored-By:` trailer naming a model or vendor
- `Generated-by:`, `Made-with:`, `AI-generated-by:` trailers
- "Claude", "Anthropic", "Codex", or similar as an author, co-author,
  committer, or contributor

Verify after every commit:

```bash
git log -1 --format='Author: %an <%ae>%nCommitter: %cn <%ce>%n%B'
```

Historical commits are never rewritten to satisfy this policy — it applies to
new commits only.

## Artifact Hygiene

`runs/`, `benchmark-results/`, proposer-generated `agents/*.py`, tool caches,
and Playwright `test-results/` are build artifacts. They must stay untracked.
Published benchmark evidence is the one exception and lives under
`benchmarks/results/<experiment-id>/`, which is committed deliberately.
