# Project status — Meta-Harness

Meta-Harness is a LangGraph-native substrate for self-improving
coding-agent harnesses. The system is built, tested and reproducible.
This page is the one-screen summary; the detail lives in
[`CAPABILITIES.md`](CAPABILITIES.md).

To check any of it yourself, in one command:

```bash
make verify          # everything provable without credentials
```

## The system

| Component | State |
|---|---|
| Outer machine — `propose → validate → benchmark → update_frontier` | Complete |
| Inner machine — `orient → plan → act → verify → submit` | Complete |
| 6-tool fixed contract + 11 harness override points | Complete |
| Claude CLI proposer with SKILL.md workflow contract | Complete |
| PostgreSQL checkpointing (`AsyncPostgresSaver`) | Complete |
| Time-travel forking with thread-scoped artifacts | Complete |
| Recording + exact replay of a recorded execution | Complete |
| Version graph over the checkpoint DAG | Complete |
| Cross-run memory (`AsyncPostgresStore`) | Complete |
| FastAPI REST + 11-event SSE closed set | Complete |
| Next.js dashboard — graph, trajectory tree, diff, frontier, memory | Complete |
| CLI (`app.cli:main`) | Complete |
| Measurement pipeline + two committed protocols | Complete |
| Optional W&B tracking, off by default | Complete |
| 5 frozen search tasks + 2 disjoint holdout tasks | Complete |

## Verification

The backend suite runs clean:

```
339 passed, 53 skipped
```

The 53 skips are the credential- and Postgres-gated integration tests;
CI runs them against a live Postgres service on every push
(`.github/workflows/backend.yml`), and asserts they actually ran rather
than silently skipping.

CI derives every capability claim from committed artifacts on each
push, so no claim on this page can drift from the code that backs it.

## The empirical measurement

One command runs the canonical measurement end to end, and needs only
`ANTHROPIC_API_KEY`:

```bash
uv run meta-harness canonical-experiment --dry-run   # plan + cost, spends nothing
uv run meta-harness canonical-experiment             # evolve → select → measure → holdout → verify → report
```

It runs 280 final trials (200 search-set, 80 holdout) and writes a
publishable result directory under `benchmarks/results/`, which CI then
re-derives from the raw trial rows on every push.

## Where to read more

- [`CAPABILITIES.md`](CAPABILITIES.md) — every capability, the code, the command that validates it
- [`INTERFACES.md`](INTERFACES.md) — cross-boundary contracts (start at §0 Amendments)
- [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md) — directory structure and naming
- [`TEAM_HANDOFF.md`](TEAM_HANDOFF.md) — onboarding
