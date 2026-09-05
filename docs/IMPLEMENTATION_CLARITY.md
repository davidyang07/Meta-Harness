# IMPLEMENTATION_CLARITY.md

> **Status note.** This document tracks the original build sequence and is
> kept for that history. For what the system does today and how each
> capability is validated, read [`CAPABILITIES.md`](CAPABILITIES.md); for the
> current cross-boundary contracts read
> [`INTERFACES.md`](INTERFACES.md) starting at §0 Amendments. To check
> the system yourself rather than read about it: `make verify`.


Purpose: provide one fast, accurate map of what is implemented now, where
contracts live, and what behavior is intentionally placeholder.

## Canonical contract order

When there is disagreement, use this precedence:

1. `docs/INTERFACES.md` (cross-component contracts)
2. `ARCHITECTURE_SECTION_1.md` (locked architecture decisions)
3. `docs/PROJECT_LAYOUT.md` (placement and naming constraints)
4. `docs/DEFINITION_OF_DONE.md` (acceptance criteria)
5. `docs/BUILD_ORDER.md` (step DoD sequencing)
6. live code in `backend/app/**` and `frontend/dashboard/src/**`

`docs/PROJECT_KNOWLEDGE_BASE.md` is comprehensive context and rationale; if it
disagrees with (1)-(5) or current code, treat it as needing an update.

## Current architecture (implemented)

- Backend: FastAPI + LangGraph state machines with `AsyncPostgresSaver`.
- Outer loop: `propose -> validate -> benchmark -> update_frontier`.
- Inner loop: `orient -> plan -> act -> verify -> submit`.
- SSE: 11-event closed set with required `thread_id` on every payload.
- Frontend: Next.js dashboard consuming run detail + SSE stream + fork API.

## Time-travel fork behavior (implemented now)

The dashboard now forks using LangGraph time-travel semantics, not a local
annotation:

- Fork requests include:
  - `parent_checkpoint_id`
  - `parent_thread_id` (when available from selected node)
  - optional `mods` and `name`
- Checkpoint resolution for selected nodes prefers:
  1. exact `thread_id + iteration`
  2. latest checkpoint in that `thread_id`
  3. fallback by summary candidate
  4. fallback by summary iteration
- If a checkpoint cannot be resolved, UI logs a clear "not persisted yet"
  message and does not send a fake fork request.
- After fork creation, branch events are tracked by thread lineage and new
  branch nodes auto-select when streamed.

Key files:

- `frontend/dashboard/src/components/TrajectoryTree.tsx`
- `frontend/dashboard/src/components/ForkModal.tsx`
- `frontend/dashboard/src/lib/api.ts`
- `frontend/dashboard/src/lib/sse.ts`
- `backend/app/api/forks.py`
- `backend/app/meta_harness/branches.py`

## Data surfaces are backed by real endpoints

Every panel in the dashboard reads from a backend route; none renders
fabricated content.

- Candidate diff is served by
  `GET /runs/{run_id}/candidates/{candidate_name}/diff`
  (`backend/app/api/runs.py`), computed against the candidate's own source
  snapshot.
- Candidate test output is served by
  `GET /runs/{run_id}/candidates/{candidate_name}/test-output`, read from the
  candidate's trial artifacts.
- Token and cost accounting is computed in
  `backend/app/meta_harness/metrics.py` and carried on every trial row.
  Unmeasured is `null`, never `0`, so an unmeasured candidate cannot be
  mistaken for a free one.
- `mode` in the dashboard is transport ("is a backend answering"), not
  provenance. Provenance lives on `run.metricsSource` and is surfaced in the
  status bar, so a live mock-bench run is never rendered as measured.

A panel with no data yet renders its empty state. That is the absence of a
result, not a stand-in for one.

## Quick verification checklist

- Fork a node in `/runs/{run_id}` and confirm a `POST /runs/{run_id}/fork` call
  is made with real checkpoint/thread context.
- Confirm fork-created events include `thread_id` and are visible in decision
  log/tree updates.
- Confirm run page remains connected to SSE and receives non-fork event types.

## Maintenance rule

When changing any of these surfaces in one PR, update all three together:

- `docs/INTERFACES.md` (if contract changed)
- relevant frontend/backend implementation files
- this clarity file if operator behavior changed
