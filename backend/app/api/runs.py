"""Run lifecycle REST endpoints."""

from __future__ import annotations

import asyncio
import difflib
import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response, status
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from app.meta_harness import runs as runs_mod
from app.meta_harness.branches import (
    BranchOwnedElsewhere,
    cancel_branch_outcome,
    forget_run,
    get_branch,
    is_live,
    list_branches,
    live_branches,
    owned_elsewhere,
)
from app.meta_harness.outer import OuterLoopRunner, initial_state
from app.meta_harness.persistence import delete_threads
from app.meta_harness.state import BASELINE_CANDIDATE_NAME
from app.streaming import channel_for_run, emit_run_event, event_registry


router = APIRouter(tags=["runs"])

#: How long a cancel waits for a run task to actually stop. Bounded, so
#: a stuck task cannot hang ``DELETE /runs/{id}`` or shutdown behind it.
#: The proposer stops its subprocess tree when its task is cancelled.
RUN_CANCEL_TIMEOUT_S = 10.0

#: Generic, path-free errors. A message that echoed the resolved path let
#: a caller probe the server's filesystem one request at a time.
_INVALID_SKILL = "invalid skill_path: expected skills/<name>/SKILL.md in this repository"
_INVALID_DOMAIN = "invalid domain: use 1-128 letters, numbers, '.', '_' or '-'"


@dataclass
class RunRecord:
    """In-process record for an API-started outer-loop run."""

    run_id: str
    thread_id: str
    status: str
    started_at: str
    domain: str
    skill_path: str | None
    budget: int
    model: str
    current_iteration: int
    run_dir: Path
    graph: Any
    runner: Any = None
    task: asyncio.Task[Any] | None = None
    error: str | None = None
    checkpointer: Any = None


class CreateRunRequest(BaseModel):
    domain: str = "coding-agent"
    skill_path: str | None = None
    budget: int = Field(default=5, ge=1)
    model: str = "opus"
    fresh: bool = True
    run_name: str | None = None
    proposer: Literal["claude", "mock"] = "claude"
    mock_bench: bool | None = None
    trials: int = Field(default=5, ge=1)
    workers: int = Field(default=3, ge=1)


run_registry: dict[str, RunRecord] = {}

#: Run ids a ``POST /runs`` is between its checks and its registration.
#: Loop-agnostic on purpose (an ``asyncio.Lock`` binds to one loop).
_creating: set[str] = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generated_run_id() -> str:
    return runs_mod.generated_run_id("run")


def _repo_root(request: Request) -> Path:
    return request.app.state.repo_root


def _eval_tasks_dir(request: Request) -> Path:
    return request.app.state.eval_tasks_dir


def _app_checkpointer(request: Request) -> Any:
    return getattr(request.app.state, "checkpointer", None)


def _app_memory_store(request: Request) -> Any:
    return getattr(request.app.state, "memory_store", None)


def _validated_domain(domain: str) -> str:
    """``domain`` is joined into ``skills/meta-harness-<domain>``: a safe name only."""
    try:
        return runs_mod.validate_domain(domain)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_INVALID_DOMAIN
        ) from None


def _resolve_skill_path(
    *,
    repo_root: Path,
    domain: str,
    skill_path: str | None,
    proposer: str,
) -> Path | None:
    """The SKILL.md the proposer will be given, or ``None`` for the mock.

    The file becomes the proposer's system prompt, so only
    ``<repo_root>/skills/<name>/SKILL.md`` is accepted — an absolute path
    or ``../`` used to turn any readable file into a prompt. An explicit
    ``skill_path`` is containment-checked even for the mock proposer, so
    bad input is rejected rather than recorded. Errors never echo a path.
    """
    try:
        resolved = (
            runs_mod.resolve_skill_path(repo_root, skill_path)
            if skill_path
            else runs_mod.default_skill_path(repo_root, domain)
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_INVALID_SKILL
        ) from None
    if proposer == "mock":
        return None
    if not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="skill not found"
        )
    return resolved


def _build_runner(
    *,
    run_dir: Path,
    repo_root: Path,
    eval_tasks_dir: Path,
    mock_proposer: bool,
    mock_bench: bool,
    trials: int,
    workers: int,
    skill_path: Path | None,
    checkpointer: Any,
    memory_store: Any = None,
) -> OuterLoopRunner:
    return OuterLoopRunner(
        run_dir=run_dir,
        repo_root=repo_root,
        eval_tasks_dir=eval_tasks_dir,
        mock_proposer=mock_proposer,
        mock_bench=mock_bench,
        trials=trials,
        bench_workers=workers,
        skill_path=skill_path,
        checkpointer=checkpointer,
        memory_store=memory_store,
    )


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def _read_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(run_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_manifest_status(run_dir: Path, **updates: Any) -> None:
    if not run_dir.is_dir():
        # The run was re-created or removed underneath this task; do not
        # resurrect a directory with a manifest-only skeleton.
        return
    manifest = _read_manifest(run_dir) or {}
    manifest.update(updates)
    runs_mod.write_json_atomic(_manifest_path(run_dir), manifest)


def _read_summary_rows(run_dir: Path, *, limit: int = 5) -> list[dict[str, Any]]:
    """Most recent evolution rows across every branch in the run."""
    rows = runs_mod.aggregate_evolution_rows(run_dir)
    return rows[-limit:] if limit else rows


def _read_all_summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    return runs_mod.aggregate_evolution_rows(run_dir)


def _candidate_artifact_dir(run_dir: Path, candidate_name: str) -> Path:
    """Locate a candidate's artifact dir in whichever branch owns it."""
    candidate_dir = runs_mod.find_candidate_dir(run_dir, candidate_name)
    if candidate_dir is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="candidate not found",
        )
    return candidate_dir


def _candidate_row(run_dir: Path, candidate_name: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in _read_all_summary_rows(run_dir)
            if row.get("candidate") == candidate_name
        ),
        None,
    )


def _candidate_source(
    *, repo_root: Path, run_dir: Path, candidate_name: str
) -> tuple[Path | None, str]:
    """Resolve a candidate's source file and the label to show in the diff.

    Prefers the branch-private snapshot — that is the source the branch
    actually benchmarked, and it cannot have been rewritten by a
    concurrently running branch. Falls back to the committed baseline.
    """
    snapshot = runs_mod.find_candidate_source(run_dir, candidate_name)
    if snapshot is not None:
        return snapshot, f"agents/{candidate_name}.py"
    try:
        safe = runs_mod.validate_artifact_name(candidate_name, kind="candidate")
    except ValueError:
        return None, f"agents/{candidate_name}.py"
    committed = repo_root / "agents" / f"{safe}.py"
    return (committed if committed.is_file() else None), f"agents/{safe}.py"


def _unified_candidate_diff(
    *,
    repo_root: Path,
    run_dir: Path,
    candidate_name: str,
) -> dict[str, str]:
    row = _candidate_row(run_dir, candidate_name) or {}
    parent = row.get("parent_candidate_name") or BASELINE_CANDIDATE_NAME
    parent_path, parent_label = _candidate_source(
        repo_root=repo_root, run_dir=run_dir, candidate_name=str(parent)
    )
    candidate_path, candidate_label = _candidate_source(
        repo_root=repo_root, run_dir=run_dir, candidate_name=candidate_name
    )
    if candidate_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="candidate source not found",
        )
    parent_text = (
        parent_path.read_text().splitlines(keepends=True)
        if parent_path is not None
        else []
    )
    candidate_text = candidate_path.read_text().splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            parent_text,
            candidate_text,
            fromfile=parent_label,
            tofile=candidate_label,
        )
    )
    return {
        "candidate": candidate_name,
        "parent": str(parent),
        "thread_id": str(row.get("thread_id") or run_dir.name),
        "from_path": parent_label,
        "to_path": candidate_label,
        "diff": diff,
    }


def _format_accuracy(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "0.0000"


def _candidate_test_output(candidate_dir: Path) -> str:
    """Render a candidate's recorded evaluation for the dashboard.

    Everything here comes from files the benchmark actually wrote. A
    candidate with no traces yields an empty string, so the UI shows an
    empty state rather than fixture text.
    """
    chunks: list[str] = []
    eval_path = candidate_dir / "eval-result.json"
    if eval_path.exists():
        eval_result = json.loads(eval_path.read_text())
        header = [
            f"candidate: {eval_result.get('candidate', candidate_dir.name)}",
            f"metrics_source: {eval_result.get('metrics_source', 'unknown')}",
            f"accuracy: {_format_accuracy(eval_result.get('accuracy'))}",
            f"tasks: {eval_result.get('n_tasks', 0)}",
            f"trials_per_task: {eval_result.get('n_trials_per_task', 0)}",
            f"llm_calls: {eval_result.get('total_llm_calls', 0)}",
            f"total_tokens: {(eval_result.get('tokens') or {}).get('total_tokens', 0)}",
        ]
        cost = eval_result.get("total_cost_usd")
        # Unknown cost is reported as unknown, never as $0.00.
        header.append(
            f"total_cost_usd: {cost}" if cost is not None else "total_cost_usd: unknown"
        )
        chunks.append("\n".join(header))
    for verify_path in sorted((candidate_dir / "traces").glob("*/*verify.json"))[:10]:
        verify = json.loads(verify_path.read_text())
        chunks.append(
            "\n".join(
                [
                    f"== {verify_path.parent.name} ==",
                    f"tests_pass: {verify.get('tests_pass', False)}",
                    str(verify.get("test_output", "")).strip(),
                ]
            )
        )
    return "\n\n".join(chunk for chunk in chunks if chunk.strip())


def _best_score(frontier: dict[str, Any] | None) -> float | None:
    if not frontier:
        return None
    best = frontier.get("_best") or {}
    score = best.get("accuracy")
    return float(score) if score is not None else None


def _root_frontier(run_dir: Path) -> dict[str, Any] | None:
    """The run's root-branch frontier (the dashboard's default view)."""
    return runs_mod.read_frontier(run_dir, run_dir.name)


def _branch_frontiers(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Every branch's frontier, keyed by thread_id."""
    return runs_mod.aggregate_frontiers(run_dir)


def _run_best_score(run_dir: Path) -> float | None:
    """Best measured accuracy across every branch in the run."""
    scores = [
        s
        for s in (_best_score(f) for f in _branch_frontiers(run_dir).values())
        if s is not None
    ]
    return max(scores) if scores else None


def _run_info_from_record(record: RunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "thread_id": record.thread_id,
        "status": record.status,
        "started_at": record.started_at,
        "domain": record.domain,
        "skill_path": record.skill_path,
        "budget": record.budget,
        "model": record.model,
        "current_iteration": record.current_iteration,
    }


def _run_info_from_files(run_dir: Path) -> dict[str, Any]:
    manifest = _read_manifest(run_dir) or {}
    frontier = _root_frontier(run_dir)
    rows = _read_summary_rows(run_dir, limit=5)
    current_iteration = manifest.get("current_iteration")
    if current_iteration is None and rows:
        current_iteration = rows[-1].get("iteration")
    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "thread_id": manifest.get("thread_id", manifest.get("run_id", run_dir.name)),
        "status": manifest.get("status", "unknown"),
        "started_at": manifest.get("started_at"),
        "domain": manifest.get("domain", "coding-agent"),
        "skill_path": manifest.get("skill_path"),
        "budget": manifest.get("budget"),
        "model": manifest.get("model"),
        "current_iteration": current_iteration or 0,
        "best_score": _best_score(frontier),
    }


def _full_run_info(run_dir: Path, record: RunRecord | None = None) -> dict[str, Any]:
    base = _run_info_from_record(record) if record else _run_info_from_files(run_dir)
    frontier = _root_frontier(run_dir)
    manifest = _read_manifest(run_dir) or {}
    branch_frontiers = _branch_frontiers(run_dir)
    base.update(
        {
            "manifest": manifest,
            "frontier_val": frontier,
            # Every branch's evolution rows, each tagged with the
            # thread_id that produced it, so the dashboard can draw the
            # whole search tree rather than one linear branch.
            "summary_rows": _read_all_summary_rows(run_dir),
            "branch_frontiers": branch_frontiers,
            "best_score": _run_best_score(run_dir),
            # "measured" | "mock" — the dashboard must be able to tell a
            # synthetic demo run from a real experiment at a glance.
            "metrics_source": manifest.get(
                "metrics_source",
                "mock" if manifest.get("mock_bench") else "measured",
            ),
        }
    )
    if record and record.error:
        base["error"] = record.error
    return base


async def _emit_checkpoint_events(record: RunRecord) -> None:
    from app.meta_harness.branches import get_state_history

    try:
        history = await get_state_history(record.graph, thread_id=record.thread_id)
    except Exception:
        return
    for checkpoint in reversed(history):
        emit_run_event(
            record.run_id,
            "checkpoint-written",
            {
                "thread_id": checkpoint.thread_id,
                "checkpoint_id": checkpoint.checkpoint_id,
                "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                "ts": checkpoint.ts,
                "node": checkpoint.node,
            },
            event_id=checkpoint.checkpoint_id,
        )


async def _execute_run(record: RunRecord, budget: int) -> None:
    """Measure the baseline, then drive the outer graph to completion.

    The baseline benchmark runs inside the task rather than in the POST
    handler so ``POST /runs`` still returns immediately, but it always
    completes *before* the first propose — the run's first candidate is
    compared against a real measurement, never against zero.
    """
    config = {
        "configurable": {"thread_id": record.thread_id},
        "recursion_limit": 200,
    }
    try:
        seeds = [await record.runner.benchmark_baseline(run_id=record.run_id)]
        state = initial_state(
            run_id=record.run_id, budget=budget, seed_candidates=seeds
        )
        final = await record.graph.ainvoke(state, config=config)
    except asyncio.CancelledError:
        record.status = "cancelled"
        # Synchronous on purpose: awaiting inside a cancellation handler
        # can itself be cancelled, and this write is what records it.
        _write_manifest_status(record.run_dir, status="cancelled")
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced through API + SSE
        record.status = "failed"
        record.error = str(exc)
        trace = traceback.format_exc()
        await asyncio.to_thread(
            _write_manifest_status, record.run_dir, status="failed", error=str(exc)
        )
        emit_run_event(
            record.run_id,
            "error",
            {
                "thread_id": record.thread_id,
                "node": "outer_loop",
                "message": str(exc),
                "traceback": trace,
            },
        )
        return

    record.status = "completed"
    record.current_iteration = int(final.get("iteration", record.current_iteration))
    await asyncio.to_thread(
        _write_manifest_status,
        record.run_dir,
        status="completed",
        current_iteration=record.current_iteration,
        finished_at=_now(),
    )
    await _emit_checkpoint_events(record)


def get_run_record(run_id: str) -> RunRecord | None:
    return run_registry.get(run_id)


def get_run_dir(request: Request, run_id: str) -> Path:
    """``runs/<run_id>`` for an existing run, or 404.

    A run is a directory with a ``manifest.json``; ``run_id`` is validated
    as an artifact name before it is joined, so traversal is a 404.
    """
    try:
        run_dir = runs_mod.make_run_path(_repo_root(request), run_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="run not found",
        ) from None
    if not _manifest_path(run_dir).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return run_dir


def resolve_run_thread(run_id: str, thread_id: str | None) -> str:
    """A thread of ``run_id``: its root thread or one of its forks, else 404.

    Every route that takes a thread id goes through this, so a request
    under one run cannot read, fork from or cancel another run's thread.
    """
    if thread_id is None or thread_id == run_id:
        return run_id
    branch = get_branch(thread_id, run_id=run_id)
    if branch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="thread not found in this run",
        )
    return branch.thread_id


async def get_run_graph(request: Request, run_id: str) -> Any:
    """The run's outer graph: the live one, or one rebuilt from its manifest.

    The rebuild reads the run's recorded config — workers, skill and
    domain included — rather than assuming the coding-agent defaults.
    """
    record = run_registry.get(run_id)
    if record is not None:
        return record.graph

    run_dir = get_run_dir(request, run_id)
    manifest = await asyncio.to_thread(_read_manifest, run_dir)
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="run manifest not found",
        )

    checkpointer = _app_checkpointer(request)
    if checkpointer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="run graph unavailable without an active in-process record",
        )

    repo_root = _repo_root(request)
    config = runs_mod.run_config_from_manifest(repo_root, manifest)
    runner = _build_runner(
        run_dir=run_dir,
        repo_root=repo_root,
        eval_tasks_dir=_eval_tasks_dir(request),
        mock_proposer=config["mock_proposer"],
        mock_bench=config["mock_bench"],
        trials=config["trials"],
        workers=config["workers"],
        skill_path=config["skill_path"],
        checkpointer=checkpointer,
        # A manifest that predates the ``memory`` key keeps the old
        # behaviour; one that says the run had no memory gets none.
        memory_store=(
            _app_memory_store(request) if manifest.get("memory", True) else None
        ),
    )
    return runner.build()


async def _stop_task(task: asyncio.Task[Any] | None) -> bool:
    """Cancel ``task`` and wait for it, bounded. True if it has stopped."""
    if task is None or task.done():
        return True
    task.cancel()
    await asyncio.wait({task}, timeout=RUN_CANCEL_TIMEOUT_S)
    return task.done()


async def cancel_active_runs() -> None:
    """Best-effort shutdown cleanup for API-started tasks.

    Cancels every live run task and waits for them together, bounded by
    ``RUN_CANCEL_TIMEOUT_S``, so shutdown cannot hang on one of them.
    """

    pending = [
        record.task
        for record in run_registry.values()
        if record.task is not None and not record.task.done()
    ]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=RUN_CANCEL_TIMEOUT_S)


def clear_run_state() -> None:
    """Clear the in-process run registry. Intended for tests."""

    run_registry.clear()
    _creating.clear()


async def _reset_run_state(request: Request, run_id: str) -> None:
    """Make ``run_id`` start clean: no old branches, checkpoints or events.

    A "fresh" re-run under an existing name used to inherit all three:
    the old branches were listed under the new run, the old SSE history
    was replayed to it, ``/checkpoints`` and the fork endpoint offered
    the old run's checkpoints, and still-running old branches kept
    writing into the new directory. Refuses (409) while any branch of the
    run is live in this or another process; nothing is deleted then.
    """
    live = live_branches(run_id)
    if live:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"run has {len(live)} live branch(es); cancel them before "
                "re-creating the run"
            ),
        )
    thread_ids = [run_id, *(b.thread_id for b in list_branches(run_id=run_id))]
    checkpointer = _app_checkpointer(request)
    if checkpointer is not None:
        try:
            await delete_threads(checkpointer, thread_ids)
        except Exception:  # noqa: BLE001 - reported, never silently kept
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="could not clear the previous run's checkpoint history",
            ) from None
    forget_run(run_id)
    event_registry.evict(channel_for_run(run_id))
    run_registry.pop(run_id, None)


@router.post("/runs", status_code=status.HTTP_201_CREATED)
async def create_run(
    payload: CreateRunRequest,
    request: Request,
    response: Response,
) -> dict[str, Any]:
    repo_root = _repo_root(request)
    domain = _validated_domain(payload.domain)
    run_id = payload.run_name if payload.run_name is not None else _generated_run_id()
    try:
        run_dir = runs_mod.make_run_path(repo_root, run_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from None
    # Every input is validated before anything on disk is touched.
    skill_path = _resolve_skill_path(
        repo_root=repo_root,
        domain=domain,
        skill_path=payload.skill_path,
        proposer=payload.proposer,
    )
    if run_id in _creating:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="run is being created",
        )
    existing = run_registry.get(run_id)
    if existing is not None and existing.task is not None and not existing.task.done():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="run already active",
        )
    if run_dir.exists() and not payload.fresh:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="run already exists",
        )

    mock_proposer = payload.proposer == "mock"
    mock_bench = payload.mock_bench if payload.mock_bench is not None else mock_proposer
    memory_store = _app_memory_store(request)
    manifest_skill = runs_mod.skill_path_for_manifest(repo_root, skill_path)
    started_at = _now()

    _creating.add(run_id)
    try:
        # A new run never inherits a previous run's branches, checkpoints
        # or stream history, whether or not its directory still exists.
        await _reset_run_state(request, run_id)
        run_dir = await asyncio.to_thread(
            runs_mod.make_run_dir, repo_root, run_id, fresh=payload.fresh
        )
        await asyncio.to_thread(
            runs_mod.write_manifest,
            run_dir,
            # The run config every rebuild/resume path reads back (the
            # same keys ``run_outer_loop`` writes for a CLI run)...
            run_id=run_id,
            thread_id=run_id,
            budget=payload.budget,
            trials=payload.trials,
            workers=payload.workers,
            mock_proposer=mock_proposer,
            mock_bench=mock_bench,
            metrics_source="mock" if mock_bench else "measured",
            domain=domain,
            skill_path=manifest_skill,
            record=False,
            memory=memory_store is not None,
            # ...plus what only the API tracks.
            proposer=payload.proposer,
            model=payload.model,
            status="running",
            started_at=started_at,
            current_iteration=0,
        )

        checkpointer = _app_checkpointer(request) or MemorySaver()
        runner = _build_runner(
            run_dir=run_dir,
            repo_root=repo_root,
            eval_tasks_dir=_eval_tasks_dir(request),
            mock_proposer=mock_proposer,
            mock_bench=mock_bench,
            trials=payload.trials,
            workers=payload.workers,
            skill_path=skill_path,
            checkpointer=checkpointer,
            memory_store=memory_store,
        )
        graph = runner.build()
        record = RunRecord(
            run_id=run_id,
            thread_id=run_id,
            status="running",
            started_at=started_at,
            domain=domain,
            skill_path=manifest_skill,
            budget=payload.budget,
            model=payload.model,
            current_iteration=0,
            run_dir=run_dir,
            graph=graph,
            runner=runner,
            checkpointer=checkpointer,
        )
        record.task = asyncio.create_task(
            _execute_run(record, payload.budget),
            name=f"run:{run_id}",
        )
        run_registry[run_id] = record
    finally:
        _creating.discard(run_id)

    response.headers["Location"] = f"/runs/{run_id}"
    return _run_info_from_record(record)


def _list_runs(runs_root: Path) -> list[dict[str, Any]]:
    """Summaries of every run under ``runs_root``. Read-only."""
    runs: list[dict[str, Any]] = []
    for run_dir in runs_mod.list_run_dirs(runs_root):
        record = run_registry.get(run_dir.name)
        try:
            info = (
                _run_info_from_record(record) if record else _run_info_from_files(run_dir)
            )
            if "best_score" not in info:
                info["best_score"] = _run_best_score(run_dir)
        except (OSError, ValueError):
            # One unreadable run (a torn manifest, say) must not take the
            # whole listing down with it.
            continue
        runs.append(
            {
                "run_id": info["run_id"],
                "thread_id": info["thread_id"],
                "status": info["status"],
                "started_at": info["started_at"],
                "current_iteration": info["current_iteration"],
                "best_score": info.get("best_score"),
            }
        )
    return runs


@router.get("/runs")
async def list_runs(request: Request) -> dict[str, Any]:
    runs_root = _repo_root(request) / "runs"
    return {"runs": await asyncio.to_thread(_list_runs, runs_root)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    run_dir = get_run_dir(request, run_id)
    return await asyncio.to_thread(_full_run_info, run_dir, run_registry.get(run_id))


@router.get("/runs/{run_id}/candidates/{candidate_name}/diff")
async def get_candidate_diff(
    run_id: str,
    candidate_name: str,
    request: Request,
) -> dict[str, str]:
    run_dir = get_run_dir(request, run_id)
    await asyncio.to_thread(_candidate_artifact_dir, run_dir, candidate_name)
    return await asyncio.to_thread(
        lambda: _unified_candidate_diff(
            repo_root=_repo_root(request),
            run_dir=run_dir,
            candidate_name=candidate_name,
        )
    )


@router.get("/runs/{run_id}/candidates/{candidate_name}/test-output")
async def get_candidate_test_output(
    run_id: str,
    candidate_name: str,
    request: Request,
) -> dict[str, str]:
    run_dir = get_run_dir(request, run_id)
    candidate_dir = await asyncio.to_thread(
        _candidate_artifact_dir, run_dir, candidate_name
    )
    return {
        "candidate": candidate_name,
        "output": await asyncio.to_thread(_candidate_test_output, candidate_dir),
    }


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, request: Request) -> dict[str, str]:
    """Cancel a run's live work: its root task and every live branch.

    Only live work is touched. A completed or failed run keeps its
    status (it used to be relabelled ``cancelled``), and a branch another
    live process drives makes the whole request a 409, before anything is
    cancelled, because this process cannot stop it. The run's SSE replay
    history is dropped once its subscribers have seen the cancellation.
    """
    run_dir = get_run_dir(request, run_id)
    record = run_registry.get(run_id)
    branches = list_branches(run_id=run_id)
    foreign = [b.thread_id for b in branches if owned_elsewhere(b)]
    if foreign:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"{len(foreign)} branch(es) of this run are running in another "
                "process; cancel them there first"
            ),
        )

    run_was_live = (
        record is not None and record.task is not None and not record.task.done()
    )
    stopped = await _stop_task(record.task) if run_was_live else True

    for branch in branches:
        if not is_live(branch):
            continue
        try:
            metadata, changed = await cancel_branch_outcome(
                branch.thread_id, run_id=run_id
            )
        except (KeyError, BranchOwnedElsewhere):
            continue
        if changed:
            emit_run_event(
                run_id,
                "branch-cancelled",
                {"thread_id": metadata.thread_id, "reason": "run-cancelled"},
            )

    if run_was_live and stopped and record.status == "running":
        # Cancelled before its first step: its handler never ran.
        record.status = "cancelled"
        await asyncio.to_thread(_write_manifest_status, run_dir, status="cancelled")

    if record is not None:
        current = record.status if stopped else "cancelling"
    else:
        manifest = await asyncio.to_thread(_read_manifest, run_dir) or {}
        current = str(manifest.get("status", "unknown"))
    event_registry.evict(channel_for_run(run_id))
    return {"status": current}
