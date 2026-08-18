"""Shared measured-benchmark core: (tasks × trials) → raw rows + aggregate.

One implementation, three callers — the outer loop's ``benchmark`` node,
``meta-harness benchmark``, and the resume-claim experiment runner. They
previously had three copies of the trial loop, and two of them wrote
placeholder ``{"input_tokens": 0, "cost_usd": 0.0}`` blocks that looked
like measurements.

Every trial here produces a raw row (see ``metrics.UsageRecorder``), and
every summary is derived mechanically from those rows. Nothing in this
module invents a number.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable

from app.meta_harness import metrics as met
from app.meta_harness import runs as runs_mod
from app.meta_harness.inner import run_inner_loop
from app.meta_harness.sandbox import sandbox_for


def discover_tasks(tasks_dir: Path) -> list[Path]:
    """Every task directory under ``tasks_dir``, sorted by id."""
    if not tasks_dir.is_dir():
        return []
    return sorted(
        d for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.json").exists()
    )


async def run_trials(
    *,
    harness_factory: Callable[[], Any],
    task_dirs: list[Path],
    trials: int,
    workers: int,
    trace_dir_for: Callable[[str, int], Path | None],
    inner_thread_id_for: Callable[[str, int], str],
    checkpointer: Any = None,
    on_trial: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run (tasks × trials) inner-loop trials and return raw metric rows.

    ``harness_factory`` is called once per trial so each trial gets a
    fresh harness instance (and therefore its own usage recorder).
    """
    work = [
        (td, json.loads((td / "task.json").read_text()), t)
        for td in task_dirs
        for t in range(1, trials + 1)
    ]
    sem = asyncio.Semaphore(max(1, workers))

    async def _one(td: Path, spec: dict[str, Any], trial_idx: int) -> dict[str, Any]:
        task_id = td.name
        trace_dir = trace_dir_for(task_id, trial_idx)
        thread_id = inner_thread_id_for(task_id, trial_idx)
        started = time.monotonic()
        async with sem:
            harness = harness_factory()
            usage = met.UsageRecorder()
            met.instrument_harness(harness, usage)
            with sandbox_for(td / "workspace") as sandbox:
                final = await run_inner_loop(
                    harness,
                    task_dict=spec,
                    workspace=sandbox,
                    trace_dir=trace_dir,
                    thread_id=thread_id,
                    checkpointer=checkpointer,
                )
        row = usage.to_trial_row(
            task_id=task_id,
            trial=trial_idx,
            passed=(final.get("score") or 0.0) >= 1.0,
            score=float(final.get("score") or 0.0),
            wall_time_s=round(time.monotonic() - started, 3),
        )
        row["inner_thread_id"] = thread_id
        if trace_dir is not None:
            runs_mod.write_json_atomic(trace_dir / "metrics.json", row)
        if on_trial is not None:
            on_trial(row)
        return row

    rows = await asyncio.gather(*[_one(td, spec, t) for td, spec, t in work])
    return list(rows)


def per_task_breakdown(
    rows: list[dict[str, Any]], task_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Pass rate + ordered per-trial outcomes for each task."""
    out: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        task_rows = sorted(
            (r for r in rows if r["task_id"] == task_id), key=lambda r: r["trial"]
        )
        outcomes = [bool(r["passed"]) for r in task_rows]
        out[task_id] = {
            "pass_rate": round(sum(outcomes) / len(outcomes), 4) if outcomes else 0.0,
            "trials": outcomes,
        }
    return out


def summarize(
    rows: list[dict[str, Any]],
    *,
    task_ids: list[str],
    trials: int,
    metrics_source: str = met.MEASURED,
) -> dict[str, Any]:
    """Assemble an ``eval-result.json`` payload from raw trial rows."""
    passes = sum(1 for r in rows if r["passed"])
    accuracy = passes / len(rows) if rows else 0.0
    return {
        "n_tasks": len(task_ids),
        "n_trials_per_task": trials,
        "accuracy": round(accuracy, 4),
        "per_task": per_task_breakdown(rows, task_ids),
        "trials": rows,
        "metrics_source": metrics_source,
        "_mock_bench": metrics_source == met.MOCK,
        **met.aggregate_trials(rows, metrics_source=metrics_source),
    }


async def benchmark_harness(
    *,
    harness_factory: Callable[[], Any],
    tasks_dir: Path,
    trials: int,
    workers: int,
    trace_root: Path | None,
    thread_prefix: str,
    checkpointer: Any = None,
    on_trial: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Benchmark one harness over a task set. Returns the eval-result payload."""
    task_dirs = discover_tasks(tasks_dir)
    if not task_dirs:
        raise ValueError(f"no tasks found in {tasks_dir}")

    def _trace_dir(task_id: str, trial: int) -> Path | None:
        if trace_root is None:
            return None
        return trace_root / f"{task_id}-trial-{trial}"

    def _thread_id(task_id: str, trial: int) -> str:
        return f"{thread_prefix}::{task_id}::trial-{trial}"

    rows = await run_trials(
        harness_factory=harness_factory,
        task_dirs=task_dirs,
        trials=trials,
        workers=workers,
        trace_dir_for=_trace_dir,
        inner_thread_id_for=_thread_id,
        checkpointer=checkpointer,
        on_trial=on_trial,
    )
    return summarize(rows, task_ids=[d.name for d in task_dirs], trials=trials)
