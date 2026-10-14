"""Shared measured-benchmark core: (tasks × trials) → raw rows + aggregate.

One implementation, three callers — the outer loop's ``benchmark`` node,
``meta-harness benchmark``, and the pass-rate experiment runner. They
previously had three copies of the trial loop, and two of them wrote
placeholder ``{"input_tokens": 0, "cost_usd": 0.0}`` blocks that looked
like measurements.

Every trial here produces a raw row (see ``metrics.UsageRecorder``), and
every summary is derived mechanically from those rows. Nothing in this
module invents a number.

A trial that raises does not take the run down with it. It becomes a row
with ``status: "error"``, a failed outcome and unknown usage (see
``metrics.error_trial_row``), and the summary reports how many there
were. Only cancellation, or a failure outside any single trial, stops a
run — and then every sibling trial is cancelled and awaited before the
exception propagates, so nothing keeps running unattended.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.meta_harness import effects as fx
from app.meta_harness import inner as inner_mod
from app.meta_harness import metrics as met
from app.meta_harness import recording as rec
from app.meta_harness import runs as runs_mod
from app.meta_harness.inner import build_inner_graph, initial_inner_state
from app.meta_harness.sandbox import sandbox_for


def discover_tasks(tasks_dir: Path) -> list[Path]:
    """Every task directory under ``tasks_dir``, sorted by id."""
    if not tasks_dir.is_dir():
        return []
    return sorted(
        d for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.json").exists()
    )


def recursion_limit_for(harness: Any) -> int:
    """LangGraph step budget for one inner-loop trial of ``harness``.

    The inner module owns the exact count (``inner.recursion_limit_for``).
    Where it does not provide one, the bound is computed from the same
    two knobs it would use: every verify retry can cost a full act phase
    plus its verify, and an act phase may take up to ``MAX_ACT_TURNS``
    steps. A fixed limit silently truncated candidates that raised either
    knob, and a truncated trial reads as a failure of the harness.
    """
    exact = getattr(inner_mod, "recursion_limit_for", None)
    if exact is not None:
        return int(exact(harness))
    act_turns = int(getattr(harness, "MAX_ACT_TURNS", 25) or 25)
    retries = int(getattr(harness, "MAX_VERIFY_RETRIES", 3) or 0)
    # orient + plan + submit, then (act turns + verify) per attempt, with
    # headroom for routing steps.
    return 3 + (retries + 1) * (act_turns + 2) + 10


@dataclass(frozen=True)
class TrialSpec:
    """One (harness, task, trial) unit of work, fully resolved."""

    task_dir: Path
    spec: dict[str, Any]
    trial: int
    harness_factory: Callable[[], Any]
    thread_id: str
    trace_dir: Path | None = None
    recording_dir: Path | None = None
    arm: str | None = None


def _grade_fields(final: dict[str, Any]) -> dict[str, Any]:
    """The grading facts a trial row carries, read defensively.

    ``grade_result`` is written by the inner loop's ``submit`` node. A
    final state without one (an older inner loop, or a scripted graph)
    reports ``None`` for each field: unknown, not "no changes".
    """
    grade = final.get("grade_result")
    if not isinstance(grade, dict):
        grade = {}
    return {
        "agent_verify_passed": grade.get("agent_verify_passed"),
        "out_of_plan_changes": grade.get("out_of_plan_changes"),
    }


async def _execute_trial(spec: TrialSpec, *, checkpointer: Any) -> dict[str, Any]:
    """Run one trial and return its raw row. Never raises for trial errors.

    Called with the worker semaphore already held, so ``wall_time_s``
    measures the trial and not the time it spent queued behind others.
    """
    task_id = spec.task_dir.name
    started = time.monotonic()
    harness: Any = None
    try:
        harness = spec.harness_factory()
        writer = rec.TapeWriter() if spec.recording_dir is not None else None
        # The effects boundary sits INSIDE the usage recorder, so a
        # later replay's recorder still sees a response (the recorded
        # one) and reports the recorded token counts.
        effects: fx.Effects = (
            fx.RecordingEffects(writer) if writer is not None else fx.Effects()
        )
        if writer is not None:
            fx.instrument_harness_for_effects(harness, effects)
        usage = met.UsageRecorder()
        met.instrument_harness(harness, usage)
        workspace = spec.task_dir / "workspace"
        with sandbox_for(workspace) as sandbox:
            task_dict = dict(spec.spec)
            # The pristine workspace the submit node grades against. It
            # never leaves this process's view of the task, and the
            # agent's sandbox is a copy, so the agent cannot edit it.
            task_dict["_source_workspace"] = str(workspace)
            if spec.trace_dir is not None:
                spec.trace_dir.mkdir(parents=True, exist_ok=True)
                task_dict["_trace_dir"] = str(spec.trace_dir)
            graph = build_inner_graph(
                harness, checkpointer=checkpointer, effects=effects
            )
            if writer is not None:
                # Imported lazily: ``replay`` imports ``inner``, which
                # this module also builds graphs from, so a module-level
                # import would close a cycle.
                from app.meta_harness import replay as replay_mod  # noqa: PLC0415

                await replay_mod.assert_thread_unused(graph, thread_id=spec.thread_id)
            final = await graph.ainvoke(
                initial_inner_state(task_dict=task_dict, workspace=sandbox),
                config={
                    "configurable": {"thread_id": spec.thread_id},
                    "recursion_limit": recursion_limit_for(harness),
                },
            )
            if writer is not None:
                from app.meta_harness import replay as replay_mod  # noqa: PLC0415

                await replay_mod.finalize_recording(
                    graph,
                    writer=writer,
                    thread_id=spec.thread_id,
                    task_dict=task_dict,
                    workspace=sandbox,
                    harness=harness,
                    usage=usage,
                    final_state=final,
                    recording_dir=spec.recording_dir,  # type: ignore[arg-type]
                )
    except Exception as exc:  # noqa: BLE001 — one trial must not abort the run
        row = met.error_trial_row(
            task_id=task_id,
            trial=spec.trial,
            error=exc,
            wall_time_s=time.monotonic() - started,
            model=getattr(harness, "MODEL", None) if harness is not None else None,
        )
        row.update({"agent_verify_passed": None, "out_of_plan_changes": None})
    else:
        row = usage.to_trial_row(
            task_id=task_id,
            trial=spec.trial,
            passed=(final.get("score") or 0.0) >= 1.0,
            score=float(final.get("score") or 0.0),
            wall_time_s=time.monotonic() - started,
            model=getattr(harness, "MODEL", None),
        )
        row.update(_grade_fields(final))

    row["inner_thread_id"] = spec.thread_id
    if spec.arm is not None:
        row["arm"] = spec.arm
    if spec.recording_dir is not None:
        row["recording_dir"] = str(spec.recording_dir)
    if spec.trace_dir is not None:
        runs_mod.write_json_atomic(spec.trace_dir / "metrics.json", row)
    return row


async def run_trial_specs(
    specs: list[TrialSpec],
    *,
    workers: int,
    checkpointer: Any = None,
    on_trial: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run trials under one worker pool, in the order given.

    Trials start in list order: the semaphore admits waiters first come,
    first served, so a caller that interleaves two arms gets them
    interleaved in time as well. Rows come back in the same order.

    Recording needs a checkpointer, because a replay is positioned by
    checkpoint; asking for a tape without one is a configuration error
    and fails before any trial starts rather than as N error rows.
    """
    for spec in specs:
        if spec.recording_dir is not None and checkpointer is None:
            raise ValueError(
                f"trial {spec.task_dir.name}/{spec.trial} asked to be recorded but "
                "no checkpointer is configured; a tape with no checkpoint history "
                "cannot be replayed from a checkpoint"
            )

    sem = asyncio.Semaphore(max(1, workers))
    rows: list[dict[str, Any] | None] = [None] * len(specs)

    async def _run(index: int, spec: TrialSpec) -> None:
        async with sem:
            row = await _execute_trial(spec, checkpointer=checkpointer)
        rows[index] = row
        if on_trial is not None:
            on_trial(row)

    tasks = [
        asyncio.create_task(_run(index, spec), name=f"trial:{spec.thread_id}")
        for index, spec in enumerate(specs)
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        # Cancellation, or a failure no single trial owns (a progress
        # callback raising): stop every sibling and wait for it, so no
        # trial keeps spending against a run that has already ended.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return [row for row in rows if row is not None]


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
    recording_dir_for: Callable[[str, int], Path | None] | None = None,
) -> list[dict[str, Any]]:
    """Run (tasks × trials) inner-loop trials and return raw metric rows.

    ``harness_factory`` is called once per trial so each trial gets a
    fresh harness instance (and therefore its own usage recorder).

    ``recording_dir_for`` opts a trial into execution recording: every
    model response, tool result and workspace observation is taped so the
    trial can later be replayed exactly (see ``recording.py``). Recording
    needs a checkpointer, because a replay is positioned by checkpoint;
    a trial that asks for one without a checkpointer fails loudly rather
    than writing a tape nothing can replay from.
    """
    specs = [
        TrialSpec(
            task_dir=td,
            spec=json.loads((td / "task.json").read_text()),
            trial=t,
            harness_factory=harness_factory,
            thread_id=inner_thread_id_for(td.name, t),
            trace_dir=trace_dir_for(td.name, t),
            recording_dir=recording_dir_for(td.name, t) if recording_dir_for else None,
        )
        for td in task_dirs
        for t in range(1, trials + 1)
    ]
    return await run_trial_specs(
        specs, workers=workers, checkpointer=checkpointer, on_trial=on_trial
    )


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
            "errored_trials": sum(1 for r in task_rows if met.is_error_row(r)),
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
    recording_root: Path | None = None,
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

    def _recording_dir(task_id: str, trial: int) -> Path | None:
        if recording_root is None:
            return None
        return recording_root / f"{task_id}-trial-{trial}"

    rows = await run_trials(
        harness_factory=harness_factory,
        task_dirs=task_dirs,
        trials=trials,
        workers=workers,
        trace_dir_for=_trace_dir,
        inner_thread_id_for=_thread_id,
        checkpointer=checkpointer,
        on_trial=on_trial,
        recording_dir_for=_recording_dir if recording_root is not None else None,
    )
    return summarize(rows, task_ids=[d.name for d in task_dirs], trials=trials)
