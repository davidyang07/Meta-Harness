"""Meta-Harness CLI entry — ``meta-harness <subcommand>``.

Real subcommands land progressively across BUILD_ORDER steps. The
``meta-harness inner`` command (step 3) runs one inner-loop trial on a
single eval task. The ``loop``, ``benchmark``, ``fork``, ``resume``,
``init``, and ``memory`` subcommands land at later steps.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv

# The meta-harness CLI imports ``agents.<n>`` dynamically at runtime.
# ``agents/`` lives at the repo root, so we add it to sys.path before
# importing anything that depends on it.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env from the repo root so ANTHROPIC_API_KEY / POSTGRES_DSN are
# available before any subcommand instantiates a harness or a saver.
load_dotenv(REPO_ROOT / ".env")


def _run_async(coro: Any) -> Any:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


app = typer.Typer(
    name="meta-harness",
    help="Meta-Harness — LangGraph-native substrate for self-improving agent harnesses.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Show the Meta-Harness version."""
    from app import __version__

    typer.echo(f"meta-harness {__version__}")


@app.command()
def inner(
    task: str = typer.Option(
        ...,
        "--task",
        help="Task id (e.g. task-001-fix-typo)",
    ),
    candidate: str = typer.Option(
        "baseline",
        "--candidate",
        help="Candidate harness module name under agents/ (default: baseline)",
    ),
    run_name: str = typer.Option(
        "inner-test",
        "--run-name",
        help="Run name for trace output dir (runs/{run_name}/...)",
    ),
    holdout: bool = typer.Option(
        False,
        "--holdout",
        help="Resolve task from eval/holdout/ instead of eval/tasks/",
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help=(
            "Tape every model response, tool result and workspace observation "
            "so this trial can later be replayed exactly with "
            "`meta-harness verify-replay`. Requires Postgres: a replay is "
            "positioned by checkpoint."
        ),
    ),
) -> None:
    """Run ONE inner-loop trial on a single task (async)."""
    from app.meta_harness import metrics as met
    from app.meta_harness import runs as runs_mod
    from app.meta_harness.inner import run_inner_loop
    from app.meta_harness.sandbox import sandbox_for

    eval_root = REPO_ROOT / "eval"
    task_dir = (eval_root / ("holdout" if holdout else "tasks")) / task
    if not task_dir.exists():
        typer.echo(f"task not found: {task_dir}", err=True)
        raise typer.Exit(1)
    task_spec = json.loads((task_dir / "task.json").read_text())

    harness = _resolve_harness_class(candidate)()
    usage = met.UsageRecorder()
    if not record:
        # In the recording path the usage recorder is installed by
        # ``record_inner_execution``, outside the effects boundary, so a
        # later replay reports the recorded token counts. Installing it
        # twice would record every call twice.
        met.instrument_harness(harness, usage)

    run_dir = runs_mod.make_run_dir(REPO_ROOT, run_name)
    trace_dir = (
        runs_mod.candidate_dir(run_dir, run_name, candidate)
        / "traces"
        / f"{task}-trial-1"
    )
    thread_id = met.inner_thread_id(
        run_id=run_name,
        thread_id=run_name,
        candidate=candidate,
        task_id=task,
        trial=1,
    )

    recording_dir = (
        runs_mod.thread_dir(run_dir, run_name)
        / "recordings"
        / f"{candidate}--{task}-trial-1"
        if record
        else None
    )

    async def _run() -> dict[str, Any]:
        if recording_dir is None:
            with sandbox_for(task_dir / "workspace") as sandbox:
                return await run_inner_loop(
                    harness,
                    task_dict=task_spec,
                    workspace=sandbox,
                    trace_dir=trace_dir,
                    thread_id=thread_id,
                )

        from app.meta_harness import replay as replay_mod  # noqa: PLC0415
        from app.meta_harness.persistence import persistence_layer  # noqa: PLC0415

        async with persistence_layer() as saver:
            with sandbox_for(task_dir / "workspace") as sandbox:
                recorded = await replay_mod.record_inner_execution(
                    harness_factory=lambda: harness,
                    task_dict=task_spec,
                    workspace=sandbox,
                    thread_id=thread_id,
                    checkpointer=saver,
                    recording_dir=recording_dir,
                    trace_dir=trace_dir,
                    usage=usage,
                )
        return recorded["final_state"]

    started = time.monotonic()
    final_state = _run_async(_run())
    row = usage.to_trial_row(
        task_id=task,
        trial=1,
        passed=(final_state.get("score") or 0.0) >= 1.0,
        score=float(final_state.get("score") or 0.0),
        wall_time_s=time.monotonic() - started,
    )
    row["inner_thread_id"] = thread_id
    runs_mod.write_json_atomic(trace_dir / "metrics.json", row)

    typer.echo(
        json.dumps(
            {
                "task": task,
                "candidate": candidate,
                "score": final_state.get("score"),
                "passed": row["passed"],
                "turn_count": final_state.get("turn_count"),
                "verify_attempts": final_state.get("verify_attempts"),
                "llm_calls": row["llm_calls"],
                "total_tokens": row["total_tokens"],
                "cost_usd": row["cost_usd"],
                "wall_time_s": row["wall_time_s"],
                "trace_dir": str(trace_dir),
                "thread_id": thread_id,
                "recording_dir": str(recording_dir) if recording_dir else None,
            },
            indent=2,
        )
    )


@app.command()
def benchmark(
    candidate: str = typer.Option(
        "baseline",
        "--candidate",
        help="Candidate harness module name under agents/",
    ),
    trials: int = typer.Option(
        5,
        "--trials",
        help="Trials per task (default: 5, matches Appendix C §C.11)",
    ),
    workers: int = typer.Option(
        5,
        "--workers",
        help="Parallel workers across (task × trial) tuples",
    ),
    run_name: str = typer.Option(
        None,
        "--run-name",
        help="Run dir under runs/. Auto-generated if omitted.",
    ),
    holdout: bool = typer.Option(
        False,
        "--holdout",
        help="Resolve tasks from eval/holdout/ instead of eval/tasks/",
    ),
) -> None:
    """Benchmark one candidate: N trials × M tasks, with measured metrics.

    Writes raw per-trial rows and an aggregate to
    ``runs/{run_name}/threads/{run_name}/candidates/{candidate}/eval-result.json``.
    Uses the same benchmark core as the outer loop's ``benchmark`` node,
    so a CLI benchmark and an in-loop benchmark are directly comparable.
    """
    import datetime

    from app.meta_harness import benchmark as bench
    from app.meta_harness import runs as runs_mod

    eval_root = REPO_ROOT / "eval"
    tasks_root = eval_root / ("holdout" if holdout else "tasks")
    task_dirs = bench.discover_tasks(tasks_root)
    if not task_dirs:
        typer.echo(f"no tasks found in {tasks_root}", err=True)
        raise typer.Exit(1)

    if run_name is None:
        run_name = "bench-" + datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

    harness_class = _resolve_harness_class(candidate)
    total = len(task_dirs) * trials
    typer.echo(
        f"benchmark: candidate={candidate}, tasks={len(task_dirs)}, "
        f"trials={trials}, total={total}, workers={workers}, run={run_name}"
    )

    run_dir = runs_mod.make_run_dir(REPO_ROOT, run_name)
    cand_dir = runs_mod.candidate_dir(run_dir, run_name, candidate)
    n_done = 0

    def _progress(row: dict[str, Any]) -> None:
        nonlocal n_done
        n_done += 1
        mark = "PASS" if row["passed"] else "FAIL"
        typer.echo(
            f"  [{n_done}/{total}] {mark} {row['task_id']} trial-{row['trial']} "
            f"({row['total_tokens']} tok, {row['wall_time_s']}s)"
        )

    eval_result = _run_async(
        bench.benchmark_harness(
            harness_factory=harness_class,
            tasks_dir=tasks_root,
            trials=trials,
            workers=workers,
            trace_root=cand_dir / ("holdout-traces" if holdout else "traces"),
            thread_prefix=f"cli-bench::{run_name}::{candidate}",
        )
    )
    eval_result["candidate"] = candidate
    eval_result["task_set"] = "holdout" if holdout else "search"
    eval_result["timestamp"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()

    eval_result_path = cand_dir / (
        "holdout-result.json" if holdout else "eval-result.json"
    )
    runs_mod.write_json_atomic(eval_result_path, eval_result)

    # Raw rows are large; keep them out of the console summary but on disk.
    summary = {k: v for k, v in eval_result.items() if k != "trials"}
    typer.echo("")
    typer.echo(json.dumps(summary, indent=2))
    typer.echo("")
    typer.echo(f"wrote {eval_result_path}")


def _resolve_harness_class(candidate: str) -> type:
    """Import a candidate harness class by name under ``agents/``."""
    import importlib

    from app.meta_harness.harness import CodingAgentHarness

    if candidate == "baseline":
        from agents.baseline import BaselineHarness

        return BaselineHarness
    try:
        mod = importlib.import_module(f"agents.{candidate}")
    except ImportError as exc:
        typer.echo(f"failed to import agents.{candidate}: {exc}", err=True)
        raise typer.Exit(1) from None
    cls = _find_harness_class(mod)
    if cls is None:
        typer.echo(
            f"agents.{candidate} does not export a CodingAgentHarness subclass",
            err=True,
        )
        raise typer.Exit(1)
    assert issubclass(cls, CodingAgentHarness)
    return cls


@app.command()
def loop(
    proposer: str = typer.Option(
        "claude",
        "--proposer",
        help="Proposer mode: 'claude' (real subprocess) or 'mock' (deterministic stub)",
    ),
    budget: int = typer.Option(
        5,
        "--budget",
        help="Number of outer-loop iterations",
    ),
    trials: int = typer.Option(
        5,
        "--trials",
        help="Trials per task during benchmark phase",
    ),
    workers: int = typer.Option(
        3,
        "--workers",
        help="Parallel workers for benchmark phase",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Wipe runs/<run-name>/ before starting",
    ),
    run_name: str = typer.Option(
        None,
        "--run-name",
        help="Run dir under runs/. Auto-generated if omitted.",
    ),
    domain: str = typer.Option(
        "coding-agent",
        "--domain",
        help="SKILL.md domain name (resolved to skills/meta-harness-<domain>/SKILL.md)",
    ),
    skill: str = typer.Option(
        None,
        "--skill",
        help="Override skill path (per INTERFACES.md §5.3)",
    ),
    mock_bench: bool = typer.Option(
        False,
        "--mock-bench",
        help=(
            "Synthesize scores instead of running the inner loop. Useful "
            "for fast outer-loop testing (BUILD_ORDER step 5 DoD)."
        ),
    ),
    holdout: bool = typer.Option(
        False,
        "--holdout",
        help=(
            "After the main loop completes, post-evaluate the best "
            "candidate against eval/holdout/ and write holdout-result.json. "
            "The proposer never sees holdout tasks (honest reporting per "
            "Appendix C §C.14). No-op when --mock-bench is set."
        ),
    ),
    persistent: bool = typer.Option(
        True,
        "--persistent/--no-persistent",
        help=(
            "Use AsyncPostgresSaver checkpointing (step 7). Disable to "
            "skip checkpoint persistence (in-memory; mock-test mode)."
        ),
    ),
    wandb: bool = typer.Option(
        None,
        "--wandb/--no-wandb",
        help=(
            "Log iterations, per-task results, token usage and the Pareto "
            "frontier to Weights & Biases. Optional; defaults to "
            "META_HARNESS_WANDB. WANDB_MODE=offline works without network."
        ),
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help=(
            "Tape every inner-loop trial for exact replay. Requires "
            "--persistent, since a replay is positioned by checkpoint."
        ),
    ),
) -> None:
    """Run the meta-harness outer loop (async).

    Step 5 DoD: ``meta-harness loop --proposer mock --mock-bench
    --budget 2 --fresh`` runs 2 iterations and writes
    pending_eval.json, frontier_val.json, evolution_summary.jsonl.
    Step 7 DoD: ``--persistent`` (default ON when POSTGRES_DSN
    resolves) checkpoints every transition; ``meta-harness resume
    <run-name>`` continues from the last checkpoint.
    """
    import asyncio
    import datetime as _dt

    from app.meta_harness.outer import run_outer_loop
    from app.meta_harness.persistence import persistence_layer
    from app.meta_harness.runs import make_run_dir

    if proposer not in {"claude", "mock"}:
        typer.echo(f"--proposer must be 'claude' or 'mock' (got {proposer!r})", err=True)
        raise typer.Exit(2)

    if run_name is None:
        run_name = "loop-" + _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_dir = make_run_dir(REPO_ROOT, run_name, fresh=fresh)
    # Search always runs on eval/tasks/. Holdout post-eval (if --holdout
    # is set) runs after the main loop completes, on eval/holdout/ — the
    # proposer never sees holdout tasks (Appendix C §C.14).
    eval_tasks_dir = REPO_ROOT / "eval" / "tasks"

    skill_path: Path | None = None
    if proposer == "claude":
        if skill:
            sp = Path(skill)
            skill_path = sp if sp.is_absolute() else (REPO_ROOT / sp).resolve()
        else:
            skill_path = REPO_ROOT / "skills" / f"meta-harness-{domain}" / "SKILL.md"
        if not skill_path.exists():
            typer.echo(f"skill not found: {skill_path}", err=True)
            raise typer.Exit(2)

    from app.meta_harness import tracking as trk

    if record and not persistent:
        typer.echo(
            "--record needs --persistent: a tape with no checkpoint history "
            "cannot be replayed from a checkpoint",
            err=True,
        )
        raise typer.Exit(2)

    tracker = trk.make_tracker(
        enabled=wandb,
        run_name=run_name,
        config={
            "proposer": proposer,
            "budget": budget,
            "trials": trials,
            "workers": workers,
            "mock_bench": mock_bench,
            "domain": domain,
        },
        tags=["outer-loop", proposer],
        job_type="loop",
    )
    if tracker.enabled:
        typer.echo(f"tracking: wandb ({tracker.run_url or 'offline'})")
    elif tracker.reason and wandb:
        typer.echo(f"tracking: off ({tracker.reason})")

    recording_root = (
        _recordings_root(run_dir, run_name) if record else None
    )

    async def _run() -> Any:
        # Open the memory store FIRST so its failure mode is isolated
        # from the loop body. Previously a single ``try`` wrapped both
        # the store entry AND ``run_outer_loop``, so any node-body
        # exception silently fell through to a second loop invocation
        # with memory dropped (double-spending the LLM budget and
        # masking the original error).
        from app.meta_harness.memory import memory_store as _mem_store
        import logging

        log = logging.getLogger("meta_harness.cli")

        if persistent:
            async with persistence_layer() as saver:
                mstore = None
                try:
                    mstore_cm = _mem_store()
                    mstore = await mstore_cm.__aenter__()
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "memory store unavailable (%s); proceeding without it. "
                        "The cross-run memory beat will be skipped this run.",
                        type(exc).__name__,
                    )
                    mstore_cm = None
                try:
                    return await run_outer_loop(
                        run_dir=run_dir,
                        repo_root=REPO_ROOT,
                        eval_tasks_dir=eval_tasks_dir,
                        mock_proposer=(proposer == "mock"),
                        mock_bench=mock_bench,
                        trials=trials,
                        bench_workers=workers,
                        budget=budget,
                        skill_path=skill_path,
                        checkpointer=saver,
                        memory_store=mstore,
                        tracker=tracker,
                        recording_root=recording_root,
                    )
                finally:
                    if mstore_cm is not None:
                        try:
                            await mstore_cm.__aexit__(None, None, None)
                        except Exception:  # noqa: BLE001
                            pass
        return await run_outer_loop(
            run_dir=run_dir,
            repo_root=REPO_ROOT,
            eval_tasks_dir=eval_tasks_dir,
            mock_proposer=(proposer == "mock"),
            mock_bench=mock_bench,
            trials=trials,
            bench_workers=workers,
            budget=budget,
            skill_path=skill_path,
            checkpointer=None,
            tracker=tracker,
        )

    final_state = _run_async(_run())
    tracker.finish()

    # Post-eval on holdout set (if requested and meaningful).
    holdout_result: dict[str, Any] | None = None
    if holdout and not mock_bench and final_state.get("best_candidate"):
        holdout_result = _run_async(
            _run_holdout_eval(
                run_dir=run_dir,
                repo_root=REPO_ROOT,
                thread_id=run_name,
                best_candidate=final_state["best_candidate"],
                trials=trials,
                workers=workers,
            )
        )

    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "iterations_completed": final_state["iteration"],
                "budget_remaining": final_state["budget_remaining"],
                "best_candidate": final_state.get("best_candidate"),
                "n_candidates": len(final_state.get("candidates") or []),
                "frontier": final_state.get("frontier"),
                "persistent": persistent,
                "holdout": holdout_result,
                "recording_root": str(recording_root) if recording_root else None,
                "tracking": (
                    {"backend": "wandb", "run_url": tracker.run_url}
                    if tracker.enabled
                    else {"backend": None, "reason": tracker.reason}
                ),
            },
            indent=2,
        )
    )


def _recordings_root(run_dir: Path, thread_id: str) -> Path:
    """Where a run's execution tapes live: one directory per branch."""
    from app.meta_harness import runs as runs_mod

    return runs_mod.thread_dir(run_dir, thread_id) / "recordings"


async def _run_holdout_eval(
    *,
    run_dir: Path,
    repo_root: Path,
    thread_id: str,
    best_candidate: str,
    trials: int,
    workers: int,
) -> dict[str, Any]:
    """Post-evaluate the best candidate on ``eval/holdout/``.

    The proposer never sees holdout tasks during search (Appendix C
    §C.14), so this is the honest generalisation number. It is written
    to the branch's own ``holdout-result.json`` and labelled
    ``task_set: "holdout"`` so no consumer can confuse it with a
    search-set result.
    """
    import datetime as _dt

    from app.meta_harness import benchmark as bench
    from app.meta_harness import runs as runs_mod

    holdout_dir = repo_root / "eval" / "holdout"
    task_dirs = bench.discover_tasks(holdout_dir)
    if not task_dirs:
        return {
            "candidate": best_candidate,
            "task_set": "holdout",
            "skipped": True,
            "reason": f"no holdout tasks found in {holdout_dir}",
        }

    try:
        harness_class = _resolve_harness_class(best_candidate)
    except typer.Exit:
        return {
            "candidate": best_candidate,
            "task_set": "holdout",
            "skipped": True,
            "reason": f"could not import agents.{best_candidate}",
        }

    cand_dir = runs_mod.candidate_dir(run_dir, thread_id, best_candidate)
    holdout_result = await bench.benchmark_harness(
        harness_factory=harness_class,
        tasks_dir=holdout_dir,
        trials=trials,
        workers=workers,
        trace_root=cand_dir / "holdout-traces",
        thread_prefix=f"holdout::{run_dir.name}::{thread_id}::{best_candidate}",
    )
    holdout_result["candidate"] = best_candidate
    holdout_result["task_set"] = "holdout"
    holdout_result["thread_id"] = thread_id
    holdout_result["n_holdout_tasks"] = len(task_dirs)
    holdout_result["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    runs_mod.write_json_atomic(
        runs_mod.thread_dir(run_dir, thread_id) / "holdout-result.json",
        holdout_result,
    )
    return holdout_result


@app.command()
def fork(
    run_name: str = typer.Argument(..., help="Run name to fork from (under runs/)"),
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        help="Parent checkpoint id (from `meta-harness checkpoints` or the API)",
    ),
    mod: list[str] = typer.Option(
        [],
        "--mod",
        help="State mod to apply at the fork point: KEY=VALUE (repeatable)",
    ),
    branch_name: str = typer.Option(
        None,
        "--name",
        help="Optional human-readable label for the branch",
    ),
    detach: bool = typer.Option(
        False,
        "--detach",
        help="Don't wait for the branch to finish; return metadata immediately",
    ),
) -> None:
    """Fork a run from a checkpoint into a concurrent branch (Appendix A).

    Reads runs/<run-name>/manifest.json to recover the original run config,
    opens the same Postgres checkpointer, calls ``branches.worktree_add``,
    and (unless ``--detach``) awaits the branch to completion.
    """
    import asyncio

    from app.meta_harness.branches import set_runs_root, worktree_add
    from app.meta_harness.outer import OuterLoopRunner
    from app.meta_harness.persistence import persistence_layer

    run_dir = REPO_ROOT / "runs" / run_name
    if not run_dir.exists():
        typer.echo(f"run not found: {run_dir}", err=True)
        raise typer.Exit(1)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"manifest.json missing in {run_dir}; cannot fork", err=True)
        raise typer.Exit(1)
    manifest = json.loads(manifest_path.read_text())
    # Persist branch metadata so the API/dashboard can see this fork.
    set_runs_root(REPO_ROOT / "runs")

    mods: dict[str, Any] = {}
    for raw in mod:
        if "=" not in raw:
            typer.echo(f"--mod must be KEY=VALUE; got {raw!r}", err=True)
            raise typer.Exit(2)
        k, _, v = raw.partition("=")
        mods[k.strip()] = v

    eval_tasks_dir = REPO_ROOT / "eval" / "tasks"
    skill_path = REPO_ROOT / "skills" / "meta-harness-coding-agent" / "SKILL.md"
    skill_path = skill_path if skill_path.exists() else None

    async def _run() -> dict[str, Any]:
        async with persistence_layer() as saver:
            runner = OuterLoopRunner(
                run_dir=run_dir,
                repo_root=REPO_ROOT,
                eval_tasks_dir=eval_tasks_dir,
                mock_proposer=manifest.get("mock_proposer", False),
                mock_bench=manifest.get("mock_bench", False),
                trials=manifest.get("trials", 5),
                bench_workers=manifest.get("workers", 3),
                skill_path=skill_path,
                checkpointer=saver,
            )
            graph = runner.build()
            metadata, task = await worktree_add(
                graph,
                run_id=run_name,
                parent_thread_id=run_name,
                parent_checkpoint_id=checkpoint,
                mods=mods,
                name=branch_name,
            )
            if not detach:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            return metadata.to_dict()

    metadata = _run_async(_run())
    typer.echo(json.dumps(metadata, indent=2, default=str))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on edits."),
) -> None:
    """Serve the FastAPI backend on a Postgres-compatible event loop.

    Prefer this over a bare ``uvicorn app.main:app``. uvicorn selects
    Windows' ProactorEventLoop by default, and psycopg's async driver
    cannot run on it — the server would come up with checkpointing
    silently degraded to in-memory, which disables checkpoint history,
    forking and branch recovery.
    """
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        loop="app.event_loop:selector_loop_factory",
    )


@app.command()
def experiment(
    candidate: str = typer.Option(
        ...,
        "--candidate",
        help="Evolved candidate harness under agents/ to compare against baseline.",
    ),
    config: str = typer.Option(
        "benchmarks/pass-rate/config.json",
        "--config",
        help="Canonical protocol file.",
    ),
    baseline: str = typer.Option(
        "baseline", "--baseline", help="Control-arm harness under agents/."
    ),
    trials: int = typer.Option(
        None,
        "--trials",
        help="Override trials per task per arm (default: from config).",
    ),
    workers: int = typer.Option(
        None, "--workers", help="Override parallel workers (default: from config)."
    ),
    output: str = typer.Option(
        None,
        "--output",
        help="Result directory. Defaults to benchmark-results/<experiment-id>/.",
    ),
    record_trials: int = typer.Option(
        0,
        "--record-trials",
        help=(
            "Tape this many trials per task per arm for exact-replay evidence. "
            "Requires Postgres; 0 (the default) records nothing."
        ),
    ),
    wandb: bool = typer.Option(
        None,
        "--wandb/--no-wandb",
        help="Log to Weights & Biases. Defaults to META_HARNESS_WANDB.",
    ),
) -> None:
    """Run the canonical two-arm pass-rate experiment.

    Both arms run the identical protocol — same tasks, same trial count,
    same worker pool, same model — so the measured difference is
    attributable to the harness. The summary is derived mechanically
    from raw per-trial rows; no target number exists anywhere in this
    command.

    THIS ISSUES REAL LLM CALLS. The committed protocol is
    5 tasks x 20 trials x 2 arms = 200 trials.
    """
    import datetime as _dt

    from app.meta_harness import benchmark as bench
    from app.meta_harness import experiment as exp
    from app.meta_harness.harness import CodingAgentHarness

    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        typer.echo(f"config not found: {config_path}", err=True)
        raise typer.Exit(2)
    cfg = exp.load_config(config_path)

    n_trials = trials if trials is not None else int(cfg["trials_per_task"])
    n_workers = workers if workers is not None else int(cfg["workers"])
    tasks_dir = REPO_ROOT / cfg["task_set"]
    task_dirs = bench.discover_tasks(tasks_dir)
    if not task_dirs:
        typer.echo(f"no tasks found in {tasks_dir}", err=True)
        raise typer.Exit(1)

    expected = cfg.get("tasks")
    if expected and sorted(d.name for d in task_dirs) != sorted(expected):
        typer.echo("task set does not match the committed protocol:", err=True)
        typer.echo(f"  expected: {sorted(expected)}", err=True)
        typer.echo(f"  found:    {sorted(d.name for d in task_dirs)}", err=True)
        raise typer.Exit(1)

    experiment_id = (
        f"{cfg['experiment']}-"
        + _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir = (
        Path(output)
        if output
        else REPO_ROOT / "benchmark-results" / experiment_id
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    baseline_cls = _resolve_harness_class(baseline)
    candidate_cls = _resolve_harness_class(candidate)
    model = getattr(candidate_cls, "MODEL", CodingAgentHarness.MODEL)

    total_per_arm = len(task_dirs) * n_trials
    for line in (
        f"experiment {experiment_id}",
        f"  tasks:     {len(task_dirs)}",
        f"  trials:    {n_trials} per task per arm",
        f"  arms:      {baseline} (control) vs {candidate}",
        f"  total:     {total_per_arm * 2} task trials",
        f"  model:     {model}",
        f"  output:    {output_dir}",
    ):
        typer.echo(line)

    done = {"baseline": 0, "candidate": 0}

    def _progress(arm: str, row: dict[str, Any]) -> None:
        done[arm] += 1
        mark = "PASS" if row["passed"] else "FAIL"
        typer.echo(
            f"  [{arm} {done[arm]}/{total_per_arm}] {mark} "
            f"{row['task_id']} trial-{row['trial']}"
        )

    from app.meta_harness import tracking as trk

    tracker = trk.make_tracker(
        enabled=wandb,
        run_name=experiment_id,
        config={
            "experiment": cfg.get("experiment"),
            "task_set": cfg.get("task_set"),
            "trials_per_task": n_trials,
            "workers": n_workers,
            "baseline": baseline,
            "candidate": candidate,
            "model": model,
        },
        tags=["experiment", str(cfg.get("experiment"))],
        job_type="experiment",
    )
    if tracker.enabled:
        typer.echo(f"  tracking:  wandb ({tracker.run_url or 'offline'})")

    async def _run() -> dict[str, Any]:
        # Recording needs a checkpointer, because a replay is positioned
        # by checkpoint. Only open Postgres when trials are being taped.
        if record_trials <= 0:
            return await exp.run_two_arm_experiment(
                repo_root=REPO_ROOT,
                config=cfg,
                experiment_id=experiment_id,
                tasks_dir=tasks_dir,
                baseline_label=baseline,
                candidate_label=candidate,
                baseline_factory=baseline_cls,
                candidate_factory=candidate_cls,
                arm_sources={
                    baseline: REPO_ROOT / "agents" / f"{baseline}.py",
                    candidate: REPO_ROOT / "agents" / f"{candidate}.py",
                },
                model=model,
                trials=n_trials,
                workers=n_workers,
                output_dir=output_dir,
                tracker=tracker,
                on_trial=_progress,
            )

        from app.meta_harness.persistence import persistence_layer

        async with persistence_layer() as saver:
            return await exp.run_two_arm_experiment(
                repo_root=REPO_ROOT,
                config=cfg,
                experiment_id=experiment_id,
                tasks_dir=tasks_dir,
                baseline_label=baseline,
                candidate_label=candidate,
                baseline_factory=baseline_cls,
                candidate_factory=candidate_cls,
                arm_sources={
                    baseline: REPO_ROOT / "agents" / f"{baseline}.py",
                    candidate: REPO_ROOT / "agents" / f"{candidate}.py",
                },
                model=model,
                trials=n_trials,
                workers=n_workers,
                output_dir=output_dir,
                checkpointer=saver,
                tracker=tracker,
                on_trial=_progress,
                record_trials_per_task=record_trials,
            )

    result = _run_async(_run())
    tracker.finish()
    summary = result["summary"]
    paths = result["paths"]

    typer.echo("")
    typer.echo(exp.render_report(summary))
    typer.echo("")
    validation = result["validation"]
    for name, ok in validation["checks"].items():
        typer.echo(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    typer.echo("")
    typer.echo(f"wrote {paths['summary']}")
    if not validation["identical_protocol"]:
        typer.echo(
            "the two arms did not run an identical protocol; the delta above "
            "is not attributable to the harness alone",
            err=True,
        )
        raise typer.Exit(1)


@app.command()
def checkpoints(
    run_name: str = typer.Argument(..., help="Run name (under runs/)"),
    thread: str = typer.Option(
        None,
        "--thread",
        help="Thread id to list. Defaults to the run's root thread.",
    ),
    limit: int = typer.Option(20, "--limit", help="Max checkpoints to show."),
) -> None:
    """List a thread's LangGraph checkpoint history (newest first)."""
    from app.meta_harness.branches import get_state_history
    from app.meta_harness.persistence import persistence_layer

    run_dir, manifest = _require_run(run_name)
    thread_id = thread or run_name

    async def _run() -> list[dict[str, Any]]:
        async with persistence_layer() as saver:
            graph = _rebuild_graph(run_dir, manifest, saver)
            history = await get_state_history(
                graph, thread_id=thread_id, limit=limit
            )
            return [record.to_dict() for record in history]

    records = _run_async(_run())
    if not records:
        typer.echo(f"no checkpoints for thread {thread_id!r}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(records, indent=2, default=str))


@app.command()
def replay(
    run_name: str = typer.Argument(..., help="Run name (under runs/)"),
    thread: str = typer.Option(
        None,
        "--thread",
        help="Thread id to replay. Defaults to the run's root thread.",
    ),
    checkpoint: str = typer.Option(
        None,
        "--checkpoint",
        help="Restore one checkpoint's exact state instead of replaying the thread.",
    ),
    limit: int = typer.Option(
        None, "--limit", help="Max checkpoints to walk when replaying a thread."
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help=(
            "Re-execute the recorded run from --checkpoint against its tape "
            "and assert exact equivalence. Exits non-zero if the replayed "
            "node sequence, per-step state hashes or final state hash differ, "
            "or if the tape is not consumed exactly."
        ),
    ),
    recording: str = typer.Option(
        None,
        "--recording",
        help=(
            "Recording directory to replay. Defaults to searching "
            "runs/<run>/ for the recording that holds --checkpoint."
        ),
    ),
) -> None:
    """Restore historical state from Postgres, or replay a recorded execution.

    Three different things, kept apart on purpose:

    - default: walk the thread's recorded transitions oldest-first. Pure
      read-back; nothing executes.
    - ``--checkpoint``: return the exact stored state at that checkpoint
      plus a SHA-256 of its canonical JSON encoding.
    - ``--checkpoint --verify``: EXACT REPLAY. Re-execute the inner-loop
      state machine from that checkpoint with every model response, tool
      result and workspace observation served from the recorded tape, and
      assert it reproduces the recording exactly.

    None of these issue a model call. Note that ``meta-harness resume`` is
    the opposite: it re-enters a graph from a checkpoint and issues FRESH
    model calls, which is a new stochastic execution, not a replay.
    """
    from app.meta_harness import replay as replay_mod
    from app.meta_harness.persistence import persistence_layer

    run_dir, manifest = _require_run(run_name)
    thread_id = thread or run_name

    if verify:
        _verify_recorded_checkpoint_replay(
            run_dir=run_dir,
            checkpoint=checkpoint,
            recording_dir=recording,
        )
        return

    async def _run() -> dict[str, Any]:
        async with persistence_layer() as saver:
            graph = _rebuild_graph(run_dir, manifest, saver)
            if checkpoint:
                return await replay_mod.restore_checkpoint(
                    graph, thread_id=thread_id, checkpoint_id=checkpoint
                )
            return await replay_mod.replay_thread(
                graph, thread_id=thread_id, limit=limit
            )

    try:
        result = _run_async(_run())
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps(result, indent=2, default=str))


def _verify_recorded_checkpoint_replay(
    *, run_dir: Path, checkpoint: str | None, recording_dir: str | None
) -> None:
    """``replay <run> --checkpoint <id> --verify``: exact replay, or exit 1."""
    from app.meta_harness import recording as rec
    from app.meta_harness import replay as replay_mod
    from app.meta_harness.inner import build_inner_graph
    from app.meta_harness.pipeline import _replay_harness_factory
    from app.meta_harness.persistence import persistence_layer

    if not checkpoint:
        typer.echo("--verify requires --checkpoint", err=True)
        raise typer.Exit(2)

    if recording_dir:
        directory = Path(recording_dir)
        if not directory.is_absolute():
            directory = REPO_ROOT / directory
        found = rec.read_recording(directory)
    else:
        found = None
        for candidate_dir in rec.discover_recordings(run_dir):
            loaded = rec.read_recording(candidate_dir)
            if any(s.checkpoint_id == checkpoint for s in loaded.steps):
                found = loaded
                break
        if found is None:
            typer.echo(
                f"no recording under {run_dir} holds checkpoint {checkpoint}. "
                "Record a run first: `meta-harness inner --record`, or "
                "`meta-harness resume-experiment --record-trials 1`.",
                err=True,
            )
            raise typer.Exit(1)

    factory = _replay_harness_factory(found, repo_root=REPO_ROOT)

    async def _run() -> dict[str, Any]:
        async with persistence_layer() as saver:
            source_graph = build_inner_graph(factory(), checkpointer=saver)
            return await replay_mod.replay_recorded_execution(
                recording=found,
                harness_factory=factory,
                checkpointer=saver,
                source_graph=source_graph,
                from_checkpoint=checkpoint,
            )

    try:
        report = _run_async(_run())
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    typer.echo(replay_mod.render_verification(report))
    if not report["verified"]:
        raise typer.Exit(1)


def _require_run(run_name: str) -> tuple[Path, dict[str, Any]]:
    """Resolve runs/<run_name> and its manifest, or exit with a message."""
    from app.meta_harness import runs as runs_mod

    try:
        run_dir = runs_mod.make_run_path(REPO_ROOT, run_name)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    if not run_dir.exists():
        typer.echo(f"run not found: {run_dir}", err=True)
        raise typer.Exit(1)
    manifest = runs_mod.read_manifest(run_dir)
    if manifest is None:
        typer.echo(f"manifest.json missing in {run_dir}", err=True)
        raise typer.Exit(1)
    return run_dir, manifest


def _rebuild_graph(run_dir: Path, manifest: dict[str, Any], saver: Any) -> Any:
    """Recompile a run's outer graph from its manifest, for read-only use."""
    from app.meta_harness.outer import OuterLoopRunner

    skill_path = REPO_ROOT / "skills" / "meta-harness-coding-agent" / "SKILL.md"
    runner = OuterLoopRunner(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=bool(manifest.get("mock_proposer", False)),
        mock_bench=bool(manifest.get("mock_bench", False)),
        trials=int(manifest.get("trials", 5)),
        bench_workers=int(manifest.get("workers", 3)),
        skill_path=skill_path if skill_path.exists() else None,
        checkpointer=saver,
    )
    return runner.build()


@app.command()
def init(
    domain: str = typer.Argument(..., help="Domain name, e.g. 'coding-agent'"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite skills/meta-harness-<domain>/ if it already exists",
    ),
) -> None:
    """Scaffold a new domain: skills/meta-harness-<domain>/SKILL.md.

    If a coding-agent skill exists at skills/meta-harness-coding-agent/SKILL.md,
    use it as a template; otherwise generate a minimal SKILL.md. Prints
    next-step guidance.
    """
    target = REPO_ROOT / "skills" / f"meta-harness-{domain}"
    skill_file = target / "SKILL.md"
    if skill_file.exists() and not force:
        typer.echo(
            f"{skill_file} already exists. Pass --force to overwrite.",
            err=True,
        )
        raise typer.Exit(1)
    target.mkdir(parents=True, exist_ok=True)

    template = REPO_ROOT / "skills" / "meta-harness-coding-agent" / "SKILL.md"
    if template.exists() and template != skill_file:
        content = template.read_text()
        content = content.replace(
            "meta-harness-coding-agent", f"meta-harness-{domain}", 1
        )
        skill_file.write_text(content)
    else:
        skill_file.write_text(
            f"""---
name: meta-harness-{domain}
description: Evolve the {domain} harness. Read past traces, form one falsifiable hypothesis, write ONE new candidate, register in pending_eval.json.
---

# Meta-Harness {domain.title()} Evolution

You are evolving the source code of a harness. Read the run's
``evolution_summary.jsonl`` and ``frontier_val.json``, then form
ONE falsifiable hypothesis and write ONE new candidate file.

## Workflow

1. **Analyze** — read prior candidates' source + traces (~10 files).
2. **Pick** — one hypothesis with the highest expected delta.
3. **Prototype** — exercise the mechanism in /tmp/ first.
4. **Implement** — copy parent → ``agents/<name>.py``, apply targeted
   change, add a self-critique block at the top.
5. **Register** — write ``pending_eval.json`` with the candidate metadata.
"""
        )

    typer.echo(
        json.dumps(
            {
                "domain": domain,
                "skill_path": str(skill_file),
                "next_steps": [
                    f"Edit {skill_file} to customize the workflow.",
                    f"Add eval tasks in eval/tasks/ if this is a new domain.",
                    f"meta-harness loop --domain {domain} --proposer mock --mock-bench --budget 2 --fresh",
                ],
            },
            indent=2,
        )
    )


@app.command()
def resume(
    run_name: str = typer.Argument(..., help="Run name to resume (under runs/)"),
) -> None:
    """Resume an interrupted ``meta-harness loop`` run from its last
    Postgres checkpoint. Reconstructs the run config from
    ``runs/{run_name}/manifest.json`` and continues with the same
    proposer / mock_bench / trials settings.
    """
    import asyncio

    from app.meta_harness.outer import resume_outer_loop
    from app.meta_harness.persistence import persistence_layer

    run_dir = REPO_ROOT / "runs" / run_name
    if not run_dir.exists():
        typer.echo(f"run not found: {run_dir}", err=True)
        raise typer.Exit(1)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"manifest.json missing in {run_dir}; cannot resume", err=True)
        raise typer.Exit(1)

    eval_tasks_dir = REPO_ROOT / "eval" / "tasks"
    skill_path: Path | None = None
    skills_default = REPO_ROOT / "skills" / "meta-harness-coding-agent" / "SKILL.md"
    if skills_default.exists():
        skill_path = skills_default

    async def _run() -> Any:
        async with persistence_layer() as saver:
            return await resume_outer_loop(
                run_dir=run_dir,
                repo_root=REPO_ROOT,
                eval_tasks_dir=eval_tasks_dir,
                checkpointer=saver,
                skill_path=skill_path,
            )

    final_state = _run_async(_run())
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "resumed": True,
                "iterations_completed": final_state["iteration"],
                "budget_remaining": final_state["budget_remaining"],
                "best_candidate": final_state.get("best_candidate"),
                "n_candidates": len(final_state.get("candidates") or []),
            },
            indent=2,
        )
    )


def _find_harness_class(mod) -> type | None:
    """Find the first ``CodingAgentHarness`` subclass exported by a module."""
    import inspect

    from app.meta_harness.harness import CodingAgentHarness

    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if (
            issubclass(obj, CodingAgentHarness)
            and obj is not CodingAgentHarness
            and obj.__module__ == mod.__name__
        ):
            return obj
    return None


# ── memory sub-app (step 8) ──────────────────────────────────────────

memory_app = typer.Typer(
    name="memory",
    help="Cross-run memory commands (PostgresStore).",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")


@memory_app.command("list")
def memory_list(
    namespace: str = typer.Option(
        "coding-agent",
        "--namespace",
        help="Domain namespace to list (e.g. 'coding-agent').",
    ),
    limit: int = typer.Option(50, "--limit", help="Max entries to return."),
) -> None:
    """List all learned patterns in a namespace."""
    import asyncio

    from app.meta_harness.memory import list_namespace, memory_store

    async def _run() -> list:
        async with memory_store() as store:
            return await list_namespace(store, domain=namespace, limit=limit)

    entries = _run_async(_run())
    if not entries:
        typer.echo(f"No patterns in namespace ('learned_patterns', '{namespace}').")
        return
    typer.echo(json.dumps(entries, indent=2, default=str))


# ──────────────────────────────────────────────────────────────────────
# Exact recorded-execution replay
# ──────────────────────────────────────────────────────────────────────


@app.command("verify-replay")
def verify_replay(
    recordings: str = typer.Argument(
        ...,
        help="Directory holding one or more recordings (searched recursively).",
    ),
    limit: int = typer.Option(
        None, "--limit", help="Verify at most N recordings."
    ),
    output: str = typer.Option(
        None,
        "--output",
        help=(
            "Write the verification report here. Defaults to "
            "docs/evidence/replay-verification.json, which is what "
            "`report resume-evidence` reads."
        ),
    ),
) -> None:
    """Re-execute recorded trials against their tapes and verify equivalence.

    Each recording is replayed twice — once from the start of the run and
    once from a mid-run checkpoint — and both must reproduce the recorded
    node sequence, the per-step state hashes and the final state hash, and
    must consume the tape exactly.

    **No model is called.** ``ReplayEffects`` has no code path that
    reaches a provider; the count of issued model calls is reported and
    must be zero.

    Exits non-zero if any replay fails to reproduce its recording.
    """
    from app.meta_harness import pipeline as pipe
    from app.meta_harness.persistence import persistence_layer

    root = Path(recordings)
    if not root.is_absolute():
        root = REPO_ROOT / root
    from app.meta_harness import recording as rec

    found = rec.discover_recordings(root)
    if not found:
        typer.echo(f"no recordings found under {root}", err=True)
        raise typer.Exit(1)

    typer.echo(f"verifying {len(found)} recording(s) under {root}")

    async def _run() -> dict[str, Any]:
        # Replaying a whole run needs only the tape. Replaying *from a
        # checkpoint* additionally needs the store that holds the
        # recorded thread, so use Postgres when it is reachable and fall
        # back to an in-memory checkpointer when it is not — reporting
        # the checkpoint replays as skipped rather than as passes.
        from app.meta_harness.persistence import healthcheck  # noqa: PLC0415

        if await healthcheck():
            async with persistence_layer() as saver:
                return await pipe.verify_recorded_replays(
                    recordings_root=root,
                    repo_root=REPO_ROOT,
                    checkpointer=saver,
                    limit=limit,
                )

        from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

        typer.echo(
            "  Postgres is not reachable: whole-run replays will be verified, "
            "replays from a stored checkpoint will be skipped."
        )
        return await pipe.verify_recorded_replays(
            recordings_root=root,
            repo_root=REPO_ROOT,
            checkpointer=InMemorySaver(),
            limit=limit,
        )

    report = _run_async(_run())

    from app.meta_harness import replay as replay_mod

    for entry in report["reports"]:
        typer.echo("")
        typer.echo(f"--- {entry['recording']} ({entry['mode']}) ---")
        typer.echo(replay_mod.render_verification(entry))

    out_path = (
        Path(output)
        if output
        else REPO_ROOT / "docs" / "evidence" / "replay-verification.json"
    )
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    from app.meta_harness import runs as runs_mod

    runs_mod.write_json_atomic(out_path, report)
    typer.echo("")
    typer.echo(f"wrote {out_path}")

    for entry in report["skipped"]:
        typer.echo("")
        typer.echo(f"--- {entry['recording']} ({entry['mode']}) SKIPPED ---")
        typer.echo(f"    {entry['reason']}")

    if not report["all_verified"]:
        typer.echo("EXACT REPLAY FAILED — see the checks above", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"all {report['replays']} replays verified "
        f"({report['replays_from_checkpoint']} from a stored checkpoint); "
        f"{report['model_calls_issued']} model calls issued"
    )
    if report["skipped"]:
        typer.echo(
            f"{len(report['skipped'])} checkpoint replay(s) skipped; see the "
            f"report for why"
        )


# ──────────────────────────────────────────────────────────────────────
# The end-to-end evidence pipeline
# ──────────────────────────────────────────────────────────────────────


@app.command("resume-experiment")
def resume_experiment(
    budget: int = typer.Option(
        5, "--budget", help="Outer-loop iterations to spend evolving candidates."
    ),
    search_trials: int = typer.Option(
        5,
        "--search-trials",
        help=(
            "Trials per task during evolution. These are the VALIDATION "
            "numbers candidate selection uses; the final experiment re-measures "
            "with fresh independent trials."
        ),
    ),
    workers: int = typer.Option(5, "--workers", help="Parallel trial workers."),
    candidate: str = typer.Option(
        None,
        "--candidate",
        help=(
            "Skip evolution and measure this already-evolved candidate under "
            "agents/. Use when a previous run already produced one."
        ),
    ),
    run_name: str = typer.Option(
        None, "--run-name", help="Run dir under runs/. Auto-generated if omitted."
    ),
    domain: str = typer.Option(
        "coding-agent", "--domain", help="SKILL.md domain for the proposer."
    ),
    record_trials: int = typer.Option(
        1,
        "--record-trials",
        help=(
            "Tape this many trials per task per arm for exact-replay evidence. "
            "0 disables recording."
        ),
    ),
    holdout: bool = typer.Option(
        True,
        "--holdout/--no-holdout",
        help="Also run the two-arm generalisation experiment on eval/holdout/.",
    ),
    wandb: bool = typer.Option(
        None,
        "--wandb/--no-wandb",
        help="Log to Weights & Biases. Defaults to META_HARNESS_WANDB.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Print the plan and a cost estimate extrapolated from measured "
            "trials already on disk, then exit without spending anything."
        ),
    ),
) -> None:
    """Evolve, select on validation only, measure, verify replay, report.

    The one command behind every number in `docs/RESUME_EVIDENCE.md`:

    1. evolve candidate harnesses on the search set with the real proposer;
    2. select the best candidate using ONLY the validation accuracy the
       outer loop measured during evolution — the final experiment has not
       run yet, so its trials cannot influence the choice;
    3. run the canonical two-arm experiment (5 tasks x 20 trials x 2 arms);
    4. run the two-arm holdout experiment on tasks the proposer never saw;
    5. verify a recorded trial replays exactly;
    6. regenerate the resume-evidence document from the artifacts.

    THIS ISSUES REAL LLM CALLS AND COSTS MONEY. Use ``--dry-run`` first.

    Nothing in this command compares a result to a target. If the measured
    improvement is below the resume's claim, the claim is reported
    unsupported and the measured number stands.
    """
    import datetime as _dt

    from app.meta_harness import evidence as ev
    from app.meta_harness import pipeline as pipe
    from app.meta_harness import runs as runs_mod
    from app.meta_harness import tracking as trk
    from app.meta_harness.outer import run_outer_loop
    from app.meta_harness.persistence import persistence_layer

    search_config = REPO_ROOT / "benchmarks" / "pass-rate" / "config.json"
    holdout_config = REPO_ROOT / "benchmarks" / "holdout" / "config.json"

    isolation = _experiment_module().check_task_set_isolation(
        search_dir=REPO_ROOT / "eval" / "tasks",
        holdout_dir=REPO_ROOT / "eval" / "holdout",
    )
    if not isolation["disjoint"]:
        typer.echo(
            "search and holdout task sets overlap on "
            f"{isolation['overlapping_tasks']}; the holdout number would be "
            "a second search-set measurement",
            err=True,
        )
        raise typer.Exit(1)

    planned = _planned_trials(search_config, holdout_config if holdout else None)
    estimate = pipe.estimate_cost(
        pipe.collect_measured_rows(REPO_ROOT), planned_trials=planned
    )

    typer.echo("plan")
    typer.echo(f"  evolve:      {budget} iterations x {search_trials} trials/task")
    typer.echo(f"  experiment:  {search_config.relative_to(REPO_ROOT)}")
    if holdout:
        typer.echo(f"  holdout:     {holdout_config.relative_to(REPO_ROOT)}")
    typer.echo(f"  final trials: {planned}")
    typer.echo(f"  cost estimate: {json.dumps(estimate, indent=2)}")

    if dry_run:
        typer.echo("")
        typer.echo("--dry-run: nothing was executed and nothing was spent.")
        return

    if run_name is None:
        run_name = "resume-" + _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    run_dir = runs_mod.make_run_dir(REPO_ROOT, run_name, fresh=True)
    output_root = REPO_ROOT / "benchmark-results"

    skill_path = REPO_ROOT / "skills" / f"meta-harness-{domain}" / "SKILL.md"
    if candidate is None and not skill_path.exists():
        typer.echo(f"skill not found: {skill_path}", err=True)
        raise typer.Exit(2)

    tracker = trk.make_tracker(
        enabled=wandb,
        run_name=run_name,
        config={
            "budget": budget,
            "search_trials": search_trials,
            "workers": workers,
            "record_trials": record_trials,
            "holdout": holdout,
        },
        tags=["resume-experiment"],
        job_type="resume-experiment",
    )
    if tracker.enabled:
        typer.echo(f"tracking: wandb ({tracker.run_url or 'offline'})")
    elif tracker.reason:
        typer.echo(f"tracking: off ({tracker.reason})")

    stages: list[Any] = []

    def _progress(arm: str, row: dict[str, Any]) -> None:
        mark = "PASS" if row["passed"] else "FAIL"
        typer.echo(f"  [{arm}] {mark} {row['task_id']} trial-{row['trial']}")

    async def _run() -> list[Any]:
        collected: list[Any] = []
        async with persistence_layer() as saver:
            if candidate is None:
                typer.echo("")
                typer.echo("stage 1/6: evolving candidates")
                final_state = await run_outer_loop(
                    run_dir=run_dir,
                    repo_root=REPO_ROOT,
                    eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
                    mock_proposer=False,
                    mock_bench=False,
                    trials=search_trials,
                    bench_workers=workers,
                    budget=budget,
                    skill_path=skill_path,
                    checkpointer=saver,
                    tracker=tracker,
                )
                collected.append(
                    pipe.StageResult(
                        "evolve",
                        "ok",
                        f"{final_state['iteration']} iterations, "
                        f"{len(final_state.get('candidates') or [])} candidates",
                        {"best_candidate": final_state.get("best_candidate")},
                    )
                )
                selection = pipe.select_candidate(final_state)
            else:
                selection = _selection_from_agents_dir(candidate)
                collected.append(
                    pipe.StageResult(
                        "evolve",
                        "skipped",
                        f"--candidate {candidate} supplied; evolution skipped",
                    )
                )

            typer.echo("")
            typer.echo(f"stage 2/6: selection — {selection['reason']}")
            runs_mod.write_json_atomic(run_dir / "selection.json", selection)
            collected.append(
                pipe.StageResult(
                    "select", "ok", selection["reason"], {"selection": selection}
                )
            )

            typer.echo("")
            typer.echo("stage 3-4/6: canonical + holdout experiments")
            collected.extend(
                await pipe.run_measurement_pipeline(
                    repo_root=REPO_ROOT,
                    search_config_path=search_config,
                    holdout_config_path=holdout_config if holdout else None,
                    selection=selection,
                    baseline_label="baseline",
                    output_root=output_root,
                    workers=workers,
                    checkpointer=saver,
                    tracker=tracker,
                    on_trial=_progress,
                    record_trials_per_task=record_trials,
                    run_holdout=holdout,
                )
            )

            search_stage = next(
                (s for s in collected if s.name == "experiment"), None
            )
            if search_stage is not None and search_stage.status == "ok":
                recordings_root = (
                    Path(search_stage.data["output_dir"]) / "recordings"
                )
                if record_trials > 0 and recordings_root.is_dir():
                    typer.echo("")
                    typer.echo("stage 5/6: verifying exact recorded replay")
                    verification = await pipe.verify_recorded_replays(
                        recordings_root=recordings_root,
                        repo_root=REPO_ROOT,
                        checkpointer=saver,
                        limit=2,
                    )
                    runs_mod.write_json_atomic(
                        REPO_ROOT / "docs" / "evidence" / "replay-verification.json",
                        verification,
                    )
                    collected.append(
                        pipe.StageResult(
                            "replay-verify",
                            "ok" if verification["all_verified"] else "failed",
                            f"{verification['replays']} replays, "
                            f"{verification['model_calls_issued']} model calls",
                            {"verification": verification},
                        )
                    )
                else:
                    collected.append(
                        pipe.StageResult(
                            "replay-verify", "skipped", "no trials were recorded"
                        )
                    )

            version = await _capture_version_graph(saver, run_dir, run_name)
            collected.append(
                pipe.StageResult(
                    "version-graph",
                    "ok",
                    f"{version['checkpoint_count']} checkpoints, "
                    f"{version['branch_count']} branches",
                    {"version_graph": {k: version[k] for k in ("checkpoint_count", "branch_count", "immutable")}},
                )
            )
        return collected

    stages = _run_async(_run())

    typer.echo("")
    typer.echo("stage 6/6: regenerating resume evidence")
    report = ev.build_report(REPO_ROOT)
    evidence_path = REPO_ROOT / "docs" / "RESUME_EVIDENCE.md"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(report["markdown"], encoding="utf-8")

    pipe.write_pipeline_log(run_dir, stages)
    tracker.finish()

    typer.echo("")
    for stage in stages:
        typer.echo(f"  [{stage.status}] {stage.name}: {stage.detail}")
    typer.echo("")
    typer.echo(f"wrote {evidence_path}")
    typer.echo(json.dumps(report["counts"], indent=2))
    if report["any_failed"]:
        raise typer.Exit(1)


def _experiment_module() -> Any:
    from app.meta_harness import experiment as exp

    return exp


def _planned_trials(search_config: Path, holdout_config: Path | None) -> int:
    """Total final-experiment trials, read from the committed protocols."""
    exp = _experiment_module()
    total = 0
    for path in (search_config, holdout_config):
        if path is None or not path.exists():
            continue
        config = exp.load_config(path)
        total += len(config.get("tasks") or []) * int(config["trials_per_task"]) * 2
    return total


def _selection_from_agents_dir(candidate: str) -> dict[str, Any]:
    """Build a selection record for a candidate supplied by hand.

    Recorded as ``operator-supplied`` rather than dressed up as a
    validation result: no validation number backs this choice, and the
    evidence document should say so.
    """
    cls = _resolve_harness_class(candidate)
    return {
        "selected": candidate,
        "selected_row": {
            "candidate": candidate,
            "label": candidate,
            "import_path": f"agents.{candidate}:{cls.__name__}",
            "source_path": str(REPO_ROOT / "agents" / f"{candidate}.py"),
            "validation_accuracy": None,
            "iteration": None,
        },
        "reason": f"operator-supplied candidate agents/{candidate}.py",
        "selection_basis": (
            "supplied on the command line; no validation measurement backs "
            "this choice"
        ),
        "table": [],
    }


async def _capture_version_graph(
    saver: Any, run_dir: Path, run_name: str, *, full: bool = False
) -> dict[str, Any]:
    """Read the run's checkpoint DAG out of Postgres and persist it as evidence."""
    from app.meta_harness import branches as br
    from app.meta_harness import runs as runs_mod
    from app.meta_harness import versioning as ver

    # Branch refs live in runs/<run>/branches.json, so the registry has to
    # know where runs/ is before it can reload them.
    br.set_runs_root(REPO_ROOT / "runs")
    manifest = runs_mod.read_manifest(run_dir) or {}
    graph = _rebuild_graph(run_dir, manifest, saver)

    version = await ver.capture_evidence(
        graph, run_id=run_name, run_dir=run_dir, full=full
    )
    runs_mod.write_json_atomic(
        REPO_ROOT / "docs" / "evidence" / "version-graph.json", version
    )
    return version


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────

report_app = typer.Typer(
    name="report",
    help="Derive evidence documents from committed artifacts.",
    no_args_is_help=True,
)
app.add_typer(report_app, name="report")


@report_app.command("resume-evidence")
def report_resume_evidence(
    output: str = typer.Option(
        "docs/RESUME_EVIDENCE.md",
        "--output",
        help="Where to write the document.",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help=(
            "Do not write. Regenerate from the artifacts and exit non-zero if "
            "the committed document disagrees. This is what CI runs."
        ),
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print the derived checks as JSON instead."
    ),
) -> None:
    """Derive PASS/FAIL for every resume claim from committed artifacts.

    Measured rows are recomputed from raw trial rows; structural rows
    compile the graphs and read their node sets; artifact rows read
    verification reports. No value in the output is hand-entered, and a
    claim with no supporting artifact reports UNSUPPORTED.
    """
    from app.meta_harness import evidence as ev

    report = ev.build_report(REPO_ROOT)

    if as_json:
        typer.echo(
            json.dumps(
                {k: v for k, v in report.items() if k != "markdown"},
                indent=2,
                default=str,
            )
        )
        return

    path = Path(output)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if check:
        if not path.exists():
            typer.echo(f"{path} does not exist; run without --check", err=True)
            raise typer.Exit(1)
        committed = ev.comparable(path.read_text(encoding="utf-8"))
        derived = ev.comparable(report["markdown"])
        if committed != derived:
            typer.echo(
                f"{path} disagrees with the artifacts it claims to summarise. "
                "Regenerate it with `uv run meta-harness report resume-evidence`.",
                err=True,
            )
            _echo_first_difference(committed, derived)
            raise typer.Exit(1)
        typer.echo(f"{path} matches the artifacts it is derived from")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report["markdown"], encoding="utf-8")
    typer.echo(f"wrote {path}")
    for check_row in report["checks"]:
        typer.echo(f"  [{check_row['status']}] {check_row['claim']}")
    typer.echo(json.dumps(report["counts"], indent=2))


def _echo_first_difference(committed: str, derived: str) -> None:
    for index, (a, b) in enumerate(
        zip(committed.splitlines(), derived.splitlines())
    ):
        if a != b:
            typer.echo(f"  first difference at line {index + 1}:", err=True)
            typer.echo(f"    committed: {a}", err=True)
            typer.echo(f"    derived:   {b}", err=True)
            return
    typer.echo("  the documents differ in length", err=True)


@report_app.command("version-graph")
def report_version_graph(
    run_name: str = typer.Argument(..., help="Run name (under runs/)"),
    output: str = typer.Option(
        "docs/evidence/version-graph.json",
        "--output",
        help="Where to write the version-graph evidence artifact.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help=(
            "Include each checkpoint's value summary. Large, and the claim "
            "does not rest on it; off by default so the artifact stays "
            "committable."
        ),
    ),
) -> None:
    """Read a run's checkpoint DAG out of Postgres and persist it as evidence.

    Records the checkpoints (immutable ids with parent references), the
    branch refs and their fork points, each branch's private working
    tree, and a re-read immutability check confirming every stored
    checkpoint still hashes to what it hashed to.
    """
    from app.meta_harness.persistence import persistence_layer

    run_dir, _ = _require_run(run_name)

    async def _run() -> dict[str, Any]:
        async with persistence_layer() as saver:
            return await _capture_version_graph(
                saver, run_dir, run_name, full=full
            )

    version = _run_async(_run())
    path = Path(output)
    if not path.is_absolute():
        path = REPO_ROOT / path
    typer.echo(
        json.dumps(
            {
                "run_id": version["run_id"],
                "checkpoint_count": version["checkpoint_count"],
                "branch_count": version["branch_count"],
                "immutable": version["immutable"],
                "checkpoints_reread": version["immutability"]["checked"],
                "threads": {k: len(v) for k, v in version["threads"].items()},
            },
            indent=2,
        )
    )
    typer.echo(f"wrote {path}")
    if not version["immutable"]:
        typer.echo("a stored checkpoint changed; versioning is broken", err=True)
        raise typer.Exit(1)


#: The committed W&B evidence artifact. Only a probe that actually
#: exercised wandb may replace it; see ``report_wandb_check``.
DEFAULT_WANDB_EVIDENCE = "docs/evidence/wandb-offline.json"


@report_app.command("wandb-check")
def report_wandb_check(
    output: str = typer.Option(
        DEFAULT_WANDB_EVIDENCE,
        "--output",
        help="Where to write the probe result.",
    ),
) -> None:
    """Probe the W&B adapter in offline mode and record the result.

    Forces ``WANDB_MODE=offline``, so the probe never touches the network
    and never needs credentials. If ``wandb`` is not installed the probe
    records that plainly rather than failing — the integration is
    optional by design, and "the repository runs without it" is exactly
    what the row in the evidence document asserts.

    Writing that result to the *committed* evidence artifact is a
    different matter, and is refused. ``docs/evidence/wandb-offline.json``
    records that an offline run was actually created (``logged: 3``); a
    probe from an environment without the extra installed would replace
    it with ``logged: 0`` and quietly downgrade committed evidence to a
    weaker claim, with the run that supported the stronger one gone. Use
    ``--output`` to record such a probe somewhere else, or install the
    extra (``uv sync --extra wandb``) to refresh the artifact for real.
    """
    from app.meta_harness import runs as runs_mod
    from app.meta_harness import tracking as trk

    result = trk.offline_probe()
    destination = _abs(output)

    if (
        not result.get("wandb_installed")
        and destination.resolve() == _abs(DEFAULT_WANDB_EVIDENCE).resolve()
        and destination.exists()
    ):
        typer.echo(json.dumps(result, indent=2))
        typer.secho(
            f"Refusing to overwrite {DEFAULT_WANDB_EVIDENCE}: wandb is not "
            "installed here, so this probe cannot support the claim the "
            "committed artifact makes. Run `uv sync --extra wandb` to "
            "refresh it, or pass --output to record this probe elsewhere.",
            err=True,
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(1)

    runs_mod.write_json_atomic(destination, result)
    typer.echo(json.dumps(result, indent=2))
    if not result["ok"]:
        raise typer.Exit(1)


@report_app.command("cost-estimate")
def report_cost_estimate(
    trials: int = typer.Option(
        None,
        "--trials",
        help="Trials to price. Defaults to the committed protocols' total.",
    ),
) -> None:
    """Price a planned experiment from measured trial rows already on disk.

    Reports ``null`` rather than a number when there is nothing measured
    to extrapolate from.
    """
    from app.meta_harness import pipeline as pipe

    planned = trials if trials is not None else _planned_trials(
        REPO_ROOT / "benchmarks" / "pass-rate" / "config.json",
        REPO_ROOT / "benchmarks" / "holdout" / "config.json",
    )
    rows = pipe.collect_measured_rows(REPO_ROOT)
    typer.echo(json.dumps(pipe.estimate_cost(rows, planned_trials=planned), indent=2))


def _abs(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":
    main()
