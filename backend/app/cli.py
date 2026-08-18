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

    async def _run() -> dict[str, Any]:
        with sandbox_for(task_dir / "workspace") as sandbox:
            return await run_inner_loop(
                harness,
                task_dict=task_spec,
                workspace=sandbox,
                trace_dir=trace_dir,
                thread_id=thread_id,
            )

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
        )

    final_state = _run_async(_run())

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
            },
            indent=2,
        )
    )


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

    from app.meta_harness.branches import worktree_add
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


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":
    main()
