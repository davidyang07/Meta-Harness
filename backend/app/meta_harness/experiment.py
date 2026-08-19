"""The canonical two-arm pass-rate experiment.

Protocol (``benchmarks/pass-rate/config.json``):

    5 frozen search tasks x 20 independent trials  = 100 baseline trials
    the same 5 tasks     x 20 independent trials  = 100 evolved trials
                                                    -------------------
                                                    200 task trials

Design rules this module enforces, because the whole point is that the
resulting number is defensible:

1. **The summary is derived, never entered.** ``summarize()`` takes raw
   trial rows and computes everything. There is no code path that
   accepts a pass rate, a delta, or a target.
2. **Both arms run the identical protocol** — same tasks, same trial
   count, same worker pool, same model — so the difference is
   attributable to the harness and nothing else.
3. **Provenance is captured, secrets are not.** Commit SHA, dirty flag,
   model id, per-task definition and test hashes, harness source hashes,
   interpreter and platform. No environment variables are copied.
4. **The statistics are stated with their limitations.** A Wald interval
   on the difference of two proportions assumes independent Bernoulli
   trials; trials are clustered within tasks, so the interval is
   reported alongside an explicit caveat rather than as a p-value.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.meta_harness import benchmark as bench
from app.meta_harness import metrics as met
from app.meta_harness import runs as runs_mod

#: Bump when the on-disk result schema changes incompatibly.
RESULT_SCHEMA_VERSION = "1.0.0"

#: 95% two-sided normal quantile, for the difference-in-proportions interval.
Z_95 = 1.959963984540054


# ── provenance ────────────────────────────────────────────────────────


def _git(*args: str, repo_root: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def file_sha256(path: Path) -> str | None:
    """Hex SHA-256 of a file, or ``None`` if it is not readable."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def hash_task(task_dir: Path) -> dict[str, Any]:
    """Content hashes that pin one task's definition and its tests.

    A published result is only meaningful if the tasks it was measured
    against can be shown not to have changed since.
    """
    spec_path = task_dir / "task.json"
    files: dict[str, str] = {}
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest = file_sha256(path)
        if digest:
            files[str(path.relative_to(task_dir)).replace("\\", "/")] = digest
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "task_id": task_dir.name,
        "task_json_sha256": file_sha256(spec_path),
        "files_sha256": files,
        "task_sha256": combined,
    }


def capture_environment(
    *,
    repo_root: Path,
    model: str,
    tasks: list[Path],
    arm_sources: dict[str, Path | None],
) -> dict[str, Any]:
    """Everything needed to judge, and re-run, a published result.

    Deliberately records no environment variables: the point is
    reproducibility, and copying the environment is how API keys end up
    in committed artifacts.
    """
    dirty = _git("status", "--porcelain", repo_root=repo_root)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git("rev-parse", "HEAD", repo_root=repo_root),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo_root=repo_root),
            "dirty": bool(dirty) if dirty is not None else None,
            "dirty_paths": sorted(
                line[3:] for line in (dirty or "").splitlines() if line[3:]
            )[:50],
        },
        "model": model,
        "pricing_source": "META_HARNESS_PRICING" if _pricing_overridden() else "builtin",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tasks": [hash_task(t) for t in tasks],
        "arm_sources": {
            name: {
                "path": str(path) if path else None,
                "sha256": file_sha256(path) if path else None,
            }
            for name, path in arm_sources.items()
        },
    }


def _pricing_overridden() -> bool:
    import os

    return bool(os.environ.get("META_HARNESS_PRICING"))


# ── statistics ────────────────────────────────────────────────────────


def wald_diff_ci(
    passes_a: int, n_a: int, passes_b: int, n_b: int, *, z: float = Z_95
) -> dict[str, Any]:
    """95% Wald interval for ``p_b - p_a`` (difference of two proportions).

    Reported with its assumptions rather than as a significance verdict:
    trials are clustered within tasks, so this interval is optimistic
    about independence.
    """
    if n_a <= 0 or n_b <= 0:
        return {"lower": None, "upper": None, "method": "wald-95", "note": "no trials"}
    p_a, p_b = passes_a / n_a, passes_b / n_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    diff = p_b - p_a
    return {
        "difference": round(diff, 6),
        "standard_error": round(se, 6),
        "lower": round(diff - z * se, 6),
        "upper": round(diff + z * se, 6),
        "confidence": 0.95,
        "method": "wald-95",
        "assumptions": (
            "independent Bernoulli trials; trials are clustered within tasks, "
            "so the true interval is wider than this one"
        ),
    }


def _arm_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trials = len(rows)
    passes = sum(1 for r in rows if r.get("passed"))
    aggregate = met.aggregate_trials(rows, metrics_source=met.MEASURED) if rows else {}
    return {
        "trials": trials,
        "passes": passes,
        "accuracy": round(passes / trials, 6) if trials else None,
        "tokens": aggregate.get("tokens"),
        "mean_total_tokens_per_trial": aggregate.get("mean_total_tokens_per_trial"),
        "median_total_tokens_per_trial": aggregate.get(
            "median_total_tokens_per_trial"
        ),
        "total_llm_calls": aggregate.get("total_llm_calls"),
        "total_wall_time_s": aggregate.get("total_wall_time_s"),
        "total_cost_usd": aggregate.get("total_cost_usd"),
        "cost_complete": aggregate.get("cost_complete", False),
    }


def _per_task(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_ids: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task_id in task_ids:
        b = [r for r in baseline_rows if r["task_id"] == task_id]
        c = [r for r in candidate_rows if r["task_id"] == task_id]
        b_pass = sum(1 for r in b if r["passed"])
        c_pass = sum(1 for r in c if r["passed"])
        b_acc = b_pass / len(b) if b else None
        c_acc = c_pass / len(c) if c else None
        out[task_id] = {
            "baseline_passes": b_pass,
            "baseline_trials": len(b),
            "baseline_accuracy": round(b_acc, 6) if b_acc is not None else None,
            "candidate_passes": c_pass,
            "candidate_trials": len(c),
            "candidate_accuracy": round(c_acc, 6) if c_acc is not None else None,
            "percentage_point_delta": (
                round((c_acc - b_acc) * 100, 4)
                if b_acc is not None and c_acc is not None
                else None
            ),
        }
    return out


def summarize(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_ids: list[str],
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    """Derive the whole summary from raw trial rows.

    This is the only place the headline number is produced, and it has
    no input other than the rows themselves.
    """
    baseline = _arm_stats(baseline_rows)
    candidate = _arm_stats(candidate_rows)
    delta_pp = (
        round((candidate["accuracy"] - baseline["accuracy"]) * 100, 4)
        if baseline["accuracy"] is not None and candidate["accuracy"] is not None
        else None
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_passes": baseline["passes"],
        "baseline_trials": baseline["trials"],
        "baseline_accuracy": baseline["accuracy"],
        "candidate_passes": candidate["passes"],
        "candidate_trials": candidate["trials"],
        "candidate_accuracy": candidate["accuracy"],
        "absolute_percentage_point_delta": delta_pp,
        "total_trials": baseline["trials"] + candidate["trials"],
        "difference_ci": wald_diff_ci(
            baseline["passes"],
            baseline["trials"],
            candidate["passes"],
            candidate["trials"],
        ),
        "per_task": _per_task(baseline_rows, candidate_rows, task_ids),
        "arms": {"baseline": baseline, "candidate": candidate},
        "metrics_source": met.MEASURED,
    }


def reported_metric_sentence(summary: dict[str, Any]) -> str:
    """The one sentence a reader may quote, built only from the summary."""
    delta = summary.get("absolute_percentage_point_delta")
    if delta is None:
        return "No measured result: the experiment produced no trials."
    return (
        f"Improved agent pass rate by {delta:+.1f} percentage points "
        f"across {summary['total_trials']} task trials "
        f"({summary['baseline_passes']}/{summary['baseline_trials']} baseline vs "
        f"{summary['candidate_passes']}/{summary['candidate_trials']} evolved)."
    )


def render_report(summary: dict[str, Any]) -> str:
    """Human-readable console report. Every number comes from the summary."""
    lines = [
        f"Baseline  ({summary['baseline_label']}): "
        f"{summary['baseline_passes']}/{summary['baseline_trials']} = "
        f"{_pct(summary['baseline_accuracy'])}",
        f"Candidate ({summary['candidate_label']}): "
        f"{summary['candidate_passes']}/{summary['candidate_trials']} = "
        f"{_pct(summary['candidate_accuracy'])}",
    ]
    delta = summary.get("absolute_percentage_point_delta")
    if delta is not None:
        lines.append(f"Absolute improvement: {delta:+.1f} percentage points")
    ci = summary.get("difference_ci") or {}
    if ci.get("lower") is not None:
        lines.append(
            f"95% CI on the difference: "
            f"[{ci['lower'] * 100:+.1f}, {ci['upper'] * 100:+.1f}] pp "
            f"({ci['method']})"
        )
    lines.append(f"Total trials: {summary['total_trials']}")
    lines.append("")
    lines.append("Per task:")
    for task_id, row in summary.get("per_task", {}).items():
        lines.append(
            f"  {task_id}: {row['baseline_passes']}/{row['baseline_trials']} -> "
            f"{row['candidate_passes']}/{row['candidate_trials']} "
            f"({row['percentage_point_delta']:+.1f} pp)"
            if row["percentage_point_delta"] is not None
            else f"  {task_id}: incomplete"
        )
    lines.append("")
    lines.append("Reported metric:")
    lines.append(f'  "{reported_metric_sentence(summary)}"')
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


# ── raw result IO ─────────────────────────────────────────────────────


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write raw trial rows as JSONL, one object per trial.

    Per-call detail is dropped: it is large, and the summary depends
    only on the per-trial aggregates, which stay verifiable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            slim = {k: v for k, v in row.items() if k != "calls"}
            fh.write(json.dumps(slim, default=str) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read raw trial rows back from JSONL."""
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


# ── the experiment ────────────────────────────────────────────────────


async def run_arm(
    *,
    arm: str,
    harness_factory: Callable[[], Any],
    tasks_dir: Path,
    trials: int,
    workers: int,
    output_dir: Path,
    experiment_id: str,
    checkpointer: Any = None,
    on_trial: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run one arm of the experiment and return its raw trial rows."""
    task_dirs = bench.discover_tasks(tasks_dir)
    traces_root = output_dir / "traces" / arm

    rows = await bench.run_trials(
        harness_factory=harness_factory,
        task_dirs=task_dirs,
        trials=trials,
        workers=workers,
        trace_dir_for=lambda task_id, trial: traces_root
        / f"{task_id}-trial-{trial}",
        inner_thread_id_for=lambda task_id, trial: met.inner_thread_id(
            run_id=experiment_id,
            thread_id=experiment_id,
            candidate=arm,
            task_id=task_id,
            trial=trial,
        ),
        checkpointer=checkpointer,
        on_trial=(lambda row: on_trial(arm, row)) if on_trial else None,
    )
    for row in rows:
        row["arm"] = arm
    return rows


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the committed canonical protocol."""
    config = json.loads(path.read_text())
    for key in ("experiment", "task_set", "trials_per_task", "workers"):
        if key not in config:
            raise ValueError(f"{path} is missing required key {key!r}")
    if int(config["trials_per_task"]) < 1:
        raise ValueError("trials_per_task must be >= 1")
    return config


def write_results(
    *,
    output_dir: Path,
    config: dict[str, Any],
    environment: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    """Persist config, provenance, raw rows and the derived summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": output_dir / "config.json",
        "environment": output_dir / "environment.json",
        "baseline_results": output_dir / "baseline-results.jsonl",
        "candidate_results": output_dir / "candidate-results.jsonl",
        "summary": output_dir / "summary.json",
        "report": output_dir / "REPORT.md",
    }
    runs_mod.write_json_atomic(paths["config"], config)
    runs_mod.write_json_atomic(paths["environment"], environment)
    write_rows(paths["baseline_results"], baseline_rows)
    write_rows(paths["candidate_results"], candidate_rows)
    runs_mod.write_json_atomic(paths["summary"], summary)
    paths["report"].write_text(
        _report_markdown(config, environment, summary), encoding="utf-8"
    )
    return paths


def _report_markdown(
    config: dict[str, Any], environment: dict[str, Any], summary: dict[str, Any]
) -> str:
    git = environment.get("git", {})
    return f"""# {config.get('experiment', 'experiment')} — measured result

Generated by `meta-harness experiment`. Every number below is derived
from `baseline-results.jsonl` and `candidate-results.jsonl` by
`app.meta_harness.experiment.summarize`; nothing here is hand-entered.

- Commit: `{git.get('commit')}` (dirty: `{git.get('dirty')}`)
- Model: `{environment.get('model')}`
- Captured: `{environment.get('captured_at')}`
- Protocol: {len(environment.get('tasks', []))} tasks x \
{config.get('trials_per_task')} trials per arm

```
{render_report(summary)}
```

## Reproducing

```bash
uv run meta-harness experiment --config benchmarks/pass-rate/config.json \\
    --candidate <candidate-name>
```

## Limitations

- Trials are clustered within tasks, so the reported Wald interval on the
  difference in proportions is optimistic about independence.
- Results are tied to the task hashes in `environment.json`. Changing a
  task invalidates comparison with this experiment.
- This measures the five frozen search tasks. Generalisation to unseen
  tasks is a separate holdout measurement.
"""
