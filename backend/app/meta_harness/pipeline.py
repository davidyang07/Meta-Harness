"""End-to-end evidence pipeline: evolve → select → measure → verify → report.

``meta-harness canonical-experiment`` is this module. It exists so the
sequence that produces a publishable number is one command and one code
path rather than a README paragraph someone follows by hand.

The ordering is the methodology, and it is deliberate:

1. **Evolve** on the search set (``eval/tasks``) with the real proposer.
2. **Select** the best candidate using *only* the validation numbers the
   outer loop measured during evolution. The final experiment has not
   run at this point, so there is no way for its trials to influence
   which candidate gets tested — this is what stops the headline number
   from being a selection artifact.
3. **Measure** the canonical two-arm experiment: baseline vs the selected
   candidate, identical protocol, fresh independent trials.
4. **Generalise** on the holdout set, which the proposer never saw.
5. **Verify** that a recorded trial replays exactly.
6. **Report** — derive the evidence document from the artifacts.

Nothing here compares a result to a target. If the measured delta is
below the resume's claim, the pipeline records the measured delta and the
evidence document marks the claim unsupported. There is no branch in this
file that reruns, reselects, or filters on the final number.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.meta_harness import benchmark as bench
from app.meta_harness import candidates as cand_mod
from app.meta_harness import experiment as exp
from app.meta_harness import metrics as met
from app.meta_harness import recording as rec
from app.meta_harness import replay as replay_mod
from app.meta_harness import runs as runs_mod
from app.meta_harness import tracking as trk
from app.meta_harness.state import BASELINE_CANDIDATE_NAME


def _now_id(prefix: str) -> str:
    return f"{prefix}-{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


@dataclass
class StageResult:
    """One pipeline stage's outcome, for the run log and the report."""

    name: str
    status: str  # "ok" | "skipped" | "failed"
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "status": self.status,
            "detail": self.detail,
            "data": self.data,
        }


# ── candidate selection ───────────────────────────────────────────────


def select_candidate(final_state: dict[str, Any]) -> dict[str, Any]:
    """Pick the candidate to test, using validation results only.

    The input is the outer loop's terminal state: every candidate it
    proposed, with the accuracy it measured on the search set during
    evolution. The final experiment has not run yet, and this function
    has no access to it.

    Returns the decision *and* the table it was made from, so a reader
    can check that the winner really is the validation winner.
    """
    candidates = list(final_state.get("candidates") or [])
    table = [
        {
            "candidate": c["name"],
            "label": c.get("label") or c["name"],
            "iteration": c.get("iteration"),
            "status": c.get("status"),
            "validation_accuracy": (c.get("scores") or {}).get("accuracy"),
            "validation_trials": (c.get("scores") or {}).get("total_trials"),
            "metrics_source": (c.get("scores") or {}).get("metrics_source"),
            "mean_total_tokens_per_trial": (c.get("scores") or {}).get(
                "mean_total_tokens_per_trial"
            ),
            "source_path": c.get("source_path"),
            "source_sha256": c.get("source_sha256"),
            "import_path": c.get("import_path"),
            "axis": c.get("axis"),
            "hypothesis": c.get("hypothesis"),
        }
        for c in candidates
    ]

    evolved = [
        row
        for row in table
        if row["candidate"] != BASELINE_CANDIDATE_NAME
        and row["validation_accuracy"] is not None
        and row["metrics_source"] == met.MEASURED
    ]
    baseline_row = next(
        (r for r in table if r["candidate"] == BASELINE_CANDIDATE_NAME), None
    )

    if not evolved:
        return {
            "selected": None,
            "reason": (
                "no evolved candidate was measured on the search set; there is "
                "nothing to compare against the baseline"
            ),
            "selection_basis": "validation accuracy on eval/tasks, measured "
            "during evolution",
            "baseline": baseline_row,
            "table": table,
        }

    # Highest validation accuracy; ties go to the cheaper candidate, then
    # to the earlier iteration so the choice is deterministic.
    winner = min(
        evolved,
        key=lambda r: (
            -float(r["validation_accuracy"]),
            float(r["mean_total_tokens_per_trial"] or float("inf")),
            int(r["iteration"] or 0),
        ),
    )
    return {
        "selected": winner["candidate"],
        "selected_row": winner,
        "reason": (
            f"highest validation accuracy on the search set "
            f"({winner['validation_accuracy']}) across "
            f"{len(evolved)} measured evolved candidates"
        ),
        "selection_basis": "validation accuracy on eval/tasks, measured during "
        "evolution; the final experiment had not run at selection time",
        "baseline": baseline_row,
        "table": table,
    }


def harness_factory_for(
    row: dict[str, Any], *, repo_root: Path
) -> Callable[[], Any]:
    """Build the harness factory for a selected candidate row.

    Prefers the branch-private source snapshot the outer loop actually
    benchmarked, so the experiment measures the same bytes the validation
    did — not whatever ``agents/<label>.py`` happens to hold now.
    """

    def _factory() -> Any:
        cls = cand_mod.load_harness_class(
            {
                "import_path": row["import_path"],
                "source_path": row.get("source_path"),
            },
            repo_root=repo_root,
        )
        cand_mod.assert_is_harness(cls, import_path=row["import_path"])
        return cls()

    return _factory


# ── cost estimation ───────────────────────────────────────────────────


def estimate_cost(
    rows: list[dict[str, Any]], *, planned_trials: int
) -> dict[str, Any]:
    """Extrapolate the cost of a planned run from measured trial rows.

    Returns ``None`` values rather than zeros when there is nothing to
    extrapolate from: an estimate built on no measurement is not a cheap
    estimate, it is a fiction.
    """
    measured = [r for r in rows if r.get("metrics_source") == met.MEASURED]
    if not measured:
        return {
            "basis_trials": 0,
            "planned_trials": planned_trials,
            "mean_total_tokens_per_trial": None,
            "mean_cost_usd_per_trial": None,
            "estimated_total_tokens": None,
            "estimated_total_cost_usd": None,
            "note": "no measured trial rows available to extrapolate from",
        }
    tokens = [int(r.get("total_tokens") or 0) for r in measured]
    costs = [r.get("cost_usd") for r in measured]
    cost_known = all(c is not None for c in costs)
    mean_tokens = sum(tokens) / len(tokens)
    mean_cost = (sum(c for c in costs if c is not None) / len(costs)) if cost_known else None
    return {
        "basis_trials": len(measured),
        "planned_trials": planned_trials,
        "mean_total_tokens_per_trial": round(mean_tokens, 2),
        "mean_cost_usd_per_trial": round(mean_cost, 6) if mean_cost is not None else None,
        "estimated_total_tokens": int(round(mean_tokens * planned_trials)),
        "estimated_total_cost_usd": (
            round(mean_cost * planned_trials, 4) if mean_cost is not None else None
        ),
        "note": (
            "extrapolated from measured trials; actual cost varies with how "
            "many act turns each trial needs"
            if cost_known
            else "token counts measured; cost unavailable because the model "
            "has no configured price (see META_HARNESS_PRICING)"
        ),
    }


def collect_measured_rows(repo_root: Path, *, limit_runs: int = 20) -> list[dict[str, Any]]:
    """Every measured trial row this repository has on disk.

    Used to price a planned experiment before spending anything. Reads
    published results first, then run artifacts.
    """
    rows: list[dict[str, Any]] = []
    results_root = repo_root / "benchmarks" / "results"
    if results_root.is_dir():
        for directory in sorted(results_root.iterdir()):
            for name in ("baseline-results.jsonl", "candidate-results.jsonl"):
                rows.extend(exp.read_rows(directory / name))
    runs_root = repo_root / "runs"
    if runs_root.is_dir():
        for run_dir in sorted(runs_root.iterdir())[-limit_runs:]:
            for metrics_path in run_dir.rglob("metrics.json"):
                try:
                    row = json.loads(metrics_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if row.get("metrics_source") == met.MEASURED:
                    rows.append(row)
    return [r for r in rows if r.get("metrics_source") == met.MEASURED]


# ── replay verification stage ─────────────────────────────────────────


async def verify_recorded_replays(
    *,
    recordings_root: Path,
    repo_root: Path,
    checkpointer: Any,
    source_graph_factory: Callable[[dict[str, Any]], Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Replay every recorded trial under ``recordings_root`` and verify it.

    Each recording is replayed twice: once from the start of the run, and
    once from a mid-run checkpoint, because those are two different
    claims — "this tape reproduces the run" and "any stored checkpoint is
    a valid entry point".
    """
    directories = rec.discover_recordings(recordings_root)
    if limit is not None:
        directories = directories[:limit]

    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for directory in directories:
        recording = rec.read_recording(directory)
        factory = _replay_harness_factory(recording, repo_root=repo_root)

        # Replaying the whole run needs nothing but the tape: the entry
        # state is in the manifest.
        full = await replay_mod.replay_recorded_execution(
            recording=recording,
            harness_factory=factory,
            checkpointer=checkpointer,
        )
        reports.append({"recording": directory.name, "mode": "full", **full})

        # Replaying *from a checkpoint* additionally needs the store that
        # holds the recorded thread. If it is not reachable, that is a
        # missing precondition, not a failed replay — record it as
        # skipped rather than letting it read as a pass or a failure.
        mid = _mid_checkpoint(recording)
        if mid is None:
            continue
        source_graph = (
            source_graph_factory(recording)
            if source_graph_factory is not None
            else replay_mod.build_inner_graph(factory(), checkpointer=checkpointer)
        )
        try:
            partial = await replay_mod.replay_recorded_execution(
                recording=recording,
                harness_factory=factory,
                checkpointer=checkpointer,
                source_graph=source_graph,
                from_checkpoint=mid,
            )
        except KeyError as exc:
            skipped.append(
                {
                    "recording": directory.name,
                    "mode": "from-checkpoint",
                    "checkpoint": mid,
                    "reason": (
                        f"the recorded thread {recording.thread_id!r} is not in "
                        f"the configured checkpoint store ({exc}). Point "
                        f"POSTGRES_DSN at the database the recording was made "
                        f"against."
                    ),
                }
            )
            continue
        reports.append(
            {"recording": directory.name, "mode": "from-checkpoint", **partial}
        )

    return {
        "recordings_root": str(recordings_root),
        "recordings_verified": len(directories),
        # What was replayed, so the report is readable without the tapes.
        # ``recorded_models`` distinguishes a provider recording from a
        # scripted one; both exercise the same replay machinery, and a
        # reader is entitled to know which they are looking at.
        "recorded_models": sorted(
            {str(r.get("recorded_model")) for r in reports if r.get("recorded_model")}
        ),
        "recorded_tasks": sorted(
            {str(r.get("recorded_task_id")) for r in reports if r.get("recorded_task_id")}
        ),
        "replays": len(reports),
        "replays_from_checkpoint": sum(
            1 for r in reports if r["mode"] == "from-checkpoint"
        ),
        "skipped": skipped,
        "all_verified": bool(reports) and all(r["verified"] for r in reports),
        "model_calls_issued": sum(r["model_calls_issued"] for r in reports),
        "reports": reports,
    }


def _mid_checkpoint(recording: rec.Recording) -> str | None:
    """A checkpoint partway through the recording, to fork the replay from."""
    with_ids = [s for s in recording.steps if s.checkpoint_id]
    if len(with_ids) < 2:
        return None
    return with_ids[len(with_ids) // 2].checkpoint_id


def _replay_harness_factory(
    recording: rec.Recording, *, repo_root: Path
) -> Callable[[], Any]:
    """Rebuild the harness class a recording was made with.

    Replay never calls the model, so the harness needs no credentials —
    but it must be the same class, because the non-model override points
    (result formatting, retry policy, act prompt) are part of the state
    machine being replayed.
    """
    module_path, _, class_name = str(
        recording.manifest.get("harness_class") or ""
    ).partition(":")
    source_path = recording.manifest.get("candidate_source_path")

    def _factory() -> Any:
        import importlib  # noqa: PLC0415
        import sys  # noqa: PLC0415

        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        if source_path:
            cls = cand_mod.load_harness_from_source(Path(source_path), class_name)
        else:
            cls = getattr(importlib.import_module(module_path), class_name)
        try:
            return cls()
        except RuntimeError:
            # The base harness refuses to construct without an API key.
            # A replay issues no model call, so build the instance
            # without running __init__ and give it the attributes the
            # non-model code paths read.
            instance = cls.__new__(cls)
            instance.api_key = None
            instance._client = None
            return instance

    return _factory


# ── the pipeline ──────────────────────────────────────────────────────


async def run_measurement_pipeline(
    *,
    repo_root: Path,
    search_config_path: Path,
    holdout_config_path: Path | None,
    selection: dict[str, Any],
    baseline_label: str,
    output_root: Path,
    trials: int | None = None,
    workers: int | None = None,
    checkpointer: Any = None,
    tracker: trk.Tracker | None = None,
    on_trial: Callable[[str, dict[str, Any]], None] | None = None,
    record_trials_per_task: int = 1,
    run_holdout: bool = True,
    baseline_factory: Callable[[], Any] | None = None,
) -> list[StageResult]:
    """Run the canonical experiment, then the holdout experiment.

    Both use the same runner, so the holdout arm is measured exactly the
    way the search arm is, and the two numbers are comparable.

    ``baseline_factory`` defaults to the committed ``agents/baseline.py``
    and exists so the pipeline can be exercised end to end without a
    provider: the real baseline refuses to construct without an API key,
    which would otherwise make the whole sequence untestable offline.
    Production callers leave it alone.
    """
    stages: list[StageResult] = []
    row = selection.get("selected_row")
    if row is None:
        stages.append(
            StageResult(
                "experiment",
                "skipped",
                selection.get("reason", "no candidate was selected"),
            )
        )
        return stages

    candidate_factory = harness_factory_for(row, repo_root=repo_root)
    baseline_factory = baseline_factory or harness_factory_for(
        {
            "import_path": "agents.baseline:BaselineHarness",
            "source_path": None,
        },
        repo_root=repo_root,
    )
    model = getattr(candidate_factory(), "MODEL", None) or "unknown"
    candidate_source = (
        Path(row["source_path"]) if row.get("source_path") else None
    )

    search_stage = await _run_one_experiment(
        repo_root=repo_root,
        config_path=search_config_path,
        stage_name="experiment",
        output_root=output_root,
        baseline_label=baseline_label,
        candidate_label=str(row["candidate"]),
        baseline_factory=baseline_factory,
        candidate_factory=candidate_factory,
        candidate_source=candidate_source,
        model=model,
        trials=trials,
        workers=workers,
        checkpointer=checkpointer,
        tracker=tracker,
        on_trial=on_trial,
        record_trials_per_task=record_trials_per_task,
    )
    stages.append(search_stage)

    if run_holdout and holdout_config_path is not None:
        stages.append(
            await _run_one_experiment(
                repo_root=repo_root,
                config_path=holdout_config_path,
                stage_name="holdout-experiment",
                output_root=output_root,
                baseline_label=baseline_label,
                candidate_label=str(row["candidate"]),
                baseline_factory=baseline_factory,
                candidate_factory=candidate_factory,
                candidate_source=candidate_source,
                model=model,
                trials=trials,
                workers=workers,
                checkpointer=checkpointer,
                tracker=tracker,
                on_trial=on_trial,
                record_trials_per_task=0,
            )
        )
    elif run_holdout:
        stages.append(
            StageResult(
                "holdout-experiment", "skipped", "no holdout protocol configured"
            )
        )
    return stages


async def _run_one_experiment(
    *,
    repo_root: Path,
    config_path: Path,
    stage_name: str,
    output_root: Path,
    baseline_label: str,
    candidate_label: str,
    baseline_factory: Callable[[], Any],
    candidate_factory: Callable[[], Any],
    candidate_source: Path | None,
    model: str,
    trials: int | None,
    workers: int | None,
    checkpointer: Any,
    tracker: trk.Tracker | None,
    on_trial: Callable[[str, dict[str, Any]], None] | None,
    record_trials_per_task: int,
) -> StageResult:
    config = exp.load_config(config_path)
    tasks_dir = repo_root / config["task_set"]
    n_trials = trials if trials is not None else int(config["trials_per_task"])
    n_workers = workers if workers is not None else int(config["workers"])

    declared = config.get("tasks")
    found = [d.name for d in bench.discover_tasks(tasks_dir)]
    if declared and sorted(found) != sorted(declared):
        return StageResult(
            stage_name,
            "failed",
            f"task set does not match the committed protocol: expected "
            f"{sorted(declared)}, found {sorted(found)}",
        )

    experiment_id = _now_id(str(config["experiment"]))
    result = await exp.run_two_arm_experiment(
        repo_root=repo_root,
        config=config,
        experiment_id=experiment_id,
        tasks_dir=tasks_dir,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_factory=baseline_factory,
        candidate_factory=candidate_factory,
        arm_sources={
            baseline_label: repo_root / "agents" / f"{baseline_label}.py",
            candidate_label: candidate_source,
        },
        model=model,
        trials=n_trials,
        workers=n_workers,
        output_dir=output_root / experiment_id,
        checkpointer=checkpointer,
        tracker=tracker,
        on_trial=on_trial,
        record_trials_per_task=record_trials_per_task,
    )
    summary = result["summary"]
    return StageResult(
        stage_name,
        "ok",
        exp.reported_metric_sentence(summary),
        {
            "experiment_id": experiment_id,
            "output_dir": result["output_dir"],
            "summary": summary,
            "validation": result["validation"],
            "environment": result["environment"],
        },
    )


def write_pipeline_log(output_dir: Path, stages: list[StageResult]) -> Path:
    """Persist the stage log beside the results it produced."""
    path = output_dir / "pipeline.json"
    runs_mod.write_json_atomic(
        path,
        {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "stages": [s.to_dict() for s in stages],
        },
    )
    return path


__all__ = [
    "StageResult",
    "collect_measured_rows",
    "estimate_cost",
    "harness_factory_for",
    "run_measurement_pipeline",
    "select_candidate",
    "verify_recorded_replays",
    "write_pipeline_log",
]
