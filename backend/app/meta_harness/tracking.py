"""Experiment tracking: one interface, a no-op default, and a W&B adapter.

Core optimization logic must not know that Weights & Biases exists. It
calls the small vocabulary in this module —
:func:`log_iteration`, :func:`log_experiment`, :func:`log_frontier`,
:func:`log_trial` — and those translate into whatever the configured
:class:`Tracker` does. Swapping W&B for something else, or for nothing,
is a change to this file only.

**Tracking is optional and off by default.** The repository, its tests
and every CLI command work with no W&B account, no credentials and no
network. ``make_tracker`` returns a :class:`NullTracker` unless tracking
is explicitly enabled, and returns one (with a stated reason) if
``wandb`` is not installed or fails to start. A tracker never raises
into a caller: a metrics backend that is down must not fail a 200-trial
experiment that costs real money.

**Offline works.** ``WANDB_MODE=offline`` is the supported way to run
this without touching the network — W&B writes a local run directory
that can be synced later, or never.

**No credentials are read, written, or logged here.** The config a
tracker receives is assembled by the caller from the same provenance
block the experiment already writes, which deliberately contains no
environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_PROJECT = "meta-harness"

#: Truthy values for ``META_HARNESS_WANDB``.
_TRUE = {"1", "true", "yes", "on"}


class Tracker:
    """No-op tracker. The default, and the shape every backend implements."""

    #: Whether anything is actually being recorded.
    enabled: bool = False
    #: Why tracking is off, when it is off for a reason worth reporting.
    reason: str | None = None
    #: Where the backend put its data, when it has a location.
    run_url: str | None = None

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Record a flat metric dict, optionally at an explicit step."""

    def log_table(
        self, name: str, columns: list[str], rows: list[list[Any]]
    ) -> None:
        """Record a tabular result (per-task breakdowns, trial rows)."""

    def log_artifact(
        self,
        name: str,
        path: Path,
        *,
        artifact_type: str = "results",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach a file or directory of evidence to the run."""

    def set_summary(self, values: dict[str, Any]) -> None:
        """Record the run's headline values."""

    def finish(self) -> None:
        """Close the run."""


class NullTracker(Tracker):
    """Explicitly-off tracker that remembers why."""

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason


class MemoryTracker(Tracker):
    """In-memory tracker used by tests and by ``--dry-run`` inspection.

    Records exactly what a real backend would have been asked to do,
    which is what makes the adapter testable without a network, an
    account, or the ``wandb`` package.
    """

    enabled = True

    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, Any], int | None]] = []
        self.tables: list[tuple[str, list[str], list[list[Any]]]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.finished = False

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))

    def log_table(
        self, name: str, columns: list[str], rows: list[list[Any]]
    ) -> None:
        self.tables.append((name, list(columns), [list(r) for r in rows]))

    def log_artifact(
        self,
        name: str,
        path: Path,
        *,
        artifact_type: str = "results",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.artifacts.append(
            {
                "name": name,
                "path": str(path),
                "type": artifact_type,
                "metadata": dict(metadata or {}),
            }
        )

    def set_summary(self, values: dict[str, Any]) -> None:
        self.summary.update(values)

    def finish(self) -> None:
        self.finished = True


class WandbTracker(Tracker):
    """Weights & Biases adapter.

    ``wandb_module`` is injectable so the translation from this
    interface into W&B's API can be tested without installing W&B and
    without a network call.
    """

    enabled = True

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        config: dict[str, Any],
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
        wandb_module: Any = None,
    ) -> None:
        self._wandb = wandb_module or _import_wandb()
        self._run = self._wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=list(tags or []),
            group=group,
            job_type=job_type,
            reinit=True,
        )
        self.run_url = getattr(self._run, "url", None)

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if step is None:
            self._wandb.log(metrics)
        else:
            self._wandb.log(metrics, step=step)

    def log_table(
        self, name: str, columns: list[str], rows: list[list[Any]]
    ) -> None:
        table = self._wandb.Table(columns=list(columns), data=[list(r) for r in rows])
        self._wandb.log({name: table})

    def log_artifact(
        self,
        name: str,
        path: Path,
        *,
        artifact_type: str = "results",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        artifact = self._wandb.Artifact(
            name=name, type=artifact_type, metadata=dict(metadata or {})
        )
        path = Path(path)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self._wandb.log_artifact(artifact)

    def set_summary(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self._run.summary[key] = value

    def finish(self) -> None:
        self._wandb.finish()


def _import_wandb() -> Any:
    import wandb  # noqa: PLC0415 — optional dependency, imported on demand

    return wandb


def tracking_requested(explicit: bool | None = None) -> bool:
    """Whether the caller asked for tracking.

    ``explicit`` (a CLI flag) wins; otherwise ``META_HARNESS_WANDB``
    decides. Off unless asked: nothing in this repository should start
    talking to a third-party service because a package happened to be
    installed.
    """
    if explicit is not None:
        return explicit
    return os.environ.get("META_HARNESS_WANDB", "").strip().lower() in _TRUE


def make_tracker(
    *,
    enabled: bool | None = None,
    project: str | None = None,
    run_name: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    job_type: str | None = None,
    wandb_module: Any = None,
) -> Tracker:
    """Return the configured tracker, or a ``NullTracker`` with a reason.

    Never raises. Tracking is instrumentation; if it cannot start, the
    work it was going to observe still has to run.
    """
    if not tracking_requested(enabled):
        return NullTracker("tracking not enabled (set META_HARNESS_WANDB=1 or --wandb)")
    try:
        return WandbTracker(
            project=project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT,
            run_name=run_name,
            config=dict(config or {}),
            tags=tags,
            group=group,
            job_type=job_type,
            wandb_module=wandb_module,
        )
    except ImportError:
        return NullTracker(
            "wandb is not installed; install the optional extra with "
            "`uv sync --extra wandb`"
        )
    except Exception as exc:  # noqa: BLE001 — instrumentation must not fail the run
        return NullTracker(f"wandb failed to start ({type(exc).__name__}: {exc})")


# ──────────────────────────────────────────────────────────────────────
# The vocabulary core logic uses. Nothing above this line is W&B-shaped.
# ──────────────────────────────────────────────────────────────────────


def log_trial(tracker: Tracker, row: dict[str, Any], *, arm: str | None = None) -> None:
    """One benchmark trial: outcome, tokens, cost, wall time."""
    if not tracker.enabled:
        return
    prefix = f"{arm}/" if arm else "trial/"
    tracker.log(
        {
            f"{prefix}passed": 1 if row.get("passed") else 0,
            f"{prefix}total_tokens": row.get("total_tokens"),
            f"{prefix}llm_calls": row.get("llm_calls"),
            f"{prefix}wall_time_s": row.get("wall_time_s"),
            **(
                {f"{prefix}cost_usd": row["cost_usd"]}
                if row.get("cost_usd") is not None
                else {}
            ),
        }
    )


def log_iteration(
    tracker: Tracker,
    *,
    iteration: int,
    candidate: str,
    accuracy: float | None,
    delta: float | None,
    accepted: bool,
    axis: str | None,
    metrics_source: str | None,
    mean_tokens: float | None = None,
    cost_usd: float | None = None,
    thread_id: str | None = None,
    branch_id: str | None = None,
    per_task: dict[str, Any] | None = None,
) -> None:
    """One outer-loop iteration: what was tried and what it scored.

    ``metrics_source`` rides along on every point so a mock-benchmarked
    iteration can never be mistaken for a measured one in a dashboard —
    the same rule the on-disk payloads follow.
    """
    if not tracker.enabled:
        return
    payload: dict[str, Any] = {
        "iteration": iteration,
        "candidate": candidate,
        "accuracy": accuracy,
        "delta": delta,
        "accepted": 1 if accepted else 0,
        "axis": axis,
        "metrics_source": metrics_source,
        "thread_id": thread_id,
        "branch_id": branch_id,
    }
    if mean_tokens is not None:
        payload["mean_total_tokens_per_trial"] = mean_tokens
    if cost_usd is not None:
        payload["cost_usd"] = cost_usd
    tracker.log(payload, step=iteration)

    if per_task:
        tracker.log_table(
            f"per_task/iteration-{iteration}",
            ["task_id", "pass_rate", "candidate", "metrics_source"],
            [
                [task_id, info.get("pass_rate"), candidate, metrics_source]
                for task_id, info in sorted(per_task.items())
            ],
        )


def log_frontier(
    tracker: Tracker, frontier: dict[str, Any], *, thread_id: str | None = None
) -> None:
    """The Pareto frontier after an iteration."""
    if not tracker.enabled:
        return
    candidates = frontier.get("candidates") or []
    tracker.log(
        {
            "frontier/size": len(frontier.get("_pareto_names") or []),
            "frontier/candidates": len(candidates),
            "frontier/best_accuracy": (frontier.get("_best") or {}).get("accuracy"),
            "frontier/metrics_source": frontier.get("metrics_source"),
            "thread_id": thread_id,
        },
        step=frontier.get("iteration"),
    )
    tracker.log_table(
        "frontier",
        ["name", "accuracy", "avg_tokens", "metrics_source", "on_frontier"],
        [
            [
                c.get("name"),
                c.get("accuracy"),
                c.get("avg_tokens"),
                c.get("metrics_source"),
                not c.get("dominated_by_names"),
            ]
            for c in candidates
        ],
    )


def log_experiment(
    tracker: Tracker,
    *,
    summary: dict[str, Any],
    environment: dict[str, Any] | None = None,
    artifacts: dict[str, Path] | None = None,
) -> None:
    """The two-arm experiment result: pass rates, delta, CI, per-task rows.

    Reads only from the derived summary, so a tracked number and a
    published number cannot disagree — they have the same single source.
    """
    if not tracker.enabled:
        return
    ci = summary.get("difference_ci") or {}
    arms = summary.get("arms") or {}
    baseline_arm = arms.get("baseline") or {}
    candidate_arm = arms.get("candidate") or {}
    tracker.set_summary(
        {
            "baseline_accuracy": summary.get("baseline_accuracy"),
            "candidate_accuracy": summary.get("candidate_accuracy"),
            "absolute_percentage_point_delta": summary.get(
                "absolute_percentage_point_delta"
            ),
            "total_trials": summary.get("total_trials"),
            "difference_ci_lower": ci.get("lower"),
            "difference_ci_upper": ci.get("upper"),
            "baseline_total_tokens": (baseline_arm.get("tokens") or {}).get(
                "total_tokens"
            ),
            "candidate_total_tokens": (candidate_arm.get("tokens") or {}).get(
                "total_tokens"
            ),
            "baseline_total_cost_usd": baseline_arm.get("total_cost_usd"),
            "candidate_total_cost_usd": candidate_arm.get("total_cost_usd"),
            "metrics_source": summary.get("metrics_source"),
            "model": (environment or {}).get("model"),
            "git_commit": ((environment or {}).get("git") or {}).get("commit"),
        }
    )
    tracker.log_table(
        "experiment/per_task",
        [
            "task_id",
            "baseline_passes",
            "baseline_trials",
            "candidate_passes",
            "candidate_trials",
            "percentage_point_delta",
        ],
        [
            [
                task_id,
                row.get("baseline_passes"),
                row.get("baseline_trials"),
                row.get("candidate_passes"),
                row.get("candidate_trials"),
                row.get("percentage_point_delta"),
            ]
            for task_id, row in sorted((summary.get("per_task") or {}).items())
        ],
    )
    for name, path in (artifacts or {}).items():
        tracker.log_artifact(
            name,
            Path(path),
            artifact_type="benchmark-results",
            metadata={"schema_version": summary.get("schema_version")},
        )


def offline_probe() -> dict[str, Any]:
    """Exercise the adapter in offline mode and report what happened.

    Forces ``WANDB_MODE=offline`` so the probe never touches the network
    and never needs credentials. A missing ``wandb`` is reported as
    ``ok`` with a reason, not as a failure: the integration is optional,
    and "the repository works without it" is the property being checked.

    Lives here rather than in the CLI so this module stays the only place
    that knows W&B exists.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_SILENT", "true")

    result: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline",
        "wandb_installed": False,
        "wandb_version": None,
        "ok": False,
        "logged": 0,
        "run_url": None,
        "detail": "",
    }
    try:
        module = _import_wandb()
    except ImportError:
        result["ok"] = True
        result["detail"] = (
            "wandb is not installed; the adapter is exercised by "
            "tests/test_tracking.py with an injected module, and the "
            "repository runs without it"
        )
        return result

    result["wandb_installed"] = True
    result["wandb_version"] = getattr(module, "__version__", None)

    tracker = make_tracker(
        enabled=True,
        run_name="wandb-offline-probe",
        config={"probe": True},
        tags=["probe"],
        job_type="probe",
    )
    if not tracker.enabled:
        result["detail"] = tracker.reason or "tracker did not start"
        return result

    tracker.log({"probe/value": 1}, step=0)
    tracker.log_table("probe/table", ["a", "b"], [[1, 2]])
    tracker.set_summary({"probe": "ok"})
    tracker.finish()

    result["ok"] = True
    result["logged"] = 3
    result["run_url"] = tracker.run_url
    result["detail"] = "offline run created; no network access required"
    return result


__all__ = [
    "DEFAULT_PROJECT",
    "MemoryTracker",
    "NullTracker",
    "Tracker",
    "WandbTracker",
    "log_experiment",
    "log_frontier",
    "log_iteration",
    "log_trial",
    "make_tracker",
    "offline_probe",
    "tracking_requested",
]
