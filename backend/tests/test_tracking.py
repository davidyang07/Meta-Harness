"""The experiment-tracking adapter.

Two properties matter more than the logging itself:

1. **Optional means optional.** No credentials, no network, no installed
   backend — everything still runs, and tracking reports why it is off
   rather than failing the work it was meant to observe.
2. **W&B does not leak into core logic.** The vocabulary core logic uses
   (``log_iteration``, ``log_frontier``, ``log_experiment``) is asserted
   against an in-memory tracker; the W&B-shaped translation is asserted
   separately with an injected fake module.

Nothing here imports ``wandb`` or touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import tracking as trk  # noqa: E402


# ── enablement ────────────────────────────────────────────────────────


def test_tracking_is_off_by_default(monkeypatch):
    monkeypatch.delenv("META_HARNESS_WANDB", raising=False)
    tracker = trk.make_tracker(run_name="r")
    assert isinstance(tracker, trk.NullTracker)
    assert tracker.enabled is False
    assert "not enabled" in (tracker.reason or "")


def test_env_var_enables_tracking(monkeypatch):
    monkeypatch.setenv("META_HARNESS_WANDB", "1")
    assert trk.tracking_requested() is True
    monkeypatch.setenv("META_HARNESS_WANDB", "off")
    assert trk.tracking_requested() is False


def test_an_explicit_flag_overrides_the_env_var(monkeypatch):
    monkeypatch.setenv("META_HARNESS_WANDB", "1")
    assert trk.tracking_requested(False) is False
    monkeypatch.delenv("META_HARNESS_WANDB", raising=False)
    assert trk.tracking_requested(True) is True


def test_a_missing_backend_degrades_to_null_with_a_reason():
    def _explode() -> Any:
        raise ImportError("no module named wandb")

    tracker = trk.make_tracker(
        enabled=True, run_name="r", wandb_module=_ImportErrorModule()
    )
    assert isinstance(tracker, trk.NullTracker)
    assert "not installed" in (tracker.reason or "")


class _ImportErrorModule:
    def init(self, **kwargs: Any) -> Any:
        raise ImportError("no module named wandb")


def test_a_backend_that_fails_to_start_never_fails_the_run():
    class _Broken:
        def init(self, **kwargs: Any) -> Any:
            raise RuntimeError("wandb: network unreachable")

    tracker = trk.make_tracker(enabled=True, run_name="r", wandb_module=_Broken())
    assert isinstance(tracker, trk.NullTracker)
    assert "failed to start" in (tracker.reason or "")
    # And it is still usable — a no-op tracker, not an exception.
    tracker.log({"x": 1})
    tracker.finish()


def test_the_null_tracker_accepts_the_whole_interface():
    tracker = trk.NullTracker()
    tracker.log({"a": 1}, step=0)
    tracker.log_table("t", ["a"], [[1]])
    tracker.log_artifact("n", Path("."), artifact_type="x", metadata={"k": "v"})
    tracker.set_summary({"a": 1})
    tracker.finish()


# ── the W&B translation, with an injected module ──────────────────────


class FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.url = "https://wandb.example/offline"


class FakeTable:
    def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
        self.columns = columns
        self.data = data


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[str] = []
        self.dirs: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)

    def add_dir(self, path: str) -> None:
        self.dirs.append(path)


class FakeWandb:
    """Records the calls a real ``wandb`` would have received."""

    Table = FakeTable
    Artifact = FakeArtifact

    def __init__(self) -> None:
        self.init_kwargs: dict[str, Any] | None = None
        self.logged: list[tuple[dict[str, Any], int | None]] = []
        self.artifacts: list[FakeArtifact] = []
        self.finished = False
        self.run = FakeRun()

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_kwargs = kwargs
        return self.run

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self.logged.append((payload, step))

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


def test_wandb_adapter_translates_the_interface():
    fake = FakeWandb()
    tracker = trk.make_tracker(
        enabled=True,
        project="proj",
        run_name="run-7",
        config={"budget": 3},
        tags=["a"],
        group="g",
        job_type="experiment",
        wandb_module=fake,
    )

    assert isinstance(tracker, trk.WandbTracker)
    assert fake.init_kwargs["project"] == "proj"
    assert fake.init_kwargs["name"] == "run-7"
    assert fake.init_kwargs["config"] == {"budget": 3}
    assert fake.init_kwargs["tags"] == ["a"]
    assert fake.init_kwargs["job_type"] == "experiment"
    assert tracker.run_url == "https://wandb.example/offline"

    tracker.log({"accuracy": 0.5}, step=2)
    assert fake.logged[-1] == ({"accuracy": 0.5}, 2)

    tracker.log({"accuracy": 0.6})
    assert fake.logged[-1] == ({"accuracy": 0.6}, None)

    tracker.log_table("per_task", ["task", "rate"], [["t1", 0.5]])
    payload, _ = fake.logged[-1]
    table = payload["per_task"]
    assert isinstance(table, FakeTable)
    assert table.columns == ["task", "rate"]
    assert table.data == [["t1", 0.5]]

    tracker.set_summary({"delta": 12.0})
    assert fake.run.summary["delta"] == 12.0

    tracker.finish()
    assert fake.finished is True


def test_wandb_adapter_attaches_files_and_directories(tmp_path: Path):
    fake = FakeWandb()
    tracker = trk.WandbTracker(
        project="p", run_name="r", config={}, wandb_module=fake
    )
    a_file = tmp_path / "summary.json"
    a_file.write_text("{}")
    a_dir = tmp_path / "results"
    a_dir.mkdir()

    tracker.log_artifact("summary", a_file, artifact_type="benchmark-results")
    tracker.log_artifact("results", a_dir)

    assert fake.artifacts[0].files == [str(a_file)]
    assert fake.artifacts[0].type == "benchmark-results"
    assert fake.artifacts[1].dirs == [str(a_dir)]


def test_wandb_project_falls_back_to_the_env_var(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "from-env")
    fake = FakeWandb()
    trk.make_tracker(enabled=True, run_name="r", wandb_module=fake)
    assert fake.init_kwargs["project"] == "from-env"


# ── the vocabulary core logic uses ────────────────────────────────────


def test_log_iteration_carries_the_metrics_source():
    """A mock-benchmarked iteration must never look measured in a dashboard."""
    tracker = trk.MemoryTracker()
    trk.log_iteration(
        tracker,
        iteration=2,
        candidate="cand-a",
        accuracy=0.7,
        delta=0.1,
        accepted=True,
        axis="exploitation",
        metrics_source="mock",
        mean_tokens=20000.0,
        thread_id="run.fork.abc",
        branch_id="abc",
        per_task={"t1": {"pass_rate": 0.8}},
    )

    payload, step = tracker.metrics[0]
    assert step == 2
    assert payload["metrics_source"] == "mock"
    assert payload["candidate"] == "cand-a"
    assert payload["accepted"] == 1
    assert payload["branch_id"] == "abc"
    assert payload["mean_total_tokens_per_trial"] == 20000.0

    name, columns, rows = tracker.tables[0]
    assert name == "per_task/iteration-2"
    assert columns == ["task_id", "pass_rate", "candidate", "metrics_source"]
    assert rows == [["t1", 0.8, "cand-a", "mock"]]


def test_log_iteration_omits_cost_when_it_was_not_measured():
    tracker = trk.MemoryTracker()
    trk.log_iteration(
        tracker,
        iteration=1,
        candidate="c",
        accuracy=0.5,
        delta=None,
        accepted=False,
        axis=None,
        metrics_source="measured",
        cost_usd=None,
    )
    payload, _ = tracker.metrics[0]
    assert "cost_usd" not in payload


def test_log_trial_reports_the_outcome_and_cost():
    tracker = trk.MemoryTracker()
    trk.log_trial(
        tracker,
        {
            "passed": True,
            "total_tokens": 21000,
            "llm_calls": 6,
            "wall_time_s": 12.5,
            "cost_usd": 0.03,
        },
        arm="candidate",
    )
    payload, _ = tracker.metrics[0]
    assert payload["candidate/passed"] == 1
    assert payload["candidate/total_tokens"] == 21000
    assert payload["candidate/cost_usd"] == 0.03


def test_log_frontier_records_pareto_membership():
    tracker = trk.MemoryTracker()
    trk.log_frontier(
        tracker,
        {
            "iteration": 3,
            "candidates": [
                {
                    "name": "a",
                    "accuracy": 0.8,
                    "avg_tokens": 1000,
                    "metrics_source": "measured",
                    "dominated_by_names": [],
                },
                {
                    "name": "b",
                    "accuracy": 0.5,
                    "avg_tokens": 2000,
                    "metrics_source": "measured",
                    "dominated_by_names": ["a"],
                },
            ],
            "_pareto_names": ["a"],
            "_best": {"accuracy": 0.8},
            "metrics_source": "measured",
        },
        thread_id="run-1",
    )
    payload, step = tracker.metrics[0]
    assert step == 3
    assert payload["frontier/size"] == 1
    assert payload["frontier/best_accuracy"] == 0.8

    _, _, rows = tracker.tables[0]
    assert rows[0][-1] is True
    assert rows[1][-1] is False


def test_log_experiment_reads_only_the_derived_summary(tmp_path: Path):
    """A tracked number and a published number have one source, so they agree."""
    tracker = trk.MemoryTracker()
    summary_file = tmp_path / "summary.json"
    summary_file.write_text("{}")
    summary = {
        "schema_version": "1.0.0",
        "baseline_accuracy": 0.63,
        "candidate_accuracy": 0.75,
        "absolute_percentage_point_delta": 12.0,
        "total_trials": 200,
        "difference_ci": {"lower": -0.01, "upper": 0.25},
        "metrics_source": "measured",
        "per_task": {"t1": {"baseline_passes": 12, "candidate_passes": 15}},
        "arms": {
            "baseline": {"tokens": {"total_tokens": 100}, "total_cost_usd": 1.0},
            "candidate": {"tokens": {"total_tokens": 120}, "total_cost_usd": 1.2},
        },
    }

    trk.log_experiment(
        tracker,
        summary=summary,
        environment={"model": "m", "git": {"commit": "abc"}},
        artifacts={"summary": summary_file},
    )

    assert tracker.summary["absolute_percentage_point_delta"] == 12.0
    assert tracker.summary["baseline_total_cost_usd"] == 1.0
    assert tracker.summary["model"] == "m"
    assert tracker.summary["git_commit"] == "abc"
    name, columns, rows = tracker.tables[0]
    assert name == "experiment/per_task"
    assert rows[0][0] == "t1"
    assert tracker.artifacts[0]["type"] == "benchmark-results"


def test_the_vocabulary_is_a_no_op_on_a_disabled_tracker():
    """Core logic calls these unconditionally; off must cost nothing."""
    tracker = trk.NullTracker()
    trk.log_trial(tracker, {"passed": True})
    trk.log_iteration(
        tracker,
        iteration=1,
        candidate="c",
        accuracy=1.0,
        delta=0.0,
        accepted=True,
        axis=None,
        metrics_source="measured",
    )
    trk.log_frontier(tracker, {"iteration": 1, "candidates": []})
    trk.log_experiment(tracker, summary={})
    assert tracker.enabled is False


def test_core_modules_do_not_import_wandb():
    """The adapter is the only place that may know W&B exists."""
    offenders = []
    for path in sorted((REPO_ROOT / "backend" / "app").rglob("*.py")):
        if path.name == "tracking.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "import wandb" in text or "wandb." in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"W&B leaked into {offenders}"
