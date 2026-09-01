"""The measurement pipeline, end to end, without a provider.

``meta-harness resume-experiment`` is the command that produces every
number in `docs/RESUME_EVIDENCE.md`, and it will be run once, blind, on a
machine with credentials and a real bill. So the parts that do not need a
provider — config loading, protocol matching, running both arms, the
methodology checks, the artifact layout, the tracker wiring, and the
evidence derivation that reads the result back — are exercised here with
scripted harnesses and a miniature protocol.

What this does not cover: the proposer subprocess and real model calls.
Those are LEVEL 2 (`scripts/live_smoke.sh`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import evidence as ev  # noqa: E402
from app.meta_harness import experiment as exp  # noqa: E402
from app.meta_harness import metrics as met  # noqa: E402
from app.meta_harness import pipeline as pipe  # noqa: E402
from app.meta_harness import tracking as trk  # noqa: E402
from tests.harness_doubles import FailingHarness, FixTypoHarness  # noqa: E402

#: The control arm. The committed baseline refuses to construct without
#: an API key, so the offline pipeline test needs a scripted stand-in
#: that behaves identically apart from where its model turns come from.
BASELINE_DOUBLE = FailingHarness

#: A protocol small enough to run in a test: one task, two trials, two
#: arms. Same shape as the committed 200-trial one, and it goes through
#: the same runner.
MINI_CONFIG = {
    "experiment": "pass-rate",
    "description": "miniature protocol used by the pipeline integration test",
    "task_set": "eval/tasks",
    "tasks": ["task-001-fix-typo"],
    "trials_per_task": 2,
    "arms": {"baseline": "baseline", "candidate": None},
    "workers": 2,
    "total_trials": 4,
}


def _mini_protocol(tmp_path: Path, *, tasks_dir: Path) -> Path:
    """Write a config whose task set is a one-task copy of the real one."""
    import shutil

    task = tasks_dir / "task-001-fix-typo"
    destination = tmp_path / "eval" / "tasks" / "task-001-fix-typo"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task, destination)

    config = {**MINI_CONFIG, "task_set": "eval/tasks"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _selection(candidate_cls: type) -> dict[str, Any]:
    """A selection record pointing at an importable scripted harness."""
    return {
        "selected": "scripted-candidate",
        "selected_row": {
            "candidate": "scripted-candidate",
            "import_path": (
                f"{candidate_cls.__module__}:{candidate_cls.__qualname__}"
            ),
            "source_path": None,
            "validation_accuracy": 1.0,
            "iteration": 1,
        },
        "reason": "test fixture",
        "selection_basis": "test fixture",
        "table": [],
    }


async def test_the_pipeline_produces_a_complete_publishable_result(tmp_path: Path):
    config_path = _mini_protocol(tmp_path, tasks_dir=REPO_ROOT / "eval" / "tasks")
    tracker = trk.MemoryTracker()

    stages = await pipe.run_measurement_pipeline(
        repo_root=tmp_path,
        search_config_path=config_path,
        holdout_config_path=None,
        selection=_selection(FixTypoHarness),
        baseline_label="baseline",
        baseline_factory=BASELINE_DOUBLE,
        output_root=tmp_path / "results",
        checkpointer=None,
        tracker=tracker,
        record_trials_per_task=0,
        run_holdout=False,
    )

    experiment_stage = next(s for s in stages if s.name == "experiment")
    assert experiment_stage.status == "ok", experiment_stage.detail

    output_dir = Path(experiment_stage.data["output_dir"])
    for name in (
        "config.json",
        "environment.json",
        "baseline-results.jsonl",
        "candidate-results.jsonl",
        "validation.json",
        "summary.json",
        "REPORT.md",
    ):
        assert (output_dir / name).exists(), f"missing {name}"

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["total_trials"] == 4
    assert summary["metrics_source"] == met.MEASURED


async def test_both_arms_are_recorded_as_having_run_one_protocol(tmp_path: Path):
    config_path = _mini_protocol(tmp_path, tasks_dir=REPO_ROOT / "eval" / "tasks")

    stages = await pipe.run_measurement_pipeline(
        repo_root=tmp_path,
        search_config_path=config_path,
        holdout_config_path=None,
        selection=_selection(FixTypoHarness),
        baseline_label="baseline",
        baseline_factory=BASELINE_DOUBLE,
        output_root=tmp_path / "results",
        run_holdout=False,
    )

    validation = next(s for s in stages if s.name == "experiment").data["validation"]
    assert validation["identical_protocol"] is True, validation["checks"]
    assert validation["checks"]["measured_only"] is True
    assert validation["baseline_completeness"]["complete"] is True
    assert validation["candidate_completeness"]["complete"] is True


async def test_the_published_summary_re_derives_from_its_raw_rows(tmp_path: Path):
    """The property CI enforces, checked against a result this code wrote."""
    config_path = _mini_protocol(tmp_path, tasks_dir=REPO_ROOT / "eval" / "tasks")

    stages = await pipe.run_measurement_pipeline(
        repo_root=tmp_path,
        search_config_path=config_path,
        holdout_config_path=None,
        selection=_selection(FixTypoHarness),
        baseline_label="baseline",
        baseline_factory=BASELINE_DOUBLE,
        output_root=tmp_path / "results",
        run_holdout=False,
    )
    output_dir = Path(next(s for s in stages if s.name == "experiment").data["output_dir"])

    published = json.loads((output_dir / "summary.json").read_text())
    environment = json.loads((output_dir / "environment.json").read_text())
    recomputed = exp.summarize(
        baseline_rows=exp.read_rows(output_dir / "baseline-results.jsonl"),
        candidate_rows=exp.read_rows(output_dir / "candidate-results.jsonl"),
        task_ids=[t["task_id"] for t in environment["tasks"]],
        baseline_label=published["baseline_label"],
        candidate_label=published["candidate_label"],
    )
    for key in (
        "baseline_passes",
        "candidate_passes",
        "baseline_accuracy",
        "candidate_accuracy",
        "absolute_percentage_point_delta",
    ):
        assert recomputed[key] == published[key]


async def test_the_delta_is_whatever_the_trials_say(tmp_path: Path):
    """A candidate that solves nothing produces a negative delta, reported.

    There is no branch anywhere in the pipeline that reruns, reselects or
    filters on the final number.
    """
    config_path = _mini_protocol(tmp_path, tasks_dir=REPO_ROOT / "eval" / "tasks")

    stages = await pipe.run_measurement_pipeline(
        repo_root=tmp_path,
        search_config_path=config_path,
        holdout_config_path=None,
        selection=_selection(FailingHarness),
        baseline_label="baseline",
        baseline_factory=BASELINE_DOUBLE,
        output_root=tmp_path / "results",
        run_holdout=False,
    )

    summary = next(s for s in stages if s.name == "experiment").data["summary"]
    assert summary["candidate_accuracy"] == 0.0
    assert summary["absolute_percentage_point_delta"] <= 0.0
    assert "percentage points" in exp.reported_metric_sentence(summary)


async def test_a_result_the_pipeline_wrote_reads_back_as_evidence(tmp_path: Path):
    """Publish where the evidence reader looks, and read the rows back."""
    config_path = _mini_protocol(tmp_path, tasks_dir=REPO_ROOT / "eval" / "tasks")

    stages = await pipe.run_measurement_pipeline(
        repo_root=tmp_path,
        search_config_path=config_path,
        holdout_config_path=None,
        selection=_selection(FixTypoHarness),
        baseline_label="baseline",
        baseline_factory=BASELINE_DOUBLE,
        output_root=tmp_path / "benchmarks" / "results",
        run_holdout=False,
    )
    assert next(s for s in stages if s.name == "experiment").status == "ok"

    loaded = ev.latest_experiment(tmp_path, experiment="pass-rate")

    assert loaded is not None
    assert loaded["reproducible"] is True
    assert loaded["mismatched_keys"] == []
    checks = {c.key: c for c in ev._measurement_checks(loaded)}
    # Four trials is not two hundred, and the row says so.
    assert checks["canonical_200_trials"].status == ev.FAIL
    assert checks["canonical_200_trials"].value == 4
    # The pass-rate rows are real measurements from real trials.
    assert checks["baseline_pass_rate"].value is not None
    assert checks["evolved_pass_rate"].value is not None


async def test_the_pipeline_logs_iterations_and_the_experiment_to_the_tracker(
    tmp_path: Path,
):
    config_path = _mini_protocol(tmp_path, tasks_dir=REPO_ROOT / "eval" / "tasks")
    tracker = trk.MemoryTracker()

    await pipe.run_measurement_pipeline(
        repo_root=tmp_path,
        search_config_path=config_path,
        holdout_config_path=None,
        selection=_selection(FixTypoHarness),
        baseline_label="baseline",
        baseline_factory=BASELINE_DOUBLE,
        output_root=tmp_path / "results",
        tracker=tracker,
        run_holdout=False,
    )

    assert tracker.summary["absolute_percentage_point_delta"] is not None
    assert tracker.summary["metrics_source"] == met.MEASURED
    assert any(name == "experiment/per_task" for name, _, _ in tracker.tables)
    assert tracker.artifacts, "the result files should be attached as artifacts"


async def test_a_protocol_whose_tasks_are_missing_fails_the_stage(tmp_path: Path):
    """A protocol that does not match the task set on disk must not run."""
    config_path = _mini_protocol(tmp_path, tasks_dir=REPO_ROOT / "eval" / "tasks")
    config = json.loads(config_path.read_text())
    config["tasks"] = ["task-999-does-not-exist"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    stages = await pipe.run_measurement_pipeline(
        repo_root=tmp_path,
        search_config_path=config_path,
        holdout_config_path=None,
        selection=_selection(FixTypoHarness),
        baseline_label="baseline",
        baseline_factory=BASELINE_DOUBLE,
        output_root=tmp_path / "results",
        run_holdout=False,
    )

    assert stages[0].status == "failed"
    assert "does not match the committed protocol" in stages[0].detail
