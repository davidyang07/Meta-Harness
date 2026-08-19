"""The pass-rate experiment: summary derivation, provenance, IO.

The single most important property here is that the headline number
cannot be authored. ``summarize`` takes raw trial rows and nothing else,
so the only way to change the reported delta is to change the trials.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from app.meta_harness import experiment as exp
from app.meta_harness import metrics as met

REPO_ROOT = Path(__file__).resolve().parents[2]


def _rows(arm: str, spec: dict[str, list[bool]]) -> list[dict[str, Any]]:
    """Build raw rows from {task_id: [pass, pass, ...]}."""
    out = []
    for task_id, outcomes in spec.items():
        for trial, passed in enumerate(outcomes, start=1):
            out.append(
                {
                    "task_id": task_id,
                    "trial": trial,
                    "arm": arm,
                    "passed": passed,
                    "score": 1.0 if passed else 0.0,
                    "wall_time_s": 1.0,
                    "metrics_source": met.MEASURED,
                    "llm_calls": 2,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 150,
                    "cost_usd": 0.001,
                    "calls": [{"index": 1, "input_tokens": 100}],
                }
            )
    return out


# ── the summary is derived, never entered ─────────────────────────────


def test_summary_is_computed_from_raw_rows():
    baseline = _rows("baseline", {"t1": [True, False], "t2": [False, False]})
    candidate = _rows("candidate", {"t1": [True, True], "t2": [True, False]})

    summary = exp.summarize(
        baseline_rows=baseline,
        candidate_rows=candidate,
        task_ids=["t1", "t2"],
        baseline_label="baseline",
        candidate_label="evolved",
    )

    assert summary["baseline_passes"] == 1
    assert summary["baseline_trials"] == 4
    assert summary["baseline_accuracy"] == 0.25
    assert summary["candidate_passes"] == 3
    assert summary["candidate_trials"] == 4
    assert summary["candidate_accuracy"] == 0.75
    assert summary["absolute_percentage_point_delta"] == 50.0
    assert summary["total_trials"] == 8


def test_changing_one_trial_changes_the_headline_number():
    """There is no path from a desired number back into the summary."""
    base = _rows("baseline", {"t1": [True, False]})
    cand = _rows("candidate", {"t1": [True, True]})
    kwargs = dict(task_ids=["t1"], baseline_label="b", candidate_label="c")

    before = exp.summarize(baseline_rows=base, candidate_rows=cand, **kwargs)
    cand[1]["passed"] = False
    after = exp.summarize(baseline_rows=base, candidate_rows=cand, **kwargs)

    assert before["absolute_percentage_point_delta"] == 50.0
    assert after["absolute_percentage_point_delta"] == 0.0


def test_summarize_accepts_no_target_or_expected_value():
    """Guard against anyone adding a 'target delta' parameter later."""
    params = set(inspect.signature(exp.summarize).parameters)
    assert params == {
        "baseline_rows",
        "candidate_rows",
        "task_ids",
        "baseline_label",
        "candidate_label",
    }


def test_per_task_breakdown_matches_the_rows():
    baseline = _rows("baseline", {"t1": [True, False], "t2": [False, False]})
    candidate = _rows("candidate", {"t1": [True, True], "t2": [True, True]})
    summary = exp.summarize(
        baseline_rows=baseline,
        candidate_rows=candidate,
        task_ids=["t1", "t2"],
        baseline_label="b",
        candidate_label="c",
    )
    t1 = summary["per_task"]["t1"]
    assert (t1["baseline_passes"], t1["baseline_trials"]) == (1, 2)
    assert (t1["candidate_passes"], t1["candidate_trials"]) == (2, 2)
    assert t1["percentage_point_delta"] == 50.0
    assert summary["per_task"]["t2"]["percentage_point_delta"] == 100.0


def test_empty_experiment_reports_no_result_rather_than_zero():
    summary = exp.summarize(
        baseline_rows=[],
        candidate_rows=[],
        task_ids=[],
        baseline_label="b",
        candidate_label="c",
    )
    assert summary["baseline_accuracy"] is None
    assert summary["absolute_percentage_point_delta"] is None
    assert "No measured result" in exp.reported_metric_sentence(summary)


# ── statistics ────────────────────────────────────────────────────────


def test_wald_interval_brackets_the_observed_difference():
    ci = exp.wald_diff_ci(63, 100, 75, 100)
    assert ci["difference"] == pytest.approx(0.12)
    assert ci["lower"] < ci["difference"] < ci["upper"]
    assert ci["confidence"] == 0.95
    # The clustering caveat is part of the payload, not a footnote.
    assert "clustered within tasks" in ci["assumptions"]


def test_wald_interval_widens_with_fewer_trials():
    wide = exp.wald_diff_ci(6, 10, 8, 10)
    narrow = exp.wald_diff_ci(600, 1000, 800, 1000)
    assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])


def test_wald_interval_handles_zero_trials():
    ci = exp.wald_diff_ci(0, 0, 0, 0)
    assert ci["lower"] is None and ci["upper"] is None


def test_reported_metric_sentence_quotes_only_measured_values():
    summary = exp.summarize(
        baseline_rows=_rows("baseline", {"t": [True, False, False, False]}),
        candidate_rows=_rows("candidate", {"t": [True, True, True, False]}),
        task_ids=["t"],
        baseline_label="baseline",
        candidate_label="evolved",
    )
    sentence = exp.reported_metric_sentence(summary)
    assert "+50.0 percentage points" in sentence
    assert "across 8 task trials" in sentence
    assert "1/4 baseline vs 3/4 evolved" in sentence


# ── provenance ────────────────────────────────────────────────────────


def test_task_hashes_change_when_a_task_changes(tmp_path: Path):
    task_dir = tmp_path / "task-x"
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "task.json").write_text('{"id": "task-x"}')
    (task_dir / "workspace" / "a.py").write_text("x = 1\n")

    first = exp.hash_task(task_dir)
    assert first["task_id"] == "task-x"
    assert "workspace/a.py" in first["files_sha256"]

    (task_dir / "workspace" / "a.py").write_text("x = 2\n")
    second = exp.hash_task(task_dir)
    assert second["task_sha256"] != first["task_sha256"]


def test_environment_capture_records_provenance_and_no_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-appear")
    env = exp.capture_environment(
        repo_root=REPO_ROOT,
        model="claude-haiku-4-5-20251001",
        tasks=sorted(
            d for d in (REPO_ROOT / "eval" / "tasks").iterdir() if d.is_dir()
        ),
        arm_sources={"baseline": REPO_ROOT / "agents" / "baseline.py"},
    )

    assert env["schema_version"] == exp.RESULT_SCHEMA_VERSION
    assert env["model"] == "claude-haiku-4-5-20251001"
    assert env["git"]["commit"], "commit SHA must be captured"
    assert isinstance(env["git"]["dirty"], bool)
    assert env["python"] and env["platform"]
    assert len(env["tasks"]) == 5
    assert env["arm_sources"]["baseline"]["sha256"]

    blob = json.dumps(env)
    assert "sk-should-never-appear" not in blob
    assert "ANTHROPIC_API_KEY" not in blob


# ── result IO ─────────────────────────────────────────────────────────


def test_rows_round_trip_without_per_call_detail(tmp_path: Path):
    rows = _rows("baseline", {"t1": [True, False]})
    path = tmp_path / "baseline-results.jsonl"
    exp.write_rows(path, rows)

    back = exp.read_rows(path)
    assert len(back) == 2
    assert "calls" not in back[0], "per-call detail is intentionally dropped"
    # Everything the summary needs survives.
    assert back[0]["task_id"] == "t1"
    assert back[0]["total_tokens"] == 150


def test_summary_recomputes_identically_from_the_written_files(tmp_path: Path):
    """Anyone can re-derive the published number from the raw rows."""
    baseline = _rows("baseline", {"t1": [True, False, False]})
    candidate = _rows("candidate", {"t1": [True, True, False]})
    kwargs = dict(task_ids=["t1"], baseline_label="b", candidate_label="c")
    summary = exp.summarize(
        baseline_rows=baseline, candidate_rows=candidate, **kwargs
    )

    paths = exp.write_results(
        output_dir=tmp_path / "out",
        config={"experiment": "t"},
        environment={"git": {"commit": "abc"}, "model": "m", "tasks": []},
        baseline_rows=baseline,
        candidate_rows=candidate,
        summary=summary,
    )

    recomputed = exp.summarize(
        baseline_rows=exp.read_rows(paths["baseline_results"]),
        candidate_rows=exp.read_rows(paths["candidate_results"]),
        **kwargs,
    )
    published = json.loads(paths["summary"].read_text())
    assert recomputed["absolute_percentage_point_delta"] == (
        published["absolute_percentage_point_delta"]
    )
    assert recomputed["baseline_passes"] == published["baseline_passes"]
    assert recomputed["candidate_passes"] == published["candidate_passes"]
    assert paths["report"].exists()


# ── the committed protocol ────────────────────────────────────────────


def test_committed_config_matches_the_documented_200_trial_protocol():
    config = exp.load_config(REPO_ROOT / "benchmarks" / "pass-rate" / "config.json")
    assert config["task_set"] == "eval/tasks"
    assert config["trials_per_task"] == 20
    assert len(config["tasks"]) == 5
    assert config["total_trials"] == 200
    # 5 tasks x 20 trials x 2 arms
    assert len(config["tasks"]) * config["trials_per_task"] * 2 == 200
    # The evolved arm is supplied at run time, never pinned in the repo.
    assert config["arms"]["candidate"] is None


def test_committed_config_task_ids_exist_on_disk():
    config = exp.load_config(REPO_ROOT / "benchmarks" / "pass-rate" / "config.json")
    for task_id in config["tasks"]:
        assert (REPO_ROOT / config["task_set"] / task_id / "task.json").is_file()


def test_load_config_rejects_an_incomplete_protocol(tmp_path: Path):
    bad = tmp_path / "config.json"
    bad.write_text(json.dumps({"experiment": "x"}))
    with pytest.raises(ValueError, match="missing required key"):
        exp.load_config(bad)
