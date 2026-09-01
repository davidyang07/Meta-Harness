"""The evidence pipeline: selection, cost estimation, stage discipline.

The property that matters most here is that candidate selection cannot
see the final experiment. ``select_candidate`` takes the outer loop's
terminal state — validation numbers measured during evolution — and
nothing else, so the trials that produce the headline number cannot
influence which candidate produces it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import metrics as met  # noqa: E402
from app.meta_harness import pipeline as pipe  # noqa: E402


def _candidate(
    name: str,
    accuracy: float | None,
    *,
    iteration: int = 1,
    tokens: float | None = 1000.0,
    source: str = met.MEASURED,
    status: str = "evaluated",
) -> dict[str, Any]:
    return {
        "name": name,
        "label": name,
        "iteration": iteration,
        "status": status,
        "axis": "exploitation",
        "hypothesis": "h",
        "import_path": f"agents.{name}:H",
        "source_path": f"/runs/x/agents/{name}.py",
        "source_sha256": "deadbeef",
        "scores": None
        if accuracy is None
        else {
            "accuracy": accuracy,
            "total_trials": 25,
            "metrics_source": source,
            "mean_total_tokens_per_trial": tokens,
        },
    }


# ── selection ─────────────────────────────────────────────────────────


def test_selection_picks_the_validation_winner():
    state = {
        "candidates": [
            _candidate("baseline", 0.5, iteration=0),
            _candidate("cand-a", 0.6, iteration=1),
            _candidate("cand-b", 0.72, iteration=2),
            _candidate("cand-c", 0.65, iteration=3),
        ]
    }

    decision = pipe.select_candidate(state)

    assert decision["selected"] == "cand-b"
    assert decision["selected_row"]["validation_accuracy"] == 0.72
    assert "validation accuracy" in decision["selection_basis"]


def test_selection_never_picks_the_baseline():
    """The baseline is the control arm, not a candidate for the treatment arm."""
    state = {
        "candidates": [
            _candidate("baseline", 0.9, iteration=0),
            _candidate("cand-a", 0.4, iteration=1),
        ]
    }
    assert pipe.select_candidate(state)["selected"] == "cand-a"


def test_selection_ignores_mock_scored_candidates():
    """A synthesized score is not a validation measurement."""
    state = {
        "candidates": [
            _candidate("baseline", 0.5, iteration=0),
            _candidate("mocked", 0.99, iteration=1, source=met.MOCK),
            _candidate("measured", 0.55, iteration=2),
        ]
    }
    assert pipe.select_candidate(state)["selected"] == "measured"


def test_selection_ignores_unscored_candidates():
    state = {
        "candidates": [
            _candidate("baseline", 0.5, iteration=0),
            _candidate("failed", None, iteration=1, status="smoke_failed"),
            _candidate("ok", 0.55, iteration=2),
        ]
    }
    assert pipe.select_candidate(state)["selected"] == "ok"


def test_selection_breaks_ties_on_cost_then_iteration():
    state = {
        "candidates": [
            _candidate("baseline", 0.5, iteration=0),
            _candidate("expensive", 0.7, iteration=1, tokens=30000.0),
            _candidate("cheap", 0.7, iteration=2, tokens=10000.0),
        ]
    }
    assert pipe.select_candidate(state)["selected"] == "cheap"


def test_selection_is_deterministic():
    state = {
        "candidates": [
            _candidate("baseline", 0.5, iteration=0),
            _candidate("a", 0.7, iteration=1, tokens=1000.0),
            _candidate("b", 0.7, iteration=2, tokens=1000.0),
        ]
    }
    assert {pipe.select_candidate(state)["selected"] for _ in range(5)} == {"a"}


def test_selection_reports_nothing_to_test_rather_than_inventing_one():
    state = {"candidates": [_candidate("baseline", 0.5, iteration=0)]}
    decision = pipe.select_candidate(state)
    assert decision["selected"] is None
    assert "nothing to compare" in decision["reason"]


def test_selection_publishes_the_table_it_decided_from():
    """A reader must be able to check that the winner really won."""
    state = {
        "candidates": [
            _candidate("baseline", 0.5, iteration=0),
            _candidate("a", 0.6, iteration=1),
            _candidate("b", 0.8, iteration=2),
        ]
    }
    decision = pipe.select_candidate(state)
    accuracies = {r["candidate"]: r["validation_accuracy"] for r in decision["table"]}
    assert accuracies == {"baseline": 0.5, "a": 0.6, "b": 0.8}
    best = max(
        (r for r in decision["table"] if r["candidate"] != "baseline"),
        key=lambda r: r["validation_accuracy"],
    )
    assert best["candidate"] == decision["selected"]


def test_select_candidate_cannot_see_a_final_experiment():
    """Guard the property by signature: validation state is the only input."""
    import inspect

    params = set(inspect.signature(pipe.select_candidate).parameters)
    assert params == {"final_state"}


# ── cost estimation ───────────────────────────────────────────────────


def _row(tokens: int, cost: float | None) -> dict[str, Any]:
    return {
        "metrics_source": met.MEASURED,
        "total_tokens": tokens,
        "cost_usd": cost,
    }


def test_cost_estimate_extrapolates_from_measured_trials():
    estimate = pipe.estimate_cost(
        [_row(20000, 0.02), _row(30000, 0.04)], planned_trials=100
    )
    assert estimate["basis_trials"] == 2
    assert estimate["mean_total_tokens_per_trial"] == 25000.0
    assert estimate["mean_cost_usd_per_trial"] == 0.03
    assert estimate["estimated_total_tokens"] == 2_500_000
    assert estimate["estimated_total_cost_usd"] == 3.0


def test_cost_estimate_reports_none_rather_than_zero_without_data():
    estimate = pipe.estimate_cost([], planned_trials=200)
    assert estimate["basis_trials"] == 0
    assert estimate["mean_total_tokens_per_trial"] is None
    assert estimate["estimated_total_cost_usd"] is None
    assert "no measured trial rows" in estimate["note"]


def test_cost_estimate_keeps_tokens_when_price_is_unknown():
    """An unpriced model gives a token estimate and an explicit null cost."""
    estimate = pipe.estimate_cost([_row(20000, None)], planned_trials=10)
    assert estimate["estimated_total_tokens"] == 200_000
    assert estimate["estimated_total_cost_usd"] is None
    assert "no configured price" in estimate["note"]


def test_cost_estimate_ignores_mock_rows():
    rows = [
        {"metrics_source": met.MOCK, "total_tokens": 999999, "cost_usd": None},
        _row(20000, 0.02),
    ]
    estimate = pipe.estimate_cost(rows, planned_trials=1)
    assert estimate["basis_trials"] == 1
    assert estimate["mean_total_tokens_per_trial"] == 20000.0


# ── stage discipline ──────────────────────────────────────────────────


async def test_the_pipeline_skips_measurement_when_nothing_was_selected(
    tmp_path: Path,
):
    """No candidate means no experiment — not an experiment against itself."""
    stages = await pipe.run_measurement_pipeline(
        repo_root=REPO_ROOT,
        search_config_path=REPO_ROOT / "benchmarks" / "pass-rate" / "config.json",
        holdout_config_path=None,
        selection={"selected": None, "reason": "no evolved candidate was measured"},
        baseline_label="baseline",
        output_root=tmp_path,
    )
    assert [s.status for s in stages] == ["skipped"]
    assert "no evolved candidate" in stages[0].detail


def test_stage_results_serialise_for_the_run_log(tmp_path: Path):
    stages = [
        pipe.StageResult("evolve", "ok", "3 iterations", {"best": "cand-a"}),
        pipe.StageResult("holdout-experiment", "skipped", "not configured"),
    ]
    path = pipe.write_pipeline_log(tmp_path, stages)

    import json

    logged = json.loads(path.read_text())
    assert [s["stage"] for s in logged["stages"]] == ["evolve", "holdout-experiment"]
    assert logged["stages"][0]["data"]["best"] == "cand-a"


def test_a_hand_supplied_candidate_is_labelled_as_unvalidated():
    """It must not be presented as if a validation measurement chose it."""
    from app.cli import _selection_from_agents_dir

    decision = _selection_from_agents_dir("baseline")
    assert decision["selected_row"]["validation_accuracy"] is None
    assert "no validation measurement" in decision["selection_basis"]


def test_collect_measured_rows_never_returns_mock_rows(tmp_path: Path):
    rows = pipe.collect_measured_rows(REPO_ROOT)
    assert all(r.get("metrics_source") == met.MEASURED for r in rows)
