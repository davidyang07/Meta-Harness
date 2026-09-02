"""The capability-evidence document must be derived, not written.

Three properties, in order of importance:

1. **Nothing is hard-coded.** No pass rate, delta, trial count or
   target threshold lives in ``evidence.py``. There is no row of the
   form "the measurement cleared X": a document that grades a number
   against a bar is a document with an interest in the answer.
2. **Measured rows are recomputed from raw trials.** A published summary
   that no longer falls out of its own trial rows fails the row it
   supports.
3. **An absent artifact reports UNSUPPORTED.** Silence is never rounded
   up to a pass.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import evidence as ev  # noqa: E402
from app.meta_harness import experiment as exp  # noqa: E402
from app.meta_harness import metrics as met  # noqa: E402


def _rows(arm: str, spec: dict[str, list[bool]]) -> list[dict[str, Any]]:
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
                }
            )
    return out


def _publish(
    root: Path,
    *,
    experiment: str = "pass-rate",
    name: str = "pass-rate-20260101T000000Z",
    baseline_spec: dict[str, list[bool]] | None = None,
    candidate_spec: dict[str, list[bool]] | None = None,
    tamper: dict[str, Any] | None = None,
) -> Path:
    """Write a published experiment directory the way the runner does."""
    baseline_spec = baseline_spec or {"t1": [True, False, False, False]}
    candidate_spec = candidate_spec or {"t1": [True, True, True, False]}
    directory = root / "benchmarks" / "results" / name
    directory.mkdir(parents=True, exist_ok=True)

    baseline = _rows("baseline", baseline_spec)
    candidate = _rows("candidate", candidate_spec)
    task_ids = sorted(baseline_spec)
    summary = exp.summarize(
        baseline_rows=baseline,
        candidate_rows=candidate,
        task_ids=task_ids,
        baseline_label="baseline",
        candidate_label="evolved",
    )
    if tamper:
        summary = {**summary, **tamper}

    exp.write_rows(directory / "baseline-results.jsonl", baseline)
    exp.write_rows(directory / "candidate-results.jsonl", candidate)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    (directory / "config.json").write_text(
        json.dumps({"experiment": experiment, "trials_per_task": len(baseline_spec[task_ids[0]])})
    )
    (directory / "environment.json").write_text(
        json.dumps(
            {
                "tasks": [{"task_id": t} for t in task_ids],
                "model": "claude-haiku-4-5-20251001",
                "git": {"commit": "abc123"},
            }
        )
    )
    (directory / "validation.json").write_text(
        json.dumps(
            exp.check_protocol_equality(
                baseline_rows=baseline,
                candidate_rows=candidate,
                task_ids=task_ids,
                trials_per_task=len(baseline_spec[task_ids[0]]),
                baseline_model="m",
                candidate_model="m",
            )
        )
    )
    return directory


def _repo(tmp_path: Path) -> Path:
    """A minimal repo root the evidence reader can work against."""
    (tmp_path / "benchmarks" / "results").mkdir(parents=True)
    return tmp_path


# ── nothing is hard-coded ─────────────────────────────────────────────


def test_no_pass_rate_or_delta_is_hard_coded_in_the_evidence_module():
    """No rate may be written down where one could be measured."""
    source = (
        REPO_ROOT / "backend" / "app" / "meta_harness" / "evidence.py"
    ).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    suspicious = re.findall(r"\b0\.\d{2,}\b", code)
    assert suspicious == [], f"hard-coded rates in evidence.py: {suspicious}"


def test_the_evidence_module_holds_no_target_threshold():
    """No constant, and no row, grades a measurement against a bar.

    A threshold row is how a report acquires an interest in its own
    answer: once there is a bar to clear, the tasks, the trial count and
    the selection rule all get chosen in its shadow. The document states
    what was measured and how to re-derive it, and stops there.
    """
    source = (
        REPO_ROOT / "backend" / "app" / "meta_harness" / "evidence.py"
    ).read_text(encoding="utf-8")
    for banned in ("CLAIMED_IMPROVEMENT", "improvement_at_least", "percentage points"):
        assert banned not in source, f"{banned!r} is back in evidence.py"

    keys = {c.key for c in ev.build_checks(REPO_ROOT)}
    assert not any("at_least" in key or "claim" in key for key in keys), keys


def test_no_check_grades_a_measurement_against_a_threshold():
    """Every row states a fact; none of them states a verdict on a target."""
    for check in ev.build_checks(REPO_ROOT):
        text = f"{check.claim} {check.detail}".lower()
        assert ">=" not in text and "at least" not in text, check.claim
        assert "target" not in text, check.claim


# ── measured rows are recomputed ──────────────────────────────────────


def test_measured_rows_are_recomputed_from_the_raw_trials(tmp_path: Path):
    repo = _repo(tmp_path)
    _publish(repo)

    checks = {c.key: c for c in ev._measurement_checks(ev.latest_experiment(repo))}

    assert checks["baseline_pass_rate"].value == 0.25
    assert checks["evolved_pass_rate"].value == 0.75
    assert checks["absolute_improvement_pp"].value == 50.0


def test_a_summary_that_disagrees_with_its_rows_fails(tmp_path: Path):
    """A number that cannot be re-derived is not evidence."""
    repo = _repo(tmp_path)
    _publish(repo, tamper={"candidate_passes": 99, "candidate_accuracy": 0.99})

    loaded = ev.latest_experiment(repo)

    assert loaded["reproducible"] is False
    assert "candidate_passes" in loaded["mismatched_keys"]
    checks = {c.key: c for c in ev._measurement_checks(loaded)}
    assert checks["canonical_200_trials"].status == ev.FAIL
    # And the reported value is the recomputed one, not the tampered one.
    assert checks["evolved_pass_rate"].value == 0.75


def test_the_trial_count_row_needs_a_full_protocol(tmp_path: Path):
    """Four trials is not two hundred, however green the run was."""
    repo = _repo(tmp_path)
    _publish(repo)
    checks = {c.key: c for c in ev._measurement_checks(ev.latest_experiment(repo))}
    assert checks["canonical_200_trials"].status == ev.FAIL
    assert checks["canonical_200_trials"].value == 8


def test_a_full_protocol_passes_the_trial_count_row(tmp_path: Path):
    repo = _repo(tmp_path)
    _publish(
        repo,
        baseline_spec={f"t{i}": [True] * 20 for i in range(1, 6)},
        candidate_spec={f"t{i}": [True] * 20 for i in range(1, 6)},
    )
    checks = {c.key: c for c in ev._measurement_checks(ev.latest_experiment(repo))}
    assert checks["canonical_200_trials"].value == 200
    assert checks["canonical_200_trials"].status == ev.PASS


def test_a_small_delta_is_reported_as_the_number_it_is(tmp_path: Path):
    """A modest improvement is a modest improvement, not a failure."""
    repo = _repo(tmp_path)
    _publish(
        repo,
        baseline_spec={"t1": [True] * 10 + [False] * 10},
        candidate_spec={"t1": [True] * 11 + [False] * 9},
    )
    checks = {c.key: c for c in ev._measurement_checks(ev.latest_experiment(repo))}
    delta = checks["absolute_improvement_pp"]

    assert delta.value == 5.0
    assert delta.status == ev.PASS
    assert "measured +5.0 pp" in delta.detail


def test_a_regression_is_reported_rather_than_suppressed(tmp_path: Path):
    """A candidate that does worse publishes a negative delta."""
    repo = _repo(tmp_path)
    _publish(
        repo,
        baseline_spec={"t1": [True] * 15 + [False] * 5},
        candidate_spec={"t1": [True] * 10 + [False] * 10},
    )
    checks = {c.key: c for c in ev._measurement_checks(ev.latest_experiment(repo))}
    delta = checks["absolute_improvement_pp"]

    assert delta.value == -25.0
    assert delta.status == ev.PASS
    assert "measured -25.0 pp" in delta.detail


def test_the_delta_row_carries_the_cluster_aware_interval(tmp_path: Path):
    """The interval that describes this design travels with the number.

    The Wald interval assumes 200 independent Bernoulli trials. The
    design has 5 tasks, so a reader given only the Wald bounds is given a
    precision the experiment cannot support.
    """
    repo = _repo(tmp_path)
    _publish(
        repo,
        baseline_spec={f"t{i}": [True] * 10 + [False] * 10 for i in range(1, 6)},
        candidate_spec={f"t{i}": [True] * 15 + [False] * 5 for i in range(1, 6)},
    )
    checks = {c.key: c for c in ev._measurement_checks(ev.latest_experiment(repo))}
    detail = checks["absolute_improvement_pp"].detail

    assert "task-cluster bootstrap interval" in detail
    assert "5 task clusters" in detail
    assert "seed" in detail


# ── absence reports UNSUPPORTED ───────────────────────────────────────


def test_no_published_experiment_reports_unsupported(tmp_path: Path):
    repo = _repo(tmp_path)
    checks = {c.key: c for c in ev._measurement_checks(None)}
    assert all(c.status == ev.UNSUPPORTED for c in checks.values())
    assert all(c.value is None for c in checks.values())
    assert "canonical-experiment" in checks["baseline_pass_rate"].detail


def test_no_holdout_experiment_reports_unsupported_not_pass():
    checks = ev._holdout_checks(None)
    assert checks[0].status == ev.UNSUPPORTED
    assert "not a substitute" in checks[0].detail


def test_holdout_is_read_from_its_own_published_experiment(tmp_path: Path):
    repo = _repo(tmp_path)
    _publish(
        repo,
        experiment="holdout",
        name="holdout-20260101T000000Z",
        baseline_spec={"task-006": [True, False]},
        candidate_spec={"task-006": [True, True]},
    )
    holdout = ev.latest_experiment(repo, experiment="holdout")
    assert holdout is not None
    assert ev._holdout_checks(holdout)[0].value == 50.0


def test_latest_experiment_selects_by_experiment_name(tmp_path: Path):
    repo = _repo(tmp_path)
    _publish(repo, experiment="pass-rate", name="pass-rate-a")
    _publish(repo, experiment="holdout", name="holdout-a")

    assert ev.latest_experiment(repo, experiment="pass-rate")["dir"].name == "pass-rate-a"
    assert ev.latest_experiment(repo, experiment="holdout")["dir"].name == "holdout-a"
    assert ev.latest_experiment(repo, experiment="nope") is None


# ── structural rows ───────────────────────────────────────────────────


def test_the_two_state_machines_are_read_from_compiled_graphs():
    nodes = ev.graph_node_sets(REPO_ROOT)
    assert nodes["inner"] == ["act", "orient", "plan", "submit", "verify"]
    assert nodes["outer"] == ["benchmark", "propose", "update_frontier", "validate"]
    assert not set(nodes["inner"]) & set(nodes["outer"])


def test_the_state_machine_row_passes_against_this_repository():
    checks = {c.key: c for c in ev.build_checks(REPO_ROOT)}
    assert checks["dual_state_machines"].status == ev.PASS


def test_trace_driven_row_checks_both_production_and_consumption():
    signals = ev.trace_driven_signals(REPO_ROOT)
    assert signals["trace_driven"] is True
    assert "score.json" in signals["trace_artifacts"]
    assert "evolution_summary" in signals["proposer_inputs"]


def test_stack_rows_read_the_declared_dependencies():
    deps = ev.declared_dependencies(REPO_ROOT)
    for name in ("langgraph", "fastapi", "pydantic", "langgraph-checkpoint-postgres"):
        assert name in deps, f"{name} is not a declared dependency"
    assert "wandb" in deps, "the optional wandb extra must stay declared"


def test_every_stack_row_passes_against_this_repository():
    checks = {c.key: c for c in ev.build_checks(REPO_ROOT)}
    for key in (
        "stack_langgraph",
        "stack_fastapi",
        "stack_docker",
        "stack_postgres",
        "stack_pydantic",
        "stack_wandb",
    ):
        assert checks[key].status == ev.PASS, f"{key}: {checks[key].detail}"


# ── the source scanner counts uses, not mentions ──────────────────────


def test_a_name_inside_a_string_or_comment_is_not_a_use():
    """The drift that broke the committed document, as a test.

    ``evidence.py`` holds every scan pattern as a string literal. While
    the scanner matched raw text it counted itself, and the document
    recorded nine LangGraph modules; once the module was skipped by name
    it recorded eight, and the two disagreed forever. Dropping literals
    and comments is the rule that makes the count mean one thing.
    """
    quotes = "'" * 3
    source = "\n".join(
        [
            "PATTERN = 'from langgraph'  # from langgraph in a comment too",
            quotes + "from langgraph in a docstring" + quotes,
            "",
        ]
    )
    assert "from langgraph" not in ev.executable_code(source)


def test_a_real_import_is_a_use():
    assert re.search(
        "from langgraph", ev.executable_code("from langgraph.graph import StateGraph\n")
    )


def test_the_scanner_special_cases_no_file_by_name():
    """No filename may be excluded; the rule has to be about content."""
    source = Path(ev.__file__).read_text(encoding="utf-8")
    assert 'path.name == "evidence.py"' not in source


def test_the_scanner_does_not_count_itself_for_a_pattern_it_only_stores():
    """``evidence.py`` names every pattern but imports none of them."""
    scanner = "backend/app/meta_harness/evidence.py"
    for pattern in ("from langgraph", "from fastapi", "from pydantic"):
        assert scanner not in ev.source_uses(REPO_ROOT, pattern), pattern


def test_the_scanner_still_finds_the_modules_that_really_use_langgraph():
    hits = ev.source_uses(REPO_ROOT, "from langgraph")
    assert "backend/app/meta_harness/inner.py" in hits
    assert "backend/app/meta_harness/outer.py" in hits
    # Derived, never asserted against a literal count: the number the
    # document carries is whatever the repository currently is.
    expected = [
        path
        for path in sorted((REPO_ROOT / "backend" / "app").rglob("*.py"))
        if re.search(
            "from langgraph", ev.executable_code(path.read_text(encoding="utf-8"))
        )
    ]
    assert len(hits) == len(expected)


def test_an_unparseable_module_is_scanned_rather_than_silently_skipped():
    """A tokenise failure must not turn into 'this file uses nothing'."""
    assert "from langgraph" in ev.executable_code("from langgraph.graph import (\n")


# ── rendering and the CI check ────────────────────────────────────────


def test_the_rendered_document_is_stable_apart_from_the_header():
    """CI compares two generations, so only the timestamp may move."""
    first = ev.render_markdown(ev.build_checks(REPO_ROOT), repo_root=REPO_ROOT)
    second = ev.render_markdown(ev.build_checks(REPO_ROOT), repo_root=REPO_ROOT)
    assert ev.comparable(first) == ev.comparable(second)


def test_comparable_strips_the_generated_at_and_commit_lines():
    a = "# t\n<!-- generated-at: 2026-01-01 -->\n<!-- commit: aaa -->\nbody"
    b = "# t\n<!-- generated-at: 2027-02-02 -->\n<!-- commit: bbb -->\nbody"
    assert ev.comparable(a) == ev.comparable(b)


def test_an_edited_document_is_detected_as_disagreeing():
    derived = ev.render_markdown(ev.build_checks(REPO_ROOT), repo_root=REPO_ROOT)
    edited = derived.replace("UNSUPPORTED", "PASS")
    assert ev.comparable(edited) != ev.comparable(derived)


def test_the_document_reports_every_claim_and_a_verdict_count():
    report = ev.build_report(REPO_ROOT)
    markdown = report["markdown"]
    for check in report["checks"]:
        assert check["claim"] in markdown
    total = sum(report["counts"].values())
    assert total == len(report["checks"])
    assert f"{report['counts']['PASS']} PASS" in markdown


def test_the_headline_never_invents_a_number_without_an_experiment(tmp_path: Path):
    repo = _repo(tmp_path)
    markdown = ev.render_markdown([], repo_root=repo)
    assert "No measured result" in markdown


@pytest.mark.parametrize("status", [ev.PASS, ev.FAIL, ev.UNSUPPORTED])
def test_every_status_renders_in_bold_in_the_table(status: str):
    check = ev.Check(key="k", claim="A claim", status=status, value=1, detail="d")
    markdown = ev.render_markdown([check], repo_root=REPO_ROOT)
    assert f"| A claim | **{status}** |" in markdown


# ── artifact-backed rows ──────────────────────────────────────────────


def test_a_missing_replay_artifact_points_at_the_command_that_makes_one():
    check = ev._replay_check(Path("/nonexistent-repo"))
    assert check.status == ev.UNSUPPORTED
    assert "verify-replay" in check.detail
    assert check.evidence == ["backend/tests/test_exact_replay.py"]


def test_a_replay_report_with_only_whole_run_replays_is_not_a_pass(tmp_path: Path):
    """'From any checkpoint' is half the claim; a partial report says so."""
    (tmp_path / "docs" / "evidence").mkdir(parents=True)
    (tmp_path / "docs" / "evidence" / "replay-verification.json").write_text(
        json.dumps(
            {
                "all_verified": True,
                "model_calls_issued": 0,
                "replays": 2,
                "replays_from_checkpoint": 0,
                "skipped": [{"reason": "Postgres unreachable"}],
            }
        )
    )
    check = ev._replay_check(tmp_path)
    assert check.status == ev.UNSUPPORTED
    assert "no replay started from a stored checkpoint" in check.detail


def test_a_fully_verified_replay_report_passes(tmp_path: Path):
    (tmp_path / "docs" / "evidence").mkdir(parents=True)
    (tmp_path / "docs" / "evidence" / "replay-verification.json").write_text(
        json.dumps(
            {
                "all_verified": True,
                "model_calls_issued": 0,
                "replays": 4,
                "replays_from_checkpoint": 2,
                "skipped": [],
            }
        )
    )
    check = ev._replay_check(tmp_path)
    assert check.status == ev.PASS
    assert check.value["from_checkpoint"] == 2


def test_a_replay_that_issued_a_model_call_fails(tmp_path: Path):
    (tmp_path / "docs" / "evidence").mkdir(parents=True)
    (tmp_path / "docs" / "evidence" / "replay-verification.json").write_text(
        json.dumps(
            {
                "all_verified": True,
                "model_calls_issued": 3,
                "replays": 2,
                "replays_from_checkpoint": 1,
            }
        )
    )
    check = ev._replay_check(tmp_path)
    assert check.status == ev.FAIL
    assert "must never" in check.detail


def test_a_mutated_checkpoint_fails_the_version_graph_row():
    check = ev._version_graph_check(
        {"checkpoint_count": 9, "branch_count": 1, "immutable": False}
    )
    assert check.status == ev.FAIL
    assert "no longer hashes" in check.detail


def test_a_version_graph_with_no_fork_does_not_prove_branching():
    check = ev._branching_check(
        {"branches": [{"thread_id": "root", "parent_checkpoint_id": None}]}
    )
    assert check.status == ev.FAIL
    assert check.value == 0


def test_a_version_graph_with_a_fork_proves_branching():
    check = ev._branching_check(
        {"branches": [{"thread_id": "root.fork.a", "parent_checkpoint_id": "1f1a"}]}
    )
    assert check.status == ev.PASS
    assert check.value == 1


def test_a_missing_version_graph_names_the_tests_that_cover_it():
    check = ev._branching_check(None)
    assert check.status == ev.UNSUPPORTED
    assert "backend/tests/test_versioning.py" in check.evidence
