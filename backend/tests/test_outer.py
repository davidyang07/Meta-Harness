"""Outer-loop end-to-end test in mock mode (BUILD_ORDER step 5 DoD).

Verifies that ``meta-harness loop --proposer mock --mock-bench
--budget 2 --fresh`` produces, under the run's root thread directory:
- pending_eval.json (current iteration)
- frontier_val.json with dominated_by_names per candidate
- evolution_summary.jsonl with parent_candidate_name per row
- per-candidate eval-result.json + status.json

...and that the measured baseline is the root of the search tree.

LLM-free; runs in <5s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import runs as runs_mod  # noqa: E402
from app.meta_harness.outer import run_outer_loop  # noqa: E402
from app.meta_harness.runs import (  # noqa: E402
    candidate_dir,
    make_run_dir,
    make_run_path,
    read_evolution_summary,
    read_frontier,
    read_pending_eval,
    thread_dir,
)
from app.meta_harness.state import BASELINE_CANDIDATE_NAME  # noqa: E402


def test_run_and_candidate_names_reject_path_traversal(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("keep")

    for name in ("..", "../outside", "nested/child", "%2E%2E", ""):
        try:
            make_run_dir(tmp_path, name, fresh=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid run name: {name}")

    assert (outside / "sentinel.txt").read_text() == "keep"
    assert make_run_path(tmp_path, "safe-run_1").name == "safe-run_1"

    run_dir = make_run_dir(tmp_path, "safe-run", fresh=True)
    for name in ("..", "../escape", "nested/child", "%2E%2E", ""):
        try:
            candidate_dir(run_dir, "safe-run", name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid candidate name: {name}")

    assert candidate_dir(run_dir, "safe-run", "_mock_iter_1").name == "_mock_iter_1"


async def test_mock_outer_loop_produces_all_files(tmp_path: Path):
    run_dir = make_run_dir(tmp_path, "test-outer", fresh=True)
    eval_tasks_dir = REPO_ROOT / "eval" / "tasks"

    final = await run_outer_loop(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=eval_tasks_dir,
        mock_proposer=True,
        mock_bench=True,
        trials=5,
        bench_workers=1,
        budget=2,
    )
    root = "test-outer"

    # Loop completed both iterations. Candidates = measured baseline + 2.
    assert final["iteration"] == 2
    assert final["budget_remaining"] == 0
    assert len(final["candidates"]) == 3
    assert final["candidates"][0]["name"] == BASELINE_CANDIDATE_NAME

    # Required filesystem artifacts, all under the root thread.
    td = thread_dir(run_dir, root)
    assert (td / "pending_eval.json").exists()
    assert (td / "frontier_val.json").exists()
    assert (td / "evolution_summary.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
    assert read_pending_eval(run_dir, root)["iteration"] == 2

    # Frontier shape: dominated_by_names per candidate (INTERFACES.md §2.2).
    frontier = read_frontier(run_dir, root)
    assert frontier["iteration"] == 2
    assert "_pareto_names" in frontier and "_best" in frontier
    # Mock results are never labelled as measurements.
    assert frontier["metrics_source"] == "mock"
    names = {c["name"] for c in frontier["candidates"]}
    assert BASELINE_CANDIDATE_NAME in names, "baseline must sit on the frontier"
    for c in frontier["candidates"]:
        assert isinstance(c["dominated_by_names"], list)

    # Evolution summary: baseline row first, then one row per iteration.
    rows = read_evolution_summary(run_dir, root)
    assert len(rows) == 3
    assert rows[0]["candidate"] == BASELINE_CANDIDATE_NAME
    assert rows[0]["iteration"] == 0
    assert rows[0]["parent_candidate_name"] is None
    # The first proposed candidate's parent is the measured baseline.
    assert rows[1]["parent_candidate_name"] == BASELINE_CANDIDATE_NAME
    for row in rows:
        assert {"iteration", "candidate", "scores", "delta", "thread_id"} <= set(row)
        assert row["thread_id"] == root
        assert row["metrics_source"] == "mock"

    # Per-candidate artifacts.
    for c in final["candidates"]:
        cand_dir = td / "candidates" / c["name"]
        assert (cand_dir / "eval-result.json").exists()
        assert (cand_dir / "status.json").exists()


async def test_baseline_is_benchmarked_before_any_candidate(tmp_path: Path):
    """The first proposed candidate's delta is measured against the baseline.

    Without a benchmarked root the first candidate is effectively
    compared against 0 and always looks like a large improvement.
    """
    run_dir = make_run_dir(tmp_path, "test-baseline-first", fresh=True)

    final = await run_outer_loop(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=4,
        bench_workers=1,
        budget=1,
    )
    rows = read_evolution_summary(run_dir, "test-baseline-first")
    baseline_row, first_row = rows[0], rows[1]

    baseline_acc = baseline_row["scores"]["accuracy"]
    assert baseline_acc > 0.0, "baseline must be a real measurement, not 0"
    assert baseline_row["delta"] == 0.0

    expected = round(first_row["scores"]["accuracy"] - baseline_acc, 4)
    assert first_row["delta"] == expected
    # And the status record names what it was compared against.
    status = json.loads(
        (
            thread_dir(run_dir, "test-baseline-first")
            / "candidates"
            / final["candidates"][-1]["name"]
            / "status.json"
        ).read_text()
    )
    assert status["compared_against"] == BASELINE_CANDIDATE_NAME
    assert status["compared_against_accuracy"] == baseline_acc


async def test_candidate_source_is_snapshotted_per_branch(tmp_path: Path):
    """What a branch benchmarks is its own snapshot, not the shared file."""
    run_dir = make_run_dir(tmp_path, "test-snapshot", fresh=True)

    final = await run_outer_loop(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=2,
        bench_workers=1,
        budget=1,
    )
    candidate = final["candidates"][-1]
    snapshot = Path(candidate["source_path"])
    assert snapshot.is_file()
    assert snapshot.parent == thread_dir(run_dir, "test-snapshot") / "agents"
    assert candidate["source_sha256"]

    # Clobbering the authored file does not change the snapshot.
    authored = REPO_ROOT / "agents" / f"{candidate['label']}.py"
    original = snapshot.read_text()
    authored.write_text("# another branch overwrote this\n")
    assert snapshot.read_text() == original


# ── a candidate that fails validation is not a measurement ────────────


async def test_a_smoke_failed_candidate_never_becomes_a_measured_zero(
    tmp_path: Path, monkeypatch
):
    """Invariant 4: a candidate that did not load has no score at all.

    It used to be turned into accuracy 0.0, marked rejected, and put on
    later frontiers as a measured point with ``metrics_source: None``.
    """
    from app.meta_harness import candidates as cand_mod

    real_load = cand_mod.load_harness_class

    def _flaky(candidate, *, repo_root):
        if candidate.get("iteration") == 1:
            raise ImportError("SyntaxError in candidate")
        return real_load(candidate, repo_root=repo_root)

    monkeypatch.setattr(cand_mod, "load_harness_class", _flaky)
    run_dir = make_run_dir(tmp_path, "test-smoke", fresh=True)

    final = await run_outer_loop(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=2,
        bench_workers=1,
        budget=2,
    )
    broken = final["candidates"][1]
    assert broken["iteration"] == 1
    assert broken["status"] == "smoke_failed"
    assert broken["delta"] is None
    assert "SyntaxError" in broken["scores"]["error"]
    # Iteration 2's parent is still the measured baseline, not the failure.
    assert final["candidates"][2]["parent"] == BASELINE_CANDIDATE_NAME

    # Neither frontier — this iteration's nor the next one's — has it.
    frontier = read_frontier(run_dir, "test-smoke")
    assert broken["name"] not in {c["name"] for c in frontier["candidates"]}
    assert all(c["metrics_source"] == "mock" for c in frontier["candidates"])

    row = read_evolution_summary(run_dir, "test-smoke")[1]
    assert row["status"] == "smoke_failed"
    assert row["outcome"].startswith("smoke_failed: ")
    assert row["scores"]["accuracy"] is None
    assert row["delta"] is None
    status = json.loads(
        (thread_dir(run_dir, "test-smoke") / "candidates" / broken["name"] / "status.json").read_text()
    )
    assert status["status"] == "smoke_failed"
    assert status["reason"] == "smoke_failed"
    assert status["accepted"] is False


# ── the proposer handoff ──────────────────────────────────────────────


async def test_an_empty_proposal_fails_the_iteration_instead_of_rebenchmarking(
    tmp_path: Path, monkeypatch
):
    """A proposer that registers nothing must not re-benchmark the baseline."""
    from app.meta_harness import proposer as prp

    def _empty(**kwargs):
        runs_mod.write_pending_eval(
            kwargs["run_dir"], kwargs["thread_id"], {"iteration": 1, "candidates": []}
        )
        return {"iteration": 1, "candidates": []}

    monkeypatch.setattr(prp, "mock_propose", _empty)
    run_dir = make_run_dir(tmp_path, "test-empty", fresh=True)

    with pytest.raises(prp.ProposalError, match="exactly one candidate"):
        await run_outer_loop(
            run_dir=run_dir,
            repo_root=REPO_ROOT,
            eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
            mock_proposer=True,
            mock_bench=True,
            trials=2,
            bench_workers=1,
            budget=1,
        )
    rows = read_evolution_summary(run_dir, "test-empty")
    assert [r["candidate"] for r in rows] == [BASELINE_CANDIDATE_NAME]
    assert rows[0]["status"] == "evaluated"


async def test_propose_clears_the_previous_handoff_first(tmp_path: Path, monkeypatch):
    from app.meta_harness import proposer as prp

    run_dir = make_run_dir(tmp_path, "test-stale", fresh=True)
    seen: list[bool] = []
    real = prp.mock_propose

    def _spy(**kwargs):
        path = thread_dir(kwargs["run_dir"], kwargs["thread_id"]) / "pending_eval.json"
        seen.append(path.exists())
        return real(**kwargs)

    monkeypatch.setattr(prp, "mock_propose", _spy)
    await run_outer_loop(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=1,
        bench_workers=1,
        budget=2,
    )
    assert seen == [False, False], "iteration 2 must not start with iteration 1's file"


# ── the run manifest is what a resume runs on ─────────────────────────


async def test_the_manifest_records_every_setting_a_resume_needs(tmp_path: Path):
    run_dir = make_run_dir(tmp_path, "test-manifest", fresh=True)
    await run_outer_loop(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=1,
        bench_workers=7,
        budget=1,
        skill_path=REPO_ROOT / "skills" / "meta-harness-coding-agent" / "SKILL.md",
        domain="coding-agent",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["workers"] == 7
    assert manifest["domain"] == "coding-agent"
    assert manifest["skill_path"] == "skills/meta-harness-coding-agent/SKILL.md"
    assert manifest["record"] is False
    assert manifest["memory"] is False
    for key in (
        "run_id",
        "thread_id",
        "budget",
        "trials",
        "mock_proposer",
        "mock_bench",
        "metrics_source",
    ):
        assert key in manifest, key


async def test_resume_runs_on_the_manifest_not_on_defaults(tmp_path: Path, monkeypatch):
    """workers, skill_path, domain and recording all survive a resume."""
    from app.meta_harness import outer as outer_mod

    run_dir = make_run_dir(tmp_path, "test-resume", fresh=True)
    runs_mod.write_manifest(
        run_dir,
        run_id="test-resume",
        thread_id="test-resume",
        budget=3,
        trials=4,
        workers=9,
        mock_proposer=False,
        mock_bench=True,
        metrics_source="mock",
        domain="other-domain",
        skill_path="skills/meta-harness-coding-agent/SKILL.md",
        record=True,
        memory=True,
    )
    captured: dict = {}

    class _Snapshot:
        values = {"iteration": 3, "candidates": []}
        next = ()

    class _Graph:
        async def aget_state(self, config):
            return _Snapshot()

    class _Runner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def build(self):
            return _Graph()

    monkeypatch.setattr(outer_mod, "OuterLoopRunner", _Runner)
    store, tracker = object(), object()
    await outer_mod.resume_outer_loop(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        checkpointer=object(),
        memory_store=store,
        tracker=tracker,
    )
    assert captured["bench_workers"] == 9
    assert captured["trials"] == 4
    assert captured["domain"] == "other-domain"
    assert captured["skill_path"] == (
        REPO_ROOT / "skills" / "meta-harness-coding-agent" / "SKILL.md"
    )
    assert captured["memory_store"] is store
    assert captured["tracker"] is tracker
    # It recorded before the interruption, so it keeps recording after.
    assert captured["recording_root"] == outer_mod.default_recording_root(run_dir)


async def test_a_baseline_whose_every_trial_raised_stops_the_run(
    tmp_path: Path, monkeypatch
):
    """An all-error baseline measured the setup, not the harness."""
    from app.meta_harness import benchmark as bench

    def _provider_down(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(bench, "build_inner_graph", _provider_down)
    task = tmp_path / "tasks" / "task-a"
    (task / "workspace").mkdir(parents=True)
    (task / "task.json").write_text(json.dumps({"id": "task-a", "test_command": "true"}))
    run_dir = make_run_dir(tmp_path, "test-dead-baseline", fresh=True)

    with pytest.raises(RuntimeError, match="every baseline trial raised"):
        await run_outer_loop(
            run_dir=run_dir,
            repo_root=REPO_ROOT,
            eval_tasks_dir=tmp_path / "tasks",
            mock_proposer=True,
            mock_bench=False,
            trials=2,
            bench_workers=1,
            budget=1,
        )
    result = json.loads(
        (
            thread_dir(run_dir, "test-dead-baseline")
            / "candidates"
            / BASELINE_CANDIDATE_NAME
            / "eval-result.json"
        ).read_text()
    )
    assert result["errored_trials"] == 2
    assert all(row["status"] == "error" for row in result["trials"])
