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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _cleanup_stubs(final) -> None:
    """Remove mock harness files the proposer authored in repo-root agents/."""
    for c in final["candidates"]:
        stub = REPO_ROOT / "agents" / f"{c.get('label') or c['name']}.py"
        if stub.exists():
            stub.unlink()


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

    _cleanup_stubs(final)


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

    _cleanup_stubs(final)


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

    _cleanup_stubs(final)
