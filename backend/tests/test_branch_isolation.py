"""Concurrent-branch isolation on the *real* Meta-Harness outer graph.

``test_branches.py`` proves the LangGraph fork primitive works on a toy
graph. That is not enough: the failure this suite exists to prevent is a
filesystem race in the artifact layer, and a toy graph writes no
artifacts.

The scenario reproduced here is the one that used to corrupt a run:

    original iter 2 proposer starts
    fork     iter 2 proposer starts
    original writes pending_eval.json
    fork     overwrites pending_eval.json
    original reads  pending_eval.json   <-- gets the fork's candidate
    original benchmarks the fork's candidate

Both branches are forked from the same checkpoint, reach the same
iteration number, and run concurrently against a shared
``AsyncPostgresSaver``.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import runs as runs_mod  # noqa: E402
from app.meta_harness.branches import (  # noqa: E402
    clear_branch_state,
    get_state_history,
    reconstruct_trajectory,
    worktree_add,
)
from app.meta_harness.outer import OuterLoopRunner, initial_state  # noqa: E402
from app.meta_harness.persistence import healthcheck, persistence_layer  # noqa: E402
from app.meta_harness.state import BASELINE_CANDIDATE_NAME  # noqa: E402


_PG_OK = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
    healthcheck()
)

pytestmark = pytest.mark.skipif(
    not _PG_OK,
    reason="Postgres not reachable at configured DSN; bring up via docker compose",
)


@pytest.fixture(autouse=True)
def _clean_branch_state():
    clear_branch_state()
    yield
    clear_branch_state()


def _runner(run_dir: Path, saver) -> OuterLoopRunner:
    return OuterLoopRunner(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=2,
        bench_workers=1,
        checkpointer=saver,
    )


async def _fork_point(graph, thread_id: str) -> str:
    """Checkpoint where the next node is ``propose`` for iteration 2.

    Forking here puts both branches on the same iteration number, which
    is exactly the collision the artifact layer has to survive.
    """
    for record in await get_state_history(graph, thread_id=thread_id):
        if record.iteration == 1 and record.next == ("propose",):
            return record.checkpoint_id
    raise AssertionError("no post-iteration-1 checkpoint with a pending propose")


async def _root_run(run_dir: Path, saver, *, budget: int = 2):
    """Seed a root run: measured baseline + one completed iteration."""
    runner = _runner(run_dir, saver)
    runs_mod.write_manifest(
        run_dir,
        run_id=run_dir.name,
        thread_id=run_dir.name,
        budget=budget,
        trials=2,
        mock_proposer=True,
        mock_bench=True,
        metrics_source="mock",
    )
    seeds = [await runner.benchmark_baseline(run_id=run_dir.name)]
    graph = runner.build()
    config = {
        "configurable": {"thread_id": run_dir.name},
        "recursion_limit": 200,
    }
    # Drive exactly one iteration, then stop, leaving `propose` pending.
    state = initial_state(
        run_id=run_dir.name, budget=budget, seed_candidates=seeds
    )
    await graph.ainvoke(state, config=config)
    return runner, graph


async def _fork_two(graph, run_dir: Path, checkpoint_id: str):
    left, left_task = await worktree_add(
        graph,
        run_id=run_dir.name,
        parent_thread_id=run_dir.name,
        parent_checkpoint_id=checkpoint_id,
        mods={"proposer_prior": "branch-left", "budget_remaining": 1},
        name="left",
    )
    right, right_task = await worktree_add(
        graph,
        run_id=run_dir.name,
        parent_thread_id=run_dir.name,
        parent_checkpoint_id=checkpoint_id,
        mods={"proposer_prior": "branch-right", "budget_remaining": 1},
        name="right",
    )
    done, pending = await asyncio.wait({left_task, right_task}, timeout=90)
    assert not pending, "branches did not finish; shared saver may have deadlocked"
    for finished in done:
        await finished
    return left, right


async def test_same_iteration_branches_keep_separate_artifacts(tmp_path: Path):
    """Two branches on iteration 2 must not share any execution artifact."""
    run_dir = runs_mod.make_run_dir(tmp_path, "iso-artifacts", fresh=True)

    async with persistence_layer() as saver:
        _, graph = await _root_run(run_dir, saver, budget=2)
        checkpoint_id = await _fork_point(graph, run_dir.name)
        left, right = await _fork_two(graph, run_dir, checkpoint_id)

    assert left.status == "completed" and right.status == "completed"
    assert left.thread_id != right.thread_id

    # 1. Each branch has its OWN pending-eval handoff, naming its OWN
    #    candidate. This is the exact overwrite that used to make one
    #    branch benchmark the other branch's candidate.
    left_pending = runs_mod.read_pending_eval(run_dir, left.thread_id)
    right_pending = runs_mod.read_pending_eval(run_dir, right.thread_id)
    assert left_pending is not None and right_pending is not None
    left_name = left_pending["candidates"][0]["name"]
    right_name = right_pending["candidates"][0]["name"]
    assert left_pending["iteration"] == right_pending["iteration"] == 2
    assert left_name != right_name

    # 2. Independent frontier state.
    left_frontier = runs_mod.read_frontier(run_dir, left.thread_id)
    right_frontier = runs_mod.read_frontier(run_dir, right.thread_id)
    assert left_frontier is not None and right_frontier is not None
    assert left_frontier["thread_id"] == left.thread_id
    assert right_frontier["thread_id"] == right.thread_id

    # 3. Proposer session logs for the same iteration do not collide.
    left_session = runs_mod.proposer_session_dir(run_dir, left.thread_id, 2)
    right_session = runs_mod.proposer_session_dir(run_dir, right.thread_id, 2)
    assert left_session != right_session
    assert (left_session / "session.json").exists()
    assert (right_session / "session.json").exists()

    # 4. Candidate artifact directories and traces do not collide.
    left_dir = runs_mod.find_candidate_dir(run_dir, _qualified(left, left_name))
    right_dir = runs_mod.find_candidate_dir(run_dir, _qualified(right, right_name))
    assert left_dir is not None and right_dir is not None
    assert left_dir != right_dir
    assert (left_dir / "eval-result.json").exists()
    assert (right_dir / "eval-result.json").exists()


def _qualified(branch, label: str) -> str:
    return runs_mod.qualify_candidate_name(label, branch.thread_id, branch.run_id)


async def test_branch_candidate_source_is_never_overwritten(tmp_path: Path):
    """Each branch benchmarks the source it authored, byte for byte."""
    run_dir = runs_mod.make_run_dir(tmp_path, "iso-source", fresh=True)

    async with persistence_layer() as saver:
        _, graph = await _root_run(run_dir, saver, budget=2)
        checkpoint_id = await _fork_point(graph, run_dir.name)
        left, right = await _fork_two(graph, run_dir, checkpoint_id)

    left_label = runs_mod.read_pending_eval(run_dir, left.thread_id)["candidates"][0][
        "name"
    ]
    right_label = runs_mod.read_pending_eval(run_dir, right.thread_id)["candidates"][0][
        "name"
    ]
    left_src = runs_mod.candidate_source_path(
        run_dir, left.thread_id, _qualified(left, left_label)
    )
    right_src = runs_mod.candidate_source_path(
        run_dir, right.thread_id, _qualified(right, right_label)
    )

    assert left_src.is_file() and right_src.is_file()
    assert left_src != right_src
    # Distinct paths, and neither lives in the shared repo-root agents/.
    assert left_src.parent != right_src.parent
    assert (REPO_ROOT / "agents") not in left_src.parents


async def test_branch_evolution_lineage_stays_attributable(tmp_path: Path):
    """Rows never mix branches, and the merged view stays attributable."""
    run_dir = runs_mod.make_run_dir(tmp_path, "iso-lineage", fresh=True)

    async with persistence_layer() as saver:
        _, graph = await _root_run(run_dir, saver, budget=2)
        checkpoint_id = await _fork_point(graph, run_dir.name)
        left, right = await _fork_two(graph, run_dir, checkpoint_id)

    root_rows = runs_mod.read_evolution_summary(run_dir, run_dir.name)
    left_rows = runs_mod.read_evolution_summary(run_dir, left.thread_id)
    right_rows = runs_mod.read_evolution_summary(run_dir, right.thread_id)

    # The root branch's log is untouched by either fork.
    assert [r["candidate"] for r in root_rows][0] == BASELINE_CANDIDATE_NAME
    assert all(r["thread_id"] == run_dir.name for r in root_rows)
    assert all(r["thread_id"] == left.thread_id for r in left_rows)
    assert all(r["thread_id"] == right.thread_id for r in right_rows)

    # Each fork contributed exactly its own iteration-2 row.
    assert [r["iteration"] for r in left_rows] == [2]
    assert [r["iteration"] for r in right_rows] == [2]
    assert left_rows[0]["candidate"] != right_rows[0]["candidate"]

    # Both forks descend from the same parent candidate — the state they
    # inherited at the fork checkpoint.
    assert left_rows[0]["parent_candidate_name"] == (
        right_rows[0]["parent_candidate_name"]
    )

    merged = runs_mod.aggregate_evolution_rows(run_dir)
    threads = {r["thread_id"] for r in merged}
    assert threads == {run_dir.name, left.thread_id, right.thread_id}
    # No candidate name appears twice across the whole run.
    names = [r["candidate"] for r in merged]
    assert len(names) == len(set(names))


async def test_reconstructed_trajectory_contains_both_branches(tmp_path: Path):
    run_dir = runs_mod.make_run_dir(tmp_path, "iso-trajectory", fresh=True)

    async with persistence_layer() as saver:
        _, graph = await _root_run(run_dir, saver, budget=2)
        checkpoint_id = await _fork_point(graph, run_dir.name)
        left, right = await _fork_two(graph, run_dir, checkpoint_id)

        trajectory = reconstruct_trajectory(run_dir.name)
        assert {t["thread_id"] for t in trajectory["threads"]} == {
            run_dir.name,
            left.thread_id,
            right.thread_id,
        }
        assert len(trajectory["edges"]) == 2
        assert all(e["source"] == run_dir.name for e in trajectory["edges"])
        assert all(
            e["parent_checkpoint_id"] == checkpoint_id for e in trajectory["edges"]
        )

        # Each branch has its own Postgres checkpoint history.
        left_history = await get_state_history(graph, thread_id=left.thread_id)
        right_history = await get_state_history(graph, thread_id=right.thread_id)
        assert left_history and right_history
        assert (
            left_history[0].values_summary["proposer_prior"] == "branch-left"
        )
        assert (
            right_history[0].values_summary["proposer_prior"] == "branch-right"
        )


async def test_branches_run_concurrently_without_deadlocking_the_saver(
    tmp_path: Path,
):
    """Two real outer-loop branches share one AsyncPostgresSaver safely."""
    run_dir = runs_mod.make_run_dir(tmp_path, "iso-concurrency", fresh=True)

    async with persistence_layer() as saver:
        _, graph = await _root_run(run_dir, saver, budget=2)
        checkpoint_id = await _fork_point(graph, run_dir.name)

        started = time.monotonic()
        left, right = await _fork_two(graph, run_dir, checkpoint_id)
        elapsed = time.monotonic() - started

    assert left.status == "completed"
    assert right.status == "completed"
    assert left.finished_at is not None and right.finished_at is not None
    # Generous ceiling: this is a liveness check, not a benchmark.
    assert elapsed < 90
