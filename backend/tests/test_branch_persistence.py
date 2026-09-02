"""Branch history must survive an API restart.

Branch metadata used to live only in the in-process ``branch_metadata``
dict, so restarting the backend erased the whole search tree even though
every LangGraph checkpoint was still in Postgres. The dashboard would
show a run with no branches.

These tests simulate a restart by clearing the in-process registries and
asserting the trajectory is still reconstructable from
``runs/<run_id>/branches.json``.

They also pin the honest distinction the docs must keep: *persisted
branch history* is durable; a *running in-process branch task* is not.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import branches as br  # noqa: E402
from app.meta_harness import runs as runs_mod  # noqa: E402
from app.meta_harness.outer import OuterLoopRunner, initial_state  # noqa: E402
from app.meta_harness.persistence import healthcheck, persistence_layer  # noqa: E402
from tests.conftest import unique_name  # noqa: E402

_PG_OK = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
    healthcheck()
)

pytestmark = pytest.mark.skipif(
    not _PG_OK,
    reason="Postgres not reachable at configured DSN; bring up via docker compose",
)


@pytest.fixture(autouse=True)
def _isolated_branch_state():
    br.clear_branch_state()
    previous = br.get_runs_root()
    yield
    br.clear_branch_state()
    br.set_runs_root(previous)


async def _root_run(runs_root: Path, name: str, saver):
    """A run with a measured baseline and one completed iteration."""
    run_dir = runs_mod.make_run_dir(runs_root.parent, name, fresh=True)
    runner = OuterLoopRunner(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=2,
        bench_workers=1,
        checkpointer=saver,
    )
    runs_mod.write_manifest(
        run_dir,
        run_id=name,
        thread_id=name,
        budget=2,
        trials=2,
        mock_proposer=True,
        mock_bench=True,
    )
    seeds = [await runner.benchmark_baseline(run_id=name)]
    graph = runner.build()
    await graph.ainvoke(
        initial_state(run_id=name, budget=2, seed_candidates=seeds),
        config={"configurable": {"thread_id": name}, "recursion_limit": 200},
    )
    return run_dir, graph


async def _fork_point(graph, thread_id: str) -> str:
    for record in await br.get_state_history(graph, thread_id=thread_id):
        if record.iteration == 1 and record.next == ("propose",):
            return record.checkpoint_id
    raise AssertionError("no forkable checkpoint")


async def test_trajectory_survives_registry_reset(tmp_path: Path):
    runs_root = tmp_path / "runs"
    br.set_runs_root(runs_root)

    async with persistence_layer() as saver:
        name = unique_name("persist-root")
        run_dir, graph = await _root_run(runs_root, name, saver)
        checkpoint_id = await _fork_point(graph, name)
        metadata, task = await br.worktree_add(
            graph,
            run_id=name,
            parent_thread_id=name,
            parent_checkpoint_id=checkpoint_id,
            mods={"proposer_prior": "durable", "budget_remaining": 1},
            name="durable-branch",
        )
        await task

    # 1. It was written to disk, not just to memory.
    persisted = json.loads((run_dir / "branches.json").read_text())
    assert persisted["run_id"] == name
    assert [b["thread_id"] for b in persisted["branches"]] == [metadata.thread_id]

    # 2. Simulate a backend restart.
    br.clear_branch_state()
    assert br.branch_metadata == {}
    assert br.branch_registry == {}

    # 3. The trajectory is still complete.
    trajectory = br.reconstruct_trajectory(name)
    assert {t["thread_id"] for t in trajectory["threads"]} == {
        name,
        metadata.thread_id,
    }
    edge = trajectory["edges"][0]
    assert edge["source"] == name
    assert edge["target"] == metadata.thread_id
    assert edge["parent_checkpoint_id"] == checkpoint_id

    # 4. Every field the API needs is present.
    branch = next(
        t for t in trajectory["threads"] if t["thread_id"] == metadata.thread_id
    )
    for key in (
        "branch_id",
        "run_id",
        "thread_id",
        "parent_thread_id",
        "parent_checkpoint_id",
        "name",
        "mods",
        "created_at",
        "started_at",
        "finished_at",
        "status",
        "error",
        "parent_candidate",
    ):
        assert key in branch, f"missing {key} in persisted branch metadata"
    assert branch["status"] == "completed"
    assert branch["mods"]["proposer_prior"] == "durable"
    assert branch["name"] == "durable-branch"

    # 5. A completed branch is not claimed to be running in this process.
    assert branch["live"] is False


async def test_running_branch_is_reported_interrupted_after_restart(tmp_path: Path):
    """We never claim an asyncio task survived the process that owned it."""
    runs_root = tmp_path / "runs"
    br.set_runs_root(runs_root)
    run_dir = runs_mod.make_run_dir(tmp_path, "persist-interrupt", fresh=True)

    running = br.BranchMetadata(
        branch_id="deadbeef",
        run_id="persist-interrupt",
        thread_id="persist-interrupt.fork.deadbeef",
        parent_thread_id="persist-interrupt",
        parent_checkpoint_id="ckpt-1",
        status="running",
        created_at="2026-01-01T00:00:00+00:00",
        started_at="2026-01-01T00:00:01+00:00",
        live=True,
    )
    br.persist_branch(running)
    br.clear_branch_state()

    # `live` is never persisted as True.
    raw = json.loads((run_dir / "branches.json").read_text())
    assert raw["branches"][0]["live"] is False

    reloaded = br.load_persisted_branches("persist-interrupt")
    assert len(reloaded) == 1
    assert reloaded[0].status == "interrupted"
    assert "checkpoint history is intact" in reloaded[0].error

    found = br.get_branch("persist-interrupt.fork.deadbeef")
    assert found is not None and found.status == "interrupted"


async def test_in_process_records_win_over_persisted_ones(tmp_path: Path):
    """A live record is fresher than disk and must not be shadowed."""
    runs_root = tmp_path / "runs"
    br.set_runs_root(runs_root)
    runs_mod.make_run_dir(tmp_path, "persist-merge", fresh=True)

    stale = br.BranchMetadata(
        branch_id="aa11",
        run_id="persist-merge",
        thread_id="persist-merge.fork.aa11",
        parent_thread_id="persist-merge",
        parent_checkpoint_id="ckpt-1",
        status="running",
        created_at="2026-01-01T00:00:00+00:00",
    )
    br.persist_branch(stale)

    fresh = br.BranchMetadata(**{**stale.to_dict(), "status": "completed"})
    br.branch_metadata[fresh.thread_id] = fresh

    listed = br.list_branches(run_id="persist-merge")
    assert [b.status for b in listed] == ["completed"]


async def test_branch_persistence_rejects_traversal_in_run_ids(tmp_path: Path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)
    br.set_runs_root(runs_root)

    evil = br.BranchMetadata(
        branch_id="bad",
        run_id="../escape",
        thread_id="../escape.fork.bad",
        parent_thread_id="../escape",
        parent_checkpoint_id=None,
        status="created",
        created_at="2026-01-01T00:00:00+00:00",
    )
    br.persist_branch(evil)  # must be a silent no-op, not a write

    assert not (runs_root.parent / "escape").exists()
    assert list(runs_root.iterdir()) == []
    assert br.load_persisted_branches("../escape") == []


async def test_persistence_is_a_no_op_when_no_runs_root_is_configured():
    br.set_runs_root(None)
    metadata = br.BranchMetadata(
        branch_id="none",
        run_id="whatever",
        thread_id="whatever.fork.none",
        parent_thread_id="whatever",
        parent_checkpoint_id=None,
        status="created",
        created_at="2026-01-01T00:00:00+00:00",
    )
    br.persist_branch(metadata)  # must not raise
    assert br.load_persisted_branches("whatever") == []


async def test_a_branch_recovered_from_disk_can_still_be_cancelled(tmp_path: Path):
    """Cancelling after a restart must not blow up on the persisted tree.

    `list_branches` and `get_branch` merge memory with disk so a restarted
    API reports the full tree, but `cancel_branch` resolved memory only.
    `DELETE /runs/<id>` iterates the merged list and cancels each branch,
    so after a restart it raised

        KeyError: unknown branch thread_id: <run>.fork.<id>

    which FastAPI turned into an HTTP 500 with a non-JSON body. Reachable
    only after a *real* restart, which is why it went unnoticed: the
    acceptance ladder's "survives a backend restart" check was killing a
    subshell and leaving the server running, so it queried the process it
    believed it had replaced.

    There is nothing to cancel for a branch this process never started.
    Marking its metadata cancelled is the whole job.
    """
    runs_root = tmp_path / "runs"
    br.set_runs_root(runs_root)

    async with persistence_layer() as saver:
        name = unique_name("cancel-after-restart")
        run_dir, graph = await _root_run(runs_root, name, saver)
        checkpoint_id = await _fork_point(graph, name)
        metadata, task = await br.worktree_add(
            graph,
            run_id=name,
            parent_thread_id=name,
            parent_checkpoint_id=checkpoint_id,
            mods={"proposer_prior": "cancel-me", "budget_remaining": 1},
            name="restart-branch",
        )
        await task

    # A fresh process: the branch exists on disk and nowhere else.
    br.clear_branch_state()
    assert br.branch_metadata == {}
    assert br.branch_registry == {}
    assert (run_dir / "branches.json").is_file()

    cancelled = await br.cancel_branch(metadata.thread_id)

    assert cancelled.thread_id == metadata.thread_id
    assert cancelled.live is False
    # The branch had already completed, so cancelling must not rewrite a
    # terminal status into "cancelled".
    assert cancelled.status == "completed"

    # And it is still cancellable through the merged view the API uses.
    assert metadata.thread_id in {b.thread_id for b in br.list_branches(run_id=name)}


async def test_cancelling_an_unknown_branch_still_raises(tmp_path: Path):
    """The 404 path in the branches API depends on this KeyError."""
    br.set_runs_root(tmp_path / "runs")
    br.clear_branch_state()
    with pytest.raises(KeyError, match="unknown branch thread_id"):
        await br.cancel_branch("no-such-run.fork.deadbeef")
