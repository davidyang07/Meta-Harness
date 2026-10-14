"""Branch records shared between processes, and cancels that really stop.

``runs/<run>/branches.json`` is written by the API *and* by any
``meta-harness fork`` running beside it. These tests pin:

- the read-modify-write is locked across processes, so concurrent
  writers do not lose each other's records;
- a running branch records its owner (pid, host, heartbeat), so another
  process reports it running rather than "interrupted", and refuses to
  mark it cancelled when it has no way to stop it;
- a cancel cancels *and awaits* the task, bounded, and says whether the
  status actually changed.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.meta_harness import branches as br
from app.meta_harness import runs as runs_mod


@pytest.fixture(autouse=True)
def _isolated_branch_state():
    br.clear_branch_state()
    previous = br.get_runs_root()
    yield
    br.clear_branch_state()
    br.set_runs_root(previous)


def _record(run_id: str, suffix: str, **overrides) -> br.BranchMetadata:
    fields = {
        "branch_id": suffix,
        "run_id": run_id,
        "thread_id": f"{run_id}.fork.{suffix}",
        "parent_thread_id": run_id,
        "parent_checkpoint_id": "ckpt-1",
        "status": "completed",
        "created_at": f"2026-01-01T00:00:{int(suffix, 16) % 60:02d}+00:00",
    }
    fields.update(overrides)
    return br.BranchMetadata(**fields)


def _persist_many(runs_root: str, run_id: str, worker: int, count: int) -> None:
    """Child-process body: persist ``count`` distinct branches."""
    br.set_runs_root(Path(runs_root))
    for i in range(count):
        br.persist_branch(_record(run_id, f"{worker:02x}{i:04x}"))


def test_concurrent_processes_do_not_lose_each_others_records(tmp_path: Path):
    runs_root = tmp_path / "runs"
    runs_mod.make_run_dir(tmp_path, "locked-run", fresh=True)

    ctx = multiprocessing.get_context("spawn")
    workers, per_worker = 4, 15
    procs = [
        ctx.Process(
            target=_persist_many,
            args=(str(runs_root), "locked-run", w, per_worker),
        )
        for w in range(workers)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
        assert proc.exitcode == 0

    data = json.loads((runs_root / "locked-run" / "branches.json").read_text())
    assert len(data["branches"]) == workers * per_worker


# ── ownership ─────────────────────────────────────────────────────────


@pytest.fixture
def other_process():
    """A live process that is not this one."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _now(delta_s: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat()


def _running_elsewhere(run_id: str, pid: int, *, heartbeat: str) -> br.BranchMetadata:
    return _record(
        run_id,
        "beef",
        status="running",
        owner_pid=pid,
        owner_host=br._HOSTNAME,
        heartbeat_at=heartbeat,
    )


def test_a_branch_another_live_process_drives_is_reported_running(
    tmp_path: Path, other_process
):
    br.set_runs_root(tmp_path / "runs")
    runs_mod.make_run_dir(tmp_path, "owned", fresh=True)
    br.persist_branch(_running_elsewhere("owned", other_process.pid, heartbeat=_now()))

    [loaded] = br.load_persisted_branches("owned")
    assert loaded.status == "running"
    assert br.owned_elsewhere(loaded)
    assert [b.thread_id for b in br.live_branches("owned")] == [loaded.thread_id]


def test_a_dead_owner_or_a_stale_heartbeat_means_interrupted(
    tmp_path: Path, other_process
):
    br.set_runs_root(tmp_path / "runs")
    runs_mod.make_run_dir(tmp_path, "dead", fresh=True)

    stale = br.HEARTBEAT_STALE_S + 5
    br.persist_branch(
        _running_elsewhere("dead", other_process.pid, heartbeat=_now(-stale))
    )
    [loaded] = br.load_persisted_branches("dead")
    assert loaded.status == "interrupted"

    br.persist_branch(_running_elsewhere("dead", other_process.pid, heartbeat=_now()))
    other_process.kill()
    other_process.wait(timeout=10)
    [loaded] = br.load_persisted_branches("dead")
    assert loaded.status == "interrupted"
    assert br.live_branches("dead") == []


async def test_cancel_refuses_a_branch_another_live_process_drives(
    tmp_path: Path, other_process
):
    br.set_runs_root(tmp_path / "runs")
    run_dir = runs_mod.make_run_dir(tmp_path, "foreign", fresh=True)
    record = _running_elsewhere("foreign", other_process.pid, heartbeat=_now())
    br.persist_branch(record)

    with pytest.raises(br.BranchOwnedElsewhere):
        await br.cancel_branch_outcome(record.thread_id, run_id="foreign")

    raw = json.loads((run_dir / "branches.json").read_text())
    assert raw["branches"][0]["status"] == "running"


# ── cancel outcome ────────────────────────────────────────────────────


async def test_cancelling_a_finished_branch_changes_nothing(tmp_path: Path):
    br.set_runs_root(tmp_path / "runs")
    runs_mod.make_run_dir(tmp_path, "finished", fresh=True)
    done = _record("finished", "aa", status="completed")
    br.persist_branch(done)

    metadata, changed = await br.cancel_branch_outcome(done.thread_id)

    assert changed is False
    assert metadata.status == "completed"


async def test_an_interrupted_branch_is_not_relabelled_cancelled(tmp_path: Path):
    br.set_runs_root(tmp_path / "runs")
    runs_mod.make_run_dir(tmp_path, "crashed", fresh=True)
    # Persisted "running" by a process that is gone (no owner recorded).
    br.persist_branch(_record("crashed", "ab", status="running"))

    metadata, changed = await br.cancel_branch_outcome("crashed.fork.ab")

    assert changed is False
    assert metadata.status == "interrupted"


async def test_get_branch_and_cancel_are_scoped_to_their_run(tmp_path: Path):
    br.set_runs_root(tmp_path / "runs")
    runs_mod.make_run_dir(tmp_path, "run-a", fresh=True)
    runs_mod.make_run_dir(tmp_path, "run-b", fresh=True)
    theirs = _record("run-b", "cd", status="completed")
    br.branch_metadata[theirs.thread_id] = theirs

    assert br.get_branch(theirs.thread_id, run_id="run-b") is theirs
    assert br.get_branch(theirs.thread_id, run_id="run-a") is None
    with pytest.raises(KeyError):
        await br.cancel_branch_outcome(theirs.thread_id, run_id="run-a")


class _StubbornGraph:
    """A graph whose run ignores the first cancellation until released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def aget_state_history(self, config, limit=None):
        yield SimpleNamespace(
            config={"configurable": {"thread_id": "stub", "checkpoint_id": "c1"}},
            parent_config=None,
            metadata={"source": "input"},
            values={"x": 1},
            next=("a",),
        )

    async def aupdate_state(self, config, values, as_node=None):
        return {"configurable": {**config["configurable"], "checkpoint_id": "c2"}}

    async def ainvoke(self, _input, config=None):
        while True:
            try:
                await self.release.wait()
                return {"done": True}
            except asyncio.CancelledError:
                if self.release.is_set():
                    raise
                continue  # swallow: this is the stuck task


async def test_cancel_waits_for_the_task_but_only_for_so_long(tmp_path: Path):
    br.set_runs_root(tmp_path / "runs")
    runs_mod.make_run_dir(tmp_path, "stub", fresh=True)
    graph = _StubbornGraph()
    metadata, task = await br.worktree_add(
        graph, run_id="stub", parent_thread_id="stub", parent_checkpoint_id="c1"
    )
    assert metadata.owner_pid == os.getpid()
    await asyncio.sleep(0.05)

    started = time.monotonic()
    _, changed = await br.cancel_branch_outcome(metadata.thread_id, timeout=0.2)
    assert time.monotonic() - started < 2
    # The task has not stopped, so nothing claims it was cancelled.
    assert changed is False
    assert metadata.status == "running"
    assert not task.done()

    graph.release.set()
    await asyncio.wait({task}, timeout=5)
    assert task.done()


async def test_cancel_reports_a_change_when_it_stops_the_task(tmp_path: Path):
    br.set_runs_root(tmp_path / "runs")
    run_dir = runs_mod.make_run_dir(tmp_path, "stoppable", fresh=True)
    graph = _StubbornGraph()
    graph.release.set()  # a cancellation is honoured immediately

    async def slow(_input, config=None):
        await asyncio.sleep(30)

    graph.ainvoke = slow  # type: ignore[assignment]
    metadata, task = await br.worktree_add(
        graph, run_id="stoppable", parent_thread_id="stoppable", parent_checkpoint_id="c1"
    )
    await asyncio.sleep(0.05)

    _, changed = await br.cancel_branch_outcome(metadata.thread_id)

    assert changed is True
    assert task.cancelled()
    assert metadata.status == "cancelled"
    raw = json.loads((run_dir / "branches.json").read_text())
    assert raw["branches"][0]["status"] == "cancelled"


async def test_a_fork_that_cannot_be_written_is_recorded_failed(tmp_path: Path):
    br.set_runs_root(tmp_path / "runs")
    run_dir = runs_mod.make_run_dir(tmp_path, "broken", fresh=True)
    graph = _StubbornGraph()

    async def refuse(config, values, as_node=None):
        raise ValueError("cannot write fork point")

    graph.aupdate_state = refuse  # type: ignore[assignment]
    with pytest.raises(ValueError):
        await br.worktree_add(
            graph, run_id="broken", parent_thread_id="broken", parent_checkpoint_id="c1"
        )

    raw = json.loads((run_dir / "branches.json").read_text())
    assert raw["branches"][0]["status"] == "failed"
    assert br.live_branches("broken") == []


async def test_forget_run_drops_finished_records_only(tmp_path: Path):
    br.set_runs_root(tmp_path / "runs")
    done = _record("forget", "aa")
    br.branch_metadata[done.thread_id] = done
    other = _record("keep", "bb")
    br.branch_metadata[other.thread_id] = other

    br.forget_run("forget")

    assert done.thread_id not in br.branch_metadata
    assert other.thread_id in br.branch_metadata
