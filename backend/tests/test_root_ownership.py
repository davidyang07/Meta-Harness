"""The process driving a run's root thread, and the status it leaves behind.

Branches record their owner in ``branches.json``; the root thread is
driven by whatever started the run (``POST /runs``, ``meta-harness loop``,
``meta-harness resume``) and records it in ``threads/<run>/owner.json``.
These tests pin:

- a claim is exclusive while its owner is alive, and released on exit;
- a dead or stale owner does not block anyone, and a record naming this
  pid but none of its claims (a recycled pid) is not a live owner;
- the outer loop writes ``status`` (running, then completed / failed /
  cancelled), so a CLI run is never reported ``unknown``, and a run left
  ``running`` by a dead process reads ``interrupted``;
- neither the API nor the CLI deletes or cancels a run another live
  process is driving.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.meta_harness import branches as br
from app.meta_harness import runs as runs_mod
from app.meta_harness.outer import drive_root_thread, run_outer_loop

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolated_branch_state():
    br.clear_branch_state()
    previous = br.get_runs_root()
    yield
    br.clear_branch_state()
    br.set_runs_root(previous)


@pytest.fixture
def other_process():
    """A live process that is not this one."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _ts(delta_s: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat()


def _write_owner(run_dir: Path, *, pid: int, heartbeat: str, claim_id: str = "x") -> None:
    runs_mod.write_json_atomic(
        br.root_owner_path(run_dir),
        {
            "run_id": run_dir.name,
            "thread_id": run_dir.name,
            "claim_id": claim_id,
            "owner_pid": pid,
            "owner_host": br._HOSTNAME,
            "claimed_at": heartbeat,
            "heartbeat_at": heartbeat,
        },
    )


# ── the claim ─────────────────────────────────────────────────────────


def test_a_claim_is_visible_exclusive_and_released(tmp_path: Path):
    run_dir = runs_mod.make_run_dir(tmp_path, "claimed", fresh=True)

    with br.claim_root_thread(run_dir) as claim:
        owner = br.root_owner(run_dir)
        assert owner is not None and owner["claim_id"] == claim.claim_id
        with pytest.raises(br.RootThreadOwned):
            br.RootThreadClaim(run_dir).acquire()

    assert br.root_owner(run_dir) is None
    assert not br.root_owner_path(run_dir).exists()


def test_another_live_process_owns_the_root(tmp_path: Path, other_process):
    run_dir = runs_mod.make_run_dir(tmp_path, "foreign", fresh=True)
    _write_owner(run_dir, pid=other_process.pid, heartbeat=_ts())

    assert br.root_owner(run_dir)["owner_pid"] == other_process.pid
    with pytest.raises(br.RootThreadOwned, match=str(other_process.pid)):
        br.RootThreadClaim(run_dir).acquire()


def test_a_dead_stale_or_recycled_owner_blocks_nobody(tmp_path: Path, other_process):
    run_dir = runs_mod.make_run_dir(tmp_path, "dead", fresh=True)

    _write_owner(
        run_dir, pid=other_process.pid, heartbeat=_ts(-(br.HEARTBEAT_STALE_S + 5))
    )
    assert br.root_owner(run_dir) is None

    # This process's pid, but not one of its claims: an earlier process
    # that happened to get the same pid.
    import os

    _write_owner(run_dir, pid=os.getpid(), heartbeat=_ts(), claim_id="not-mine")
    assert br.root_owner(run_dir) is None

    _write_owner(run_dir, pid=other_process.pid, heartbeat=_ts())
    other_process.kill()
    other_process.wait(timeout=10)
    assert br.root_owner(run_dir) is None

    # ...and the next driver simply takes over.
    with br.claim_root_thread(run_dir):
        assert br.root_owner(run_dir)["owner_pid"] == os.getpid()


def test_release_never_resurrects_a_removed_run(tmp_path: Path):
    run_dir = runs_mod.make_run_dir(tmp_path, "removed", fresh=True)
    claim = br.RootThreadClaim(run_dir).acquire()
    shutil.rmtree(run_dir)

    claim.release()

    assert not run_dir.exists()


# ── the status a run leaves behind ────────────────────────────────────


async def test_a_cli_loop_records_running_then_completed(tmp_path: Path):
    run_dir = runs_mod.make_run_dir(tmp_path, "cli-run", fresh=True)
    seen: list[str | None] = []

    real_ainvoke = None

    from app.meta_harness import outer

    original_build = outer.OuterLoopRunner.build

    def _build(self):
        graph = original_build(self)
        nonlocal real_ainvoke
        real_ainvoke = graph.ainvoke

        async def _observed(*args, **kwargs):
            seen.append(runs_mod.read_manifest(run_dir).get("status"))
            seen.append((br.root_owner(run_dir) or {}).get("owner_pid"))
            return await real_ainvoke(*args, **kwargs)

        graph.ainvoke = _observed
        return graph

    outer.OuterLoopRunner.build = _build
    try:
        final = await run_outer_loop(
            run_dir=run_dir,
            repo_root=REPO_ROOT,
            eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
            mock_proposer=True,
            mock_bench=True,
            trials=1,
            bench_workers=1,
            budget=1,
        )
    finally:
        outer.OuterLoopRunner.build = original_build
        for stub in (REPO_ROOT / "agents").glob("_mock_*.py"):
            stub.unlink(missing_ok=True)

    import os

    assert seen == ["running", os.getpid()]
    manifest = runs_mod.read_manifest(run_dir)
    assert manifest["status"] == "completed"
    assert manifest["current_iteration"] == final["iteration"]
    assert manifest["finished_at"]
    assert br.root_owner(run_dir) is None


async def test_a_failed_or_cancelled_drive_says_so(tmp_path: Path):
    run_dir = runs_mod.make_run_dir(tmp_path, "ended", fresh=True)
    runs_mod.write_manifest(run_dir, run_id="ended", status="running")

    async def _boom():
        raise RuntimeError("proposer exploded")

    with pytest.raises(RuntimeError):
        await drive_root_thread(run_dir, _boom)
    manifest = runs_mod.read_manifest(run_dir)
    assert (manifest["status"], manifest["error"]) == ("failed", "proposer exploded")
    assert br.root_owner(run_dir) is None

    async def _cancelled():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await drive_root_thread(run_dir, _cancelled)
    assert runs_mod.read_manifest(run_dir)["status"] == "cancelled"


async def test_a_drive_refused_by_another_owner_touches_nothing(
    tmp_path: Path, other_process
):
    run_dir = runs_mod.make_run_dir(tmp_path, "theirs", fresh=True)
    runs_mod.write_manifest(run_dir, run_id="theirs", status="running")
    _write_owner(run_dir, pid=other_process.pid, heartbeat=_ts())
    ran = False

    async def _drive():
        nonlocal ran
        ran = True

    with pytest.raises(br.RootThreadOwned):
        await drive_root_thread(run_dir, _drive)

    assert not ran
    assert runs_mod.read_manifest(run_dir)["status"] == "running"
    assert br.root_owner(run_dir)["owner_pid"] == other_process.pid


# ── the API: status, re-creation and cancel ───────────────────────────


@pytest.fixture
def api(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.runs import clear_run_state
    from app.main import create_app
    from app.streaming import event_registry

    monkeypatch.delenv("META_HARNESS_API_TOKEN", raising=False)
    clear_run_state()
    event_registry.clear()
    repo_root = REPO_ROOT / "backend" / ".api-test-tmp" / f"repo-{uuid.uuid4().hex[:8]}"
    repo_root.mkdir(parents=True)
    app = create_app(
        repo_root=repo_root,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        use_persistence=False,
    )
    try:
        with TestClient(app) as client:
            yield client, repo_root
    finally:
        clear_run_state()
        event_registry.clear()
        shutil.rmtree(repo_root, ignore_errors=True)


def _cli_style_run(repo_root: Path, name: str, **manifest) -> Path:
    run_dir = runs_mod.make_run_dir(repo_root, name, fresh=True)
    runs_mod.write_manifest(run_dir, run_id=name, thread_id=name, **manifest)
    return run_dir


def test_a_running_run_nobody_drives_reads_interrupted(api, other_process):
    client, repo_root = api
    run_dir = _cli_style_run(repo_root, "crashed", status="running")

    assert client.get("/runs/crashed").json()["status"] == "interrupted"
    [listed] = client.get("/runs").json()["runs"]
    assert listed["status"] == "interrupted"

    _write_owner(run_dir, pid=other_process.pid, heartbeat=_ts())
    assert client.get("/runs/crashed").json()["status"] == "running"


def test_a_cli_run_is_not_reported_unknown(api):
    client, repo_root = api
    _cli_style_run(repo_root, "done", status="completed")
    assert client.get("/runs/done").json()["status"] == "completed"


def test_the_api_will_not_recreate_or_cancel_a_run_driven_elsewhere(
    api, other_process
):
    client, repo_root = api
    run_dir = _cli_style_run(repo_root, "busy-cli", status="running")
    sentinel = run_dir / "sentinel.txt"
    sentinel.write_text("keep me")
    _write_owner(run_dir, pid=other_process.pid, heartbeat=_ts())

    rerun = client.post(
        "/runs",
        json={
            "run_name": "busy-cli",
            "fresh": True,
            "proposer": "mock",
            "mock_bench": True,
            "budget": 1,
            "trials": 1,
            "workers": 1,
        },
    )
    assert rerun.status_code == 409
    assert str(other_process.pid) in rerun.json()["detail"]
    assert sentinel.read_text() == "keep me"

    assert client.delete("/runs/busy-cli").status_code == 409
    assert runs_mod.read_manifest(run_dir)["status"] == "running"


# ── the CLI ───────────────────────────────────────────────────────────


@pytest.fixture
def driven_run(other_process):
    name = f"test-driven-{uuid.uuid4().hex[:8]}"
    run_dir = runs_mod.make_run_dir(REPO_ROOT, name, fresh=True)
    runs_mod.write_manifest(run_dir, run_id=name, thread_id=name, status="running")
    _write_owner(run_dir, pid=other_process.pid, heartbeat=_ts())
    sentinel = run_dir / "sentinel.txt"
    sentinel.write_text("keep me")
    try:
        yield name, sentinel
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_loop_fresh_refuses_a_run_another_process_drives(driven_run):
    name, sentinel = driven_run

    result = CliRunner().invoke(
        cli_app,
        ["loop", "--fresh", "--run-name", name, "--proposer", "mock", "--mock-bench",
         "--no-persistent", "--budget", "1"],
    )

    assert result.exit_code == 1, result.output
    assert "being driven by pid" in result.output
    assert sentinel.read_text() == "keep me"


def test_loop_without_fresh_refuses_cleanly_too(driven_run):
    name, sentinel = driven_run

    result = CliRunner().invoke(
        cli_app,
        ["loop", "--run-name", name, "--proposer", "mock", "--mock-bench",
         "--no-persistent", "--budget", "1"],
    )

    assert result.exit_code == 1, result.output
    assert "driven by another process" in result.output
    assert "Traceback" not in result.output
    manifest = json.loads((sentinel.parent / "manifest.json").read_text())
    assert manifest["status"] == "running"
