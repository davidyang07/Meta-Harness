"""The run event history is durable, thread-scoped and idempotent.

It used to live only in the API process's memory: a backend restart
emptied every run's decision log, and events emitted by a
``meta-harness loop`` in another process were never seen at all. These
tests pin:

- every event is appended to its own thread's ``events.jsonl`` (never a
  run-level file two branches would share);
- a new registry (a restarted process) replays that history in order,
  including what another process wrote, and honours ``Last-Event-ID``;
- an event id is recorded once, so a node re-executed on resume does not
  duplicate its events, and the outer loop's ids ignore the clock;
- a late event never recreates a run directory that is gone.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import pytest

from app.meta_harness import runs as runs_mod
from app.meta_harness.outer import _event_id
from app.streaming import (
    EVENTS_FILENAME,
    EventRegistry,
    RunEventLog,
    channel_for_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry(runs_root: Path) -> EventRegistry:
    """A registry as one process sees it, persisting under ``runs_root``."""
    registry = EventRegistry()
    registry.set_event_log(RunEventLog(runs_root))
    return registry


def _payload(chunk: str) -> dict:
    return json.loads(chunk.split("data: ", 1)[1])


@pytest.fixture
def run(tmp_path: Path):
    runs_mod.make_run_dir(tmp_path, "durable", fresh=True)
    return tmp_path / "runs", channel_for_run("durable")


def test_events_are_written_to_their_own_threads(run):
    runs_root, channel = run
    registry = _registry(runs_root)
    fork = "durable.fork.ab12cd34"

    registry.emit(channel, "state-update", {"thread_id": "durable", "n": 1})
    registry.emit(channel, "fork-created", {"thread_id": fork, "n": 2})

    run_dir = runs_root / "durable"
    root_log = runs_mod.thread_dir_path(run_dir, "durable") / EVENTS_FILENAME
    fork_log = runs_mod.thread_dir_path(run_dir, fork) / EVENTS_FILENAME
    assert [json.loads(l)["data"]["n"] for l in root_log.read_text().splitlines()] == [1]
    assert [json.loads(l)["data"]["n"] for l in fork_log.read_text().splitlines()] == [2]
    assert not (run_dir / EVENTS_FILENAME).exists()


async def test_a_restarted_process_replays_the_history_in_order(run):
    runs_root, channel = run
    before = _registry(runs_root)
    first = before.emit(channel, "state-update", {"thread_id": "durable", "n": 1})
    before.emit(channel, "fork-created", {"thread_id": "durable.fork.x", "n": 2})
    before.emit(channel, "state-update", {"thread_id": "durable", "n": 3})

    after = _registry(runs_root)  # nothing in memory
    assert [e.data["n"] for e in after.history(channel)] == [1, 2, 3]

    stream = after.subscribe(channel, heartbeat_interval=0.05)
    replayed = [_payload(await anext(stream))["n"] for _ in range(3)]
    await stream.aclose()
    assert replayed == [1, 2, 3]

    resumed = after.subscribe(
        channel, last_event_id=first.event_id, heartbeat_interval=0.05
    )
    assert _payload(await anext(resumed))["n"] == 2
    await resumed.aclose()


async def test_events_another_process_wrote_are_replayed(run):
    runs_root, channel = run
    api = _registry(runs_root)
    api.emit(channel, "fork-created", {"thread_id": "durable.fork.x", "n": 1})
    cli = _registry(runs_root)  # a `meta-harness loop` beside the API
    cli.emit(channel, "state-update", {"thread_id": "durable", "n": 2})

    stream = api.subscribe(channel, heartbeat_interval=0.05)
    got = [_payload(await anext(stream))["n"] for _ in range(2)]
    await stream.aclose()
    assert got == [1, 2]


async def test_an_event_emitted_during_a_durable_replay_is_not_lost(run):
    runs_root, channel = run
    registry = _registry(runs_root)
    registry.emit(channel, "state-update", {"thread_id": "durable", "n": 1})

    stream = registry.subscribe(channel, heartbeat_interval=0.2)
    got = [_payload(await anext(stream))["n"]]
    registry.emit(channel, "state-update", {"thread_id": "durable", "n": 2})
    got.append(_payload(await anext(stream))["n"])
    await stream.aclose()

    assert got == [1, 2]


def test_a_repeated_event_id_is_recorded_once_even_across_a_restart(run):
    runs_root, channel = run
    payload = {"thread_id": "durable", "node": "benchmark", "iteration": 1}
    first = _registry(runs_root)
    first.emit(channel, "state-update", dict(payload), event_id="same")
    first.emit(channel, "state-update", dict(payload), event_id="same")

    # The node is re-executed by a resumed process.
    resumed = _registry(runs_root)
    resumed.emit(channel, "state-update", dict(payload), event_id="same")

    log = runs_mod.thread_dir_path(runs_root / "durable", "durable") / EVENTS_FILENAME
    assert len(log.read_text().splitlines()) == 1
    assert [e.event_id for e in resumed.history(channel)] == ["same"]


def test_outer_loop_event_ids_are_content_not_clock():
    payload = {"thread_id": "r", "node": "validate", "iteration": 2, "summary": {}}
    a = _event_id("state-update", {**payload, "ts": "2026-01-01T00:00:00+00:00"})
    b = _event_id("state-update", {**payload, "ts": "2026-06-01T00:00:00+00:00"})
    assert a == b
    assert a != _event_id("state-update", {**payload, "iteration": 3})
    assert a != _event_id("iteration-complete", payload)


def test_a_late_event_never_recreates_a_removed_run(run):
    runs_root, channel = run
    registry = _registry(runs_root)
    shutil.rmtree(runs_root / "durable")

    registry.emit(channel, "state-update", {"thread_id": "durable", "n": 1})

    assert not (runs_root / "durable").exists()


def test_a_torn_last_line_is_skipped(run):
    runs_root, channel = run
    registry = _registry(runs_root)
    registry.emit(channel, "state-update", {"thread_id": "durable", "n": 1})
    log = runs_mod.thread_dir_path(runs_root / "durable", "durable") / EVENTS_FILENAME
    with open(log, "a", encoding="utf-8") as fh:
        fh.write('{"event_type": "state-upd')

    assert [e.data["n"] for e in _registry(runs_root).history(channel)] == [1]


# ── through the API: the decision log survives a restart ──────────────


def test_a_backend_restart_keeps_the_runs_event_history(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.runs import clear_run_state
    from app.main import create_app
    from app.meta_harness.branches import clear_branch_state
    from app.streaming import event_registry

    monkeypatch.delenv("META_HARNESS_API_TOKEN", raising=False)
    repo_root = REPO_ROOT / "backend" / ".api-test-tmp" / f"repo-{uuid.uuid4().hex[:8]}"
    (repo_root / "agents").mkdir(parents=True)
    (repo_root / "agents" / "__init__.py").write_text("")
    (repo_root / "agents" / "baseline.py").write_text(
        "from app.meta_harness.harness import CodingAgentHarness\n\n"
        "class BaselineHarness(CodingAgentHarness):\n"
        "    pass\n"
    )

    def _app():
        return create_app(
            repo_root=repo_root,
            eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
            use_persistence=False,
        )

    try:
        with TestClient(_app()) as client:
            created = client.post(
                "/runs",
                json={
                    "run_name": "survivor",
                    "proposer": "mock",
                    "mock_bench": True,
                    "budget": 1,
                    "trials": 1,
                    "workers": 1,
                },
            )
            assert created.status_code == 201, created.text
            deadline = time.monotonic() + 30
            while client.get("/runs/survivor").json()["status"] != "completed":
                assert time.monotonic() < deadline
                time.sleep(0.05)
            before = [e.event_id for e in event_registry.history(channel_for_run("survivor"))]
        assert len(before) > 5

        # A new process: nothing in memory.
        clear_run_state()
        clear_branch_state()
        event_registry.clear()
        with TestClient(_app()):
            after = [e.event_id for e in event_registry.history(channel_for_run("survivor"))]
        assert after == before
    finally:
        clear_run_state()
        clear_branch_state()
        event_registry.clear()
        for module in [m for m in sys.modules if m == "agents" or m.startswith("agents.")]:
            sys.modules.pop(module, None)
        shutil.rmtree(repo_root, ignore_errors=True)
