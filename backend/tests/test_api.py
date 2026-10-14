"""FastAPI REST behavior tests (BUILD_ORDER step 10)."""

from __future__ import annotations

import shutil
import sys
import time
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.runs import clear_run_state
from app.main import create_app
from app.meta_harness.branches import clear_branch_state
from app.streaming import channel_for_run, event_registry


REPO_ROOT = Path(__file__).resolve().parents[2]


TEST_TMP_ROOT = REPO_ROOT / "backend" / ".api-test-tmp"


def _make_test_repo() -> Path:
    for module_name in list(sys.modules):
        if module_name == "agents" or module_name.startswith("agents."):
            sys.modules.pop(module_name, None)
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    repo_root = TEST_TMP_ROOT / f"repo-{uuid.uuid4().hex[:8]}"
    agents_dir = repo_root / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "__init__.py").write_text("")
    (agents_dir / "baseline.py").write_text(
        "from app.meta_harness.harness import CodingAgentHarness\n\n"
        "class BaselineHarness(CodingAgentHarness):\n"
        "    pass\n"
    )
    return repo_root


def _cleanup_repo(repo_root: Path) -> None:
    if repo_root.exists():
        shutil.rmtree(repo_root)


def _cleanup(repo_root: Path, run_name: str) -> None:
    run_dir = repo_root / "runs" / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    for path in (repo_root / "agents").glob("_mock_iter_*.py"):
        path.unlink()


def _wait_for_status(client: TestClient, run_name: str, status: str) -> dict:
    deadline = time.monotonic() + 5
    last = {}
    while time.monotonic() < deadline:
        response = client.get(f"/runs/{run_name}")
        response.raise_for_status()
        last = response.json()
        if last["status"] == status:
            return last
        time.sleep(0.05)
    raise AssertionError(f"run did not reach {status}; last={last}")


def test_run_ids_reject_path_traversal():
    clear_run_state()
    clear_branch_state()
    event_registry.clear()
    repo_root = _make_test_repo()
    outside = repo_root / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("keep")

    app = create_app(
        repo_root=repo_root,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        use_persistence=False,
    )
    try:
        with TestClient(app) as client:
            for run_name in ("..", "../outside", "nested/child", "%2E%2E", ""):
                response = client.post(
                    "/runs",
                    json={
                        "budget": 1,
                        "fresh": True,
                        "run_name": run_name,
                        "proposer": "mock",
                        "mock_bench": True,
                        "trials": 1,
                        "workers": 1,
                    },
                )
                assert response.status_code == 422

            for path in ("/runs/..", "/runs/%2E%2E", "/runs/nested%2Fchild"):
                response = client.get(path)
                assert response.status_code in {404, 405}

        assert (outside / "sentinel.txt").read_text() == "keep"
    finally:
        clear_run_state()
        clear_branch_state()
        event_registry.clear()
        _cleanup_repo(repo_root)


def test_dev_dashboard_origin_can_read_api():
    clear_run_state()
    clear_branch_state()
    event_registry.clear()
    repo_root = _make_test_repo()

    app = create_app(
        repo_root=repo_root,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        use_persistence=False,
    )
    try:
        with TestClient(app) as client:
            response = client.get(
                "/health",
                headers={"Origin": "http://localhost:3000"},
            )
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    finally:
        clear_run_state()
        clear_branch_state()
        event_registry.clear()
        _cleanup_repo(repo_root)


def test_run_checkpoint_fork_branch_memory_api_flow():
    clear_run_state()
    clear_branch_state()
    event_registry.clear()
    run_name = f"api-test-{uuid.uuid4().hex[:8]}"
    repo_root = _make_test_repo()
    _cleanup(repo_root, run_name)

    app = create_app(
        repo_root=repo_root,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        use_persistence=False,
    )
    try:
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["persistence"] == "memory"

            created = client.post(
                "/runs",
                json={
                    "domain": "coding-agent",
                    "budget": 1,
                    "model": "mock",
                    "fresh": True,
                    "run_name": run_name,
                    "proposer": "mock",
                    "mock_bench": True,
                    "trials": 1,
                    "workers": 1,
                },
            )
            assert created.status_code == 201
            assert created.headers["Location"] == f"/runs/{run_name}"
            body = created.json()
            assert body["run_id"] == run_name
            assert body["thread_id"] == run_name
            assert body["status"] == "running"

            info = _wait_for_status(client, run_name, "completed")
            assert info["current_iteration"] == 1
            assert info["frontier_val"]["iteration"] == 1

            diff = client.get(f"/runs/{run_name}/candidates/_mock_iter_1/diff")
            assert diff.status_code == 200
            assert "agents/_mock_iter_1.py" in diff.json()["diff"]

            test_output = client.get(
                f"/runs/{run_name}/candidates/_mock_iter_1/test-output"
            )
            assert test_output.status_code == 200
            assert "accuracy:" in test_output.json()["output"]

            listed = client.get("/runs")
            assert listed.status_code == 200
            assert any(item["run_id"] == run_name for item in listed.json()["runs"])

            checkpoints = client.get(f"/runs/{run_name}/checkpoints")
            assert checkpoints.status_code == 200
            checkpoint_rows = checkpoints.json()["checkpoints"]
            assert checkpoint_rows
            checkpoint_id = next(
                row["checkpoint_id"]
                for row in reversed(checkpoint_rows)
                if row["iteration"] == 0
            )

            detail = client.get(f"/runs/{run_name}/checkpoints/{checkpoint_id}")
            assert detail.status_code == 200
            assert detail.json()["checkpoint_id"] == checkpoint_id
            assert "state" in detail.json()

            forked = client.post(
                f"/runs/{run_name}/fork",
                json={
                    "parent_checkpoint_id": checkpoint_id,
                    "mods": {"proposer_prior": "api test prior"},
                },
            )
            assert forked.status_code == 202
            fork_thread_id = forked.json()["thread_id"]
            assert fork_thread_id.startswith(f"{run_name}.fork.")

            branches = client.get(f"/runs/{run_name}/branches")
            assert branches.status_code == 200
            assert any(
                branch["thread_id"] == fork_thread_id
                for branch in branches.json()["branches"]
            )

            trajectory = client.get(f"/runs/{run_name}/trajectory")
            assert trajectory.status_code == 200
            assert trajectory.json()["trajectory"]["run_id"] == run_name

            cancel = client.post(f"/runs/{run_name}/branches/{fork_thread_id}/cancel")
            assert cancel.status_code == 200
            assert cancel.json()["status"] in {"cancelled", "completed"}
            # A mock branch may finish before the cancel lands; then the
            # cancel changed nothing and must not announce that it did.
            branch_was_cancelled = cancel.json()["status"] == "cancelled"

            memory = client.get("/memory/coding-agent")
            assert memory.status_code == 200
            assert memory.json()["entries"] == []

            search = client.post(
                "/memory/coding-agent/search",
                json={"query": "schema drift retry", "limit": 5},
            )
            assert search.status_code == 200
            assert search.json()["results"] == []

            event_types = {
                event.event_type
                for event in event_registry.history(channel_for_run(run_name))
            }
            assert {
                "state-update",
                "checkpoint-written",
                "candidate-created",
                "validate-result",
                "eval-result",
                "frontier-updated",
                "iteration-complete",
                "fork-created",
            }.issubset(event_types)
            assert ("branch-cancelled" in event_types) == branch_was_cancelled
    finally:
        clear_run_state()
        clear_branch_state()
        event_registry.clear()
        # Remove stale sys.path entries and cached modules so
        # later test_outer.py imports resolve cleanly.
        stale = str(repo_root)
        sys.path[:] = [p for p in sys.path if p != stale]
        for module_name in list(sys.modules):
            if module_name == "agents" or module_name.startswith("agents."):
                sys.modules.pop(module_name, None)
        _cleanup_repo(repo_root)


# ── shared fixtures for the run-lifecycle and safety tests ────────────

import asyncio  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from app.api import runs as runs_api  # noqa: E402
from app.meta_harness import branches as br  # noqa: E402
from app.meta_harness.persistence import healthcheck  # noqa: E402

_I2_KEYS = {
    "run_id",
    "thread_id",
    "budget",
    "trials",
    "workers",
    "mock_proposer",
    "mock_bench",
    "metrics_source",
    "domain",
    "skill_path",
    "record",
    "memory",
}


def _forget_agents(repo_root: Path) -> None:
    stale = str(repo_root)
    sys.path[:] = [p for p in sys.path if p != stale]
    for module_name in list(sys.modules):
        if module_name == "agents" or module_name.startswith("agents."):
            sys.modules.pop(module_name, None)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv("META_HARNESS_API_TOKEN", raising=False)
    clear_run_state()
    clear_branch_state()
    event_registry.clear()
    repo_root = _make_test_repo()
    app = create_app(
        repo_root=repo_root,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        use_persistence=False,
    )
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(client=client, repo_root=repo_root, app=app)
    finally:
        clear_run_state()
        clear_branch_state()
        event_registry.clear()
        _forget_agents(repo_root)
        _cleanup_repo(repo_root)


def _mock_run(client: TestClient, run_name: str | None, **overrides):
    body = {
        "budget": 1,
        "fresh": True,
        "proposer": "mock",
        "mock_bench": True,
        "trials": 1,
        "workers": 1,
        **overrides,
    }
    if run_name is not None:
        body["run_name"] = run_name
    return client.post("/runs", json=body)


def _completed_run(client: TestClient, run_name: str) -> dict:
    created = _mock_run(client, run_name)
    assert created.status_code == 201, created.text
    return _wait_for_status(client, run_name, "completed")


def _wait_for_branch(client: TestClient, run_name: str, thread_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for branch in client.get(f"/runs/{run_name}/branches").json()["branches"]:
            if branch["thread_id"] == thread_id and branch["status"] not in {
                "created",
                "running",
            }:
                return branch
        time.sleep(0.05)
    raise AssertionError(f"branch {thread_id} never finished")


def _fork(client: TestClient, run_name: str, **extra) -> str:
    rows = client.get(f"/runs/{run_name}/checkpoints").json()["checkpoints"]
    checkpoint = next(r["checkpoint_id"] for r in reversed(rows) if r["iteration"] == 0)
    forked = client.post(
        f"/runs/{run_name}/fork", json={"parent_checkpoint_id": checkpoint, **extra}
    )
    assert forked.status_code == 202, forked.text
    return forked.json()["thread_id"]


@pytest.fixture
def no_execution(monkeypatch):
    """Create runs without executing them: no proposer is ever started.

    A ``claude`` binary may be on PATH; a test about request validation
    must not be able to reach it.
    """

    async def _idle(record, budget):  # noqa: ARG001
        await asyncio.sleep(3600)

    monkeypatch.setattr(runs_api, "_execute_run", _idle)


@pytest.fixture
def other_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ── finding: skill_path / domain containment ──────────────────────────


def test_skill_path_outside_skills_is_refused_without_echoing_it(api, no_execution):
    outside = api.repo_root.parent / f"outside-{uuid.uuid4().hex[:6]}"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET")
    try:
        for proposer in ("claude", "mock"):
            for skill_path in (
                str(outside / "secret.txt"),
                f"../{outside.name}/secret.txt",
                "/proc/self/environ",
                "/etc/does-not-exist",
            ):
                response = _mock_run(
                    api.client, "skill-probe", proposer=proposer, skill_path=skill_path
                )
                assert response.status_code == 400, (proposer, skill_path)
                detail = response.json()["detail"]
                assert outside.name not in detail and "/etc" not in detail
                assert "secret" not in detail

        missing = _mock_run(
            api.client,
            "skill-probe",
            proposer="claude",
            skill_path="skills/not-there/SKILL.md",
        )
        assert missing.status_code == 400
        assert missing.json()["detail"] == "skill not found"

        for domain in ("../../outside", "a/b", ".."):
            bad = _mock_run(api.client, "skill-probe", domain=domain)
            assert bad.status_code == 400, domain

        assert not (api.repo_root / "runs" / "skill-probe").exists()
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_a_repo_skill_is_recorded_repo_relative(api, no_execution):
    skill = api.repo_root / "skills" / "meta-harness-demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\n---\n")

    created = _mock_run(api.client, "skill-ok", proposer="claude", domain="demo")
    assert created.status_code == 201, created.text
    assert created.json()["skill_path"] == "skills/meta-harness-demo/SKILL.md"

    manifest = json.loads(
        (api.repo_root / "runs" / "skill-ok" / "manifest.json").read_text()
    )
    assert manifest["skill_path"] == "skills/meta-harness-demo/SKILL.md"
    assert manifest["domain"] == "demo"
    assert manifest["mock_proposer"] is False


# ── finding: one run config, recorded the same way everywhere (I2) ────


def test_the_manifest_records_the_shared_run_config(api, no_execution):
    created = _mock_run(api.client, "manifest-keys", workers=4, trials=2)
    assert created.status_code == 201

    manifest = json.loads(
        (api.repo_root / "runs" / "manifest-keys" / "manifest.json").read_text()
    )
    assert _I2_KEYS <= set(manifest)
    assert manifest["workers"] == 4 and isinstance(manifest["workers"], int)
    assert manifest["trials"] == 2
    assert manifest["domain"] == "coding-agent"
    assert manifest["skill_path"] is None  # the mock proposer reads no skill
    assert manifest["record"] is False
    assert manifest["memory"] is False  # no store attached in this app
    assert manifest["metrics_source"] == "mock"


def test_two_unnamed_runs_never_share_an_id(api, no_execution):
    first = _mock_run(api.client, None).json()["run_id"]
    second = _mock_run(api.client, None).json()["run_id"]

    assert first != second
    assert (api.repo_root / "runs" / first / "manifest.json").is_file()
    assert (api.repo_root / "runs" / second / "manifest.json").is_file()


# ── finding: GETs do not write ────────────────────────────────────────


def test_listing_runs_reports_only_runs_and_writes_nothing(api):
    stray = api.repo_root / "runs" / "notes"
    stray.mkdir(parents=True)
    (stray / "todo.txt").write_text("not a run")
    before = sorted(p.relative_to(api.repo_root) for p in stray.rglob("*"))

    listed = api.client.get("/runs")
    assert listed.status_code == 200
    assert [r["run_id"] for r in listed.json()["runs"]] == []
    assert api.client.get("/runs/notes").status_code == 404
    assert api.client.get("/runs/notes/branches").status_code == 404

    assert sorted(p.relative_to(api.repo_root) for p in stray.rglob("*")) == before


def test_the_stream_only_serves_existing_runs(api):
    assert api.client.get("/runs/nope/stream").status_code == 404
    assert api.client.get("/runs/%2E%2E/stream").status_code in {404, 405}


# ── finding: branch routes check run ownership ────────────────────────


def test_a_run_cannot_reach_another_runs_branches(api):
    _completed_run(api.client, "own-a")
    _completed_run(api.client, "own-b")
    theirs = _fork(api.client, "own-b")

    cancel = api.client.post(f"/runs/own-a/branches/{theirs}/cancel")
    assert cancel.status_code == 404
    assert "branch-cancelled" not in {
        e.event_type for e in event_registry.history(channel_for_run("own-a"))
    }

    rows = api.client.get("/runs/own-a/checkpoints").json()["checkpoints"]
    checkpoint = rows[-1]["checkpoint_id"]
    cross_fork = api.client.post(
        "/runs/own-a/fork",
        json={"parent_checkpoint_id": checkpoint, "parent_thread_id": theirs},
    )
    assert cross_fork.status_code == 404
    assert (
        api.client.get(f"/runs/own-a/checkpoints?thread_id={theirs}").status_code
        == 404
    )
    assert (
        api.client.get("/runs/own-b/checkpoints?thread_id=own-a").status_code == 404
    )

    _wait_for_branch(api.client, "own-b", theirs)
    own = api.client.get(f"/runs/own-b/checkpoints?thread_id={theirs}")
    assert own.status_code == 200
    assert own.json()["checkpoints"]
    assert {r["thread_id"] for r in own.json()["checkpoints"]} == {theirs}


def test_a_fork_cannot_change_the_runs_identity(api):
    _completed_run(api.client, "ident")
    rows = api.client.get("/runs/ident/checkpoints").json()["checkpoints"]
    checkpoint = rows[-1]["checkpoint_id"]

    for mods in ({"run_id": "someone-else"}, {"not_a_state_key": 1}):
        response = api.client.post(
            "/runs/ident/fork", json={"parent_checkpoint_id": checkpoint, "mods": mods}
        )
        assert response.status_code == 422, mods
    assert api.client.get("/runs/ident/branches").json()["branches"] == []


# ── finding: a fresh re-run starts clean ──────────────────────────────


def test_a_fresh_rerun_inherits_nothing(api):
    _completed_run(api.client, "rerun")
    old_fork = _fork(api.client, "rerun")
    _wait_for_branch(api.client, "rerun", old_fork)
    assert api.client.get("/runs/rerun/branches").json()["branches"]

    _completed_run(api.client, "rerun")

    assert api.client.get("/runs/rerun/branches").json()["branches"] == []
    threads = api.client.get("/runs/rerun/trajectory").json()["trajectory"]["threads"]
    assert [t["thread_id"] for t in threads] == ["rerun"]
    history = event_registry.history(channel_for_run("rerun"))
    assert "fork-created" not in {e.event_type for e in history}
    assert all(e.data.get("thread_id") != old_fork for e in history)
    assert (
        api.client.get(f"/runs/rerun/checkpoints?thread_id={old_fork}").status_code
        == 404
    )


def _persist_foreign_branch(repo_root: Path, run_name: str, pid: int) -> str:
    br.set_runs_root(repo_root / "runs")
    thread_id = f"{run_name}.fork.f0f0f0f0"
    br.persist_branch(
        br.BranchMetadata(
            branch_id="f0f0f0f0",
            run_id=run_name,
            thread_id=thread_id,
            parent_thread_id=run_name,
            parent_checkpoint_id="ckpt",
            status="running",
            created_at=datetime.now(timezone.utc).isoformat(),
            owner_pid=pid,
            owner_host=br._HOSTNAME,
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    return thread_id


def test_a_fresh_rerun_is_refused_while_a_branch_is_live(api, other_process):
    _completed_run(api.client, "busy")
    thread_id = _persist_foreign_branch(api.repo_root, "busy", other_process.pid)

    rerun = _mock_run(api.client, "busy")
    assert rerun.status_code == 409
    branches = api.client.get("/runs/busy/branches").json()["branches"]
    assert [(b["thread_id"], b["status"]) for b in branches] == [(thread_id, "running")]

    # ...and nobody here can claim to have cancelled it.
    assert api.client.post(f"/runs/busy/branches/{thread_id}/cancel").status_code == 409
    assert api.client.delete("/runs/busy").status_code == 409


# ── finding: cancel only what is live ─────────────────────────────────


def test_delete_leaves_a_finished_run_alone(api):
    _completed_run(api.client, "finished")

    response = api.client.delete("/runs/finished")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert api.client.get("/runs/finished").json()["status"] == "completed"


def test_delete_stops_a_live_run(api, no_execution):
    assert _mock_run(api.client, "live").status_code == 201
    record = runs_api.run_registry["live"]
    assert not record.task.done()

    response = api.client.delete("/runs/live")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert record.task.done()
    manifest = json.loads((api.repo_root / "runs" / "live" / "manifest.json").read_text())
    assert manifest["status"] == "cancelled"


# ── finding: memory namespace and limits ──────────────────────────────


class _FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int]] = []

    async def asearch(self, namespace, limit=10):
        self.calls.append((tuple(namespace), limit))
        return []


def test_the_memory_namespace_reported_is_the_one_queried(api):
    store = _FakeStore()
    api.app.state.memory_store = store
    try:
        for path in ("/memory/learned_patterns:coding-agent", "/memory/coding-agent"):
            body = api.client.get(path).json()
            assert body["implemented"] is True
            assert body["namespace"] == ["learned_patterns", "coding-agent"]
        assert [ns for ns, _ in store.calls] == [
            ("learned_patterns", "coding-agent"),
            ("learned_patterns", "coding-agent"),
        ]

        assert api.client.get("/memory/coding-agent?limit=201").status_code == 422
        too_many = api.client.post(
            "/memory/coding-agent/search", json={"query": "x", "limit": 201}
        )
        assert too_many.status_code == 422
    finally:
        api.app.state.memory_store = None


# ── Postgres: a fresh re-run drops the old checkpoint threads ─────────

_PG_OK = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
    healthcheck()
)


@pytest.mark.skipif(
    not _PG_OK,
    reason="Postgres not reachable at configured DSN; bring up via docker compose",
)
def test_a_fresh_rerun_drops_old_checkpoints_on_postgres(monkeypatch):
    monkeypatch.delenv("META_HARNESS_API_TOKEN", raising=False)
    clear_run_state()
    clear_branch_state()
    event_registry.clear()
    repo_root = _make_test_repo()
    run_name = f"pg-rerun-{uuid.uuid4().hex[:8]}"
    app = create_app(
        repo_root=repo_root,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        use_persistence=True,
    )
    try:
        with TestClient(app) as client:
            assert client.get("/health").json()["persistence"] == "postgres"
            _completed_run(client, run_name)
            first = client.get(f"/runs/{run_name}/checkpoints").json()["checkpoints"]
            fork = _fork(client, run_name)
            _wait_for_branch(client, run_name, fork)

            _completed_run(client, run_name)
            second = client.get(f"/runs/{run_name}/checkpoints").json()["checkpoints"]

            assert len(second) == len(first)
            assert not {r["checkpoint_id"] for r in first} & {
                r["checkpoint_id"] for r in second
            }
            assert (
                client.get(f"/runs/{run_name}/checkpoints?thread_id={fork}").status_code
                == 404
            )
    finally:
        clear_run_state()
        clear_branch_state()
        event_registry.clear()
        _forget_agents(repo_root)
        _cleanup_repo(repo_root)
