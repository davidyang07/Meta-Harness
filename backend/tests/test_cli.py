"""CLI smoke tests (BUILD_ORDER step 12).

These tests cover the typer subcommands without firing real LLM calls
or requiring Postgres beyond what the rest of the suite already needs.
The deeper integration coverage (live ``inner``, real ``loop`` with
the claude CLI) is exercised by ``test_inner.py`` and the demo
dry-run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert "meta-harness" in result.output


def test_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in ("inner", "benchmark", "loop", "fork", "init", "resume", "memory", "version"):
        assert name in result.output, f"--help output missing subcommand: {name}"


def test_loop_mock_no_persistent_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    """``loop --proposer mock --mock-bench --no-persistent`` should run
    end-to-end and write the canonical artifacts."""
    # Run in the actual repo (not a temp root — agents/ + eval/tasks/
    # both live there), but use a fresh run_name under runs/.
    run_name = "test-cli-loop-" + tmp_path.name[-8:]
    result = runner.invoke(
        app,
        [
            "loop",
            "--proposer",
            "mock",
            "--mock-bench",
            "--budget",
            "1",
            "--fresh",
            "--no-persistent",
            "--run-name",
            run_name,
        ],
    )
    try:
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["iterations_completed"] == 1
        assert data["budget_remaining"] == 0
        run_dir = REPO_ROOT / "runs" / run_name
        assert (run_dir / "manifest.json").exists()
        # Execution artifacts live under the run's root thread.
        td = run_dir / "threads" / run_name
        assert (td / "frontier_val.json").exists()
        assert (td / "evolution_summary.jsonl").exists()
        assert (td / "pending_eval.json").exists()
    finally:
        run_dir = REPO_ROOT / "runs" / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        for stub in (REPO_ROOT / "agents").glob("_mock_*.py"):
            stub.unlink()


@pytest.mark.asyncio
async def test_loop_mock_cli_inside_running_event_loop(tmp_path: Path) -> None:
    run_name = "test-cli-async-" + tmp_path.name[-8:]
    result = runner.invoke(
        app,
        [
            "loop",
            "--proposer",
            "mock",
            "--mock-bench",
            "--budget",
            "1",
            "--fresh",
            "--no-persistent",
            "--run-name",
            run_name,
        ],
    )
    try:
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["iterations_completed"] == 1
    finally:
        run_dir = REPO_ROOT / "runs" / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        for stub in (REPO_ROOT / "agents").glob("_mock_*.py"):
            stub.unlink()


def test_init_scaffolds_skill_md() -> None:
    """``init test-cli-domain`` must create skills/meta-harness-test-cli-domain/SKILL.md
    with renamed frontmatter."""
    domain = "test-cli-domain-" + str(abs(hash("seed")) % 100000)
    target_dir = REPO_ROOT / "skills" / f"meta-harness-{domain}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    try:
        result = runner.invoke(app, ["init", domain])
        assert result.exit_code == 0, result.output
        skill_path = target_dir / "SKILL.md"
        assert skill_path.exists()
        first_lines = skill_path.read_text().splitlines()[:3]
        assert any(f"name: meta-harness-{domain}" in line for line in first_lines), (
            f"frontmatter name not rewritten: {first_lines}"
        )
        # Re-running without --force should error.
        result_again = runner.invoke(app, ["init", domain])
        assert result_again.exit_code != 0
        # With --force it should succeed.
        result_force = runner.invoke(app, ["init", domain, "--force"])
        assert result_force.exit_code == 0, result_force.output
    finally:
        if target_dir.exists():
            shutil.rmtree(target_dir)


def test_fork_unknown_run_errors_cleanly() -> None:
    """``fork`` on a missing run must exit non-zero with a clear error
    rather than crash."""
    result = runner.invoke(
        app,
        ["fork", "definitely-not-a-real-run-name", "--checkpoint", "x"],
    )
    assert result.exit_code != 0
    assert "run not found" in result.output


def test_fork_requires_mod_in_key_value_form() -> None:
    """``--mod`` without ``=`` must error with code 2."""
    # Use any run name; we'll fail at mod parsing before run-dir lookup
    # only if the run actually exists. Otherwise --mod check happens
    # after run-dir check. Test the path where the run dir exists:
    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    fake_run = runs_dir / "test-cli-fork-malformed-mod"
    fake_run.mkdir(exist_ok=True)
    (fake_run / "manifest.json").write_text("{}")
    try:
        result = runner.invoke(
            app,
            [
                "fork",
                fake_run.name,
                "--checkpoint",
                "x",
                "--mod",
                "no_equals_sign",
            ],
        )
        assert result.exit_code != 0
    finally:
        if fake_run.exists():
            shutil.rmtree(fake_run)


@pytest.mark.skipif(
    not (REPO_ROOT / "benchmarks" / "holdout" / "config.json").exists(),
    reason="holdout protocol not present",
)
def test_holdout_task_files_well_formed() -> None:
    """Every holdout task the protocol names has task.json + workspace/."""
    config = json.loads(
        (REPO_ROOT / "benchmarks" / "holdout" / "config.json").read_text()
    )
    assert config["tasks"], "holdout protocol lists no tasks"
    for task in config["tasks"]:
        task_dir = REPO_ROOT / "eval" / "holdout" / task
        assert (task_dir / "task.json").exists(), f"{task}/task.json missing"
        assert (task_dir / "workspace").is_dir(), f"{task}/workspace/ missing"
        spec = json.loads((task_dir / "task.json").read_text())
        assert spec["id"] == task
        assert "instruction" in spec
        assert "test_command" in spec


# ── the W&B probe may not downgrade committed evidence ────────────────


def _probe_result(*, installed: bool) -> dict:
    return {
        "checked_at": "2026-01-01T00:00:00+00:00",
        "mode": "offline",
        "wandb_installed": installed,
        "wandb_version": "0.29.0" if installed else None,
        "ok": True,
        "logged": 3 if installed else 0,
        "run_url": None,
        "detail": "probe",
    }


def test_wandb_check_refuses_to_downgrade_the_committed_artifact(monkeypatch):
    """A probe from an env without the extra must not replace the evidence.

    `docs/evidence/wandb-offline.json` records that an offline run was
    really created. Running the probe in a base checkout produces
    `logged: 0`, and writing that over the artifact would quietly swap a
    supported claim for a weaker one, with the run that supported the
    stronger one gone. This was not hypothetical -- it happened by simply
    running the command.
    """
    from app.meta_harness import tracking as trk

    artifact = REPO_ROOT / "docs" / "evidence" / "wandb-offline.json"
    before = artifact.read_bytes()

    monkeypatch.setattr(trk, "offline_probe", lambda: _probe_result(installed=False))
    result = runner.invoke(app, ["report", "wandb-check"])

    assert result.exit_code == 1, result.output
    assert "Refusing to overwrite" in result.output
    assert artifact.read_bytes() == before, "the committed artifact was modified"


def test_wandb_check_writes_the_artifact_when_wandb_really_ran(
    monkeypatch, tmp_path: Path
):
    """The guard is about provenance, not about blocking the command."""
    from app.meta_harness import tracking as trk

    monkeypatch.setattr(trk, "offline_probe", lambda: _probe_result(installed=True))
    destination = tmp_path / "wandb.json"
    result = runner.invoke(
        app, ["report", "wandb-check", "--output", str(destination)]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(destination.read_text())["logged"] == 3


def test_wandb_check_can_always_record_elsewhere(monkeypatch, tmp_path: Path):
    """--output is the documented way out, and must not be blocked."""
    from app.meta_harness import tracking as trk

    monkeypatch.setattr(trk, "offline_probe", lambda: _probe_result(installed=False))
    destination = tmp_path / "probe.json"
    result = runner.invoke(
        app, ["report", "wandb-check", "--output", str(destination)]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(destination.read_text())["wandb_installed"] is False


# ── serve refuses an unauthenticated public bind ──────────────────────


@pytest.fixture
def fake_uvicorn(monkeypatch):
    import uvicorn

    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append(k))
    return calls


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_serve_refuses_a_public_bind_without_a_token(monkeypatch, fake_uvicorn, host):
    monkeypatch.delenv("META_HARNESS_API_TOKEN", raising=False)

    result = runner.invoke(app, ["serve", "--host", host])

    assert result.exit_code == 2
    assert "refusing to serve" in result.output
    assert "META_HARNESS_API_TOKEN" in result.output
    assert fake_uvicorn == []


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_serve_allows_loopback_without_a_token(monkeypatch, fake_uvicorn, host):
    monkeypatch.delenv("META_HARNESS_API_TOKEN", raising=False)

    result = runner.invoke(app, ["serve", "--host", host])

    assert result.exit_code == 0, result.output
    assert [c["host"] for c in fake_uvicorn] == [host]


def test_serve_binds_publicly_once_a_token_is_set(monkeypatch, fake_uvicorn):
    monkeypatch.setenv("META_HARNESS_API_TOKEN", "s3cret")

    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.output
    assert [c["host"] for c in fake_uvicorn] == ["0.0.0.0"]


def test_serve_bounds_its_graceful_shutdown(monkeypatch, fake_uvicorn):
    """An open event stream must not keep a stopped server alive forever."""
    monkeypatch.delenv("META_HARNESS_API_TOKEN", raising=False)

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0, result.output
    [call] = fake_uvicorn
    assert 0 < call["timeout_graceful_shutdown"] <= 30


# ── loop validates every input before deleting anything ───────────────


@pytest.fixture
def existing_run():
    import uuid

    name = f"test-cli-keep-{uuid.uuid4().hex[:8]}"
    run_dir = REPO_ROOT / "runs" / name
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "sentinel.txt"
    sentinel.write_text("keep me")
    try:
        yield name, sentinel
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--skill", "skills/not-a-skill/SKILL.md"],
        ["--skill", "../outside/SKILL.md"],
        ["--skill", "/etc/passwd"],
        ["--domain", "../escape"],
        ["--proposer", "nonsense"],
        ["--record", "--no-persistent"],
    ],
)
def test_loop_fresh_with_a_bad_input_deletes_nothing(existing_run, bad_args):
    name, sentinel = existing_run

    result = runner.invoke(app, ["loop", "--fresh", "--run-name", name, *bad_args])

    assert result.exit_code == 2, result.output
    assert sentinel.read_text() == "keep me"


# ── fork --detach is refused instead of cancelling its own branch ─────


def test_fork_detach_is_refused_with_a_pointer_to_the_api(existing_run):
    name, sentinel = existing_run
    (sentinel.parent / "manifest.json").write_text(json.dumps({"run_id": name}))

    result = runner.invoke(app, ["fork", name, "--checkpoint", "x", "--detach"])

    assert result.exit_code == 2
    assert "POST /runs/<run>/fork" in result.output
    assert not (sentinel.parent / "branches.json").exists()


# ── rebuild and resume read the run's own config (I2) ─────────────────


@pytest.fixture
def scratch_skill():
    import uuid

    domain = f"cli-test-{uuid.uuid4().hex[:8]}"
    skill_dir = REPO_ROOT / "skills" / f"meta-harness-{domain}"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: scratch\n---\n")
    try:
        yield domain, (skill_dir / "SKILL.md").resolve()
    finally:
        shutil.rmtree(skill_dir, ignore_errors=True)


def test_rebuild_graph_uses_the_runs_own_workers_and_skill(monkeypatch, scratch_skill):
    from app import cli
    from app.meta_harness import outer

    domain, skill = scratch_skill
    seen: dict = {}

    class _Runner:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def build(self):
            return "graph"

    monkeypatch.setattr(outer, "OuterLoopRunner", _Runner)
    manifest = {"domain": domain, "workers": 7, "trials": 2, "mock_proposer": False}

    assert cli._rebuild_graph(REPO_ROOT / "runs" / "x", manifest, saver=None) == "graph"
    assert seen["bench_workers"] == 7
    assert seen["trials"] == 2
    assert seen["skill_path"] == skill


def test_resume_passes_the_recorded_config(monkeypatch, scratch_skill):
    import contextlib
    import uuid

    from app.meta_harness import outer, persistence

    domain, skill = scratch_skill
    name = f"test-cli-resume-{uuid.uuid4().hex[:8]}"
    run_dir = REPO_ROOT / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "workers": 4,
                "trials": 2,
                "mock_proposer": False,
                "domain": domain,
                "skill_path": f"skills/meta-harness-{domain}/SKILL.md",
                "record": True,
                "memory": False,
            }
        )
    )
    seen: dict = {}

    async def fake_resume(
        *,
        run_dir,
        repo_root,
        eval_tasks_dir,
        checkpointer,
        skill_path=None,
        memory_store=None,
        tracker=None,
        recording_root=None,
        bench_workers=None,
    ):
        seen.update(
            skill_path=skill_path,
            memory_store=memory_store,
            tracker=tracker,
            recording_root=recording_root,
            bench_workers=bench_workers,
        )
        return {"iteration": 1, "budget_remaining": 0, "candidates": []}

    @contextlib.asynccontextmanager
    async def fake_layer():
        yield object()

    monkeypatch.setattr(outer, "resume_outer_loop", fake_resume)
    monkeypatch.setattr(persistence, "persistence_layer", fake_layer)
    try:
        result = runner.invoke(app, ["resume", name, "--no-wandb"])

        assert result.exit_code == 0, result.output
        assert seen["skill_path"] == skill
        assert seen["bench_workers"] == 4
        assert seen["memory_store"] is None  # the run had no memory store
        assert seen["tracker"] is not None
        assert seen["recording_root"] is not None
        assert seen["recording_root"].name == "recordings"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ── evidence paths and --output (I6) ──────────────────────────────────


def test_evidence_reports_carry_no_absolute_paths():
    from app import cli

    report = {
        "recordings_root": str(REPO_ROOT / "runs" / "replay-evidence"),
        "reports": [
            {"recording": "task-1-trial-1", "recording_dir": str(REPO_ROOT / "x")},
        ],
        "nested": {"thread_dir": str(REPO_ROOT / "runs" / "r" / "threads" / "r")},
    }

    portable = cli._portable_paths(report)

    assert portable["recordings_root"] == "runs/replay-evidence"
    assert portable["reports"][0] == {"recording": "task-1-trial-1", "recording_dir": "x"}
    assert portable["nested"]["thread_dir"] == "runs/r/threads/r"


def _pg_ok() -> bool:
    import asyncio

    from app.meta_harness.persistence import healthcheck

    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        healthcheck()
    )


@pytest.mark.skipif(not _pg_ok(), reason="Postgres not reachable at configured DSN")
def test_report_version_graph_writes_where_it_is_told(tmp_path: Path) -> None:
    import uuid

    name = f"test-cli-vg-{uuid.uuid4().hex[:8]}"
    committed = REPO_ROOT / "docs" / "evidence" / "version-graph.json"
    before = committed.read_bytes() if committed.exists() else None
    destination = tmp_path / "vg.json"
    try:
        loop = runner.invoke(
            app,
            [
                "loop", "--proposer", "mock", "--mock-bench", "--budget", "1",
                "--fresh", "--run-name", name, "--trials", "1", "--workers", "1",
            ],
        )
        assert loop.exit_code == 0, loop.output

        result = runner.invoke(
            app, ["report", "version-graph", name, "--output", str(destination)]
        )
        assert result.exit_code == 0, result.output
        assert f"wrote {destination}" in result.output
        assert destination.is_file()
        assert (committed.read_bytes() if committed.exists() else None) == before

        written = json.loads(destination.read_text())
        assert written["checkpoint_count"] > 0
        for thread in written["branch_artifacts"].values():
            assert thread["thread_dir"].startswith(f"runs/{name}/threads/")
            for source in thread["candidate_sources"].values():
                assert not Path(source["path"]).is_absolute()
                assert "\\" not in source["path"]
    finally:
        shutil.rmtree(REPO_ROOT / "runs" / name, ignore_errors=True)
        for stub in (REPO_ROOT / "agents").glob("_mock_*.py"):
            stub.unlink()
