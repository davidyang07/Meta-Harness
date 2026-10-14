"""``canonical-experiment`` / ``resume-experiment``: inputs first, no surprise deletes.

Every expensive stage is stubbed, so nothing here calls a model, opens
Postgres or runs a trial. What is checked is the order of operations and
where things are written:

- invalid input (an unknown skill domain or candidate) is rejected before
  anything on disk changes;
- an existing run is deleted only with an explicit ``--fresh``;
- ``resume-experiment`` resumes and never deletes;
- results go to ``benchmarks/results/`` and the evidence document to
  ``evidence.DOCUMENT_PATH`` — the places stage 6 reads from.
"""

from __future__ import annotations

import contextlib
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from app import cli
from app.meta_harness import evidence as ev
from app.meta_harness import outer as outer_mod
from app.meta_harness import persistence
from app.meta_harness import pipeline as pipe

runner = CliRunner()


class _Stop(Exception):
    """Raised by a stub to end the command once the behaviour is observed."""


@pytest.fixture
def existing_run():
    name = f"test-canon-{uuid.uuid4().hex[:8]}"
    run_dir = cli.REPO_ROOT / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"run_id": "%s"}' % name)
    sentinel = run_dir / "keep-me.txt"
    sentinel.write_text("evolution history")
    yield name, sentinel
    shutil.rmtree(run_dir, ignore_errors=True)


@pytest.fixture
def no_postgres(monkeypatch: pytest.MonkeyPatch):
    @contextlib.asynccontextmanager
    async def _layer():
        yield None

    monkeypatch.setattr(persistence, "persistence_layer", _layer)


def test_an_unknown_skill_is_rejected_before_the_run_is_touched(existing_run):
    name, sentinel = existing_run
    result = runner.invoke(
        cli.app,
        ["canonical-experiment", "--run-name", name, "--fresh", "--domain", "no-such-domain"],
    )
    assert result.exit_code == 2, result.output
    assert "skill not found" in result.output
    assert sentinel.read_text() == "evolution history"


def test_an_unknown_candidate_is_rejected_before_the_run_is_touched(existing_run):
    name, sentinel = existing_run
    result = runner.invoke(
        cli.app,
        ["canonical-experiment", "--run-name", name, "--fresh", "--candidate", "no_such_agent"],
    )
    assert result.exit_code == 1, result.output
    assert sentinel.read_text() == "evolution history"


def test_an_existing_run_is_refused_without_fresh(existing_run):
    name, sentinel = existing_run
    result = runner.invoke(cli.app, ["canonical-experiment", "--run-name", name])
    assert result.exit_code == 1, result.output
    assert "--fresh" in result.output and "resume-experiment" in result.output
    assert sentinel.read_text() == "evolution history"


def test_fresh_is_the_one_thing_that_deletes_a_run(
    existing_run, no_postgres, monkeypatch: pytest.MonkeyPatch
):
    name, sentinel = existing_run

    async def _evolve(**kwargs: Any) -> Any:
        raise _Stop

    monkeypatch.setattr(outer_mod, "run_outer_loop", _evolve)
    result = runner.invoke(cli.app, ["canonical-experiment", "--run-name", name, "--fresh"])
    assert isinstance(result.exception, _Stop), result.output
    assert not sentinel.exists()


def test_resume_experiment_resumes_and_never_deletes(
    existing_run, no_postgres, monkeypatch: pytest.MonkeyPatch
):
    name, sentinel = existing_run
    seen: dict[str, Any] = {}

    async def _resume(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise _Stop

    async def _fresh_start(**kwargs: Any) -> Any:
        raise AssertionError("an existing run must be resumed, not restarted")

    monkeypatch.setattr(outer_mod, "resume_outer_loop", _resume)
    monkeypatch.setattr(outer_mod, "run_outer_loop", _fresh_start)
    result = runner.invoke(cli.app, ["resume-experiment", "--run-name", name])

    assert isinstance(result.exception, _Stop), result.output
    assert seen["run_dir"].name == name
    assert sentinel.read_text() == "evolution history"


def test_results_and_evidence_land_where_stage_six_reads_them(
    existing_run, no_postgres, monkeypatch: pytest.MonkeyPatch
):
    name, _ = existing_run
    calls: dict[str, Any] = {}
    document = f"runs/{name}/CAPABILITY_EVIDENCE.md"

    async def _evolve(**kwargs: Any) -> dict[str, Any]:
        return {"iteration": 0, "candidates": [], "best_candidate": None}

    async def _measure(**kwargs: Any) -> list[Any]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr(outer_mod, "run_outer_loop", _evolve)
    monkeypatch.setattr(pipe, "run_measurement_pipeline", _measure)
    monkeypatch.setattr(cli, "_capture_version_graph", _fake_version_graph)
    monkeypatch.setattr(ev, "DOCUMENT_PATH", document)

    result = runner.invoke(
        cli.app, ["canonical-experiment", "--run-name", name, "--fresh", "--no-holdout"]
    )

    assert result.exit_code in (0, 1), result.output
    assert calls["output_root"] == cli.REPO_ROOT / "benchmarks" / "results"
    assert (cli.REPO_ROOT / document).is_file()
    assert not (cli.REPO_ROOT / "docs" / "docs").exists()


async def _fake_version_graph(saver: Any, run_dir: Path, run_name: str, **_: Any):
    return {"checkpoint_count": 0, "branch_count": 0, "immutable": True}
