"""Shared fixtures for the backend suite.

The outer loop's proposer authors candidate harnesses into the repo's
real ``agents/`` directory — that is the SKILL.md contract, and tests
exercise it for real rather than stubbing it out. Cleaning those files
up used to be each test's own job, and the loops all looked like::

    for c in final["candidates"]:
        (REPO_ROOT / "agents" / f"{c['name']}.py").unlink(missing_ok=True)

Now that ``final["candidates"]`` includes the measured ``baseline``
candidate, that pattern deletes the committed ``agents/baseline.py`` and
every later test fails with ``No module named 'agents.baseline'``.

``_clean_generated_agents`` removes exactly the files a test created and
refuses to touch anything that was present before the session started,
so no test can delete committed source.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from app.meta_harness import sandbox

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "agents"


def _agent_files() -> set[Path]:
    if not AGENTS_DIR.is_dir():
        return set()
    return {p for p in AGENTS_DIR.glob("*.py")}


@pytest.fixture(scope="session")
def committed_agent_files() -> set[Path]:
    """The ``agents/*.py`` files that existed before the suite ran."""
    return _agent_files()


@pytest.fixture(autouse=True)
def _clean_generated_agents(committed_agent_files: set[Path]):
    """Delete proposer-authored candidate files created during one test."""
    yield
    for path in _agent_files() - committed_agent_files:
        try:
            path.unlink()
        except OSError:
            pass


def unique_name(prefix: str) -> str:
    """A run/thread name unique to this invocation.

    Postgres checkpoint history is keyed by thread_id and outlives the
    test process. A fixed name accumulates checkpoints across suite runs,
    which makes any assertion about history shape flap on the second run
    against the same database.
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def workspace_outside_git():
    """A workspace directory that sits outside every git working tree.

    ``tmp_path`` is repository-local (see ``backend/conftest.py``), which
    is right for almost everything but wrong for the handful of tests
    that assert on ``git apply``'s *unconfigured* behaviour. Inside a
    working tree git always converts line endings on write — the
    attributes only choose which ending — so "leave the file's own
    endings alone" is reachable only with no repository above the
    workspace at all.

    That is also the real execution environment: ``sandbox.make_sandbox_dir``
    puts every task workspace under ``sandbox_root()``, which is the
    platform temp root and outside this checkout. Building the fixture on
    the same root keeps the test measuring what production does.
    """
    workspace = sandbox.sandbox_root() / f"meta-harness-test-{uuid.uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
