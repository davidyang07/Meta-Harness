"""The frozen evaluation tasks are well-formed and gradeable.

A trial is graded in a fresh copy of the task's pristine ``workspace/``
into which only the files named by ``expected_files_changed`` are copied
from the agent's workspace; tests, configs and new files the agent wrote
are ignored. That makes ``expected_files_changed`` part of the grading
contract, so every task must name the files a solution changes, those
files must exist, and the pristine workspace must genuinely fail.

The benchmark protocols under ``benchmarks/`` must list exactly the tasks
on disk, and no task may state how well anything is expected to do on it
(AGENTS.md invariant 15).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import experiment as exp  # noqa: E402

SEARCH_DIR = REPO_ROOT / "eval" / "tasks"
HOLDOUT_DIR = REPO_ROOT / "eval" / "holdout"
PROTOCOLS = {
    "pass-rate": (REPO_ROOT / "benchmarks" / "pass-rate" / "config.json", SEARCH_DIR),
    "holdout": (REPO_ROOT / "benchmarks" / "holdout" / "config.json", HOLDOUT_DIR),
}

TIERS = {"bug-fix", "implement-spec", "restructure", "robustness", "multi-file"}
TASK_ID = re.compile(r"^task-(\d{3})-[a-z0-9]+(?:-[a-z0-9]+)*$")

#: Key fragments that would make a task spec state an expected outcome.
#: ``expected_files_changed`` is the one ``expected_`` key allowed: it names
#: the files a solution touches, not how well anything should do.
OUTCOME_KEY = re.compile(
    r"pass_?rate|target|score|threshold|outcome|baseline|best_known|calibration"
    r"|^expected_(?!files_changed$)",
    re.IGNORECASE,
)

#: pytest exit codes that mean "the task's tests ran and did not pass":
#: 1 = tests failed, 2 = collection error (e.g. a function the task asks
#: for does not exist yet). Anything else - usage error, no tests
#: collected, command not found - means the task is broken, not unsolved.
TESTS_DID_NOT_PASS = {1, 2}

#: Per pristine run. A task's own tests finish in well under a second.
PRISTINE_TIMEOUT_S = 60

_HAVE_PYTEST_TIMEOUT = importlib.util.find_spec("pytest_timeout") is not None


def _bounded(fn):
    """Cap the subprocess-running test where pytest-timeout is installed.

    The cap covers the module fixture that runs every task, so it allows a
    few rounds of the per-run timeout rather than one.
    """
    return pytest.mark.timeout(PRISTINE_TIMEOUT_S * 4)(fn) if _HAVE_PYTEST_TIMEOUT else fn


def _task_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "task.json").exists())


ALL_TASKS = _task_dirs(SEARCH_DIR) + _task_dirs(HOLDOUT_DIR)


def _label(task_dir: Path) -> str:
    return f"{task_dir.parent.name}/{task_dir.name}"


def _spec(task_dir: Path) -> dict:
    return json.loads((task_dir / "task.json").read_text(encoding="utf-8"))


def test_both_task_sets_are_present():
    assert _task_dirs(SEARCH_DIR), "no search tasks under eval/tasks"
    assert _task_dirs(HOLDOUT_DIR), "no holdout tasks under eval/holdout"


@pytest.mark.parametrize("task_dir", ALL_TASKS, ids=_label)
def test_task_spec_declares_the_grading_contract(task_dir: Path):
    spec = _spec(task_dir)
    assert spec.get("id") == task_dir.name
    assert TASK_ID.match(spec["id"]), f"{spec['id']} does not follow task-NNN-name"
    assert spec.get("tier") in TIERS, f"unknown tier {spec.get('tier')!r}"
    for field in ("instruction", "test_command"):
        assert isinstance(spec.get(field), str) and spec[field].strip(), field

    expected = spec.get("expected_files_changed")
    assert isinstance(expected, list) and expected, "expected_files_changed is empty"
    assert len(set(expected)) == len(expected), "expected_files_changed has duplicates"
    for rel in expected:
        assert isinstance(rel, str) and rel
        path = PurePosixPath(rel)
        assert not path.is_absolute() and ".." not in path.parts and "\\" not in rel, rel


@pytest.mark.parametrize("task_dir", ALL_TASKS, ids=_label)
def test_expected_files_exist_and_are_not_the_contract(task_dir: Path):
    """Each graded file ships in the pristine workspace.

    Grading copies only these files out of the agent's workspace, so a
    task whose solution needs a brand-new file cannot be passed, and a
    task that lists its own tests would let a solution rewrite them.
    """
    workspace = task_dir / "workspace"
    for rel in _spec(task_dir)["expected_files_changed"]:
        assert (workspace / rel).is_file(), f"{rel} is not in the pristine workspace"
        assert not rel.startswith("tests/"), f"{rel}: the tests are the contract"
        assert rel != "pytest.ini", "pytest.ini is part of the contract"


@pytest.mark.parametrize("task_dir", ALL_TASKS, ids=_label)
def test_workspace_has_the_test_layout_its_command_runs(task_dir: Path):
    workspace = task_dir / "workspace"
    assert (workspace / "pytest.ini").is_file()
    assert (workspace / "tests" / "__init__.py").is_file()
    command = shlex.split(_spec(task_dir)["test_command"])
    assert command[0] == "pytest"
    targets = [arg for arg in command[1:] if arg.endswith(".py")]
    assert targets, "test_command names no test file"
    for target in targets:
        assert (workspace / target).is_file(), f"{target} does not exist"


@pytest.mark.parametrize("task_dir", ALL_TASKS, ids=_label)
def test_no_task_declares_an_expected_outcome(task_dir: Path):
    offenders = [key for key in _spec(task_dir) if OUTCOME_KEY.search(key)]
    assert offenders == [], f"expected-outcome fields in {task_dir.name}: {offenders}"


def test_outcome_key_pattern_catches_what_it_is_for():
    for key in ("expected_pass_rate", "target", "expected_score", "baseline_pass_rate",
                "target_pass_rate", "best_known_pass_rate", "calibration_note"):
        assert OUTCOME_KEY.search(key), key
    for key in ("id", "tier", "instruction", "test_command", "expected_files_changed"):
        assert not OUTCOME_KEY.search(key), key


def test_search_and_holdout_ids_are_disjoint():
    search = {d.name for d in _task_dirs(SEARCH_DIR)}
    holdout = {d.name for d in _task_dirs(HOLDOUT_DIR)}
    assert not (search & holdout), f"in both sets: {sorted(search & holdout)}"


def test_task_numbers_are_unique_across_both_sets():
    numbers = [TASK_ID.match(d.name).group(1) for d in ALL_TASKS if TASK_ID.match(d.name)]
    assert len(numbers) == len(ALL_TASKS)
    assert len(set(numbers)) == len(numbers), "two tasks share a number"


@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
def test_benchmark_configs_list_exactly_the_tasks_on_disk(protocol: str):
    config_path, task_root = PROTOCOLS[protocol]
    config = exp.load_config(config_path)
    assert (REPO_ROOT / config["task_set"]).resolve() == task_root.resolve()
    declared = config["tasks"]
    assert len(set(declared)) == len(declared), "a task is listed twice"
    assert sorted(declared) == [d.name for d in _task_dirs(task_root)]
    assert config["total_trials"] == len(declared) * int(config["trials_per_task"]) * 2


def _run_pristine(task_dir: Path, root: Path) -> subprocess.CompletedProcess:
    """Run the unmodified task exactly as grading would, in a scratch copy."""
    workspace = root / task_dir.parent.name / task_dir.name
    shutil.copytree(
        task_dir / "workspace",
        workspace,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        _spec(task_dir)["test_command"],
        shell=True,
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=PRISTINE_TIMEOUT_S,
    )


@pytest.fixture(scope="module")
def pristine_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Future]:
    """Every task's pristine run, started together so the module stays fast.

    Each run is a separate pytest process, and its start-up dominates; run
    one after another they cost the suite several seconds for no benefit.
    """
    root = tmp_path_factory.mktemp("pristine")
    workers = max(2, min(8, os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {_label(t): pool.submit(_run_pristine, t, root) for t in ALL_TASKS}


@_bounded
@pytest.mark.parametrize("task_dir", ALL_TASKS, ids=_label)
def test_pristine_workspace_fails_its_test_command(
    task_dir: Path, pristine_runs: dict[str, Future]
):
    proc = pristine_runs[_label(task_dir)].result()
    output = (proc.stdout + proc.stderr)[-2000:]
    assert proc.returncode != 0, f"{task_dir.name} passes with no change at all"
    assert proc.returncode in TESTS_DID_NOT_PASS, (
        f"{task_dir.name}: exit {proc.returncode} means the task is broken, "
        f"not unsolved:\n{output}"
    )
