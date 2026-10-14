"""The grade is computed where the agent cannot reach it.

The agent's own verify runs in the workspace it edits, so anything that
decides how pytest behaves there is agent-writable. These tests pin the
exploits that used to score 1.0 — a ``conftest.py`` that rewrites every
report to ``passed``, edited tests, deleted tests — at 0.0, and pin the
honest route at 1.0, through ``grading.grade_workspace``, through the
inner graph's ``submit``, and through ``eval/score.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import grading  # noqa: E402
from app.meta_harness import inner  # noqa: E402
from app.meta_harness.tools import execute_tool  # noqa: E402
from tests.harness_doubles import (  # noqa: E402
    ScriptedHarness,
    load_task,
    task_workspace,
)

TASK_ID = "task-001-fix-typo"
SOURCE = task_workspace(REPO_ROOT, TASK_ID)
TEST_COMMAND = "pytest tests/test_calculator.py -q -p no:cacheprovider"

#: A hookwrapper that turns every test report into a pass.
CONFTEST_EXPLOIT = (
    "import pytest\n"
    "@pytest.hookimpl(hookwrapper=True)\n"
    "def pytest_runtest_makereport(item, call):\n"
    "    out = yield\n"
    "    rep = out.get_result()\n"
    "    rep.outcome = 'passed'\n"
)

FIX = (
    "--- a/calculator.py\n"
    "+++ b/calculator.py\n"
    "@@ -4,3 +4,3 @@\n"
    " def add(a: int, b: int) -> int:\n"
    "-    return a - b\n"
    "+    return a + b\n"
    " \n"
)


@pytest.fixture
def agent_ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "agent"
    shutil.copytree(SOURCE, workspace)
    return workspace


def _grade(agent: Path, expected: list[str] | None = None, **kw: Any) -> dict[str, Any]:
    return grading.grade_workspace(
        source_workspace=SOURCE,
        agent_workspace=agent,
        test_command=kw.pop("test_command", TEST_COMMAND),
        expected_files_changed=["calculator.py"] if expected is None else expected,
        **kw,
    )


def _agent_verify(workspace: Path) -> bool:
    return inner._run_verify_subprocess(workspace, TEST_COMMAND)[0]


# ── the exploits ──────────────────────────────────────────────────────


def test_the_conftest_exploit_fools_verify_but_scores_zero(agent_ws: Path):
    assert execute_tool(
        "write_file", agent_ws, path="conftest.py", content=CONFTEST_EXPLOIT
    )["status"] == "ok"

    # The exploit is real where the agent's own verify runs.
    assert _agent_verify(agent_ws) is True
    result = _grade(agent_ws)

    assert result["passed"] is False
    assert result["out_of_plan_changes"] == ["conftest.py"]
    assert result["carried_files"] == ["calculator.py"]


def test_editing_the_tests_scores_zero(agent_ws: Path):
    (agent_ws / "tests" / "test_calculator.py").write_text(
        "def test_nothing():\n    pass\n", encoding="utf-8"
    )
    assert _agent_verify(agent_ws) is True

    result = _grade(agent_ws)

    assert result["passed"] is False
    assert result["out_of_plan_changes"] == ["tests/test_calculator.py"]


def test_deleting_the_tests_scores_zero(agent_ws: Path):
    shutil.rmtree(agent_ws / "tests")

    result = _grade(agent_ws)

    assert result["passed"] is False
    assert "tests/test_calculator.py" in result["out_of_plan_changes"]
    assert "tests/__init__.py" in result["out_of_plan_changes"]


def test_a_new_pytest_config_cannot_skip_the_tests(agent_ws: Path):
    (agent_ws / "pytest.ini").write_text(
        "[pytest]\naddopts = --co\n", encoding="utf-8"
    )
    result = _grade(agent_ws)
    assert result["passed"] is False
    assert result["out_of_plan_changes"] == ["pytest.ini"]


def test_a_symlinked_expected_file_is_not_followed(agent_ws: Path, tmp_path: Path):
    if not hasattr(os, "symlink"):
        pytest.skip("no symlinks")
    fixed = tmp_path / "elsewhere.py"
    fixed.write_text(
        (SOURCE / "calculator.py").read_text(encoding="utf-8").replace(
            "def add(a: int, b: int) -> int:\n    return a - b",
            "def add(a: int, b: int) -> int:\n    return a + b",
        ),
        encoding="utf-8",
    )
    (agent_ws / "calculator.py").unlink()
    try:
        (agent_ws / "calculator.py").symlink_to(fixed)
    except OSError:
        pytest.skip("symlinks not permitted here")

    result = _grade(agent_ws)

    assert result["passed"] is False
    assert result["carried_files"] == []


# ── the honest route ──────────────────────────────────────────────────


def test_a_correct_fix_scores_one(agent_ws: Path):
    assert execute_tool("apply_patch", agent_ws, path="calculator.py", patch=FIX)[
        "status"
    ] == "ok"

    result = _grade(agent_ws)

    assert result["passed"] is True, result["output"]
    assert result["exit_code"] == 0
    assert result["carried_files"] == ["calculator.py"]
    assert result["out_of_plan_changes"] == []


def test_a_fix_with_extra_files_still_passes_and_reports_them(agent_ws: Path):
    execute_tool("apply_patch", agent_ws, path="calculator.py", patch=FIX)
    (agent_ws / "notes.txt").write_text("scratch\n", encoding="utf-8")
    (agent_ws / "tests" / "test_calculator.py").unlink()

    result = _grade(agent_ws)

    assert result["passed"] is True, result["output"]
    assert result["out_of_plan_changes"] == ["notes.txt", "tests/test_calculator.py"]


def test_deleting_an_expected_file_deletes_it_in_the_copy(agent_ws: Path):
    (agent_ws / "calculator.py").unlink()

    result = _grade(agent_ws)

    assert result["passed"] is False
    assert result["carried_files"] == ["calculator.py"]
    assert "calculator" in result["output"]


def test_test_run_side_effects_are_not_out_of_plan(agent_ws: Path):
    execute_tool("apply_patch", agent_ws, path="calculator.py", patch=FIX)
    assert inner._run_verify_subprocess(agent_ws, "pytest tests -q")[0]
    assert any(agent_ws.rglob("__pycache__"))

    assert _grade(agent_ws)["out_of_plan_changes"] == []


# ── the grader runs under the sandbox's rules ─────────────────────────


def _mini_task(tmp_path: Path, test_body: str) -> Path:
    source = tmp_path / "mini"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_env.py").write_text(test_body, encoding="utf-8")
    return source


def test_the_grade_runs_with_a_scrubbed_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-leak")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tok-must-not-leak")
    source = _mini_task(
        tmp_path,
        "import os\n"
        "def test_env():\n"
        "    assert 'ANTHROPIC_API_KEY' not in os.environ\n"
        "    assert 'SOME_SERVICE_TOKEN' not in os.environ\n",
    )
    result = grading.grade_workspace(
        source_workspace=source,
        agent_workspace=source,
        test_command="pytest -q -p no:cacheprovider",
        expected_files_changed=[],
    )
    assert result["passed"] is True, result["output"]
    assert "must-not-leak" not in json.dumps(result)


def test_the_grade_honours_its_timeout(tmp_path: Path):
    source = _mini_task(tmp_path, "def test_x():\n    pass\n")
    result = grading.grade_workspace(
        source_workspace=source,
        agent_workspace=source,
        test_command="sleep 30",
        expected_files_changed=[],
        timeout_sec=1,
    )
    assert result["passed"] is False
    assert result["timed_out"] is True
    assert result["exit_code"] is None


def test_the_result_is_plain_json(agent_ws: Path):
    """It is recorded on the tape and must hash identically when replayed."""
    result = _grade(agent_ws)
    assert json.loads(json.dumps(result)) == result


# ── expected_files_changed ────────────────────────────────────────────


def test_expected_files_are_normalised_to_posix():
    entries = ["geometry\\__init__.py", "a.py", "a.py"]
    assert grading.normalize_expected_files(entries) == ["geometry/__init__.py", "a.py"]


@pytest.mark.parametrize(
    "bad", ["/etc/passwd", "../outside.py", "a/../../b.py", "C:\\x.py", ""]
)
def test_expected_files_must_stay_inside_the_workspace(bad: str):
    with pytest.raises(ValueError):
        grading.normalize_expected_files([bad])


def test_task_timeout_defaults_and_overrides():
    assert grading.timeout_for_task({}) == grading.DEFAULT_TEST_TIMEOUT_SEC
    assert grading.timeout_for_task({"timeout_sec": 7}) == 7
    assert grading.timeout_for_task({"timeout_sec": "nonsense"}) == (
        grading.DEFAULT_TEST_TIMEOUT_SEC
    )


# ── through the inner graph ───────────────────────────────────────────


class ConftestExploitHarness(ScriptedHarness):
    """Never touches calculator.py; plants the conftest and declares victory."""

    SCRIPT = [
        [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "write_file",
                "input": {"path": "conftest.py", "content": CONFTEST_EXPLOIT},
            }
        ],
        [{"type": "tool_use", "id": "t2", "name": "task_complete", "input": {}}],
    ]


class HonestFixHarness(ScriptedHarness):
    SCRIPT = [
        [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "apply_patch",
                "input": {"path": "calculator.py", "patch": FIX},
            }
        ],
        [{"type": "tool_use", "id": "t2", "name": "task_complete", "input": {}}],
    ]


async def _run(harness: Any, agent_ws: Path, trace_dir: Path | None = None) -> dict:
    return await inner.run_inner_loop(
        harness,
        task_dict=load_task(REPO_ROOT, TASK_ID),
        workspace=agent_ws,
        trace_dir=trace_dir,
    )


async def test_submit_scores_the_grade_not_the_agents_verify(agent_ws: Path, tmp_path):
    final = await _run(ConftestExploitHarness(), agent_ws, tmp_path / "trace")

    assert final["verify_result"]["tests_pass"] is True
    assert final["score"] == 0.0
    grade = final["grade_result"]
    assert set(grade) == {
        "passed",
        "output",
        "carried_files",
        "out_of_plan_changes",
        "agent_verify_passed",
    }
    assert grade["passed"] is False
    assert grade["agent_verify_passed"] is True
    assert grade["out_of_plan_changes"] == ["conftest.py"]
    # verify_result no longer claims what nothing measured.
    assert final["verify_result"]["out_of_plan_changes"] == ["conftest.py"]
    assert final["verify_result"]["lint_pass"] is None
    score = json.loads((tmp_path / "trace" / "score.json").read_text())
    assert score["passed"] is False and score["agent_verify_passed"] is True
    assert (tmp_path / "trace" / "grade.json").is_file()


async def test_submit_scores_an_honest_fix_one(agent_ws: Path):
    final = await _run(HonestFixHarness(), agent_ws)
    assert final["score"] == 1.0, final["grade_result"]["output"]
    assert final["grade_result"]["carried_files"] == ["calculator.py"]
    assert final["grade_result"]["out_of_plan_changes"] == []


async def test_a_trial_without_a_source_workspace_is_refused(agent_ws: Path):
    task = load_task(REPO_ROOT, TASK_ID)
    del task["_source_workspace"]
    harness = HonestFixHarness()
    with pytest.raises(ValueError, match="_source_workspace"):
        await inner.run_inner_loop(harness, task_dict=task, workspace=agent_ws)
    assert harness.calls == [], "refused before any model call"


async def test_submit_itself_refuses_to_grade_without_the_source(agent_ws: Path):
    task = load_task(REPO_ROOT, TASK_ID)
    del task["_source_workspace"]
    state = inner.initial_inner_state(task_dict=task, workspace=agent_ws)
    state["verify_result"] = {"tests_pass": True}
    with pytest.raises(ValueError, match="will not fall back"):
        await inner.submit(state, HonestFixHarness())


# ── eval/score.py uses the same grader ────────────────────────────────


def test_eval_score_rejects_the_conftest_exploit(agent_ws: Path):
    from eval import score as score_mod  # noqa: PLC0415

    (agent_ws / "conftest.py").write_text(CONFTEST_EXPLOIT, encoding="utf-8")
    result = score_mod.score_task(TASK_ID, workspace_override=agent_ws)
    assert result["passed"] is False
    assert result["out_of_plan_changes"] == ["conftest.py"]

    execute_tool("apply_patch", agent_ws, path="calculator.py", patch=FIX)
    assert score_mod.score_task(TASK_ID, workspace_override=agent_ws)["passed"] is True


def test_eval_score_honours_the_task_timeout_and_always_exits_zero(
    tmp_path: Path, monkeypatch, capsys
):
    from eval import score as score_mod  # noqa: PLC0415

    task_dir = tmp_path / "tasks" / "slow"
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"id": "slow", "test_command": "sleep 30", "timeout_sec": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(score_mod, "TASKS_DIR", tmp_path / "tasks")

    assert score_mod.main(["--task", "slow"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["timed_out"] is True and result["passed"] is False

    assert score_mod.main(["--task", "does-not-exist"]) == 0
    assert "error" in json.loads(capsys.readouterr().out)
