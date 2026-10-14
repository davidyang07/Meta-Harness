"""eval/score.py — score a workspace against a task's pytest contract.

Usage:
    python -m eval.score --task task-001-fix-typo
    python -m eval.score --task task-001-fix-typo --workspace /tmp/agent-output
    python -m eval.score --task task-006-... --holdout

Scoring uses the same grader as the inner loop's ``submit``
(``app.meta_harness.grading``): the task's pristine ``workspace/`` is
copied to a fresh sandbox, only the files the task lists in
``expected_files_changed`` are carried over from ``--workspace``, and
the task's ``test_command`` runs there under the sandbox's limits. A
``conftest.py`` or an edited test in the scored workspace therefore
changes nothing. When ``--workspace`` is omitted the pristine workspace
is scored as-is — this is what produces the baseline (failing) result
for unfixed tasks.

The test run is bounded by the task's ``timeout_sec`` (120s when the
task declares none), with or without ``--workspace``.

Returns: a JSON dict with task, passed, score, stdout (last 4000 chars),
stderr (last 2000 chars), exit_code (``null`` on a timeout), timed_out,
carried_files and out_of_plan_changes; or task, passed, score and error
when the task cannot be scored at all. The script always exits 0; the
JSON is the result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.meta_harness import grading

EVAL_ROOT = Path(__file__).resolve().parent
TASKS_DIR = EVAL_ROOT / "tasks"
HOLDOUT_DIR = EVAL_ROOT / "holdout"


def score_task(
    task_id: str,
    *,
    workspace_override: Path | None = None,
    holdout: bool = False,
    timeout_sec: int | None = None,
) -> dict:
    """Grade a workspace for ``task_id`` and return a structured score dict.

    ``timeout_sec`` overrides the task's own ``timeout_sec``.
    """
    task_dir = (HOLDOUT_DIR if holdout else TASKS_DIR) / task_id
    if not task_dir.exists():
        raise FileNotFoundError(f"task not found: {task_dir}")

    task_spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    source = task_dir / "workspace"
    timeout = int(timeout_sec) if timeout_sec else grading.timeout_for_task(task_spec)
    grade = grading.grade_workspace(
        source_workspace=source,
        agent_workspace=Path(workspace_override) if workspace_override else source,
        test_command=task_spec.get("test_command", "pytest -q"),
        expected_files_changed=task_spec.get("expected_files_changed"),
        timeout_sec=timeout,
    )
    return {
        "task": task_id,
        "passed": grade["passed"],
        "score": 1.0 if grade["passed"] else 0.0,
        "exit_code": grade["exit_code"],
        "timed_out": grade["timed_out"],
        "stdout": grade["stdout"],
        "stderr": grade["stderr"],
        "carried_files": grade["carried_files"],
        "out_of_plan_changes": grade["out_of_plan_changes"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval.score",
        description="Score a candidate workspace against a task's pytest contract.",
    )
    parser.add_argument("--task", required=True, help="task id (e.g. task-001-fix-typo)")
    parser.add_argument(
        "--workspace",
        help="override workspace dir (e.g. an agent's output dir). "
        "Omit to score the pristine task workspace as-is.",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="resolve task from eval/holdout/ instead of eval/tasks/",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="override the task's timeout_sec for the test run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace = Path(args.workspace) if args.workspace else None
    try:
        result = score_task(
            args.task,
            workspace_override=workspace,
            holdout=args.holdout,
            timeout_sec=args.timeout_sec,
        )
    except (OSError, ValueError) as exc:
        result = {
            "task": args.task,
            "passed": False,
            "score": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
