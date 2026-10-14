"""Grading: score a trial where the agent cannot reach the grader.

The agent's own ``verify`` phase runs the task's tests inside the
workspace the agent has been editing. That is the right place for a
feedback signal and the wrong place for a score: anything that decides
how pytest behaves — a ``conftest.py``, a ``pytest.ini``, the test files
themselves — is agent-writable there. A ``conftest.py`` whose hookwrapper
rewrites every report to ``passed`` made verify green while the bug
stayed in place, and the trial scored 1.0.

So the score comes from here instead:

1. Copy the task's **pristine** ``workspace/`` into a fresh sandbox.
2. Carry over, from the agent's workspace, **only** the files the task
   declares in ``expected_files_changed``. An expected file the agent
   deleted is deleted in the copy. Nothing else the agent wrote — new
   files, edited tests, config — reaches the grader.
3. Run the task's ``test_command`` there, through the same sandbox
   runner as every tool command: its own process group, a scrubbed
   environment, rlimits and a timeout.

Every file the agent changed, added or deleted outside
``expected_files_changed`` is reported as ``out_of_plan_changes``, as
workspace-relative POSIX paths, so a reader can see what was ignored.

**Trust boundary.** ``test_command`` and ``expected_files_changed`` come
from ``eval/tasks/<id>/task.json``, committed repository content. The
command runs through a shell for that reason only; see
``inner._run_verify_subprocess``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from app.meta_harness.sandbox import run_in_sandbox, sandbox_for

#: Wall-clock budget for one test run when the task declares none.
DEFAULT_TEST_TIMEOUT_SEC = 120

_STDOUT_LIMIT = 4000
_STDERR_LIMIT = 2000
_OUTPUT_LIMIT = 4000

#: Directories that test runs write as a side effect. They are not
#: changes the agent made to the task, and listing them would bury the
#: ones it did make.
_IGNORED_DIRS = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis"}
)
_IGNORED_SUFFIXES = (".pyc", ".pyo")

#: The grading copy leaves caches behind too. A stale ``.pyc`` matching a
#: carried file's size and mtime would otherwise be imported instead of
#: the carried source.
_COPY_IGNORE = shutil.ignore_patterns(
    *_IGNORED_DIRS, *(f"*{suffix}" for suffix in _IGNORED_SUFFIXES)
)


def timeout_for_task(task: Mapping[str, Any]) -> int:
    """The task's declared ``timeout_sec``, or the default."""
    try:
        value = int(task.get("timeout_sec") or DEFAULT_TEST_TIMEOUT_SEC)
    except (TypeError, ValueError):
        value = DEFAULT_TEST_TIMEOUT_SEC
    return max(1, value)


def normalize_expected_files(entries: Sequence[str] | None) -> list[str]:
    """Validate ``expected_files_changed`` as workspace-relative POSIX paths.

    Raises ``ValueError`` for an absolute path or one that climbs out of
    the workspace: a task spec that names one is broken, not gameable.
    """
    normalized: list[str] = []
    for raw in entries or []:
        text = str(raw).replace("\\", "/").strip()
        posix = PurePosixPath(text)
        if (
            not text
            or posix.is_absolute()
            or ".." in posix.parts
            or (len(text) > 1 and text[1] == ":")
        ):
            raise ValueError(
                f"expected_files_changed entry {raw!r} is not a "
                "workspace-relative path"
            )
        rel = posix.as_posix()
        if rel not in normalized:
            normalized.append(rel)
    return normalized


def _ignored(rel: PurePosixPath) -> bool:
    return any(part in _IGNORED_DIRS for part in rel.parts[:-1]) or rel.name.endswith(
        _IGNORED_SUFFIXES
    )


def _fingerprints(root: Path) -> dict[str, str]:
    """``{posix relative path: fingerprint}`` for every file under ``root``.

    Symlinks are fingerprinted by their target text and never followed,
    so a link out of the workspace is reported, not read.
    """
    found: dict[str, str] = {}
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        base = Path(dirpath)
        names = list(filenames) + [
            d for d in dirnames if (base / d).is_symlink()
        ]
        for name in names:
            path = base / name
            rel = PurePosixPath(path.relative_to(root).as_posix())
            if _ignored(rel):
                continue
            try:
                if path.is_symlink():
                    found[rel.as_posix()] = "link:" + os.readlink(path)
                else:
                    found[rel.as_posix()] = (
                        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                    )
            except OSError:
                found[rel.as_posix()] = "unreadable"
    return found


def changed_files(source_workspace: Path, agent_workspace: Path) -> list[str]:
    """Every file added, modified or deleted relative to the pristine source."""
    before = _fingerprints(source_workspace)
    after = _fingerprints(agent_workspace)
    return sorted(
        rel
        for rel in set(before) | set(after)
        if before.get(rel) != after.get(rel)
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def _carry_over(
    agent_workspace: Path, grade_dir: Path, expected: list[str]
) -> list[str]:
    """Apply the agent's version of each expected file to the grading copy.

    Only regular files are carried. A symlink or a directory where an
    expected file should be is left at its pristine version: the grader
    reads file contents, never a link the agent pointed somewhere else.
    """
    carried: list[str] = []
    for rel in expected:
        parts = PurePosixPath(rel).parts
        src = agent_workspace.joinpath(*parts)
        dst = grade_dir.joinpath(*parts)
        if not _inside(dst.parent, grade_dir):
            continue
        if src.is_symlink() or (src.exists() and not src.is_file()):
            continue
        if src.is_file():
            if not _inside(src, agent_workspace):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            shutil.copyfile(src, dst, follow_symlinks=False)
            carried.append(rel)
        elif dst.exists() or dst.is_symlink():
            # The agent deleted an expected file; delete it here too.
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
            carried.append(rel)
    return carried


def _text(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return str(stream)


def grade_workspace(
    *,
    source_workspace: Path,
    agent_workspace: Path,
    test_command: str,
    expected_files_changed: Sequence[str] | None,
    timeout_sec: int = DEFAULT_TEST_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Grade ``agent_workspace`` against a pristine copy of the task.

    Returns plain JSON types only — the result is recorded on the
    execution tape and must hash identically when replayed::

        {"passed", "exit_code", "timed_out", "stdout", "stderr", "output",
         "carried_files", "out_of_plan_changes"}
    """
    source_workspace = Path(source_workspace)
    agent_workspace = Path(agent_workspace)
    if not source_workspace.is_dir():
        raise ValueError(f"source workspace is not a directory: {source_workspace}")
    expected = normalize_expected_files(expected_files_changed)
    out_of_plan = [
        rel
        for rel in changed_files(source_workspace, agent_workspace)
        if rel not in expected
    ]

    with sandbox_for(source_workspace, ignore=_COPY_IGNORE) as grade_dir:
        carried = _carry_over(agent_workspace, grade_dir, expected)
        try:
            proc = run_in_sandbox(grade_dir, test_command, timeout_sec=timeout_sec)
            exit_code: int | None = proc.returncode
            stdout, stderr = proc.stdout, proc.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            exit_code = None
            stdout = _text(exc.stdout)
            stderr = _text(exc.stderr) + f"\n[timeout after {timeout_sec}s]"
            timed_out = True

    return {
        "passed": exit_code == 0,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout": stdout[-_STDOUT_LIMIT:],
        "stderr": stderr[-_STDERR_LIMIT:],
        "output": (stdout + "\n" + stderr)[-_OUTPUT_LIMIT:],
        "carried_files": sorted(carried),
        "out_of_plan_changes": out_of_plan,
    }


__all__ = [
    "DEFAULT_TEST_TIMEOUT_SEC",
    "changed_files",
    "grade_workspace",
    "normalize_expected_files",
    "timeout_for_task",
]
