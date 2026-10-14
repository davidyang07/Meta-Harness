"""The 6 fixed inner-loop tools (``INTERFACES.md`` §3).

Tools are the contract with the evaluator and **cannot be overridden**
by candidates. Each function takes the workspace ``Path`` plus tool-
specific args and returns a structured dict (``{"status": ...}``).

The ``TOOL_SCHEMAS`` list is the JSON Schema dict passed to the
Anthropic API as ``tools=...``; ``TOOL_DISPATCH`` maps tool names to
the Python implementation.

**A tool call never raises.** ``execute_tool`` validates the model's
arguments against ``TOOL_SCHEMAS`` and turns every failure — a bad
argument, a binary file, a timeout, an unexpected exception — into a
``{"status": "error", "error_type": ...}`` result. An exception here
would escape ``act``, kill the trial and, before the benchmark caught
per-trial errors, the whole benchmark; the model is always owed a
``tool_result`` instead.

Every subprocess a tool starts goes through ``sandbox.run_in_sandbox``:
its own process group, a scrubbed environment, no stdin and a timeout.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from app.meta_harness.sandbox import run_in_sandbox

# Any output captured beyond these limits is truncated. Matches the
# truncation thresholds discussed in Appendix C §C.5 (Finding 6).
_STDOUT_LIMIT = 8000
_STDERR_LIMIT = 2000
_FILE_LINE_HARD_LIMIT = 2000  # Read errors over this without explicit range
_GREP_OUTPUT_LIMIT = 8000
_GREP_TIMEOUT_S = 30
_GIT_APPLY_TIMEOUT_S = 30


# ──────────────────────────────────────────────────────────────────────
# JSON schemas (verbatim from INTERFACES.md §3, kept in sync there).
# ──────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file from the workspace, with optional line range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer", "description": "Inclusive; -1 = EOF"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "apply_patch",
        "description": (
            "Apply a unified-diff patch to a file. Patches are surgical and preserve "
            "unchanged lines exactly. Use this to make targeted edits rather than "
            "rewriting whole files. The patch must apply cleanly; fuzz matching is "
            "disabled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "patch": {
                    "type": "string",
                    "description": "Unified diff format (the same format as `git diff`).",
                },
            },
            "required": ["path", "patch"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a new file. Errors if the file already exists — use apply_patch "
            "to modify existing files. Use this only for files that do not yet exist "
            "in the workspace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_bash",
        "description": (
            "Run a bash command in the sandboxed workspace. Returns stdout, stderr, "
            "exit_code, and duration_ms. Commands run with a 30s default timeout "
            "(max 120s). The workspace is reset between tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_sec": {"type": "integer", "default": 30, "maximum": 120},
            },
            "required": ["command"],
        },
    },
    {
        "name": "grep_search",
        "description": (
            "Search files in the workspace using ripgrep. Returns file paths and "
            "matching lines with line numbers. Prefer this over reading many files "
            "individually."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern."},
                "path": {"type": "string", "default": "."},
                "file_glob": {"type": "string", "description": "e.g. '*.py'"},
                "context_lines": {"type": "integer", "default": 2, "maximum": 10},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "task_complete",
        "description": (
            "Signal that the task is done. Call this when you believe the task is "
            "solved AND tests pass. The harness will run final verification."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

TOOL_NAMES = [s["name"] for s in TOOL_SCHEMAS]


# ──────────────────────────────────────────────────────────────────────
# Implementations.
# ──────────────────────────────────────────────────────────────────────


def _resolve_in_workspace(workspace: Path, path: str) -> Path | None:
    """Resolve ``path`` against ``workspace`` and return the absolute path.

    Returns None if the resolved path escapes the workspace (path
    traversal protection).
    """
    workspace = workspace.resolve()
    resolved = (workspace / path).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return None
    return resolved


def read_file(
    workspace: Path,
    path: str,
    *,
    start_line: int = 1,
    end_line: int = -1,
) -> dict[str, Any]:
    """Read a file. Returns line-numbered content. Errors on >2000 lines
    when no explicit range is given."""
    target = _resolve_in_workspace(workspace, path)
    if target is None:
        return {
            "status": "error",
            "error_type": "invalid_path",
            "error_message": f"path '{path}' escapes the workspace",
        }
    if not target.exists():
        return {
            "status": "error",
            "error_type": "file_not_found",
            "error_message": f"file not found: {path}",
        }
    if not target.is_file():
        return {
            "status": "error",
            "error_type": "not_a_file",
            "error_message": f"not a regular file: {path}",
        }

    try:
        raw = target.read_bytes()
    except OSError as exc:
        return {
            "status": "error",
            "error_type": "read_failed",
            "error_message": f"could not read {path}: {exc}",
        }
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "status": "error",
            "error_type": "binary_file",
            "error_message": (
                f"{path} is not UTF-8 text ({len(raw)} bytes); use run_bash "
                "(e.g. `file` or `xxd | head`) to inspect it"
            ),
        }
    # Universal newlines, as text-mode reading always did.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    n = len(lines)

    explicit_range = (start_line != 1) or (end_line not in (-1, n))
    if not explicit_range and n > _FILE_LINE_HARD_LIMIT:
        return {
            "status": "error",
            "error_type": "file_too_large",
            "error_message": (
                f"file has {n} lines (>{_FILE_LINE_HARD_LIMIT}); specify a "
                "range or use grep_search to find the relevant section"
            ),
            "n_lines": n,
        }

    if end_line == -1:
        end_line = n
    if start_line < 1 or start_line > n:
        return {
            "status": "error",
            "error_type": "invalid_range",
            "error_message": f"start_line {start_line} out of [1, {n}]",
        }
    if end_line < start_line:
        return {
            "status": "error",
            "error_type": "invalid_range",
            "error_message": f"end_line {end_line} < start_line {start_line}",
        }

    selected = lines[start_line - 1 : end_line]
    numbered = "\n".join(
        f"{start_line + i:6}→{line}" for i, line in enumerate(selected)
    )
    return {
        "status": "ok",
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "n_lines": n,
        "content": numbered,
    }


def write_file(workspace: Path, path: str, content: str) -> dict[str, Any]:
    """Create a new file. Errors if file already exists."""
    target = _resolve_in_workspace(workspace, path)
    if target is None:
        return {
            "status": "error",
            "error_type": "invalid_path",
            "error_message": f"path '{path}' escapes the workspace",
        }
    if target.exists():
        return {
            "status": "error",
            "error_type": "file_exists",
            "error_message": (
                f"file '{path}' already exists. Use apply_patch to modify it."
            ),
        }
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        return {
            "status": "error",
            "error_type": "invalid_content",
            "error_message": f"content is not encodable as UTF-8: {exc}",
        }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Bytes, not text mode: the model's "\n" must stay "\n" on
        # Windows too, or a later LF patch no longer matches the file.
        with target.open("xb") as fh:
            fh.write(data)
    except FileExistsError:
        return {
            "status": "error",
            "error_type": "file_exists",
            "error_message": (
                f"file '{path}' already exists. Use apply_patch to modify it."
            ),
        }
    except OSError as exc:
        return {
            "status": "error",
            "error_type": "write_failed",
            "error_message": str(exc),
        }
    return {
        "status": "ok",
        "path": path,
        "bytes_written": len(data),
    }


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.MULTILINE)


def _workspace_relative_path(workspace: Path, target: Path) -> str:
    """Return a normalized POSIX path for ``target`` relative to workspace."""
    return target.resolve().relative_to(workspace.resolve()).as_posix()


def _normalize_patch_header_path(raw: str) -> str | None:
    """Normalize a path from a unified-diff file header.

    Returns ``None`` for ``/dev/null``. Rejects absolute paths and path
    traversal; the caller compares the result against the tool's declared
    ``path`` argument before invoking ``git apply``.
    """
    path = raw.strip()
    if "\t" in path:
        path = path.split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    posix = PurePosixPath(path)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"patch header path escapes workspace: {raw!r}")
    return posix.as_posix()


_HUNK_START = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")

#: Git extended headers ``apply_patch`` refuses. Each makes a patch do
#: something other than edit the declared file's content in place —
#: rename or copy it, create or delete a file, change a mode, or carry a
#: binary payload — and a rename-only patch has no ``---``/``+++`` lines
#: at all, which is how a path check that only read those lines was
#: bypassed.
_FORBIDDEN_PATCH_HEADERS = (
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "deleted file mode",
    "new file mode",
    "old mode ",
    "new mode ",
    "similarity index ",
    "dissimilarity index ",
    "GIT binary patch",
    "Binary files ",
)


def _scan_patch_headers(
    patch_text: str,
) -> tuple[set[str], bool, list[str], list[str]]:
    """Read a patch's file headers the way ``git apply`` delimits them.

    Hunk bodies are skipped by their declared line counts, so a removed
    line that happens to begin ``--- `` is not mistaken for a header.
    Returns ``(paths, mentions_dev_null, diff_git_lines, forbidden_lines)``.
    Raises ``ValueError`` for a header path that escapes the workspace.
    """
    paths: set[str] = set()
    dev_null = False
    git_lines: list[str] = []
    forbidden: list[str] = []
    old_left = new_left = 0
    for line in patch_text.splitlines():
        if old_left > 0 or new_left > 0:
            tag = line[:1]
            if tag == "-":
                old_left -= 1
            elif tag == "+":
                new_left -= 1
            elif tag != "\\":  # context, including an empty line
                old_left -= 1
                new_left -= 1
            continue
        hunk = _HUNK_START.match(line)
        if hunk:
            old_left = int(hunk.group(1) if hunk.group(1) is not None else 1)
            new_left = int(hunk.group(2) if hunk.group(2) is not None else 1)
            continue
        if line.startswith(("--- ", "+++ ")):
            normalized = _normalize_patch_header_path(line[4:])
            if normalized is None:
                dev_null = True
            else:
                paths.add(normalized)
        elif line.startswith("diff --git "):
            git_lines.append(line)
        elif line.startswith(_FORBIDDEN_PATCH_HEADERS):
            forbidden.append(line)
    return paths, dev_null, git_lines, forbidden


def _patch_header_paths(patch_text: str) -> set[str]:
    """Return file paths mentioned by ``---`` / ``+++`` patch headers."""
    return _scan_patch_headers(patch_text)[0]


def _patch_error(error_type: str, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error_type": error_type,
        "error_message": message,
        "context_echo": None,
    }


def _validate_single_file_patch(
    workspace: Path,
    target: Path,
    patch_text: str,
) -> dict[str, Any] | None:
    """Return an error dict unless ``patch_text`` only edits ``target``.

    A patch must name the declared path in its ``---``/``+++`` headers
    and nothing else; it may not create, delete, rename, copy, re-mode or
    binary-patch anything. Every one of those used to reach ``git apply``
    unchecked whenever the patch had no ``---``/``+++`` lines to compare.
    """
    try:
        patch_paths, dev_null, git_lines, forbidden = _scan_patch_headers(
            patch_text
        )
    except ValueError as exc:
        return _patch_error("invalid_patch_path", str(exc))
    expected = _workspace_relative_path(workspace, target)
    if forbidden:
        return _patch_error(
            "unsupported_patch",
            f"apply_patch only edits the content of {expected!r}; it does not "
            f"accept {forbidden[0].strip()!r}. Use write_file to create a "
            "file and run_bash to move or delete one.",
        )
    if dev_null:
        return _patch_error(
            "unsupported_patch",
            "a /dev/null header creates or deletes a file; apply_patch only "
            f"edits {expected!r} in place",
        )
    if not patch_paths:
        return _patch_error(
            "invalid_patch",
            "the patch has no '--- a/<path>' / '+++ b/<path>' file headers; "
            f"start it with '--- a/{expected}' and '+++ b/{expected}'",
        )
    if patch_paths != {expected}:
        return _patch_error(
            "path_mismatch",
            f"patch headers target {sorted(patch_paths)!r}, but tool path is "
            f"{expected!r}. apply_patch only accepts single-file patches "
            "for the declared path.",
        )
    allowed_git_lines = {
        f"diff --git a/{expected} b/{expected}",
        f"diff --git {expected} {expected}",
    }
    for line in git_lines:
        if line.rstrip() not in allowed_git_lines:
            return _patch_error(
                "path_mismatch",
                f"{line.strip()!r} does not name {expected!r}; apply_patch only "
                "accepts single-file patches for the declared path.",
            )
    return None


def _extract_context_echo(
    workspace: Path, path: str, patch_text: str
) -> dict[str, Any] | None:
    """On a patch context_mismatch, read the file at the first hunk's
    expected range and return ``{path, start_line, end_line, content}``.

    Returns None if we can't parse the hunk header or read the file.
    """
    target = _resolve_in_workspace(workspace, path)
    if target is None or not target.exists() or not target.is_file():
        return None
    match = _HUNK_HEADER.search(patch_text)
    if not match:
        return None
    start = int(match.group(1))
    length = int(match.group(2) or "1")
    if length == 0:
        length = 1
    end = start + length - 1
    try:
        file_lines = target.read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return None
    if start < 1 or start > len(file_lines):
        return None
    end = min(end, len(file_lines))
    snippet = "\n".join(file_lines[start - 1 : end])
    return {
        "path": path,
        "start_line": start,
        "end_line": end,
        "content": snippet,
    }


def _run_git_apply(
    workspace: Path, args: list[str]
) -> subprocess.CompletedProcess[str] | None:
    """``git apply <args>`` in the workspace; ``None`` on timeout."""
    try:
        return run_in_sandbox(
            workspace, ["git", "apply", *args], timeout_sec=_GIT_APPLY_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return None


def apply_patch(workspace: Path, path: str, patch: str) -> dict[str, Any]:
    """Apply a unified-diff patch via ``git apply``. Returns
    ``context_echo`` on context_mismatch."""
    target = _resolve_in_workspace(workspace, path)
    if target is None:
        return {
            "status": "error",
            "error_type": "invalid_path",
            "error_message": f"path '{path}' escapes the workspace",
            "context_echo": None,
        }
    if not target.exists():
        return {
            "status": "error",
            "error_type": "file_not_found",
            "error_message": (
                f"file '{path}' not found. Use write_file to create new files."
            ),
            "context_echo": None,
        }
    patch_error = _validate_single_file_patch(workspace, target, patch)
    if patch_error is not None:
        return patch_error

    # ``newline=""`` and an explicit encoding, so the patch reaches
    # ``git apply`` byte-for-byte as the model wrote it. In text mode
    # Python rewrites every "\n" to os.linesep, which on Windows turns an
    # LF unified diff into a CRLF one; git then reads the trailing CR as
    # part of each context line and every patch fails as
    # ``context_mismatch``. That failure does not crash anything — it
    # just makes the inner loop's primary edit tool silently unusable on
    # Windows, and depresses the measured pass rate with it.
    try:
        patch.encode("utf-8")
    except UnicodeEncodeError as exc:
        return _patch_error("invalid_patch", f"patch is not encodable as UTF-8: {exc}")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, newline="", encoding="utf-8"
    ) as patch_file:
        patch_file.write(patch)
        patch_path = patch_file.name

    try:
        check = _run_git_apply(workspace, ["--check", patch_path])
        if check is None:
            return _patch_error(
                "timeout", f"git apply --check timed out after {_GIT_APPLY_TIMEOUT_S}s"
            )
        if check.returncode != 0:
            stderr = check.stderr.strip()
            if "patch does not apply" in stderr or "while searching for" in stderr:
                error_type = "context_mismatch"
                context_echo = _extract_context_echo(workspace, path, patch)
                if context_echo is not None:
                    msg = (
                        f"Patch context did not match at lines "
                        f"{context_echo['start_line']}-{context_echo['end_line']}. "
                        f"The file currently reads:\n{context_echo['content']}\n"
                        "Edit the patch to match this and retry."
                    )
                else:
                    msg = stderr or "patch context did not match"
                return {
                    "status": "error",
                    "error_type": error_type,
                    "error_message": msg,
                    "context_echo": context_echo,
                }
            return {
                "status": "error",
                "error_type": "invalid_patch",
                "error_message": stderr or "git apply --check failed",
                "context_echo": None,
            }

        apply_proc = _run_git_apply(workspace, [patch_path])
        if apply_proc is None:
            return _patch_error(
                "timeout", f"git apply timed out after {_GIT_APPLY_TIMEOUT_S}s"
            )
        if apply_proc.returncode != 0:
            return {
                "status": "error",
                "error_type": "invalid_patch",
                "error_message": apply_proc.stderr.strip()
                or "git apply failed after passing --check",
                "context_echo": None,
            }
        return {"status": "ok", "path": path, "patch_applied": True}
    finally:
        Path(patch_path).unlink(missing_ok=True)


def run_bash(
    workspace: Path,
    command: str,
    *,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    """Run a bash command in the sandbox. Returns stdout/stderr/exit_code/
    duration_ms. Capped at 120s."""
    timeout_sec = min(max(1, timeout_sec), 120)
    started = time.monotonic()
    try:
        proc = run_in_sandbox(workspace, command, timeout_sec=timeout_sec)
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "status": "ok",
            "stdout": proc.stdout[-_STDOUT_LIMIT:],
            "stderr": proc.stderr[-_STDERR_LIMIT:],
            "exit_code": proc.returncode,
            "duration_ms": duration_ms,
        }
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        out = exc.stdout or b""
        err = exc.stderr or b""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return {
            "status": "error",
            "error_type": "timeout",
            "error_message": f"command timed out after {timeout_sec}s",
            "stdout": out[-_STDOUT_LIMIT:],
            "stderr": err[-_STDERR_LIMIT:],
            "duration_ms": duration_ms,
        }


def grep_search(
    workspace: Path,
    pattern: str,
    *,
    path: str = ".",
    file_glob: str | None = None,
    context_lines: int = 2,
) -> dict[str, Any]:
    """Search using ripgrep (or ``grep -rn`` fallback).

    The pattern is passed with ``-e`` and the search root after ``--``,
    so neither can be read as an option: a pattern of ``-v`` used to be
    taken as a flag, leaving no search root, and the search ran over the
    *server's* working directory instead of the workspace. The command
    runs with ``cwd`` set to the workspace and reports workspace-relative
    paths.
    """
    context_lines = max(0, min(context_lines, 10))
    target = _resolve_in_workspace(workspace, path)
    if target is None:
        return {
            "status": "error",
            "error_type": "invalid_path",
            "error_message": f"path '{path}' escapes the workspace",
        }
    if not target.exists():
        return {
            "status": "error",
            "error_type": "file_not_found",
            "error_message": f"path not found: {path}",
        }
    root = _workspace_relative_path(workspace, target)

    if shutil.which("rg"):
        cmd: list[str] = ["rg", "--no-config", "-n", "-C", str(context_lines)]
        if file_glob:
            cmd += ["-g", file_glob]
        cmd += ["-e", pattern, "--", root]
    else:
        cmd = ["grep", "-rn", f"-C{context_lines}", "-E"]
        if file_glob:
            cmd.append(f"--include={file_glob}")
        cmd += ["-e", pattern, "--", root]

    try:
        proc = run_in_sandbox(workspace, cmd, timeout_sec=_GREP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error_type": "timeout",
            "error_message": f"search timed out after {_GREP_TIMEOUT_S}s",
            "command": " ".join(cmd),
        }
    matches = proc.stdout[-_GREP_OUTPUT_LIMIT:]
    result: dict[str, Any] = {
        "status": "ok",
        "matches": matches,
        "match_count": matches.count("\n"),
        "exit_code": proc.returncode,
        "command": " ".join(cmd),
    }
    if proc.stderr.strip():
        result["stderr"] = proc.stderr[-_STDERR_LIMIT:]
    return result


def task_complete() -> dict[str, Any]:
    """Sentinel — signals task is done. No-op return."""
    return {"status": "ok", "signal": "task_complete"}


# ──────────────────────────────────────────────────────────────────────
# Dispatch.
# ──────────────────────────────────────────────────────────────────────


TOOL_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "read_file": read_file,
    "write_file": write_file,
    "apply_patch": apply_patch,
    "run_bash": run_bash,
    "grep_search": grep_search,
    "task_complete": task_complete,
}


_SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {s["name"]: s for s in TOOL_SCHEMAS}

#: JSON-schema type → the Python types that satisfy it. ``bool`` is a
#: subclass of ``int`` and is excluded from "integer" explicitly.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def _invalid_arguments(name: str, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error_type": "invalid_arguments",
        "error_message": f"{name}: {message}",
    }


def validate_tool_arguments(
    name: str, arguments: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Check ``arguments`` against the tool's JSON schema.

    Returns ``(clean_arguments, None)`` or ``(None, error_result)``.
    An optional argument the model sent as ``null`` is treated as absent.
    Types are checked, not coerced: ``"start_line": "2"`` is an error the
    model can read and correct, not a ``TypeError`` that ends the trial.
    """
    schema = _SCHEMA_BY_NAME[name]["input_schema"]
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))

    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        return None, _invalid_arguments(
            name,
            f"unexpected argument(s) {unknown}; accepted: {sorted(properties)}",
        )
    clean = {
        key: value
        for key, value in arguments.items()
        if value is not None or key in required
    }
    missing = sorted(required - set(clean))
    if missing:
        return None, _invalid_arguments(name, f"missing required argument(s) {missing}")
    for key, value in clean.items():
        expected = properties[key].get("type")
        allowed = _JSON_TYPES.get(expected)
        if allowed is None:
            continue
        wrong = not isinstance(value, allowed) or (
            expected in ("integer", "number") and isinstance(value, bool)
        )
        if wrong:
            return None, _invalid_arguments(
                name,
                f"argument {key!r} must be {expected}, got "
                f"{type(value).__name__} {value!r:.80}",
            )
    return clean, None


def execute_tool(name: str, workspace: Path, **kwargs: Any) -> dict[str, Any]:
    """Dispatch a tool call by name. ``task_complete`` ignores ``workspace``.

    Never raises: invalid arguments and any failure inside a tool come
    back as a structured error result the model can act on.
    """
    if name not in TOOL_DISPATCH:
        return {
            "status": "error",
            "error_type": "unknown_tool",
            "error_message": f"unknown tool: {name}",
        }
    arguments, error = validate_tool_arguments(name, kwargs)
    if error is not None:
        return error
    fn = TOOL_DISPATCH[name]
    try:
        if name == "task_complete":
            return fn()
        return fn(workspace, **(arguments or {}))
    except Exception as exc:  # noqa: BLE001 — the model is owed a tool_result
        return {
            "status": "error",
            "error_type": "tool_exception",
            "error_message": f"{name} failed: {type(exc).__name__}: {exc}",
        }
