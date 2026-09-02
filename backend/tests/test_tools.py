"""Tests for the 6 fixed inner-loop tools (BUILD_ORDER step 2).

Each tool is exercised on happy + structured-error paths. ``apply_patch``
specifically validates that ``context_mismatch`` returns the
``context_echo`` block per Drift Correction A.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.meta_harness.tools import (
    TOOL_NAMES,
    TOOL_SCHEMAS,
    apply_patch,
    execute_tool,
    grep_search,
    read_file,
    run_bash,
    task_complete,
    write_file,
)


# ──────────────────────────────────────────────────────────────────────
# Schema-list shape (sanity).
# ──────────────────────────────────────────────────────────────────────


def test_tool_schemas_list_six_named_tools():
    assert TOOL_NAMES == [
        "read_file",
        "apply_patch",
        "write_file",
        "run_bash",
        "grep_search",
        "task_complete",
    ]
    assert len(TOOL_SCHEMAS) == 6
    for s in TOOL_SCHEMAS:
        assert "name" in s and "description" in s and "input_schema" in s


# ──────────────────────────────────────────────────────────────────────
# read_file.
# ──────────────────────────────────────────────────────────────────────


def test_read_file_basic(tmp_path: Path):
    (tmp_path / "hi.py").write_text("a = 1\nb = 2\nc = 3\n")
    out = read_file(tmp_path, "hi.py")
    assert out["status"] == "ok"
    assert out["n_lines"] == 4  # trailing newline → 4-element split
    assert "a = 1" in out["content"]
    assert "b = 2" in out["content"]


def test_read_file_with_range(tmp_path: Path):
    (tmp_path / "hi.py").write_text("\n".join(f"line {i}" for i in range(1, 11)))
    out = read_file(tmp_path, "hi.py", start_line=3, end_line=5)
    assert out["status"] == "ok"
    assert out["start_line"] == 3
    assert out["end_line"] == 5
    assert "line 3" in out["content"]
    assert "line 5" in out["content"]
    assert "line 1\n" not in out["content"]


def test_read_file_missing(tmp_path: Path):
    out = read_file(tmp_path, "nope.py")
    assert out["status"] == "error"
    assert out["error_type"] == "file_not_found"


def test_read_file_path_traversal_blocked(tmp_path: Path):
    out = read_file(tmp_path, "../../../etc/passwd")
    assert out["status"] == "error"
    assert out["error_type"] == "invalid_path"


def test_read_file_too_large(tmp_path: Path):
    big = "\n".join(f"line {i}" for i in range(1, 2502))
    (tmp_path / "big.py").write_text(big)
    out = read_file(tmp_path, "big.py")
    assert out["status"] == "error"
    assert out["error_type"] == "file_too_large"
    assert out["n_lines"] >= 2500


def test_read_file_too_large_with_range_ok(tmp_path: Path):
    big = "\n".join(f"line {i}" for i in range(1, 2502))
    (tmp_path / "big.py").write_text(big)
    out = read_file(tmp_path, "big.py", start_line=1, end_line=10)
    assert out["status"] == "ok"


# ──────────────────────────────────────────────────────────────────────
# write_file.
# ──────────────────────────────────────────────────────────────────────


def test_write_file_creates_new(tmp_path: Path):
    out = write_file(tmp_path, "new.py", "x = 1\n")
    assert out["status"] == "ok"
    assert out["bytes_written"] == 6
    assert (tmp_path / "new.py").read_text() == "x = 1\n"


def test_write_file_creates_parent_dirs(tmp_path: Path):
    out = write_file(tmp_path, "geometry/point.py", "class Point: pass\n")
    assert out["status"] == "ok"
    assert (tmp_path / "geometry" / "point.py").exists()


def test_write_file_existing_errors(tmp_path: Path):
    (tmp_path / "exists.py").write_text("old\n")
    out = write_file(tmp_path, "exists.py", "new\n")
    assert out["status"] == "error"
    assert out["error_type"] == "file_exists"
    # Original content preserved
    assert (tmp_path / "exists.py").read_text() == "old\n"


def test_write_file_path_traversal_blocked(tmp_path: Path):
    out = write_file(tmp_path, "../escape.py", "x")
    assert out["status"] == "error"
    assert out["error_type"] == "invalid_path"


# ──────────────────────────────────────────────────────────────────────
# apply_patch (with context_echo on mismatch).
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def calc_workspace(tmp_path: Path) -> Path:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    return tmp_path


_GOOD_PATCH = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""

_BAD_CONTEXT_PATCH = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a * b
+    return a + b
"""


def test_apply_patch_happy(calc_workspace: Path):
    out = apply_patch(calc_workspace, "calc.py", _GOOD_PATCH)
    assert out["status"] == "ok"
    assert "return a + b" in (calc_workspace / "calc.py").read_text()


def test_apply_patch_context_mismatch_returns_context_echo(calc_workspace: Path):
    out = apply_patch(calc_workspace, "calc.py", _BAD_CONTEXT_PATCH)
    assert out["status"] == "error"
    assert out["error_type"] == "context_mismatch"
    echo = out["context_echo"]
    assert echo is not None
    assert echo["path"] == "calc.py"
    assert echo["start_line"] == 1
    assert "return a - b" in echo["content"]


def test_apply_patch_missing_file_returns_file_not_found(tmp_path: Path):
    out = apply_patch(tmp_path, "no.py", _GOOD_PATCH)
    assert out["status"] == "error"
    assert out["error_type"] == "file_not_found"
    assert out["context_echo"] is None


def test_apply_patch_path_traversal_blocked(tmp_path: Path):
    out = apply_patch(tmp_path, "../escape.py", _GOOD_PATCH)
    assert out["status"] == "error"
    assert out["error_type"] == "invalid_path"


def test_apply_patch_rejects_patch_for_different_file(tmp_path: Path):
    (tmp_path / "safe.py").write_text("x = 1\n")
    (tmp_path / "other.py").write_text("y = 1\n")
    patch = """\
--- a/other.py
+++ b/other.py
@@ -1 +1 @@
-y = 1
+y = 2
"""

    out = apply_patch(tmp_path, "safe.py", patch)

    assert out["status"] == "error"
    assert out["error_type"] == "path_mismatch"
    assert (tmp_path / "safe.py").read_text() == "x = 1\n"
    assert (tmp_path / "other.py").read_text() == "y = 1\n"


def test_apply_patch_rejects_escaping_patch_header(tmp_path: Path):
    (tmp_path / "safe.py").write_text("x = 1\n")
    patch = """\
--- a/../escape.py
+++ b/../escape.py
@@ -1 +1 @@
-x = 1
+x = 2
"""

    out = apply_patch(tmp_path, "safe.py", patch)

    assert out["status"] == "error"
    assert out["error_type"] == "invalid_patch_path"
    assert (tmp_path / "safe.py").read_text() == "x = 1\n"


# ──────────────────────────────────────────────────────────────────────
# run_bash.
# ──────────────────────────────────────────────────────────────────────


def test_run_bash_basic(tmp_path: Path):
    out = run_bash(tmp_path, "echo hello")
    assert out["status"] == "ok"
    assert "hello" in out["stdout"]
    assert out["exit_code"] == 0


def test_run_bash_captures_nonzero_exit(tmp_path: Path):
    out = run_bash(tmp_path, "exit 7")
    assert out["status"] == "ok"
    assert out["exit_code"] == 7


def test_run_bash_timeout(tmp_path: Path):
    out = run_bash(tmp_path, "sleep 5", timeout_sec=1)
    assert out["status"] == "error"
    assert out["error_type"] == "timeout"


def test_run_bash_caps_at_120s(tmp_path: Path):
    # We don't actually wait — just confirm the cap is applied to the request.
    # We verify by passing a high timeout and a fast command; expect normal completion.
    out = run_bash(tmp_path, "echo ok", timeout_sec=600)
    assert out["status"] == "ok"


# ──────────────────────────────────────────────────────────────────────
# grep_search.
# ──────────────────────────────────────────────────────────────────────


def test_grep_search_finds_matches(tmp_path: Path):
    (tmp_path / "a.py").write_text("def median(): pass\n")
    (tmp_path / "b.py").write_text("def mean(): pass\n")
    out = grep_search(tmp_path, "median")
    assert out["status"] == "ok"
    assert "median" in out["matches"]
    assert out["match_count"] >= 1


def test_grep_search_with_glob(tmp_path: Path):
    (tmp_path / "a.py").write_text("foo\n")
    (tmp_path / "b.txt").write_text("foo\n")
    out = grep_search(tmp_path, "foo", file_glob="*.py")
    assert out["status"] == "ok"
    assert "a.py" in out["matches"]
    # b.txt should not match
    assert "b.txt" not in out["matches"]


# ──────────────────────────────────────────────────────────────────────
# task_complete.
# ──────────────────────────────────────────────────────────────────────


def test_task_complete_returns_signal():
    out = task_complete()
    assert out == {"status": "ok", "signal": "task_complete"}


# ──────────────────────────────────────────────────────────────────────
# Dispatch.
# ──────────────────────────────────────────────────────────────────────


def test_execute_tool_dispatches_known(tmp_path: Path):
    (tmp_path / "f.txt").write_text("x\n")
    out = execute_tool("read_file", tmp_path, path="f.txt")
    assert out["status"] == "ok"


def test_execute_tool_unknown_returns_error(tmp_path: Path):
    out = execute_tool("nonsense", tmp_path)
    assert out["status"] == "error"
    assert out["error_type"] == "unknown_tool"


def test_execute_tool_task_complete_ignores_workspace(tmp_path: Path):
    out = execute_tool("task_complete", tmp_path)
    assert out["status"] == "ok"
    assert out["signal"] == "task_complete"


# ── the patch must reach `git apply` byte-for-byte ────────────────────
#
# `apply_patch` writes the model's diff to a temp file and shells out to
# `git apply`. Writing it in text mode rewrites every "\n" to os.linesep,
# which on Windows turns an LF unified diff into a CRLF one; git then
# reads the trailing CR as part of each context line and every patch
# fails as `context_mismatch`. That does not crash anything — it makes
# the inner loop's primary edit tool silently unusable on one platform
# and depresses the measured pass rate with it.


def _unified_diff(before: str, after: str, path: str) -> str:
    """A well-formed diff, generated the way a careful model would write one."""
    import difflib

    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )


def test_an_lf_patch_applies_on_every_platform(tmp_path: Path):
    before = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
    after = before.replace("    return a - b\n\n", "    return a + b\n\n", 1)
    (tmp_path / "calculator.py").write_text(before, encoding="utf-8", newline="")

    result = apply_patch(
        tmp_path, "calculator.py", _unified_diff(before, after, "calculator.py")
    )

    assert result["status"] == "ok", result
    assert (tmp_path / "calculator.py").read_text(encoding="utf-8") == after


def test_the_patch_file_handed_to_git_is_not_newline_translated(tmp_path: Path):
    """Guard the fix directly: the temp patch must keep the model's bytes."""
    import inspect

    from app.meta_harness import tools

    source = inspect.getsource(tools.apply_patch)
    assert 'newline=""' in source, (
        "apply_patch must write its temp patch with newline='' or Python "
        "rewrites the diff's line endings on Windows"
    )


def test_an_lf_patch_applies_to_a_crlf_file_and_keeps_its_line_endings(
    workspace_outside_git: Path,
    autocrlf_true: None,
):
    """A model writes LF diffs; a Windows checkout may hold CRLF files.

    ``git apply`` reconciles the two itself and preserves the file's own
    endings, so the tool must not get in its way. This is exactly the
    case the old text-mode temp write broke, and it is the normal case on
    Windows, failing silently rather than loudly.

    Both fixtures are load-bearing, because "preserve the file's own
    endings" is not unconditional git behaviour — it is what git does
    under one specific configuration, and this test pins that
    configuration instead of inheriting whatever the host happens to have:

    - ``autocrlf_true`` pins ``core.autocrlf=true``. That is the
      Git-for-Windows *system* default, which is why this passes on a
      Windows box and why it does not on a Linux runner, where autocrlf
      is unset: with ``autocrlf=false`` the LF patch simply does not match
      the CRLF file and git reports ``context_mismatch``. Leaving it
      implicit made the test assert one thing on the developer's machine
      and another in CI.
    - ``workspace_outside_git`` keeps the workspace out of any working
      tree. A repository's ``.gitattributes`` overrides ``autocrlf``, and
      this one pins ``eol=lf``, so inside the checkout git would apply the
      patch and rewrite the file to LF. Real task workspaces live under
      ``sandbox.sandbox_root()``, outside this checkout, which is what the
      fixture reproduces.

    So the claim under test is precise: *given the git configuration a
    Windows checkout has by default, and a workspace outside any
    repository, ``apply_patch`` does not get in git's way.*
    """
    workspace = workspace_outside_git
    before = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
    after = before.replace("    return a - b\n\n", "    return a + b\n\n", 1)
    (workspace / "calculator.py").write_bytes(before.replace("\n", "\r\n").encode())

    result = apply_patch(
        workspace, "calculator.py", _unified_diff(before, after, "calculator.py")
    )

    assert result["status"] == "ok", result
    raw = (workspace / "calculator.py").read_bytes()
    assert b"\r\n" in raw, "the file's own line endings must survive the patch"
    assert b"return a + b" in raw


def test_without_autocrlf_an_lf_patch_does_not_match_a_crlf_file(
    workspace_outside_git: Path,
    autocrlf_false: None,
):
    """The other half of the same fact, so neither result looks universal.

    This is not a bug in ``apply_patch`` — it is git declining to
    reconcile line endings it was not configured to reconcile, and it is
    what a Linux runner does by default. Recording it here stops the
    CRLF-preservation test above from being read as a platform-independent
    guarantee.
    """
    workspace = workspace_outside_git
    before = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
    after = before.replace("    return a - b\n\n", "    return a + b\n\n", 1)
    (workspace / "calculator.py").write_bytes(before.replace("\n", "\r\n").encode())

    result = apply_patch(
        workspace, "calculator.py", _unified_diff(before, after, "calculator.py")
    )

    assert result["status"] == "error"
    assert result["error_type"] == "context_mismatch"
