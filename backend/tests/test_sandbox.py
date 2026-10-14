"""Sandbox lifecycle tests (BUILD_ORDER step 2)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.meta_harness.sandbox import (
    DEFAULT_RLIMIT_CPU,
    DEFAULT_RLIMIT_RAM,
    SANDBOX_PREFIX,
    cleanup_sandbox,
    is_secret_env_name,
    make_sandbox_dir,
    populate_sandbox,
    run_in_sandbox,
    sandbox_for,
    sandbox_root,
    scrubbed_env,
)


def test_make_sandbox_dir_creates_unique_temp_dir():
    a = make_sandbox_dir()
    b = make_sandbox_dir()
    try:
        assert a.exists() and a.is_dir()
        assert b.exists() and b.is_dir()
        assert a != b
        assert a.name.startswith(SANDBOX_PREFIX)
        assert a.parent == sandbox_root()
        assert b.parent == sandbox_root()
    finally:
        cleanup_sandbox(a)
        cleanup_sandbox(b)


def test_cleanup_sandbox_is_idempotent():
    sandbox = make_sandbox_dir()
    cleanup_sandbox(sandbox)
    assert not sandbox.exists()
    cleanup_sandbox(sandbox)  # second call must not raise


def test_populate_sandbox_copies_workspace(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    (src / "tests").mkdir()
    (src / "tests" / "test_a.py").write_text("def test(): pass\n")

    sandbox = make_sandbox_dir()
    try:
        populate_sandbox(sandbox, src)
        assert (sandbox / "a.py").read_text() == "x = 1\n"
        assert (sandbox / "tests" / "test_a.py").exists()
    finally:
        cleanup_sandbox(sandbox)


def test_sandbox_for_context_manager_cleans_up(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("hello\n")
    captured: Path | None = None
    with sandbox_for(src) as sb:
        captured = sb
        assert (sb / "f.txt").exists()
    assert captured is not None and not captured.exists()


def test_run_in_sandbox_captures_stdout_stderr_exit_code():
    sandbox = make_sandbox_dir()
    try:
        proc = run_in_sandbox(sandbox, "echo hello && echo err 1>&2 && exit 3")
        assert proc.returncode == 3
        assert "hello" in proc.stdout
        assert "err" in proc.stderr
    finally:
        cleanup_sandbox(sandbox)


def test_run_in_sandbox_enforces_timeout():
    sandbox = make_sandbox_dir()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_in_sandbox(sandbox, "sleep 5", timeout_sec=1)
    finally:
        cleanup_sandbox(sandbox)


# ── containment ───────────────────────────────────────────────────────

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


@pytest.fixture
def sandbox_dir():
    sandbox = make_sandbox_dir()
    try:
        yield sandbox
    finally:
        cleanup_sandbox(sandbox)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); it is dead for our purposes.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().split()[2] != "Z"
    except OSError:
        return True


def _wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@posix_only
def test_a_timeout_kills_the_whole_process_tree(sandbox_dir: Path):
    """``sleep 37 & echo started`` used to leave the sleep running."""
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_in_sandbox(
            sandbox_dir, "sleep 37 & echo $! > bg.pid; echo started; wait", timeout_sec=1
        )
    assert "started" in caught.value.stdout
    pid = int((sandbox_dir / "bg.pid").read_text())
    assert _wait_dead(pid), f"background process {pid} outlived the timeout"


@posix_only
def test_background_jobs_do_not_outlive_a_normal_exit(sandbox_dir: Path):
    started = time.monotonic()
    proc = run_in_sandbox(
        sandbox_dir, "sleep 41 >/dev/null 2>&1 & echo $! > bg.pid", timeout_sec=10
    )
    assert proc.returncode == 0
    assert time.monotonic() - started < 5
    pid = int((sandbox_dir / "bg.pid").read_text())
    assert _wait_dead(pid), f"background process {pid} outlived its command"


def test_secrets_never_reach_a_sandboxed_command(sandbox_dir: Path, monkeypatch):
    secrets = {
        "ANTHROPIC_API_KEY": "sk-ant-secret",
        "OPENAI_API_KEY": "sk-other-secret",
        "GITHUB_TOKEN": "ghp-secret",
        "MY_SERVICE_SECRET": "svc-secret",
        "WANDB_API_KEY": "wandb-secret",
        "WANDB_ENTITY": "wandb-entity-secret",
        "POSTGRES_DSN": "postgresql://user:pw-secret@host/db",
        "AUTH0_CLIENT_ID": "auth0-secret",
        "META_HARNESS_API_TOKEN": "api-token-secret",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HARMLESS_SETTING", "visible")

    proc = run_in_sandbox(sandbox_dir, "env")

    for name, value in secrets.items():
        assert value not in proc.stdout, name
        assert f"{name}=" not in proc.stdout, name
    assert "HARMLESS_SETTING=visible" in proc.stdout
    assert "PATH=" in proc.stdout


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("ANTHROPIC_API_KEY", True),
        ("anthropic_api_key", True),
        ("SLACK_BOT_TOKEN", True),
        ("CLIENT_SECRET", True),
        ("AWS_SECRET_ACCESS_KEY", True),
        ("WANDB_MODE", True),
        ("AUTH0_DOMAIN", True),
        ("POSTGRES_DSN", True),
        ("PATH", False),
        ("HOME", False),
        ("GIT_CONFIG_COUNT", False),
        ("TOKENIZERS_PARALLELISM", False),
    ],
)
def test_secret_env_names(name: str, secret: bool):
    assert is_secret_env_name(name) is secret


def test_scrubbed_env_keeps_everything_else():
    env = {"PATH": "/bin", "ANTHROPIC_API_KEY": "x", "LANG": "C.UTF-8"}
    assert scrubbed_env(env) == {"PATH": "/bin", "LANG": "C.UTF-8"}


def test_a_command_reading_stdin_sees_eof_instead_of_hanging(sandbox_dir: Path):
    started = time.monotonic()
    proc = run_in_sandbox(sandbox_dir, "cat", timeout_sec=10)
    assert proc.returncode == 0
    assert proc.stdout == ""
    assert time.monotonic() - started < 5


def test_an_argv_command_runs_without_a_shell(sandbox_dir: Path):
    script = "import sys; print(sys.argv[1:])"
    proc = run_in_sandbox(sandbox_dir, [sys.executable, "-c", script, "-v", "$HOME"])
    assert proc.stdout.strip() == "['-v', '$HOME']"


@pytest.mark.skipif(
    sys.platform in ("win32", "darwin"), reason="RLIMIT_AS is applied on Linux only"
)
def test_rlimits_are_applied_in_the_child(sandbox_dir: Path):
    proc = run_in_sandbox(sandbox_dir, "ulimit -v; ulimit -t")
    memory_kb, cpu_s = proc.stdout.split()
    assert int(memory_kb) == DEFAULT_RLIMIT_RAM // 1024
    assert int(cpu_s) == DEFAULT_RLIMIT_CPU


def test_rlimits_do_not_use_preexec_fn():
    """``preexec_fn`` is unsafe in threads, and every tool call is in one."""
    import inspect

    from app.meta_harness import sandbox

    assert "preexec_fn" not in inspect.getsource(sandbox.run_in_sandbox)


def test_concurrent_calls_from_worker_threads(sandbox_dir: Path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda i: run_in_sandbox(sandbox_dir, f"echo {i}").stdout.strip(),
                range(16),
            )
        )
    assert results == [str(i) for i in range(16)]


def test_populate_can_skip_caches(tmp_path: Path):
    src = tmp_path / "src"
    (src / "pkg" / "__pycache__").mkdir(parents=True)
    (src / "pkg" / "__pycache__" / "m.cpython-313.pyc").write_bytes(b"\0")
    (src / "pkg" / "m.py").write_text("x = 1\n")
    (src / "__pycache__").mkdir()
    with sandbox_for(src, ignore=shutil.ignore_patterns("__pycache__")) as sb:
        assert (sb / "pkg" / "m.py").exists()
        assert not any(sb.rglob("__pycache__"))
