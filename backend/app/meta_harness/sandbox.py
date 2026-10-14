"""Process-isolated sandbox for inner-loop tool execution.

Per Appendix C §C.6.3:
- Each task gets a fresh ``meta-harness-task-{uuid}/`` directory under the
  platform temp root (``/tmp`` on Linux/macOS, ``%TEMP%`` on Windows).
- Commands run with ``cwd=task_dir`` and a wall-clock timeout.
- rlimit 512MB RAM + 60s CPU on Unix.
- **Process isolation only** — no Docker, no network restriction, no
  filesystem jail, no binary allowlist. These are honest limits we
  surface in user-facing docs; production-grade isolation is roadmap.

What every sandboxed command gets, and why:

- **Its own process group** (``start_new_session`` on POSIX,
  ``CREATE_NEW_PROCESS_GROUP`` on Windows). A timeout kills the whole
  tree, not just the shell: ``sleep 37 & echo started`` used to leave
  the ``sleep`` running after the call returned. On POSIX the group is
  also reaped after a normal exit, so a background job cannot outlive
  the tool call that started it.
- **A scrubbed environment.** API keys, tokens, secrets, the Postgres
  DSN, W&B and Auth0 settings are dropped, so ``env`` run by a model
  cannot copy a credential into a trace or a recorded tape.
- **No stdin.** A command that waits for input fails instead of hanging
  until its timeout.
- **rlimits applied in the child, after fork.** ``preexec_fn`` is not
  safe to use from threads, and every tool call runs in a worker thread
  (``asyncio.to_thread``). The limits are instead set by a tiny Python
  trampoline that the child execs into before the real command, so
  nothing runs between fork and exec in the parent's address space.

The two layers:
- ``make_sandbox_dir`` / ``populate_sandbox`` / ``cleanup_sandbox`` /
  ``sandbox_for`` — sandbox lifecycle (create, copy task workspace, clean).
- ``run_in_sandbox`` — low-level exec used by ``tools.run_bash``, the
  fixed tool binaries (``rg``, ``git apply``), the inner loop's verify
  phase and the grader.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_IS_WINDOWS = sys.platform == "win32"

# rlimit support is Unix-only; on other platforms we no-op.
try:
    import resource as _resource  # noqa: F401 — presence check only

    _HAS_RLIMIT = True
except ImportError:  # Windows
    _HAS_RLIMIT = False


SANDBOX_PREFIX = "meta-harness-task-"
DEFAULT_RLIMIT_RAM = 512 * 1024 * 1024  # 512 MB
DEFAULT_RLIMIT_CPU = 60  # seconds

#: How long to wait for a killed tree's pipes to drain before giving up.
_DRAIN_TIMEOUT_S = 5


def sandbox_root() -> Path:
    """Return the platform temp root sandboxes are created under.

    ``tempfile.gettempdir()`` resolves ``/tmp`` on Linux/macOS and
    ``%TEMP%`` on Windows. Hardcoding ``/tmp`` produced ``C:\\tmp`` on
    Windows, which is outside the OS-managed temp tree and is not
    cleaned up by the platform.
    """
    return Path(tempfile.gettempdir())


def make_sandbox_dir() -> Path:
    """Create a fresh ``<temp>/meta-harness-task-{uuid}/`` and return it."""
    sandbox = sandbox_root() / f"{SANDBOX_PREFIX}{uuid.uuid4().hex}"
    sandbox.mkdir(parents=True, exist_ok=False)
    return sandbox


#: ``shutil.copytree`` ignore callable: ``(directory, names) -> names to skip``.
CopyIgnore = Callable[[str, list[str]], Iterable[str]]


def populate_sandbox(
    sandbox_dir: Path, source_workspace: Path, *, ignore: CopyIgnore | None = None
) -> None:
    """Copy a task's pristine workspace into the sandbox.

    ``ignore`` follows ``shutil.copytree``'s convention and applies at
    every level, the top one included.
    """
    if not source_workspace.is_dir():
        raise ValueError(f"source workspace is not a directory: {source_workspace}")
    entries = list(source_workspace.iterdir())
    skipped = (
        set(ignore(str(source_workspace), [e.name for e in entries])) if ignore else set()
    )
    for entry in entries:
        if entry.name in skipped:
            continue
        target = sandbox_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, ignore=ignore)
        else:
            shutil.copy2(entry, target)


def cleanup_sandbox(sandbox_dir: Path) -> None:
    """Remove a sandbox directory. Idempotent and tolerant of missing dir."""
    shutil.rmtree(sandbox_dir, ignore_errors=True)


@contextmanager
def sandbox_for(
    source_workspace: Path, *, ignore: CopyIgnore | None = None
) -> Iterator[Path]:
    """Context manager: fresh sandbox, populated from ``source_workspace``,
    cleaned up on exit.
    """
    sandbox = make_sandbox_dir()
    try:
        populate_sandbox(sandbox, source_workspace, ignore=ignore)
        yield sandbox
    finally:
        cleanup_sandbox(sandbox)


# ── environment ───────────────────────────────────────────────────────

#: Variables never passed to a sandboxed command, matched case-insensitively
#: (Windows environment names are case-insensitive).
_SECRET_ENV_NAMES = frozenset({"ANTHROPIC_API_KEY", "POSTGRES_DSN", "DATABASE_URL"})
_SECRET_ENV_PREFIXES = ("WANDB_", "AUTH0_")
_SECRET_ENV_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_SECRET_KEY",
    "_ACCESS_KEY",
    "_PASSWORD",
)


def is_secret_env_name(name: str) -> bool:
    """True when ``name`` looks like a credential or a private endpoint."""
    upper = name.upper()
    return (
        upper in _SECRET_ENV_NAMES
        or upper.startswith(_SECRET_ENV_PREFIXES)
        or upper.endswith(_SECRET_ENV_SUFFIXES)
    )


def scrubbed_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """``base`` (default: this process' environment) minus every secret."""
    source = os.environ if base is None else base
    return {k: v for k, v in source.items() if not is_secret_env_name(k)}


# ── execution ─────────────────────────────────────────────────────────

#: Runs in the child: set the limits, then exec the real command. Errors
#: setting a limit are ignored (best-effort, as before); an exec failure
#: is reported the way a shell reports a missing command.
_RLIMIT_TRAMPOLINE = (
    "import os, resource, sys\n"
    "ram, cpu = int(sys.argv[1]), int(sys.argv[2])\n"
    "for lim, val in ((resource.RLIMIT_AS, ram), (resource.RLIMIT_CPU, cpu)):\n"
    "    if val > 0:\n"
    "        try:\n"
    "            resource.setrlimit(lim, (val, val))\n"
    "        except (ValueError, OSError):\n"
    "            pass\n"
    "try:\n"
    "    os.execvp(sys.argv[3], sys.argv[3:])\n"
    "except OSError as exc:\n"
    "    sys.stderr.write(f'{sys.argv[3]}: {exc}\\n')\n"
    "    sys.exit(127)\n"
)


def _limited_argv(argv: list[str]) -> list[str]:
    """Prefix ``argv`` with the rlimit trampoline where rlimits exist.

    macOS's ``RLIMIT_AS`` enforcement is unreliable for Python child
    processes (Python's own address-space footprint can already exceed
    the cap before the child runs anything), so only the CPU limit is
    applied there and the wall-clock timeout does the rest.
    """
    if not _HAS_RLIMIT or _IS_WINDOWS or not sys.executable:
        return argv
    ram = 0 if sys.platform == "darwin" else DEFAULT_RLIMIT_RAM
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        _RLIMIT_TRAMPOLINE,
        str(ram),
        str(DEFAULT_RLIMIT_CPU),
        *argv,
    ]


def _popen_args(
    command: str | Sequence[str], *, rlimits: bool
) -> tuple[str | list[str], bool]:
    """``(args, shell)`` for ``Popen``: a string runs through the shell."""
    if isinstance(command, str):
        if _IS_WINDOWS:
            return command, True
        argv = ["/bin/sh", "-c", command]
    else:
        argv = [str(part) for part in command]
    return (_limited_argv(argv) if rlimits else argv), False


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    """Kill ``proc`` and everything it started. Never raises."""
    if _IS_WINDOWS:
        try:
            subprocess.run(  # noqa: S603,S607 — fixed system binary
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.kill()
        except OSError:
            pass
        return
    try:
        # ``start_new_session`` made the child a group leader, so its pid
        # is the group id even after the leader itself has exited.
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _as_text(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return str(stream)


def run_in_sandbox(
    sandbox_dir: Path,
    command: str | Sequence[str],
    *,
    timeout_sec: float = 30,
    rlimits: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command in the sandbox and return its captured output.

    ``command`` is a shell string (run by ``/bin/sh -c``, or ``cmd.exe``
    on Windows) or an argv list (no shell). Raises
    ``subprocess.TimeoutExpired`` — carrying whatever output was captured
    — after killing the command's whole process tree.

    ``rlimits=False`` skips the memory/CPU caps for trusted fixed
    binaries; process-group cleanup, the scrubbed environment and the
    closed stdin apply to every command regardless.
    """
    args, shell = _popen_args(command, rlimits=rlimits)
    platform_kwargs: dict[str, object] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
        if _IS_WINDOWS
        else {"start_new_session": True}
    )
    proc = subprocess.Popen(  # noqa: S602,S603 — sandboxed, see module docstring
        args,
        shell=shell,
        cwd=sandbox_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=scrubbed_env(),
        **platform_kwargs,  # type: ignore[arg-type]
    )
    with proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            _kill_process_tree(proc)
            try:
                # Retrying communicate() after a timeout loses no output.
                stdout, stderr = proc.communicate(timeout=_DRAIN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # Something escaped the group and still holds a pipe.
                stdout, stderr = _as_text(exc.stdout), _as_text(exc.stderr)
            raise subprocess.TimeoutExpired(
                command, timeout_sec, output=stdout, stderr=stderr
            ) from None
        except BaseException:
            _kill_process_tree(proc)
            raise
        if not _IS_WINDOWS:
            # Reap background jobs the command left running.
            _kill_process_tree(proc)
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
