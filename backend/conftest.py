"""Backend pytest config: a temp root that is chosen, not assumed, plus .env.

**`.env`** — lets ``test_inner.py``'s live LLM test pick up
``ANTHROPIC_API_KEY`` without the user having to export it in their shell.

**The temp root.** ``tmp_path`` is the suite's most-used fixture, and where
it lives has broken this suite twice in two different ways. Both are worth
recording, because the fix is shaped by them:

1. pytest's default root is ``tempfile.gettempdir()/pytest-of-<user>``. On
   one Windows box that *specific directory* had a broken ACL, and every
   ``tmp_path`` test errored with ``PermissionError: [WinError 5]`` — 159
   of them, reading as 159 broken tests rather than one broken directory
   the suite did not have to use.
2. Moving the root inside the checkout fixed that and introduced a worse
   failure: the repository lives under OneDrive, and sync holds brief locks
   on files it is uploading. A measured hammer of 400 atomic
   write-and-replace cycles failed 3 times inside the checkout and 0 times
   in the system temp — enough, across a suite that does thousands, to fail
   roughly one full run in three with
   ``PermissionError: [WinError 5] ... os.replace``. An intermittently red
   suite is worse than a squarely broken directory.

So the root is *probed* rather than assumed. Candidates are tried in order
and each must survive a real create-write-replace before it is accepted:

1. ``$META_HARNESS_PYTEST_BASETEMP`` — explicit control, for anyone whose
   machine disagrees with all of the below.
2. ``<system temp>/meta-harness-pytest`` — not sync-backed, and a directory
   of our own, so a broken ``pytest-of-<user>`` left by some other tool is
   never touched. This is the default, and it is what makes "the broken
   global temp directory is not required" true: it is not used.
3. ``backend/.pytest-tmp`` — repository-local fallback, for a machine whose
   system temp is genuinely unusable. Correct but sync-exposed here, which
   is why it is the fallback rather than the default.

**Each session gets its own directory under the chosen root.** Pointing
``--basetemp`` at one fixed path looks simpler, but pytest *wipes* the
basetemp it is given at session start, so a single lingering handle from a
previous run makes the next session fail in its entirety with
``PermissionError: [WinError 32] ... being used by another process``. That
also happened here: 375 errors, none about the code under test. pytest's own
answer is numbered directories that rotate, and this reproduces it — a fresh
``session-<pid>-<counter>`` per run, older ones reaped best-effort, so a
locked leftover is skipped rather than fatal.

``--basetemp`` given explicitly on the command line still wins.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

#: Our own directory under the platform temp. Deliberately not pytest's
#: ``pytest-of-<user>``, which another tool may have left unusable.
SYSTEM_TEMP_DIRNAME = "meta-harness-pytest"

#: Repository-local fallback root. Gitignored.
BASETEMP_DIRNAME = ".pytest-tmp"

#: Environment override, tried before anything else.
BASETEMP_ENV_VAR = "META_HARNESS_PYTEST_BASETEMP"

#: Previous sessions kept for post-mortem on a failure. Beyond this the
#: oldest are removed — best-effort, because one that is still locked must
#: not break the session that is starting.
KEEP_SESSIONS = 3


def _is_usable(root: Path) -> bool:
    """Can this directory actually host ``tmp_path``?

    Checked by doing the things the suite does — create a directory, write
    a file, and atomically replace it — rather than by inspecting
    permissions, because the failures that matter here (a broken ACL, a
    sync client holding a handle) do not show up in a mode bit.
    """
    probe = root / f".probe-{os.getpid()}"
    try:
        probe.mkdir(parents=True, exist_ok=True)
        target = probe / "probe.txt"
        target.write_text("probe", encoding="utf-8")
        fd, staged = tempfile.mkstemp(dir=str(probe), prefix=".probe.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("probe")
        os.replace(staged, target)
    except OSError:
        return False
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    return True


def _candidate_roots(backend_root: Path) -> list[Path]:
    override = os.environ.get(BASETEMP_ENV_VAR)
    roots = [Path(override)] if override else []
    roots.append(Path(tempfile.gettempdir()) / SYSTEM_TEMP_DIRNAME)
    roots.append(backend_root / BASETEMP_DIRNAME)
    return roots


def _reap_old_sessions(root: Path) -> None:
    """Delete all but the newest ``KEEP_SESSIONS`` session directories.

    Every failure here is ignored on purpose. Cleanup is a courtesy; a
    directory that cannot be removed is a leftover, not a reason to fail a
    run that has not started yet.
    """
    try:
        sessions = sorted(
            (path for path in root.glob("session-*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        return
    for stale in sessions[:-KEEP_SESSIONS] if len(sessions) > KEEP_SESSIONS else []:
        shutil.rmtree(stale, ignore_errors=True)


def _session_basetemp(backend_root: Path) -> Path:
    """A directory this session alone owns, under the first usable root."""
    roots = _candidate_roots(backend_root)
    for root in roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if not _is_usable(root):
            continue
        _reap_old_sessions(root)
        # The pid alone is not unique enough: pytest wipes whatever path it
        # is handed, and a recycled pid would wipe a concurrent run's
        # directory.
        return root / f"session-{os.getpid()}-{time.time_ns()}"
    # Nothing was usable. Return the last candidate so the failure names a
    # path, rather than silently falling back to the default that item 1 of
    # the module docstring says may be broken.
    return roots[-1] / f"session-{os.getpid()}-{time.time_ns()}"


def pytest_configure(config):
    backend_root = Path(__file__).resolve().parent
    repo_root = backend_root.parent

    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    if not config.option.basetemp:
        config.option.basetemp = str(_session_basetemp(backend_root))


@pytest.fixture(scope="session", autouse=True)
def _pin_basetemp_git_attributes(tmp_path_factory: pytest.TempPathFactory):
    """Give fixtures git's no-repository line-ending behaviour.

    Only needed when the temp root landed *inside* the checkout, where
    ``git`` invocations under ``tmp_path`` would otherwise resolve this
    repository's ``.gitattributes`` (which pins ``eol=lf``). ``text=auto``
    is what git uses with no repository and no configuration, and an
    attributes file in a subdirectory overrides its parents.

    ``-text`` would *not* do: it disables the CRLF reconciliation that
    ``git apply`` performs on an LF patch against a CRLF file, which is the
    behaviour ``tests/test_tools.py`` exists to protect.
    """
    basetemp = tmp_path_factory.getbasetemp()
    repo_root = Path(__file__).resolve().parents[1]
    if repo_root in basetemp.resolve().parents:
        (basetemp / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
