"""Time-travel branch helpers (BUILD_ORDER step 9).

This module is intentionally backend-internal. Step 10 can call these
functions from FastAPI routers later; no HTTP or SSE assumptions live here.

The core primitive is ``worktree_add``:
- read a parent checkpoint from LangGraph state history,
- create a new thread id,
- apply user state modifications with ``graph.aupdate_state``,
- resume the fork with ``graph.ainvoke(None, fork_config)`` in an
  ``asyncio.create_task``.

**Branch history is durable; branch execution is not.** Every metadata
transition is mirrored to ``runs/<run_id>/branches.json``, so restarting
the API still reconstructs the full branch tree. The in-process
``branch_registry`` holds live ``asyncio.Task`` handles, and those do
*not* survive a restart — a branch that was ``running`` when the process
died is reported as ``interrupted``, never as still running.

More than one process writes ``branches.json``: the API and any
``meta-harness fork`` running beside it. Every read-modify-write holds a
cross-process lock, and a running branch records the pid and host that
own it plus a heartbeat, so a branch another live process is driving is
reported as running, not interrupted, and cannot be marked cancelled by
a process that has no way to stop it.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import os
import socket
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from langgraph.graph import END, START

from app.meta_harness import runs as runs_mod

log = logging.getLogger("meta_harness.branches")

BranchStatus = Literal[
    "created", "running", "completed", "failed", "cancelled", "interrupted"
]

INPUT_NODE = "__input__"

#: Statuses of a branch that some process may still be driving.
ACTIVE_STATUSES = frozenset({"created", "running"})

#: How long a cancel waits for a branch task to actually stop. Bounded,
#: so one stuck task cannot hang the API request (or shutdown) behind it.
CANCEL_TIMEOUT_S = 10.0

#: A running branch refreshes ``heartbeat_at`` this often...
HEARTBEAT_INTERVAL_S = 10.0
#: ...and is considered dead once its heartbeat is older than this.
HEARTBEAT_STALE_S = 45.0

#: How long a writer waits for another process's ``branches.json`` lock.
LOCK_TIMEOUT_S = 10.0

_HOSTNAME = socket.gethostname()


@dataclass
class BranchMetadata:
    """In-process metadata for one forked LangGraph thread."""

    branch_id: str
    run_id: str
    thread_id: str
    parent_thread_id: str | None
    parent_checkpoint_id: str | None
    status: BranchStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    cancelled_at: str | None = None
    error: str | None = None
    mods: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    result: dict[str, Any] | None = None
    #: Candidate the branch inherited at its fork point, for lineage.
    parent_candidate: str | None = None
    #: True while an in-process task is driving this branch. Never
    #: persisted as True — see ``_rehydrate_status``.
    live: bool = False
    #: The process driving the branch, and when it last proved it was
    #: alive. Lets another process tell a live branch from a dead one.
    owner_pid: int | None = None
    owner_host: str | None = None
    heartbeat_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable shape for API/CLI callers."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BranchMetadata":
        """Rebuild from persisted JSON, ignoring unknown/legacy keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class CheckpointRecord:
    """Projected LangGraph checkpoint history row.

    Shape mirrors ``INTERFACES.md`` section 6.2 while staying independent
    of FastAPI.
    """

    checkpoint_id: str
    thread_id: str
    ts: str | None
    node: str | None
    iteration: int | None
    values_summary: dict[str, Any]
    parent_checkpoint_id: str | None
    next: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable shape for API/CLI callers."""
        data = asdict(self)
        data["next"] = list(self.next)
        return data


branch_registry: dict[str, asyncio.Task[Any]] = {}
branch_metadata: dict[str, BranchMetadata] = {}

#: Where each run's branch history is persisted. Set by the API/CLI so
#: this module keeps no opinion about repo layout.
_runs_root: Path | None = None

BRANCHES_FILENAME = "branches.json"


def set_runs_root(path: Path | None) -> None:
    """Point branch persistence at a ``runs/`` directory (or disable it)."""
    global _runs_root
    _runs_root = Path(path) if path is not None else None


def get_runs_root() -> Path | None:
    """The configured ``runs/`` root, or ``None`` when persistence is off."""
    return _runs_root


def _branches_path(run_id: str) -> Path | None:
    """``runs/<run_id>/branches.json``, or ``None`` if it can't be resolved.

    ``run_id`` reaches this module from HTTP paths, so it is validated
    as an artifact name before it is joined — a traversal attempt
    resolves to ``None`` rather than a path outside ``runs/``.
    """
    if _runs_root is None:
        return None
    try:
        runs_mod.validate_artifact_name(run_id, kind="run")
    except ValueError:
        return None
    root = _runs_root.resolve()
    run_dir = (root / run_id).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError:
        return None
    if not run_dir.is_dir():
        return None
    return run_dir / BRANCHES_FILENAME


def _read_persisted(run_id: str) -> dict[str, dict[str, Any]]:
    path = _branches_path(run_id)
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    branches = data.get("branches") if isinstance(data, dict) else None
    if not isinstance(branches, list):
        return {}
    return {
        str(b["thread_id"]): b
        for b in branches
        if isinstance(b, dict) and b.get("thread_id")
    }


if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _try_lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def branches_lock(path: Path, *, timeout: float | None = None) -> Iterator[None]:
    """Hold an exclusive, cross-process lock on ``path``'s sibling lockfile.

    ``flock`` on POSIX and ``msvcrt.locking`` on Windows; both are
    released by the OS if the holder dies, so a crashed writer cannot
    leave a stale lock behind. Each acquisition opens its own descriptor,
    which makes the lock exclusive between threads of one process too.
    Raises ``TimeoutError`` (an ``OSError``) if it cannot be acquired.
    """
    lock_path = path.with_name(f".{path.name}.lock")
    deadline = time.monotonic() + (LOCK_TIMEOUT_S if timeout is None else timeout)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while True:
            try:
                _try_lock(fd)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for {lock_path.name}"
                    ) from None
                time.sleep(0.01)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def persist_branch(metadata: BranchMetadata) -> None:
    """Mirror one branch's metadata into ``runs/<run_id>/branches.json``.

    The read-modify-write runs under ``branches_lock``: a CLI fork and the
    API writing the same file used to lose each other's records. The
    record is snapshotted *inside* the lock, so whichever writer goes last
    writes the newest state rather than one captured before it waited.

    Best-effort: branch history is an observability artifact, and a
    filesystem hiccup must never abort a running branch.
    """
    path = _branches_path(metadata.run_id)
    if path is None:
        return
    try:
        with branches_lock(path):
            existing = _read_persisted(metadata.run_id)
            record = metadata.to_dict()
            # ``live`` is process state. Persisting it True would make a
            # branch look running forever after a crash.
            record["live"] = False
            existing[metadata.thread_id] = record
            runs_mod.write_json_atomic(
                path,
                {
                    "run_id": metadata.run_id,
                    "updated_at": _now(),
                    "branches": sorted(
                        existing.values(),
                        key=lambda b: str(b.get("created_at", "")),
                    ),
                },
            )
    except OSError as exc:
        log.warning(
            "could not persist branch %s (%s); branch history on disk is stale",
            metadata.thread_id,
            exc,
        )


def _parse_ts(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _pid_alive(pid: int) -> bool:
    """Best-effort "does this pid exist on this host"."""
    if pid <= 0:
        return False
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        # ``os.kill(pid, 0)`` *terminates* the process on Windows. Ask the
        # kernel instead, and fall back to trusting the heartbeat.
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return True
                return code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _owner_alive(record: dict[str, Any]) -> bool:
    """Is the process that owns this branch record demonstrably still alive?

    This process answers for itself from its task registry. Another
    process on this host must have a fresh heartbeat *and* an existing
    pid (a recycled pid alone proves nothing); a process on another host
    can only be judged by its heartbeat.
    """
    pid = record.get("owner_pid")
    beat = _parse_ts(record.get("heartbeat_at"))
    if not isinstance(pid, int) or beat is None:
        return False
    host = record.get("owner_host")
    fresh = (datetime.now(timezone.utc) - beat).total_seconds() < HEARTBEAT_STALE_S
    if host not in (None, _HOSTNAME):
        return fresh
    if pid == os.getpid():
        task = branch_registry.get(str(record.get("thread_id")))
        return task is not None and not task.done()
    return fresh and _pid_alive(pid)


def owned_elsewhere(metadata: BranchMetadata) -> bool:
    """True if another live process is driving this branch."""
    if metadata.status not in ACTIVE_STATUSES or metadata.owner_pid is None:
        return False
    if metadata.owner_pid == os.getpid() and metadata.owner_host in (None, _HOSTNAME):
        return False
    return _owner_alive(metadata.to_dict())


class BranchOwnedElsewhere(RuntimeError):
    """A branch another live process is driving; this one cannot stop it."""

    def __init__(self, metadata: BranchMetadata) -> None:
        super().__init__(
            f"branch {metadata.thread_id} is running in another process "
            f"(pid {metadata.owner_pid} on {metadata.owner_host}); "
            "cancel it from that process"
        )
        self.metadata = metadata


def _rehydrate_status(record: dict[str, Any]) -> dict[str, Any]:
    """A persisted ``running`` branch is an interrupted one, unless its
    owner is demonstrably alive.

    Its asyncio task died with the process. Reporting it as running
    would be a lie the UI then shows as a live spinner. A branch another
    process (a ``meta-harness fork`` beside the API) is still driving,
    with a fresh heartbeat, really is running and is reported so.
    """
    record = dict(record)
    if record.get("status") in ACTIVE_STATUSES and not _owner_alive(record):
        record["status"] = "interrupted"
        record["error"] = record.get("error") or (
            "process exited while the branch was running; "
            "checkpoint history is intact, the task is not"
        )
    return record


def load_persisted_branches(run_id: str) -> list[BranchMetadata]:
    """Read a run's branch history from disk, independent of memory."""
    return [
        BranchMetadata.from_dict(_rehydrate_status(record))
        for record in sorted(
            _read_persisted(run_id).values(),
            key=lambda b: str(b.get("created_at", "")),
        )
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configurable(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    return config.get("configurable", {})


def _checkpoint_id(config: dict[str, Any] | None) -> str | None:
    return _configurable(config).get("checkpoint_id")


def _thread_id(config: dict[str, Any] | None) -> str | None:
    raw = _configurable(config).get("thread_id")
    return str(raw) if raw is not None else None


def _values_summary(values: dict[str, Any]) -> dict[str, Any]:
    """Small summary safe for checkpoint list views."""
    candidates = values.get("candidates") or []
    return {
        "iteration": values.get("iteration"),
        "budget_remaining": values.get("budget_remaining"),
        "best_candidate": values.get("best_candidate"),
        "frontier_count": len(values.get("frontier") or []),
        "candidates_count": len(candidates),
        "proposer_prior": values.get("proposer_prior"),
    }


def _snapshot_values(snapshot: Any) -> dict[str, Any]:
    values = getattr(snapshot, "values", {}) or {}
    return values if isinstance(values, dict) else {}


def _snapshot_next(snapshot: Any) -> tuple[str, ...]:
    return tuple(getattr(snapshot, "next", ()) or ())


async def get_state_history(
    graph: Any,
    *,
    thread_id: str,
    limit: int | None = None,
) -> list[CheckpointRecord]:
    """Return projected checkpoint history for one LangGraph thread.

    The order matches ``graph.aget_state_history``: newest checkpoint first.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshots = [
        snapshot
        async for snapshot in graph.aget_state_history(config, limit=limit)
    ]
    by_id = {_checkpoint_id(s.config): s for s in snapshots}
    records: list[CheckpointRecord] = []
    for snapshot in snapshots:
        checkpoint_id = _checkpoint_id(snapshot.config)
        if checkpoint_id is None:
            continue
        parent_checkpoint_id = _checkpoint_id(getattr(snapshot, "parent_config", None))
        parent_snapshot = by_id.get(parent_checkpoint_id)
        node = _infer_snapshot_node(snapshot, parent_snapshot)
        values = _snapshot_values(snapshot)
        records.append(
            CheckpointRecord(
                checkpoint_id=checkpoint_id,
                thread_id=_thread_id(snapshot.config) or thread_id,
                ts=getattr(snapshot, "created_at", None),
                node=node,
                iteration=values.get("iteration"),
                values_summary=_values_summary(values),
                parent_checkpoint_id=parent_checkpoint_id,
                next=_snapshot_next(snapshot),
                metadata=dict(getattr(snapshot, "metadata", {}) or {}),
            )
        )
    return records


async def get_checkpoint_state(
    graph: Any,
    *,
    thread_id: str,
    checkpoint_id: str,
) -> dict[str, Any]:
    """Return the full state at one checkpoint."""
    snapshot = await _find_snapshot(
        graph,
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
    )
    return copy.deepcopy(_snapshot_values(snapshot))


async def worktree_add(
    graph: Any,
    *,
    run_id: str,
    parent_thread_id: str,
    parent_checkpoint_id: str,
    mods: dict[str, Any] | None = None,
    name: str | None = None,
    recursion_limit: int = 200,
) -> tuple[BranchMetadata, asyncio.Task[Any]]:
    """Fork a checkpoint into a new concurrent branch.

    Returns ``(metadata, task)``. The task is also stored in
    ``branch_registry[metadata.thread_id]``.
    """
    mods = dict(mods or {})
    parent_snapshot = await _find_snapshot(
        graph,
        thread_id=parent_thread_id,
        checkpoint_id=parent_checkpoint_id,
    )
    as_node = await _infer_as_node_for_fork(graph, parent_snapshot)
    fork_values = copy.deepcopy(_snapshot_values(parent_snapshot))
    fork_values.update(mods)

    branch_id = uuid.uuid4().hex[:8]
    thread_id = f"{parent_thread_id}.fork.{branch_id}"
    metadata = BranchMetadata(
        branch_id=branch_id,
        run_id=run_id,
        thread_id=thread_id,
        parent_thread_id=parent_thread_id,
        parent_checkpoint_id=parent_checkpoint_id,
        status="created",
        created_at=_now(),
        mods=mods,
        name=name,
        parent_candidate=fork_values.get("best_candidate"),
        owner_pid=os.getpid(),
        owner_host=_HOSTNAME,
        heartbeat_at=_now(),
    )
    branch_metadata[thread_id] = metadata
    # File IO (fsync, a lock another process may hold) stays off the
    # event loop: this runs inside the fork request handler.
    await asyncio.to_thread(persist_branch, metadata)

    try:
        fork_config = await graph.aupdate_state(
            {"configurable": {"thread_id": thread_id}},
            fork_values,
            as_node=as_node,
        )
    except BaseException as exc:
        # Never leave a branch persisted as "created" with nothing behind it.
        metadata.status = "failed"
        metadata.finished_at = _now()
        metadata.error = str(exc) or type(exc).__name__
        persist_branch(metadata)
        raise

    if metadata.status == "cancelled":
        # Cancelled while the fork point was being written: do not start it.
        task = asyncio.create_task(asyncio.sleep(0), name=f"worktree:{thread_id}")
        branch_registry[thread_id] = task
        return metadata, task

    metadata.status = "running"
    metadata.started_at = _now()
    metadata.live = True
    metadata.heartbeat_at = metadata.started_at
    # Synchronous, and no await until the task is registered: a cancel
    # (or a fresh re-run's liveness check) must never observe a running
    # branch it cannot find a task for.
    persist_branch(metadata)

    task = asyncio.create_task(
        _run_branch(graph, metadata, fork_config, recursion_limit),
        name=f"worktree:{thread_id}",
    )
    task.add_done_callback(_consume_task_exception)
    branch_registry[thread_id] = task
    return metadata, task


async def cancel_branch(
    thread_id: str, *, run_id: str | None = None, timeout: float | None = None
) -> BranchMetadata:
    """Cancel a branch task if this process owns one, and mark it cancelled.

    See ``cancel_branch_outcome``, which also says whether anything
    changed. This wrapper keeps the original return shape.

    Resolved through ``get_branch`` rather than the in-memory registry
    alone, because a branch that outlived the process that created it
    exists only on disk. ``list_branches`` and ``get_branch`` already
    merge both sources so a restarted API reports the full tree; this
    used to be the one path that did not, and it failed hard:

        DELETE /runs/<id>  ->  cancel_branch(<persisted branch>)
                           ->  KeyError: unknown branch thread_id
                           ->  HTTP 500, non-JSON body

    That is reachable only after a real restart, which is why it stayed
    hidden — the acceptance ladder's "survives a backend restart" check
    was not restarting anything.

    Having no task here is not an error: the branch is not running in
    this process, so there is nothing to cancel and marking the metadata
    cancelled is the whole job. A thread id unknown to *both* memory and
    disk still raises ``KeyError``, which is what the API's 404 path
    depends on.
    """
    metadata, _changed = await cancel_branch_outcome(
        thread_id, run_id=run_id, timeout=timeout
    )
    return metadata


async def cancel_branch_outcome(
    thread_id: str, *, run_id: str | None = None, timeout: float | None = None
) -> tuple[BranchMetadata, bool]:
    """Stop a branch and report whether its status actually changed.

    - A task this process drives is cancelled **and awaited**, for at most
      ``timeout`` seconds (``CANCEL_TIMEOUT_S`` by default). The task's
      own ``CancelledError`` handler records the cancellation, so a task
      that finished first stays ``completed`` and nothing changes. A task
      that has not stopped when the wait ends is left ``running``; the
      cancel is still in flight and will be recorded when it lands.
    - A branch another live process drives raises ``BranchOwnedElsewhere``:
      this process cannot stop it, so it must not claim to have.
    - A terminal or ``interrupted`` branch is left exactly as it is.
      Relabelling it ``cancelled`` stops nothing.

    ``run_id`` scopes the lookup: a branch of another run is unknown here
    and raises ``KeyError``.
    """
    metadata = get_branch(thread_id, run_id=run_id)
    if metadata is None:
        raise KeyError(f"unknown branch thread_id: {thread_id}")
    before = metadata.status
    task = branch_registry.get(thread_id)
    if task is not None and not task.done():
        task.cancel()
        await asyncio.wait(
            {task}, timeout=CANCEL_TIMEOUT_S if timeout is None else timeout
        )
        if task.done() and metadata.status in ACTIVE_STATUSES:
            # The task ended without running ``_run_branch``'s handler
            # (cancelled before its first step).
            _mark_cancelled(metadata)
    elif before in ACTIVE_STATUSES:
        if owned_elsewhere(metadata):
            raise BranchOwnedElsewhere(metadata)
        if branch_metadata.get(thread_id) is metadata:
            # An in-process record with no task to stop (cancelled while
            # its fork point was still being written).
            _mark_cancelled(metadata)
    changed = before != "cancelled" and metadata.status == "cancelled"
    return metadata, changed


def is_live(metadata: BranchMetadata) -> bool:
    """True if some process — this one or another live one — drives it."""
    task = branch_registry.get(metadata.thread_id)
    if task is not None and not task.done():
        return True
    if (
        branch_metadata.get(metadata.thread_id) is metadata
        and metadata.status in ACTIVE_STATUSES
    ):
        return True
    return owned_elsewhere(metadata)


def live_branches(run_id: str) -> list[BranchMetadata]:
    """Every branch of ``run_id`` still being driven by a live process."""
    return [b for b in list_branches(run_id=run_id) if is_live(b)]


def forget_run(run_id: str) -> None:
    """Drop this process's in-memory branch records for ``run_id``.

    Used when a run is re-created fresh: its old branches must not show
    up in the new run. Handles to tasks that are still running are kept
    (callers refuse a fresh re-run while any branch is live).
    """
    for thread_id, metadata in list(branch_metadata.items()):
        if metadata.run_id != run_id:
            continue
        task = branch_registry.get(thread_id)
        if task is not None and not task.done():
            continue
        branch_metadata.pop(thread_id, None)
        branch_registry.pop(thread_id, None)


def list_branches(*, run_id: str | None = None) -> list[BranchMetadata]:
    """List branch metadata, merging in-process state with persisted history.

    In-process records win: they are strictly fresher than what is on
    disk. Persisted-only records fill in branches created by an earlier
    process, so a restarted API still reports the full branch tree.
    """
    merged: dict[str, BranchMetadata] = {}
    if run_id is not None:
        for persisted in load_persisted_branches(run_id):
            merged[persisted.thread_id] = persisted
    else:
        for known_run in {b.run_id for b in branch_metadata.values()}:
            for persisted in load_persisted_branches(known_run):
                merged[persisted.thread_id] = persisted

    for live in branch_metadata.values():
        if run_id is not None and live.run_id != run_id:
            continue
        merged[live.thread_id] = live
    return sorted(merged.values(), key=lambda b: b.created_at)


def get_branch(thread_id: str, *, run_id: str | None = None) -> BranchMetadata | None:
    """Return branch metadata by thread id, from memory or persisted history.

    With ``run_id``, only a branch *of that run* is returned: a thread id
    belonging to another run is unknown here.
    """
    live = branch_metadata.get(thread_id)
    if live is not None:
        if run_id is not None and live.run_id != run_id:
            return None
        return live
    if run_id is None:
        # Fork thread ids are "<parent>.fork.<id>"; the run id is the
        # leftmost segment of that chain.
        run_id = thread_id.split(".fork.")[0]
    for persisted in load_persisted_branches(run_id):
        if persisted.thread_id == thread_id:
            return persisted
    return None


def reconstruct_trajectory(run_id: str) -> dict[str, Any]:
    """Build the run's branch tree from memory + persisted history.

    Survives an API restart: branch records come from
    ``runs/<run_id>/branches.json`` when the in-process registry is
    empty. ``live`` distinguishes a branch this process is currently
    driving from one that merely has persisted history.
    """
    branches = list_branches(run_id=run_id)
    threads: dict[str, dict[str, Any]] = {
        run_id: {
            "thread_id": run_id,
            "run_id": run_id,
            "parent_thread_id": None,
            "parent_checkpoint_id": None,
            "status": "root",
            "branch_id": None,
            "name": "root",
            "live": run_id in branch_registry,
        }
    }
    edges: list[dict[str, Any]] = []
    for branch in branches:
        record = branch.to_dict()
        task = branch_registry.get(branch.thread_id)
        record["live"] = task is not None and not task.done()
        threads[branch.thread_id] = record
        edges.append(
            {
                "source": branch.parent_thread_id,
                "target": branch.thread_id,
                "parent_checkpoint_id": branch.parent_checkpoint_id,
                "parent_candidate": branch.parent_candidate,
            }
        )
    return {
        "run_id": run_id,
        "threads": list(threads.values()),
        "edges": edges,
    }


async def cancel_all_branches(timeout: float | None = None) -> None:
    """Best-effort cleanup helper for tests and shutdown hooks.

    Cancels every task this process drives and waits for them together,
    for at most ``timeout`` seconds, so shutdown cannot hang on one task.
    """
    pending = [t for t in branch_registry.values() if not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(
            pending, timeout=CANCEL_TIMEOUT_S if timeout is None else timeout
        )
    for thread_id, task in list(branch_registry.items()):
        metadata = branch_metadata.get(thread_id)
        if metadata and metadata.status == "running" and task.done():
            _mark_cancelled(metadata)


def clear_branch_state() -> None:
    """Clear in-process registries. Call only after tasks are stopped."""
    branch_registry.clear()
    branch_metadata.clear()


async def _heartbeat(metadata: BranchMetadata) -> None:
    """Refresh ``heartbeat_at`` on disk while the branch runs."""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        metadata.heartbeat_at = _now()
        try:
            await asyncio.to_thread(persist_branch, metadata)
        except Exception:  # noqa: BLE001 - a missed beat must not kill the branch
            pass


async def _run_branch(
    graph: Any,
    metadata: BranchMetadata,
    fork_config: dict[str, Any],
    recursion_limit: int,
) -> dict[str, Any]:
    heartbeat = asyncio.create_task(
        _heartbeat(metadata), name=f"heartbeat:{metadata.thread_id}"
    )
    try:
        final = await graph.ainvoke(
            None,
            config={**fork_config, "recursion_limit": recursion_limit},
        )
    except asyncio.CancelledError:
        _mark_cancelled(metadata)
        raise
    except Exception as exc:  # noqa: BLE001 - branch metadata captures failures
        metadata.status = "failed"
        metadata.finished_at = _now()
        metadata.error = str(exc)
        metadata.live = False
        persist_branch(metadata)
        raise
    finally:
        heartbeat.cancel()
    # Terminal writes are synchronous on purpose: with no await between
    # the status change and the task finishing, nobody can see a
    # "completed" branch whose task is still running — or have a late
    # write land in a run directory that was re-created in between.
    metadata.status = "completed"
    metadata.finished_at = _now()
    metadata.live = False
    metadata.result = final if isinstance(final, dict) else {"result": final}
    persist_branch(metadata)
    return final


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    """Mark task exceptions observed when API callers do not await them."""
    if task.cancelled():
        return
    task.exception()


def _mark_cancelled(metadata: BranchMetadata) -> None:
    metadata.status = "cancelled"
    metadata.cancelled_at = _now()
    metadata.finished_at = metadata.finished_at or metadata.cancelled_at
    metadata.live = False
    persist_branch(metadata)


async def _find_snapshot(
    graph: Any,
    *,
    thread_id: str,
    checkpoint_id: str,
) -> Any:
    config = {"configurable": {"thread_id": thread_id}}
    async for snapshot in graph.aget_state_history(config):
        if _checkpoint_id(snapshot.config) == checkpoint_id:
            return snapshot
    raise KeyError(f"checkpoint {checkpoint_id!r} not found in thread {thread_id!r}")


async def _infer_as_node_for_fork(graph: Any, snapshot: Any) -> str | None:
    metadata = dict(getattr(snapshot, "metadata", {}) or {})
    if metadata.get("source") == "input":
        return INPUT_NODE

    parent_config = getattr(snapshot, "parent_config", None)
    if parent_config is None:
        return None

    parent_snapshot = await graph.aget_state(parent_config)
    parent_next = _snapshot_next(parent_snapshot)
    if len(parent_next) != 1:
        return None
    previous_node = parent_next[0]
    if previous_node == START:
        return INPUT_NODE
    if previous_node == END:
        return None
    return previous_node


def _infer_snapshot_node(snapshot: Any, parent_snapshot: Any | None) -> str | None:
    metadata = dict(getattr(snapshot, "metadata", {}) or {})
    source = metadata.get("source")
    if source == "input":
        return INPUT_NODE
    if source == "fork":
        return "fork"
    if parent_snapshot is None:
        return None
    parent_next = _snapshot_next(parent_snapshot)
    if len(parent_next) != 1:
        return None
    node = parent_next[0]
    if node == START:
        return INPUT_NODE
    if node == END:
        return None
    return node
