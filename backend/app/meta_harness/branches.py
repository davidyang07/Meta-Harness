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
"""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START

from app.meta_harness import runs as runs_mod

BranchStatus = Literal[
    "created", "running", "completed", "failed", "cancelled", "interrupted"
]

INPUT_NODE = "__input__"


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


def persist_branch(metadata: BranchMetadata) -> None:
    """Mirror one branch's metadata into ``runs/<run_id>/branches.json``.

    Best-effort: branch history is an observability artifact, and a
    filesystem hiccup must never abort a running branch.
    """
    path = _branches_path(metadata.run_id)
    if path is None:
        return
    try:
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
                    existing.values(), key=lambda b: str(b.get("created_at", ""))
                ),
            },
        )
    except OSError:
        pass


def _rehydrate_status(record: dict[str, Any]) -> dict[str, Any]:
    """A persisted ``running`` branch is really an interrupted one.

    Its asyncio task died with the process. Reporting it as running
    would be a lie the UI then shows as a live spinner.
    """
    record = dict(record)
    if record.get("status") in {"created", "running"}:
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
    )
    branch_metadata[thread_id] = metadata
    persist_branch(metadata)

    fork_config = await graph.aupdate_state(
        {"configurable": {"thread_id": thread_id}},
        fork_values,
        as_node=as_node,
    )
    metadata.status = "running"
    metadata.started_at = _now()
    metadata.live = True
    persist_branch(metadata)

    task = asyncio.create_task(
        _run_branch(graph, metadata, fork_config, recursion_limit),
        name=f"worktree:{thread_id}",
    )
    task.add_done_callback(_consume_task_exception)
    branch_registry[thread_id] = task
    return metadata, task


async def cancel_branch(thread_id: str) -> BranchMetadata:
    """Cancel a branch task if this process owns one, and mark it cancelled.

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
    metadata = get_branch(thread_id)
    if metadata is None:
        raise KeyError(f"unknown branch thread_id: {thread_id}")
    task = branch_registry.get(thread_id)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    if metadata.status not in {"completed", "failed", "cancelled"}:
        _mark_cancelled(metadata)
    return metadata


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
    """Return branch metadata by thread id, from memory or persisted history."""
    live = branch_metadata.get(thread_id)
    if live is not None:
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


async def cancel_all_branches() -> None:
    """Best-effort cleanup helper for tests and shutdown hooks."""
    for thread_id in list(branch_registry):
        metadata = branch_metadata.get(thread_id)
        task = branch_registry.get(thread_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if metadata and metadata.status == "running":
            _mark_cancelled(metadata)


def clear_branch_state() -> None:
    """Clear in-process registries. Call only after tasks are stopped."""
    branch_registry.clear()
    branch_metadata.clear()


async def _run_branch(
    graph: Any,
    metadata: BranchMetadata,
    fork_config: dict[str, Any],
    recursion_limit: int,
) -> dict[str, Any]:
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
