"""Run filesystem lifecycle: ``runs/{run_id}/`` layout + helpers.

Execution state is **thread-scoped**. Every artifact a running branch
writes lives under ``runs/{run_id}/threads/{thread_slug}/`` so two
branches forked from the same checkpoint cannot overwrite each other's
pending evaluation, frontier, evolution log, proposer session, or
candidate traces — even when both are on the same iteration number.

Layout (per Appendix C §C.10 + INTERFACES.md §2)::

    runs/{run_id}/
    ├── manifest.json                 # run config (run-level, written once)
    ├── branches.json                 # durable branch metadata registry
    └── threads/{thread_slug}/
        ├── thread.json               # thread_id ↔ slug mapping + lineage
        ├── pending_eval.json         # proposer→benchmark handoff
        ├── frontier_val.json         # this branch's Pareto frontier
        ├── evolution_summary.jsonl   # this branch's append-only log
        ├── agents/{candidate}.py     # per-branch candidate source snapshot
        ├── candidates/{candidate}/
        │   ├── eval-result.json
        │   ├── status.json
        │   └── traces/{task-id}-trial-{N}/...
        └── proposer-sessions/iter-{N}/

Run-level views (the dashboard's whole-search-tree read) are produced by
the ``aggregate_*`` readers, which merge across threads and tag every row
with its originating ``thread_id``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")

THREADS_DIRNAME = "threads"


def validate_artifact_name(name: str, *, kind: str = "artifact") -> str:
    """Validate a filesystem artifact name used under runs/."""
    if not SAFE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid {kind} name {name!r}; use 1-128 letters, numbers, '.', '_', or '-'"
        )
    return name


def _contained_child(parent: Path, name: str, *, kind: str) -> Path:
    validate_artifact_name(name, kind=kind)
    resolved_parent = parent.resolve()
    child = (resolved_parent / name).resolve()
    try:
        child.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError(f"invalid {kind} path: {name!r}") from exc
    return child


# ── atomic writes ─────────────────────────────────────────────────────


def write_json_atomic(path: Path, payload: Any) -> None:
    """Serialize ``payload`` to ``path`` atomically.

    Concurrent branches poll each other's artifacts through the REST API
    while they are being written. A plain ``write_text`` lets a reader
    observe a truncated file and blow up on ``json.loads``; writing to a
    sibling temp file and ``os.replace``-ing it makes the swap atomic on
    both POSIX and Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ── thread scoping ────────────────────────────────────────────────────


def thread_slug(thread_id: str) -> str:
    """Return a filesystem-safe, collision-free directory name for a thread.

    LangGraph fork thread ids look like ``run-x.fork.ab12cd34`` and nest
    on repeated forks, so they can outgrow the 128-char artifact-name
    limit. Anything that is not directly usable is replaced by a
    ``t-<sha256[:32]>`` slug, which stays unique and stable across
    processes.
    """
    if SAFE_NAME_RE.fullmatch(thread_id):
        return thread_id
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:32]
    return f"t-{digest}"


def thread_dir(run_dir: Path, thread_id: str) -> Path:
    """Return (and create) the artifact directory for one thread."""
    slug = thread_slug(thread_id)
    root = run_dir / THREADS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    d = _contained_child(root, slug, kind="thread")
    d.mkdir(parents=True, exist_ok=True)
    (d / "candidates").mkdir(exist_ok=True)
    (d / "agents").mkdir(exist_ok=True)
    (d / "proposer-sessions").mkdir(exist_ok=True)
    marker = d / "thread.json"
    if not marker.exists():
        write_json_atomic(
            marker,
            {
                "thread_id": thread_id,
                "slug": slug,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return d


def list_thread_dirs(run_dir: Path) -> list[Path]:
    """Return every thread artifact directory in the run, sorted by slug."""
    root = run_dir / THREADS_DIRNAME
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())


def thread_id_for_dir(td: Path) -> str:
    """Recover the logical thread id from a thread artifact directory."""
    marker = td / "thread.json"
    if marker.exists():
        try:
            return str(json.loads(marker.read_text())["thread_id"])
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return td.name


# ── run lifecycle ─────────────────────────────────────────────────────


def make_run_path(repo_root: Path, run_name: str) -> Path:
    """Return the validated path for ``runs/{run_name}``."""
    return _contained_child(repo_root / "runs", run_name, kind="run")


def make_run_dir(repo_root: Path, run_name: str, *, fresh: bool = False) -> Path:
    """Create or return the run directory. Wipes if ``fresh=True``."""
    run_dir = make_run_path(repo_root, run_name)
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / THREADS_DIRNAME).mkdir(exist_ok=True)
    # The run's root thread shares the run id.
    thread_dir(run_dir, run_name)
    return run_dir


def write_manifest(run_dir: Path, **fields: Any) -> None:
    """Write run manifest with run config + start time."""
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)


def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Read ``manifest.json`` if present."""
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ── thread-scoped artifacts ───────────────────────────────────────────


def record_evolution_row(
    run_dir: Path, thread_id: str, row: dict[str, Any]
) -> None:
    """Record one candidate row in this thread's evolution_summary.jsonl.

    Every branch owns its own log file, so writes never interleave across
    branches and lineage cannot be mixed up. The ``thread_id`` is stamped
    on the row so aggregated views stay attributable.

    **Idempotent on (iteration, candidate).** A LangGraph node that is
    interrupted after its side effects but before its checkpoint commits
    is re-executed on resume. A plain append therefore produced a second
    row for the same iteration — observed as
    ``duplicate iterations in summary: [0, 1, 2, 3, 3]`` after cancelling
    a run mid-``update_frontier`` and resuming it. The re-execution's row
    replaces the earlier one rather than joining it, because the rerun is
    the authoritative result.
    """
    td = thread_dir(run_dir, thread_id)
    path = td / "evolution_summary.jsonl"
    row = {**row, "thread_id": thread_id}
    key = (row.get("iteration"), row.get("candidate"))

    existing = [
        json.loads(line)
        for line in (path.read_text().splitlines() if path.exists() else [])
        if line.strip()
    ]
    kept = [
        r for r in existing if (r.get("iteration"), r.get("candidate")) != key
    ]
    if len(kept) == len(existing):
        # Nothing to replace: the common path is a genuine append.
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return

    body = "".join(
        json.dumps(r, default=str) + "\n" for r in [*kept, row]
    )
    _write_text_atomic(path, body)


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace a text file atomically (see ``write_json_atomic``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_evolution_summary(run_dir: Path, thread_id: str) -> list[dict[str, Any]]:
    """Read one thread's evolution rows in write order."""
    path = thread_dir(run_dir, thread_id) / "evolution_summary.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def write_pending_eval(
    run_dir: Path, thread_id: str, payload: dict[str, Any]
) -> None:
    """Write this thread's ``pending_eval.json`` (proposer → benchmark)."""
    write_json_atomic(thread_dir(run_dir, thread_id) / "pending_eval.json", payload)


def read_pending_eval(run_dir: Path, thread_id: str) -> dict[str, Any] | None:
    """Read this thread's ``pending_eval.json`` if present, else None."""
    path = thread_dir(run_dir, thread_id) / "pending_eval.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_frontier(
    run_dir: Path, thread_id: str, frontier: dict[str, Any]
) -> None:
    """Write this thread's ``frontier_val.json``."""
    write_json_atomic(thread_dir(run_dir, thread_id) / "frontier_val.json", frontier)


def read_frontier(run_dir: Path, thread_id: str) -> dict[str, Any] | None:
    """Read this thread's ``frontier_val.json`` if present."""
    path = thread_dir(run_dir, thread_id) / "frontier_val.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def proposer_session_dir(run_dir: Path, thread_id: str, iteration: int) -> Path:
    """Return (and create) this thread's proposer session dir for an iteration."""
    d = thread_dir(run_dir, thread_id) / "proposer-sessions" / f"iter-{iteration}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def candidate_dir(run_dir: Path, thread_id: str, candidate_name: str) -> Path:
    """Return this thread's directory for a candidate; create if missing."""
    parent = thread_dir(run_dir, thread_id) / "candidates"
    parent.mkdir(parents=True, exist_ok=True)
    d = _contained_child(parent, candidate_name, kind="candidate")
    d.mkdir(parents=True, exist_ok=True)
    return d


def candidate_source_path(
    run_dir: Path, thread_id: str, candidate_name: str
) -> Path:
    """Return the per-thread snapshot path for a candidate's source file."""
    parent = thread_dir(run_dir, thread_id) / "agents"
    parent.mkdir(parents=True, exist_ok=True)
    return _contained_child(parent, f"{candidate_name}.py", kind="candidate source")


def write_status(
    run_dir: Path, thread_id: str, candidate_name: str, status: dict[str, Any]
) -> None:
    """Write a candidate's ``status.json`` inside its thread."""
    write_json_atomic(
        candidate_dir(run_dir, thread_id, candidate_name) / "status.json", status
    )


# ── aggregate (whole search tree) views ───────────────────────────────


def aggregate_evolution_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Every evolution row across every thread, tagged with ``thread_id``.

    Rows are ordered by (iteration, thread slug) so the dashboard can
    reconstruct a stable search tree regardless of interleaving between
    concurrently running branches.
    """
    rows: list[dict[str, Any]] = []
    for td in list_thread_dirs(run_dir):
        tid = thread_id_for_dir(td)
        path = td / "evolution_summary.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("thread_id", tid)
            rows.append(row)
    rows.sort(key=lambda r: (r.get("iteration") or 0, str(r.get("thread_id", ""))))
    return rows


def aggregate_frontiers(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Map of ``thread_id`` → that thread's frontier_val payload."""
    out: dict[str, dict[str, Any]] = {}
    for td in list_thread_dirs(run_dir):
        path = td / "frontier_val.json"
        if path.exists():
            out[thread_id_for_dir(td)] = json.loads(path.read_text())
    return out


def find_candidate_dir(run_dir: Path, candidate_name: str) -> Path | None:
    """Locate a candidate's artifact dir across all threads in the run.

    Candidate names are globally unique within a run (see
    ``qualify_candidate_name``), so at most one thread owns any given
    name and the lookup is unambiguous.
    """
    try:
        safe = validate_artifact_name(candidate_name, kind="candidate")
    except ValueError:
        return None
    for td in list_thread_dirs(run_dir):
        d = td / "candidates" / safe
        if d.is_dir():
            return d
    return None


def find_candidate_source(run_dir: Path, candidate_name: str) -> Path | None:
    """Locate a candidate's per-thread source snapshot across all threads."""
    try:
        safe = validate_artifact_name(candidate_name, kind="candidate")
    except ValueError:
        return None
    for td in list_thread_dirs(run_dir):
        p = td / "agents" / f"{safe}.py"
        if p.is_file():
            return p
    return None


def qualify_candidate_name(label: str, thread_id: str, run_id: str) -> str:
    """Return a run-unique candidate name for a proposer-supplied label.

    On the run's root thread the label is used verbatim, which keeps
    single-branch runs readable. On a forked branch a short, stable
    suffix derived from the thread id is appended so two branches that
    reach the same iteration — and therefore propose the same default
    label — cannot claim the same artifact directory or source file.

    Idempotent: a proposer that already branch-qualified its label (the
    mock proposer does, so its authored ``agents/<label>.py`` files do
    not collide either) gets that label back unchanged rather than
    ``name__abc12345__abc12345``.
    """
    validate_artifact_name(label, kind="candidate")
    if thread_id == run_id:
        return label
    suffix = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:8]
    if label.endswith(f"__{suffix}"):
        return label
    qualified = f"{label}__{suffix}"
    return validate_artifact_name(qualified, kind="candidate")
