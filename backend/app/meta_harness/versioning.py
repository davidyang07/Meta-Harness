"""The run version graph: checkpoints as commits, branches as refs.

The persistence layer already stores what a version-control system
stores. This module is the read model that says so out loud, and the
place to look when the question is "what does Git-style versioning
actually mean here":

| Git | Meta-Harness |
|---|---|
| commit | a LangGraph checkpoint, identified by an immutable `checkpoint_id` |
| parent commit | `parent_checkpoint_id` on the same or an ancestor thread |
| branch ref | a `thread_id`, registered in `branches.json` with its fork point |
| checkout -b `<sha>` | `branches.worktree_add(parent_checkpoint_id=...)` |
| working tree | `runs/<run>/threads/<thread>/`, private per branch |
| commit contents | the checkpointed state, hashed by `replay.state_hash` |
| `git diff A B` | :func:`diff_checkpoints` |

Two properties this module reads out rather than assumes:

1. **Checkpoint identity is immutable.** Nothing rewrites a checkpoint;
   a fork writes new ones on a new thread. So a checkpoint id is a
   permanent address for one state, and :func:`verify_immutability` can
   confirm a stored state still hashes to what it hashed to before.
2. **A branch owns its artifacts.** Candidate source is snapshotted per
   branch (``runs.candidate_source_path``), so two branches forked from
   one checkpoint that propose the same label still benchmark different
   files. :func:`branch_artifacts` surfaces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.meta_harness import branches as br
from app.meta_harness import runs as runs_mod


@dataclass
class VersionNode:
    """One checkpoint in the version graph."""

    checkpoint_id: str
    thread_id: str
    parent_checkpoint_id: str | None
    node: str | None
    ts: str | None
    iteration: int | None
    values_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "node": self.node,
            "ts": self.ts,
            "iteration": self.iteration,
            "values_summary": self.values_summary,
        }


async def thread_versions(graph: Any, *, thread_id: str) -> list[VersionNode]:
    """A thread's checkpoints, oldest first."""
    history = await br.get_state_history(graph, thread_id=thread_id)
    return [
        VersionNode(
            checkpoint_id=r.checkpoint_id,
            thread_id=r.thread_id,
            parent_checkpoint_id=r.parent_checkpoint_id,
            node=r.node,
            ts=r.ts,
            iteration=r.iteration,
            values_summary=r.values_summary,
        )
        for r in reversed(history)
    ]


async def version_graph(
    graph: Any, *, run_id: str, extra_threads: list[str] | None = None
) -> dict[str, Any]:
    """Assemble the whole run's checkpoint DAG plus its branch refs.

    Threads come from the durable branch registry, so the graph survives
    a backend restart — the same property ``branches.load_persisted_branches``
    gives the dashboard.
    """
    metadata = br.list_branches(run_id=run_id)
    thread_ids: list[str] = [run_id]
    for meta in metadata:
        if meta.thread_id not in thread_ids:
            thread_ids.append(meta.thread_id)
    for thread_id in extra_threads or []:
        if thread_id not in thread_ids:
            thread_ids.append(thread_id)

    nodes: list[VersionNode] = []
    per_thread: dict[str, list[str]] = {}
    for thread_id in thread_ids:
        try:
            versions = await thread_versions(graph, thread_id=thread_id)
        except Exception:  # noqa: BLE001 — a thread with no history is not an error
            versions = []
        per_thread[thread_id] = [v.checkpoint_id for v in versions]
        nodes.extend(versions)

    known = {n.checkpoint_id for n in nodes}
    edges = [
        {
            "from": n.parent_checkpoint_id,
            "to": n.checkpoint_id,
            "kind": (
                "sequential"
                if n.parent_checkpoint_id in known
                else "root"
                if n.parent_checkpoint_id is None
                else "external-parent"
            ),
        }
        for n in nodes
    ]
    fork_edges = [
        {
            "from": meta.parent_checkpoint_id,
            "to": meta.thread_id,
            "kind": "fork",
            "branch_id": meta.branch_id,
        }
        for meta in metadata
        if meta.parent_checkpoint_id
    ]

    return {
        "run_id": run_id,
        "threads": per_thread,
        "checkpoints": [n.to_dict() for n in nodes],
        "edges": edges + fork_edges,
        "branches": [m.to_dict() for m in metadata],
        "checkpoint_count": len(nodes),
        "branch_count": len(metadata),
    }


# ── diffing ───────────────────────────────────────────────────────────


def diff_states(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Key-level diff between two checkpointed states.

    Values are compared by their canonical-JSON hash so a nested change
    registers without the diff carrying the whole nested value; the
    hashes are reported so a reader can confirm the comparison.
    """
    from app.meta_harness.replay import state_hash  # noqa: PLC0415 — cycle

    before_keys, after_keys = set(before), set(after)
    changed = [
        {
            "key": key,
            "before_sha256": state_hash(before[key]),
            "after_sha256": state_hash(after[key]),
        }
        for key in sorted(before_keys & after_keys)
        if state_hash(before[key]) != state_hash(after[key])
    ]
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": changed,
        "identical": not changed
        and before_keys == after_keys,
        "before_sha256": state_hash(before),
        "after_sha256": state_hash(after),
    }


async def diff_checkpoints(
    graph: Any,
    *,
    thread_a: str,
    checkpoint_a: str,
    thread_b: str,
    checkpoint_b: str,
) -> dict[str, Any]:
    """``git diff`` between two checkpoints, on the same thread or across two."""
    state_a = await br.get_checkpoint_state(
        graph, thread_id=thread_a, checkpoint_id=checkpoint_a
    )
    state_b = await br.get_checkpoint_state(
        graph, thread_id=thread_b, checkpoint_id=checkpoint_b
    )
    return {
        "a": {"thread_id": thread_a, "checkpoint_id": checkpoint_a},
        "b": {"thread_id": thread_b, "checkpoint_id": checkpoint_b},
        **diff_states(state_a, state_b),
    }


async def verify_immutability(
    graph: Any, *, thread_id: str, expected: dict[str, str]
) -> dict[str, Any]:
    """Re-read checkpoints and confirm each still hashes to ``expected``.

    ``expected`` maps checkpoint id → state hash, as recorded earlier.
    A mismatch means a stored checkpoint changed underneath us, which
    would invalidate every published replay and fork claim — so it is
    reported as a failure, never rounded off.
    """
    from app.meta_harness.replay import state_hash  # noqa: PLC0415 — cycle

    results = []
    for checkpoint_id, expected_hash in expected.items():
        state = await br.get_checkpoint_state(
            graph, thread_id=thread_id, checkpoint_id=checkpoint_id
        )
        actual = state_hash(state)
        results.append(
            {
                "checkpoint_id": checkpoint_id,
                "expected_sha256": expected_hash,
                "actual_sha256": actual,
                "ok": actual == expected_hash,
            }
        )
    return {
        "thread_id": thread_id,
        "checked": len(results),
        "immutable": all(r["ok"] for r in results),
        "results": results,
    }


# ── evidence capture ──────────────────────────────────────────────────


async def capture_evidence(
    graph: Any, *, run_id: str, run_dir: Path, full: bool = False
) -> dict[str, Any]:
    """Capture a run's version graph as a committable evidence artifact.

    Re-reads **every** thread's checkpoints and confirms each still
    hashes to what it hashed to a moment ago. That is a weaker statement
    than "immutable forever", and it is the strongest one a single
    process can make: nothing between the two reads rewrote a checkpoint.

    ``full`` keeps the per-checkpoint value summaries. The default drops
    them: they are large, they are the only part that grows with run
    length, and the claim — immutable ids, parent references, branch refs
    with fork points, isolated working trees — does not rest on them.
    """
    from app.meta_harness.replay import state_hash  # noqa: PLC0415 — cycle

    version = await version_graph(graph, run_id=run_id)

    per_thread: list[dict[str, Any]] = []
    all_immutable = True
    checked = 0
    for thread_id in version["threads"]:
        expected: dict[str, str] = {}
        for checkpoint in version["checkpoints"]:
            if checkpoint["thread_id"] != thread_id:
                continue
            state = await br.get_checkpoint_state(
                graph, thread_id=thread_id, checkpoint_id=checkpoint["checkpoint_id"]
            )
            expected[checkpoint["checkpoint_id"]] = state_hash(state)
        result = await verify_immutability(
            graph, thread_id=thread_id, expected=expected
        )
        checked += result["checked"]
        all_immutable = all_immutable and result["immutable"]
        per_thread.append(
            {
                "thread_id": thread_id,
                "checked": result["checked"],
                "immutable": result["immutable"],
                "mismatched": [r for r in result["results"] if not r["ok"]],
            }
        )

    if not full:
        version["checkpoints"] = [
            {k: v for k, v in c.items() if k != "values_summary"}
            for c in version["checkpoints"]
        ]
        # A branch's ``result`` is its whole terminal state, including
        # every candidate's per-trial rows. The claim here is about refs
        # and fork points, so keep the ref and drop the payload — that is
        # the difference between a committable artifact and a megabyte.
        version["branches"] = [
            {k: v for k, v in b.items() if k != "result"}
            for b in version["branches"]
        ]

    version["immutable"] = all_immutable
    version["immutability"] = {
        "checked": checked,
        "immutable": all_immutable,
        "threads": per_thread,
        "method": (
            "every stored checkpoint on every thread was re-read and its "
            "canonical-JSON SHA-256 compared against the hash taken moments "
            "earlier in the same process"
        ),
    }
    version["branch_artifacts"] = branch_artifacts(run_dir)
    version["values_summaries_included"] = full
    return version


# ── per-branch working trees ──────────────────────────────────────────


def branch_artifacts(run_dir: Path) -> dict[str, dict[str, Any]]:
    """What each branch's private working tree holds.

    The isolation guarantee is structural — every path is under
    ``threads/<slug>/`` — so this reader is also the check: two branches
    listing the same candidate label must still list different source
    files with different hashes.
    """
    out: dict[str, dict[str, Any]] = {}
    for thread_path in runs_mod.list_thread_dirs(run_dir):
        thread_id = runs_mod.thread_id_for_dir(thread_path)
        agents_dir = thread_path / "agents"
        sources = {}
        if agents_dir.is_dir():
            for source in sorted(agents_dir.glob("*.py")):
                from app.meta_harness.candidates import source_sha256  # noqa: PLC0415

                sources[source.stem] = {
                    "path": str(source),
                    "sha256": source_sha256(source),
                }
        candidates_dir = thread_path / "candidates"
        out[thread_id] = {
            "thread_dir": str(thread_path),
            "candidate_sources": sources,
            "candidates": sorted(
                d.name for d in candidates_dir.iterdir() if d.is_dir()
            )
            if candidates_dir.is_dir()
            else [],
            "has_frontier": (thread_path / "frontier_val.json").exists(),
            "evolution_rows": len(
                runs_mod.read_evolution_summary(run_dir, thread_id)
            ),
        }
    return out


__all__ = [
    "VersionNode",
    "branch_artifacts",
    "diff_checkpoints",
    "diff_states",
    "thread_versions",
    "verify_immutability",
    "version_graph",
]
