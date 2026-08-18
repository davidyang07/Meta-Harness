"""Checkpoint state restoration and recorded-event replay.

**What this guarantees, precisely.**

``restore_checkpoint`` reads a stored LangGraph checkpoint and returns
the exact state that was persisted at that point in the run. Restoring
the same checkpoint twice yields byte-identical canonical JSON, and
``state_hash`` proves it. Branching from a restored checkpoint
(``branches.worktree_add``) starts from that exact state.

``replay_events`` walks a thread's checkpoint history in forward order
and yields the recorded transitions **without re-invoking any model**.
It is a deterministic replay of what was recorded.

**What this does not guarantee.** Re-executing the graph from a restored
checkpoint issues fresh LLM calls, and those are not deterministic:
sampling, model updates, and provider-side changes all move the output.
Nothing here claims byte-identical regeneration of stochastic model
output, and the docs must not either. The accurate phrasing is
"checkpoint recovery and branching from historical states".
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, AsyncIterator

from app.meta_harness.branches import (
    CheckpointRecord,
    get_checkpoint_state,
    get_state_history,
)


def canonical_json(value: Any) -> str:
    """Deterministic JSON encoding used for state hashing.

    Sorted keys, no insignificant whitespace, non-JSON values coerced by
    ``str`` so a state containing e.g. a ``Path`` still hashes stably.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def state_hash(state: Any) -> str:
    """SHA-256 over the canonical JSON encoding of a state."""
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


async def restore_checkpoint(
    graph: Any, *, thread_id: str, checkpoint_id: str
) -> dict[str, Any]:
    """Return the exact stored state at one checkpoint, plus its hash.

    Raises ``KeyError`` if the checkpoint is not in the thread's history.
    """
    state = await get_checkpoint_state(
        graph, thread_id=thread_id, checkpoint_id=checkpoint_id
    )
    return {
        "thread_id": thread_id,
        "checkpoint_id": checkpoint_id,
        "state": state,
        "state_hash": state_hash(state),
    }


async def replay_events(
    graph: Any, *, thread_id: str, limit: int | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Yield a thread's recorded transitions oldest-first.

    Pure read-back of persisted checkpoints — no node body runs and no
    model is called.
    """
    history: list[CheckpointRecord] = await get_state_history(
        graph, thread_id=thread_id, limit=limit
    )
    for index, record in enumerate(reversed(history)):
        yield {
            "index": index,
            "checkpoint_id": record.checkpoint_id,
            "parent_checkpoint_id": record.parent_checkpoint_id,
            "thread_id": record.thread_id,
            "node": record.node,
            "iteration": record.iteration,
            "ts": record.ts,
            "next": list(record.next),
            "values_summary": record.values_summary,
        }


async def replay_thread(
    graph: Any, *, thread_id: str, limit: int | None = None
) -> dict[str, Any]:
    """Materialise a whole thread's recorded transitions with a digest.

    The ``transitions_hash`` lets a reader confirm two replays of the
    same thread produced the same sequence.
    """
    events = [
        event
        async for event in replay_events(graph, thread_id=thread_id, limit=limit)
    ]
    return {
        "thread_id": thread_id,
        "checkpoint_count": len(events),
        "transitions": events,
        "transitions_hash": state_hash(events),
        "guarantee": (
            "recorded state restoration and event replay; no model calls are "
            "re-issued and no stochastic output is regenerated"
        ),
    }
