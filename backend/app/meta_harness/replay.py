"""Checkpoint restoration, recorded-event replay, and exact recorded-execution replay.

Three different things share the word "replay" in this codebase, and the
difference between them is the difference between an honest claim and an
overclaim. They are kept apart deliberately:

1. **Checkpoint restoration** (``restore_checkpoint``) — read the state
   stored at one checkpoint. Restoring the same checkpoint twice yields
   byte-identical canonical JSON; ``state_hash`` proves it.

2. **Recorded-event replay** (``replay_events`` / ``replay_thread``) —
   walk a thread's persisted checkpoints in forward order and yield the
   transitions that were recorded. Nothing executes.

3. **Exact recorded-execution replay** (``replay_recorded_execution``) —
   re-execute the inner-loop state machine from a stored checkpoint with
   every external input served from a recorded tape
   (``recording.py``). The graph really runs: nodes execute, the
   conditional edge out of ``verify`` is recomputed, LangGraph writes a
   fresh checkpoint per super-step. What it does *not* do is call a
   model, run a tool, or touch the filesystem — those all come from the
   tape, and ``ReplayEffects`` has no code path that reaches the world.
   The result is verified: same node sequence, same per-step state
   hashes, same final state hash, and the tape consumed exactly.

**Resuming is not replaying.** ``meta-harness resume`` and
``branches.worktree_add`` re-enter a graph from a checkpoint and issue
*fresh* model calls. That is a new stochastic execution which happens to
start from an old state; it is not reproducible and nothing here claims
it is. Only a run that was recorded can be replayed exactly, and only
against its own tape.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from app.meta_harness import effects as fx
from app.meta_harness import metrics as met
from app.meta_harness import recording as rec
from app.meta_harness.branches import (
    CheckpointRecord,
    get_checkpoint_state,
    get_state_history,
)
from app.meta_harness.inner import build_inner_graph, initial_inner_state

#: The inner graph's node names. Checkpoints produced by anything else
#: (``__input__``, a fork's ``update`` write) are structural, not steps.
INNER_NODES = ("orient", "plan", "act", "verify", "submit")


def _json_default(value: Any) -> Any:
    """Encode values canonical JSON cannot represent natively.

    LangChain ``BaseMessage`` objects are projected to their meaningful
    fields rather than ``repr``'d, so a state hash compares message
    content and identity instead of a pretty-printed representation.
    """
    content = getattr(value, "content", None)
    message_type = getattr(value, "type", None)
    if content is not None and message_type is not None:
        return {
            "__message__": str(message_type),
            "id": getattr(value, "id", None),
            "content": content,
        }
    return str(value)


def canonical_json(value: Any) -> str:
    """Deterministic JSON encoding used for state hashing.

    Sorted keys, no insignificant whitespace, non-JSON values coerced by
    ``_json_default`` so a state containing e.g. a ``Path`` or a
    LangChain message still hashes stably.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
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


# ──────────────────────────────────────────────────────────────────────
# Exact recorded-execution replay
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionStep:
    """One graph node execution, joined to the checkpoint it produced."""

    index: int
    node: str
    checkpoint_id: str
    parent_checkpoint_id: str | None
    state_hash: str
    tape_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def collect_execution_steps(
    graph: Any, *, thread_id: str
) -> list[ExecutionStep]:
    """Project a thread's checkpoint history into ordered node executions.

    Oldest first, structural checkpoints (``__input__``, fork writes)
    filtered out, each carrying the SHA-256 of the full state that node
    produced.
    """
    history = await get_state_history(graph, thread_id=thread_id)
    steps: list[ExecutionStep] = []
    for record in reversed(history):
        if record.node not in INNER_NODES:
            continue
        state = await get_checkpoint_state(
            graph, thread_id=thread_id, checkpoint_id=record.checkpoint_id
        )
        steps.append(
            ExecutionStep(
                index=len(steps),
                node=str(record.node),
                checkpoint_id=record.checkpoint_id,
                parent_checkpoint_id=record.parent_checkpoint_id,
                state_hash=state_hash(state),
            )
        )
    return steps


async def assert_thread_unused(graph: Any, *, thread_id: str) -> None:
    """Refuse to record onto a thread that already has checkpoint history.

    A recording pairs its Nth taped node completion with the Nth
    checkpoint on the thread. Recording onto a thread that already has
    checkpoints leaves the two out of alignment, and the failure surfaces
    later as an unpositionable replay. Catch it here, where the message
    can say what to do about it.
    """
    existing = await get_state_history(graph, thread_id=thread_id, limit=1)
    if existing:
        raise rec.RecordingError(
            f"thread {thread_id!r} already has checkpoint history, so a "
            f"recording made on it could not be positioned. Checkpoint "
            f"history outlives the process: use a fresh thread id, or a "
            f"fresh run name."
        )


async def finalize_recording(
    graph: Any,
    *,
    writer: rec.TapeWriter,
    thread_id: str,
    task_dict: dict[str, Any],
    workspace: Path | str,
    harness: Any,
    usage: met.UsageRecorder,
    final_state: dict[str, Any],
    recording_dir: Path,
    manifest_extra: dict[str, Any] | None = None,
) -> Path:
    """Join a tape to its thread's checkpoint history and persist both.

    The join is what makes "replay from checkpoint X" answerable: each
    taped node completion is paired with the checkpoint that node wrote
    and with the hash of the state it produced. A mismatch in length or
    order means the tape and the history disagree about what happened,
    which is a bug in recording rather than something to paper over — so
    it raises.
    """
    steps = await collect_execution_steps(graph, thread_id=thread_id)
    marks = writer.steps
    if len(steps) != len(marks):
        raise rec.RecordingError(
            f"recording {thread_id}: {len(marks)} node completions were taped "
            f"but the checkpoint history has {len(steps)}. The tape and the "
            f"checkpoint history must agree or replay cannot be positioned."
        )
    for step, mark in zip(steps, marks, strict=True):
        if step.node != mark.node:
            raise rec.RecordingError(
                f"recording {thread_id}: step {step.index} is {step.node!r} in "
                f"the checkpoint history but {mark.node!r} on the tape"
            )

    writer.steps = [
        rec.StepMark(
            index=mark.index,
            node=mark.node,
            tape_length=mark.tape_length,
            checkpoint_id=step.checkpoint_id,
            state_hash=step.state_hash,
        )
        for step, mark in zip(steps, marks, strict=True)
    ]
    manifest = rec.build_manifest(
        recording_id=recording_dir.name,
        thread_id=thread_id,
        task=task_dict,
        workspace_path=str(workspace),
        model=getattr(harness, "MODEL", None),
        harness_class=f"{type(harness).__module__}:{type(harness).__qualname__}",
        extra={
            "task": task_dict,
            "final_state_sha256": state_hash(final_state),
            "usage": usage.totals(),
            **(manifest_extra or {}),
        },
    )
    return rec.write_recording(recording_dir, manifest=manifest, writer=writer)


async def record_inner_execution(
    *,
    harness_factory: Callable[[], Any],
    task_dict: dict[str, Any],
    workspace: Path,
    thread_id: str,
    checkpointer: Any,
    recording_dir: Path,
    trace_dir: Path | None = None,
    manifest_extra: dict[str, Any] | None = None,
    usage: met.UsageRecorder | None = None,
) -> dict[str, Any]:
    """Run one inner-loop trial, recording every nondeterministic input.

    Requires a checkpointer: the tape alone is not enough, because
    "replay from checkpoint X" needs X to exist. The recording pairs each
    tape offset with the checkpoint the corresponding node produced.

    Returns the run's final state, its hash, and the recording directory.
    """
    if checkpointer is None:
        raise ValueError(
            "record_inner_execution requires a checkpointer: an execution "
            "with no checkpoint history cannot be replayed from a checkpoint"
        )

    writer = rec.TapeWriter()
    effects = fx.RecordingEffects(writer)

    harness = harness_factory()
    # Order matters. The effects boundary must sit *inside* the usage
    # recorder, so a replayed run's recorder still sees a response (the
    # recorded one) and reports the recorded token counts. Wrapping the
    # other way round would make every replayed trial report zero usage.
    fx.instrument_harness_for_effects(harness, effects)
    usage = usage or met.UsageRecorder()
    met.instrument_harness(harness, usage)

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        task_dict = {**task_dict, "_trace_dir": str(trace_dir)}

    graph = build_inner_graph(harness, checkpointer=checkpointer, effects=effects)
    await assert_thread_unused(graph, thread_id=thread_id)
    final_state = await graph.ainvoke(
        initial_inner_state(task_dict=task_dict, workspace=workspace),
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100},
    )

    await finalize_recording(
        graph,
        writer=writer,
        thread_id=thread_id,
        task_dict=task_dict,
        workspace=workspace,
        harness=harness,
        usage=usage,
        final_state=final_state,
        recording_dir=recording_dir,
        manifest_extra=manifest_extra,
    )

    return {
        "recording_dir": str(recording_dir),
        "thread_id": thread_id,
        "final_state": final_state,
        "final_state_sha256": state_hash(final_state),
        "steps": [s.to_dict() for s in writer.steps],
        "tape_entries": len(writer.entries),
    }


class ReplayVerificationError(AssertionError):
    """Exact replay did not reproduce the recorded execution."""


async def replay_recorded_execution(
    *,
    recording: rec.Recording,
    harness_factory: Callable[[], Any],
    checkpointer: Any,
    source_graph: Any | None = None,
    from_checkpoint: str | None = None,
    replay_thread_id: str | None = None,
) -> dict[str, Any]:
    """Re-execute a recorded run against its tape and verify equivalence.

    ``from_checkpoint`` selects where the continuation starts. ``None``
    replays the whole recorded run from its entry state; otherwise the
    checkpoint must be one the recording knows about, and only the
    transitions after it are replayed.

    Issues no model calls, runs no tools, and touches no workspace: every
    external input comes from ``recording``. Returns a verification
    report; the ``verified`` field is the whole point, and
    ``verify=True`` on the CLI turns a ``False`` into a non-zero exit.
    """
    start_index, start_step = _resolve_start(recording, from_checkpoint)
    reader = recording.reader_from_step(start_index)
    effects = fx.ReplayEffects(reader)

    harness = harness_factory()
    fx.instrument_harness_for_effects(harness, effects)
    usage = met.UsageRecorder()
    met.instrument_harness(harness, usage)

    graph = build_inner_graph(harness, checkpointer=checkpointer, effects=effects)
    thread_id = replay_thread_id or (
        f"{recording.thread_id}::replay::{uuid.uuid4().hex[:8]}"
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100,
    }

    divergence: str | None = None
    final_state: dict[str, Any] | None = None
    try:
        if start_step is None:
            task = dict(recording.manifest.get("task") or {})
            workspace = str(recording.manifest.get("workspace_path") or "")
            final_state = await graph.ainvoke(
                initial_inner_state(task_dict=task, workspace=workspace),
                config=config,
            )
        else:
            if source_graph is None:
                raise ValueError(
                    "replaying from a checkpoint needs source_graph so the "
                    "recorded state at that checkpoint can be restored"
                )
            state = await get_checkpoint_state(
                source_graph,
                thread_id=recording.thread_id,
                checkpoint_id=start_step.checkpoint_id,
            )
            await graph.aupdate_state(
                {"configurable": {"thread_id": thread_id}},
                state,
                as_node=start_step.node,
            )
            final_state = await graph.ainvoke(None, config=config)
    except rec.ReplayDivergence as exc:
        divergence = str(exc)

    replayed_steps = await collect_execution_steps(graph, thread_id=thread_id)
    expected = [
        s for s in recording.steps if start_index is None or s.index > start_index
    ]

    checks = _build_checks(
        recording=recording,
        expected=expected,
        replayed=replayed_steps,
        reader=reader,
        final_state=final_state,
        divergence=divergence,
    )
    verified = all(c["ok"] for c in checks)

    return {
        "recording_id": recording.recording_id,
        "recorded_thread_id": recording.thread_id,
        # Provenance of the recording, carried into the report so a reader
        # can see what was replayed without opening the tape — in
        # particular whether the recorded model turns came from a
        # provider or from a script.
        "recorded_model": recording.manifest.get("model"),
        "recorded_harness_class": recording.manifest.get("harness_class"),
        "recorded_task_id": recording.manifest.get("task_id"),
        "recorded_with": recording.manifest.get("recorded_with"),
        "replay_thread_id": thread_id,
        "from_checkpoint": from_checkpoint,
        "start_step_index": start_index,
        "model_calls_issued": 0,
        "replayed_nodes": [s.node for s in replayed_steps],
        "recorded_nodes": [s.node for s in expected],
        "recorded_final_state_sha256": recording.manifest.get("final_state_sha256"),
        "replayed_final_state_sha256": (
            state_hash(final_state) if final_state is not None else None
        ),
        "tape_entries_consumed": reader.consumed,
        "tape_entries_remaining": reader.remaining(),
        "usage": usage.totals(),
        "checks": checks,
        "verified": verified,
        "divergence": divergence,
        "guarantee": (
            "exact replay of a recorded execution: the inner-loop graph "
            "re-executed with every model response, tool result and "
            "filesystem observation served from the recorded tape; no "
            "external model call was issued"
        ),
    }


def _resolve_start(
    recording: rec.Recording, from_checkpoint: str | None
) -> tuple[int | None, rec.StepMark | None]:
    if from_checkpoint is None:
        return None, None
    for mark in recording.steps:
        if mark.checkpoint_id == from_checkpoint:
            return mark.index, mark
    known = [m.checkpoint_id for m in recording.steps if m.checkpoint_id]
    raise KeyError(
        f"checkpoint {from_checkpoint!r} is not in recording "
        f"{recording.recording_id!r}. Known checkpoints: {known}"
    )


def _build_checks(
    *,
    recording: rec.Recording,
    expected: list[rec.StepMark],
    replayed: list[ExecutionStep],
    reader: rec.TapeReader,
    final_state: dict[str, Any] | None,
    divergence: str | None,
) -> list[dict[str, Any]]:
    """Every equivalence assertion, each reported with its evidence."""
    recorded_final = recording.manifest.get("final_state_sha256")
    replayed_final = state_hash(final_state) if final_state is not None else None

    mismatched_states = [
        {
            "index": exp.index,
            "node": exp.node,
            "recorded": exp.state_hash,
            "replayed": got.state_hash,
        }
        for exp, got in zip(expected, replayed, strict=False)
        if exp.state_hash is not None and exp.state_hash != got.state_hash
    ]

    return [
        {
            "check": "no_divergence",
            "ok": divergence is None,
            "detail": divergence or "the tape served every request in order",
        },
        {
            "check": "node_sequence_identical",
            "ok": [s.node for s in expected] == [s.node for s in replayed],
            "detail": {
                "recorded": [s.node for s in expected],
                "replayed": [s.node for s in replayed],
            },
        },
        {
            "check": "per_step_state_hashes_identical",
            "ok": not mismatched_states
            and len(expected) == len(replayed),
            "detail": mismatched_states
            or f"{len(replayed)} steps matched their recorded state hash",
        },
        {
            "check": "final_state_byte_identical",
            "ok": (
                recorded_final is not None
                and replayed_final is not None
                and recorded_final == replayed_final
            ),
            "detail": {"recorded": recorded_final, "replayed": replayed_final},
        },
        {
            "check": "tape_fully_consumed",
            "ok": reader.remaining() == 0,
            "detail": (
                f"{reader.consumed} entries consumed, "
                f"{reader.remaining()} left unread"
            ),
        },
    ]


def render_verification(report: dict[str, Any]) -> str:
    """Human-readable verdict for ``meta-harness replay --verify``."""
    lines = [
        f"recording:      {report['recording_id']}",
        f"recorded thread {report['recorded_thread_id']}",
        f"replay thread   {report['replay_thread_id']}",
        f"from checkpoint {report['from_checkpoint'] or '(start of run)'}",
        f"model calls:    {report['model_calls_issued']}",
        "",
    ]
    for check in report["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        lines.append(f"  [{mark}] {check['check']}")
        if not check["ok"]:
            lines.append(f"         {json.dumps(check['detail'], default=str)}")
    lines.append("")
    lines.append(
        "EXACT REPLAY VERIFIED" if report["verified"] else "EXACT REPLAY FAILED"
    )
    return "\n".join(lines)


__all__ = [
    "ExecutionStep",
    "assert_thread_unused",
    "INNER_NODES",
    "ReplayVerificationError",
    "canonical_json",
    "collect_execution_steps",
    "finalize_recording",
    "record_inner_execution",
    "render_verification",
    "replay_events",
    "replay_recorded_execution",
    "replay_thread",
    "restore_checkpoint",
    "state_hash",
]
