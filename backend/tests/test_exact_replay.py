"""Exact recorded-execution replay.

The claim under test is the strong one: re-executing a recorded
inner-loop run against its tape reproduces the *same* transitions in the
same order and the *same* final state byte-for-byte, while issuing no
model call at all.

The graph really runs here. Nodes execute, tools are dispatched through
the effects boundary, the conditional edge out of ``verify`` is
recomputed from replayed state, and LangGraph writes a fresh checkpoint
per super-step. Only the world is fake — and ``ReplayEffects`` has no
code path that reaches it, which is asserted directly.

These tests need no API key and no network. A ``ScriptedHarness``
supplies the model's turns; everything downstream of it — the six tools,
the pytest subprocess, the workspace — is the real implementation.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import effects as fx  # noqa: E402
from app.meta_harness import recording as rec  # noqa: E402
from app.meta_harness import replay as replay_mod  # noqa: E402
from app.meta_harness.inner import build_inner_graph  # noqa: E402
from app.meta_harness.persistence import healthcheck, persistence_layer  # noqa: E402
from tests.conftest import unique_name as unique  # noqa: E402
from tests.harness_doubles import (  # noqa: E402
    DifferentPromptHarness,
    FailingHarness,
    FixTypoHarness,
    NoRetryHarness,
    load_task,
    task_workspace,
)

TASK_ID = "task-001-fix-typo"


def _workspace(tmp_path: Path) -> Path:
    """A disposable copy of the task workspace, outside the repo."""
    destination = tmp_path / "ws"
    shutil.copytree(task_workspace(REPO_ROOT, TASK_ID), destination)
    return destination


async def _record(
    tmp_path: Path,
    saver,
    *,
    harness=FixTypoHarness,
    thread: str | None = None,
) -> tuple[rec.Recording, dict]:
    workspace = _workspace(tmp_path)
    result = await replay_mod.record_inner_execution(
        harness_factory=harness,
        task_dict=load_task(REPO_ROOT, TASK_ID),
        workspace=workspace,
        thread_id=thread or unique("rec"),
        checkpointer=saver,
        recording_dir=tmp_path / "recordings" / "r1",
    )
    return rec.read_recording(tmp_path / "recordings" / "r1"), result


# ── the effects boundary ──────────────────────────────────────────────


async def test_replay_effects_never_invoke_the_producer():
    """The 'no model call' guarantee is a property of the code, not a promise."""
    writer = rec.TapeWriter()
    key = rec.effect_key(rec.KIND_TOOL, {"tool": "read_file"})
    writer.append(rec.KIND_TOOL, key, {"status": "ok"})
    effects = fx.ReplayEffects(rec.TapeReader(writer.entries))

    called = False

    async def _produce():
        nonlocal called
        called = True
        return {"status": "live"}

    result = await effects.observe(rec.KIND_TOOL, {"tool": "read_file"}, _produce)

    assert result == {"status": "ok"}
    assert called is False, "ReplayEffects must never reach the world"


def test_replay_effects_suppress_trace_writes(tmp_path: Path):
    """A replay must not overwrite the artifacts of the run it replays."""
    effects = fx.ReplayEffects(rec.TapeReader([]))
    target = tmp_path / "traces" / "score.json"
    effects.write_trace(target, "{}")
    effects.append_trace(target, "{}\n")
    assert not target.exists()


async def test_recording_effects_pass_the_real_value_through():
    writer = rec.TapeWriter()
    effects = fx.RecordingEffects(writer)

    async def _produce():
        return {"status": "ok", "content": "real"}

    result = await effects.observe(rec.KIND_TOOL, {"tool": "read_file"}, _produce)

    assert result == {"status": "ok", "content": "real"}
    assert len(writer.entries) == 1
    assert writer.entries[0].payload == {"status": "ok", "content": "real"}


# ── in-memory round trips (no Postgres needed) ────────────────────────


async def test_a_recorded_run_replays_exactly_from_the_start(tmp_path: Path):
    recording, recorded = await _record(tmp_path, InMemorySaver())

    report = await replay_mod.replay_recorded_execution(
        recording=recording,
        harness_factory=FixTypoHarness,
        checkpointer=InMemorySaver(),
    )

    assert report["verified"], report["checks"]
    assert report["divergence"] is None
    assert report["model_calls_issued"] == 0
    assert report["replayed_nodes"] == report["recorded_nodes"]
    assert report["tape_entries_remaining"] == 0
    assert (
        report["replayed_final_state_sha256"]
        == recorded["final_state_sha256"]
        == report["recorded_final_state_sha256"]
    )


async def test_every_equivalence_check_passes_individually(tmp_path: Path):
    recording, _ = await _record(tmp_path, InMemorySaver())
    report = await replay_mod.replay_recorded_execution(
        recording=recording,
        harness_factory=FixTypoHarness,
        checkpointer=InMemorySaver(),
    )
    by_name = {c["check"]: c for c in report["checks"]}
    assert set(by_name) == {
        "no_divergence",
        "node_sequence_identical",
        "per_step_state_hashes_identical",
        "final_state_byte_identical",
        "tape_fully_consumed",
    }
    for name, check in by_name.items():
        assert check["ok"], f"{name}: {check['detail']}"


async def test_replay_reproduces_the_verify_act_retry_loop(tmp_path: Path):
    """The conditional edge is recomputed, not replayed from a list."""
    saver = InMemorySaver()
    workspace = _workspace(tmp_path)
    thread = unique("retry")
    await replay_mod.record_inner_execution(
        harness_factory=FailingHarness,
        task_dict=load_task(REPO_ROOT, TASK_ID),
        workspace=workspace,
        thread_id=thread,
        checkpointer=saver,
        recording_dir=tmp_path / "recordings" / "retry",
    )
    recording = rec.read_recording(tmp_path / "recordings" / "retry")

    # The recorded run failed verify and looped back to act.
    assert recording.continuation_nodes(None).count("act") > 1

    report = await replay_mod.replay_recorded_execution(
        recording=recording,
        harness_factory=FailingHarness,
        checkpointer=InMemorySaver(),
    )
    assert report["verified"], report["checks"]
    assert report["replayed_nodes"].count("verify") > 1


async def test_replay_is_repeatable(tmp_path: Path):
    recording, _ = await _record(tmp_path, InMemorySaver())
    first = await replay_mod.replay_recorded_execution(
        recording=recording,
        harness_factory=FixTypoHarness,
        checkpointer=InMemorySaver(),
    )
    second = await replay_mod.replay_recorded_execution(
        recording=recording,
        harness_factory=FixTypoHarness,
        checkpointer=InMemorySaver(),
    )
    assert first["verified"] and second["verified"]
    assert (
        first["replayed_final_state_sha256"] == second["replayed_final_state_sha256"]
    )


async def test_a_different_request_is_refused_by_the_tape(tmp_path: Path):
    """The tape checks what the graph asks for, not just what comes next.

    A replay cannot diverge in the model's *output* — that is served from
    the tape. It can diverge in what the harness *asks*, which is what an
    override point changes; here the act prompt differs, so the very
    first act request has a different key.
    """
    recording, _ = await _record(tmp_path, InMemorySaver())

    report = await replay_mod.replay_recorded_execution(
        recording=recording,
        harness_factory=DifferentPromptHarness,
        checkpointer=InMemorySaver(),
    )

    assert report["verified"] is False
    assert report["divergence"] is not None
    assert "divergence" in report["divergence"]
    assert not [c for c in report["checks"] if c["check"] == "no_divergence"][0]["ok"]


async def test_a_different_routing_policy_fails_verification(tmp_path: Path):
    """The conditional edge is recomputed, so changing it changes the replay."""
    await replay_mod.record_inner_execution(
        harness_factory=FailingHarness,
        task_dict=load_task(REPO_ROOT, TASK_ID),
        workspace=_workspace(tmp_path),
        thread_id=unique("route"),
        checkpointer=InMemorySaver(),
        recording_dir=tmp_path / "recordings" / "route",
    )
    recording = rec.read_recording(tmp_path / "recordings" / "route")

    report = await replay_mod.replay_recorded_execution(
        recording=recording,
        harness_factory=NoRetryHarness,
        checkpointer=InMemorySaver(),
    )

    assert report["verified"] is False
    assert report["replayed_nodes"] != report["recorded_nodes"]


async def test_a_truncated_tape_fails_verification(tmp_path: Path):
    recording, _ = await _record(tmp_path, InMemorySaver())
    truncated = rec.Recording(
        manifest=recording.manifest,
        entries=recording.entries[:-1],
        steps=recording.steps,
    )

    report = await replay_mod.replay_recorded_execution(
        recording=truncated,
        harness_factory=FixTypoHarness,
        checkpointer=InMemorySaver(),
    )
    assert report["verified"] is False


async def test_replay_reports_the_recorded_token_usage(tmp_path: Path):
    """A replayed trial's metrics must be the recorded trial's metrics."""
    recording, recorded = await _record(tmp_path, InMemorySaver())
    report = await replay_mod.replay_recorded_execution(
        recording=recording,
        harness_factory=FixTypoHarness,
        checkpointer=InMemorySaver(),
    )
    assert report["usage"]["total_tokens"] > 0
    assert report["usage"] == recording.manifest["usage"]


async def test_recording_requires_a_checkpointer(tmp_path: Path):
    with pytest.raises(ValueError, match="requires a checkpointer"):
        await replay_mod.record_inner_execution(
            harness_factory=FixTypoHarness,
            task_dict=load_task(REPO_ROOT, TASK_ID),
            workspace=_workspace(tmp_path),
            thread_id="no-saver",
            checkpointer=None,
            recording_dir=tmp_path / "recordings" / "x",
        )


# ── replay from a stored checkpoint ───────────────────────────────────


async def test_replay_from_every_recorded_checkpoint_reproduces_the_run(
    tmp_path: Path,
):
    """Any stored checkpoint is a valid entry point, not just the first."""
    saver = InMemorySaver()
    recording, recorded = await _record(tmp_path, saver)
    source_graph = build_inner_graph(FixTypoHarness(), checkpointer=saver)

    checkpoints = [s for s in recording.steps if s.checkpoint_id]
    assert len(checkpoints) >= 4

    for step in checkpoints[:-1]:  # the last step is the terminal state
        report = await replay_mod.replay_recorded_execution(
            recording=recording,
            harness_factory=FixTypoHarness,
            checkpointer=InMemorySaver(),
            source_graph=source_graph,
            from_checkpoint=step.checkpoint_id,
        )
        assert report["verified"], (step.node, report["checks"])
        assert report["model_calls_issued"] == 0
        assert report["recorded_nodes"] == recording.continuation_nodes(step.index)
        assert (
            report["replayed_final_state_sha256"] == recorded["final_state_sha256"]
        )


async def test_replay_from_a_checkpoint_consumes_only_the_continuation(
    tmp_path: Path,
):
    saver = InMemorySaver()
    recording, _ = await _record(tmp_path, saver)
    source_graph = build_inner_graph(FixTypoHarness(), checkpointer=saver)
    mid = [s for s in recording.steps if s.node == "plan"][0]

    report = await replay_mod.replay_recorded_execution(
        recording=recording,
        harness_factory=FixTypoHarness,
        checkpointer=InMemorySaver(),
        source_graph=source_graph,
        from_checkpoint=mid.checkpoint_id,
    )

    assert report["verified"]
    assert report["tape_entries_consumed"] == len(recording.entries) - mid.tape_length
    assert "orient" not in report["replayed_nodes"]


async def test_replay_from_an_unknown_checkpoint_raises(tmp_path: Path):
    recording, _ = await _record(tmp_path, InMemorySaver())
    with pytest.raises(KeyError, match="not in recording"):
        await replay_mod.replay_recorded_execution(
            recording=recording,
            harness_factory=FixTypoHarness,
            checkpointer=InMemorySaver(),
            source_graph=object(),
            from_checkpoint="not-a-checkpoint",
        )


async def test_replay_from_a_checkpoint_requires_the_source_graph(tmp_path: Path):
    recording, _ = await _record(tmp_path, InMemorySaver())
    step = [s for s in recording.steps if s.checkpoint_id][0]
    with pytest.raises(ValueError, match="needs source_graph"):
        await replay_mod.replay_recorded_execution(
            recording=recording,
            harness_factory=FixTypoHarness,
            checkpointer=InMemorySaver(),
            from_checkpoint=step.checkpoint_id,
        )


# ── the tape and the checkpoint history must agree ────────────────────


async def test_every_taped_step_is_joined_to_a_checkpoint(tmp_path: Path):
    recording, _ = await _record(tmp_path, InMemorySaver())
    assert recording.steps
    for step in recording.steps:
        assert step.checkpoint_id, f"step {step.index} has no checkpoint"
        assert step.state_hash, f"step {step.index} has no state hash"
    ids = [s.checkpoint_id for s in recording.steps]
    assert len(set(ids)) == len(ids), "checkpoint ids must be unique per step"


async def test_recorded_nodes_follow_the_inner_graph_shape(tmp_path: Path):
    recording, _ = await _record(tmp_path, InMemorySaver())
    nodes = [s.node for s in recording.steps]
    assert nodes[:3] == ["orient", "plan", "act"]
    assert nodes[-1] == "submit"
    assert set(nodes) <= set(replay_mod.INNER_NODES)


# ── the same guarantees against real Postgres ─────────────────────────

_PG_OK = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
    healthcheck()
)

pg_only = pytest.mark.skipif(
    not _PG_OK,
    reason="Postgres not reachable at configured DSN; bring up via docker compose",
)


@pg_only
async def test_exact_replay_round_trips_through_postgres(tmp_path: Path):
    """The stored-checkpoint claim is about Postgres, so prove it there."""
    thread = unique("pg-replay")
    async with persistence_layer() as saver:
        recording, recorded = await _record(tmp_path, saver, thread=thread)
        source_graph = build_inner_graph(FixTypoHarness(), checkpointer=saver)
        mid = [s for s in recording.steps if s.checkpoint_id][2]

        full = await replay_mod.replay_recorded_execution(
            recording=recording,
            harness_factory=FixTypoHarness,
            checkpointer=saver,
            replay_thread_id=unique("pg-replay-full"),
        )
        partial = await replay_mod.replay_recorded_execution(
            recording=recording,
            harness_factory=FixTypoHarness,
            checkpointer=saver,
            source_graph=source_graph,
            from_checkpoint=mid.checkpoint_id,
            replay_thread_id=unique("pg-replay-mid"),
        )

    assert full["verified"], full["checks"]
    assert partial["verified"], partial["checks"]
    assert full["model_calls_issued"] == partial["model_calls_issued"] == 0
    assert (
        full["replayed_final_state_sha256"]
        == partial["replayed_final_state_sha256"]
        == recorded["final_state_sha256"]
    )


@pg_only
async def test_replay_does_not_mutate_the_recorded_thread(tmp_path: Path):
    """Replaying onto a new thread leaves the original history untouched."""
    thread = unique("pg-immutable")
    async with persistence_layer() as saver:
        recording, _ = await _record(tmp_path, saver, thread=thread)
        graph = build_inner_graph(FixTypoHarness(), checkpointer=saver)
        before = await replay_mod.collect_execution_steps(graph, thread_id=thread)

        await replay_mod.replay_recorded_execution(
            recording=recording,
            harness_factory=FixTypoHarness,
            checkpointer=saver,
            replay_thread_id=unique("pg-immutable-replay"),
        )

        after = await replay_mod.collect_execution_steps(graph, thread_id=thread)

    assert [s.checkpoint_id for s in before] == [s.checkpoint_id for s in after]
    assert [s.state_hash for s in before] == [s.state_hash for s in after]


# ── recording onto a used thread ──────────────────────────────────────


async def test_recording_refuses_a_thread_that_already_has_history(tmp_path: Path):
    """A tape and a checkpoint history must line up, or replay cannot start.

    Checkpoint history outlives the process, so a fixed thread id
    accumulates across runs. Caught here, where the message can say what
    to do; caught later, it looks like a corrupt recording.
    """
    saver = InMemorySaver()
    thread = unique("reused")
    await _record(tmp_path, saver, thread=thread)

    with pytest.raises(rec.RecordingError, match="already has checkpoint history"):
        await replay_mod.record_inner_execution(
            harness_factory=FixTypoHarness,
            task_dict=load_task(REPO_ROOT, TASK_ID),
            workspace=_workspace(tmp_path / "second"),
            thread_id=thread,
            checkpointer=saver,
            recording_dir=tmp_path / "recordings" / "second",
        )
