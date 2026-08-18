"""Checkpoint state restoration + recorded-event replay.

These tests pin down exactly what "replay" means in this project:

- restoring a stored checkpoint returns identical state, provably, via a
  canonical-JSON SHA-256;
- walking a thread's recorded transitions is deterministic and issues no
  model calls.

They deliberately do NOT assert that re-executing the graph reproduces
identical model output — it doesn't, and no documentation may claim it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import replay as replay_mod  # noqa: E402
from app.meta_harness import runs as runs_mod  # noqa: E402
from app.meta_harness.branches import get_state_history  # noqa: E402
from app.meta_harness.outer import OuterLoopRunner, initial_state  # noqa: E402
from app.meta_harness.persistence import healthcheck, persistence_layer  # noqa: E402
from tests.conftest import unique_name as unique  # noqa: E402


# ── pure helpers (no Postgres needed) ─────────────────────────────────


def test_canonical_json_is_order_independent():
    a = {"b": 1, "a": [3, 2], "c": {"y": 1, "x": 2}}
    b = {"c": {"x": 2, "y": 1}, "a": [3, 2], "b": 1}
    assert replay_mod.canonical_json(a) == replay_mod.canonical_json(b)
    assert replay_mod.state_hash(a) == replay_mod.state_hash(b)


def test_state_hash_detects_any_difference():
    base = {"iteration": 2, "best_candidate": "x"}
    assert replay_mod.state_hash(base) != replay_mod.state_hash(
        {"iteration": 2, "best_candidate": "y"}
    )
    # List order is meaningful and must change the hash.
    assert replay_mod.state_hash({"c": [1, 2]}) != replay_mod.state_hash(
        {"c": [2, 1]}
    )


def test_state_hash_handles_non_json_values():
    """A state holding a Path must still hash, and hash stably."""
    state = {"dir": Path("/tmp/x"), "n": 1}
    assert replay_mod.state_hash(state) == replay_mod.state_hash(dict(state))


# ── Postgres-backed round trips ───────────────────────────────────────

_PG_OK = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
    healthcheck()
)

pg_only = pytest.mark.skipif(
    not _PG_OK,
    reason="Postgres not reachable at configured DSN; bring up via docker compose",
)


async def _seeded_run(tmp_path: Path, name: str, saver, *, budget: int = 2):
    run_dir = runs_mod.make_run_dir(tmp_path, name, fresh=True)
    runner = OuterLoopRunner(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=2,
        bench_workers=1,
        checkpointer=saver,
    )
    runs_mod.write_manifest(
        run_dir,
        run_id=name,
        thread_id=name,
        budget=budget,
        trials=2,
        mock_proposer=True,
        mock_bench=True,
    )
    seeds = [await runner.benchmark_baseline(run_id=name)]
    graph = runner.build()
    await graph.ainvoke(
        initial_state(run_id=name, budget=budget, seed_candidates=seeds),
        config={"configurable": {"thread_id": name}, "recursion_limit": 200},
    )
    return run_dir, graph


@pg_only
async def test_restored_checkpoint_state_is_identical(tmp_path: Path):
    """hash(saved state) == hash(restored state), across repeated reads."""
    name = unique("replay-identity")
    async with persistence_layer() as saver:
        _, graph = await _seeded_run(tmp_path, name, saver)
        history = await get_state_history(graph, thread_id=name)
        target = history[len(history) // 2]

        first = await replay_mod.restore_checkpoint(
            graph, thread_id=name, checkpoint_id=target.checkpoint_id
        )
        second = await replay_mod.restore_checkpoint(
            graph, thread_id=name, checkpoint_id=target.checkpoint_id
        )

    assert first["state_hash"] == second["state_hash"]
    assert first["state"] == second["state"]
    assert first["state_hash"] == replay_mod.state_hash(first["state"])
    # A real state, not an empty dict masquerading as one.
    assert "iteration" in first["state"]


@pg_only
async def test_different_checkpoints_have_different_state_hashes(tmp_path: Path):
    name = unique("replay-distinct")
    async with persistence_layer() as saver:
        _, graph = await _seeded_run(tmp_path, name, saver)
        history = await get_state_history(graph, thread_id=name)
        hashes = set()
        for record in history:
            restored = await replay_mod.restore_checkpoint(
                graph,
                thread_id=name,
                checkpoint_id=record.checkpoint_id,
            )
            hashes.add(restored["state_hash"])

    # The run advanced through several distinct states.
    assert len(hashes) >= 3


@pg_only
async def test_restore_rejects_an_unknown_checkpoint(tmp_path: Path):
    name = unique("replay-missing")
    async with persistence_layer() as saver:
        _, graph = await _seeded_run(tmp_path, name, saver)
        with pytest.raises(KeyError):
            await replay_mod.restore_checkpoint(
                graph, thread_id=name, checkpoint_id="not-a-checkpoint"
            )


@pg_only
async def test_event_replay_is_ordered_and_deterministic(tmp_path: Path):
    name = unique("replay-events")
    async with persistence_layer() as saver:
        _, graph = await _seeded_run(tmp_path, name, saver)
        first = await replay_mod.replay_thread(graph, thread_id=name)
        second = await replay_mod.replay_thread(graph, thread_id=name)

    assert first["transitions_hash"] == second["transitions_hash"]
    assert first["checkpoint_count"] >= 4
    # Oldest first, contiguous indices.
    indices = [t["index"] for t in first["transitions"]]
    assert indices == list(range(len(indices)))
    iterations = [
        t["iteration"] for t in first["transitions"] if t["iteration"] is not None
    ]
    assert iterations == sorted(iterations)
    # The guarantee string never promises stochastic reproduction.
    assert "no model calls are re-issued" in first["guarantee"]


@pg_only
async def test_replay_does_not_advance_the_thread(tmp_path: Path):
    """Replay is read-only: it must not append checkpoints."""
    name = unique("replay-readonly")
    async with persistence_layer() as saver:
        _, graph = await _seeded_run(tmp_path, name, saver)
        before = await get_state_history(graph, thread_id=name)
        await replay_mod.replay_thread(graph, thread_id=name)
        await replay_mod.restore_checkpoint(
            graph,
            thread_id=name,
            checkpoint_id=before[0].checkpoint_id,
        )
        after = await get_state_history(graph, thread_id=name)

    assert len(after) == len(before)
    assert [r.checkpoint_id for r in after] == [r.checkpoint_id for r in before]
