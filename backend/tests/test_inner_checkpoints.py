"""Inner-loop LangGraph checkpointing in Postgres.

The project claims *two* checkpointed state machines. The outer graph
was always compiled with the AsyncPostgresSaver; the inner graph
accepted a ``checkpointer`` argument that the real benchmark path never
passed, so inner node transitions were never persisted.

These tests run the inner graph with a stubbed LLM (no API key, no
network) and assert its transitions land in Postgres under a thread id
that identifies the exact (run, branch, candidate, task, trial) that
produced them.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import benchmark as bench  # noqa: E402
from app.meta_harness import metrics as met  # noqa: E402
from app.meta_harness.harness import CodingAgentHarness  # noqa: E402
from app.meta_harness.inner import build_inner_graph, run_inner_loop  # noqa: E402
from app.meta_harness.persistence import healthcheck, persistence_layer  # noqa: E402
from tests.conftest import unique_name  # noqa: E402

_PG_OK = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
    healthcheck()
)

pytestmark = pytest.mark.skipif(
    not _PG_OK,
    reason="Postgres not reachable at configured DSN; bring up via docker compose",
)


class _Block:
    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _Usage:
    input_tokens = 120
    output_tokens = 40
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _Response:
    def __init__(self, content: list[Any]) -> None:
        self.content = content
        self.usage = _Usage()
        self.model = "claude-haiku-4-5-20251001"


class StubHarness(CodingAgentHarness):
    """A harness whose LLM calls are canned. No API key, no network."""

    MAX_ACT_TURNS = 1

    def __init__(self) -> None:  # noqa: D107 - deliberately skips the API key check
        self.api_key = "stub"
        self._client = None

    async def _call_llm(self, messages, tools, *, tool_choice=None):
        if tool_choice is not None:  # plan phase — forced submit_plan
            return _Response(
                [
                    _Block(
                        type="tool_use",
                        id="plan-1",
                        name="submit_plan",
                        input={"summary": "stub plan", "steps": []},
                    )
                ]
            )
        return _Response(
            [_Block(type="tool_use", id="done-1", name="task_complete", input={})]
        )


def _task(workspace: Path) -> dict[str, Any]:
    (workspace / "ok.py").write_text("VALUE = 1\n")
    return {
        "id": "stub-task",
        "instruction": "do nothing",
        # Trivially green so verify short-circuits to submit.
        "test_command": f'"{sys.executable}" -c "pass"',
    }


async def test_inner_graph_transitions_persist_in_postgres(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    task = _task(workspace)
    thread_id = met.inner_thread_id(
        run_id=unique_name("run-x"),
        thread_id="run-x.fork.beef",
        candidate="cand-1",
        task_id="stub-task",
        trial=3,
    )

    async with persistence_layer() as saver:
        final = await run_inner_loop(
            StubHarness(),
            task_dict=task,
            workspace=workspace,
            trace_dir=tmp_path / "trace",
            thread_id=thread_id,
            checkpointer=saver,
        )
        assert final["score"] == 1.0

        history = [
            snapshot
            async for snapshot in saver.alist(
                config={"configurable": {"thread_id": thread_id}}
            )
        ]

    # orient -> plan -> act -> verify -> submit, plus the input write.
    assert len(history) >= 5, f"expected >=5 inner checkpoints, got {len(history)}"


async def test_inner_thread_ids_identify_their_trial(tmp_path: Path):
    """Every inner checkpoint is attributable back to one exact trial."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    task = _task(workspace)
    thread_id = met.inner_thread_id(
        run_id=(_ry := unique_name("run-y")),
        thread_id=_ry,
        candidate="cand-2",
        task_id="stub-task",
        trial=7,
    )

    async with persistence_layer() as saver:
        await run_inner_loop(
            StubHarness(),
            task_dict=task,
            workspace=workspace,
            trace_dir=tmp_path / "trace",
            thread_id=thread_id,
            checkpointer=saver,
        )

    parsed = met.parse_inner_thread_id(thread_id)
    assert parsed == {
        "run_id": _ry,
        "thread_id": _ry,
        "candidate": "cand-2",
        "task_id": "stub-task",
        "trial": 7,
    }


async def test_two_branches_do_not_share_inner_checkpoint_threads(tmp_path: Path):
    """Same candidate label on two branches => distinct inner histories."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    task = _task(workspace)

    run_id = unique_name("run-z")
    ids = [
        met.inner_thread_id(
            run_id=run_id,
            thread_id=branch,
            candidate="same-label",
            task_id="stub-task",
            trial=1,
        )
        for branch in (run_id, f"{run_id}.fork.abcd")
    ]
    assert ids[0] != ids[1]

    async with persistence_layer() as saver:
        for thread_id in ids:
            await run_inner_loop(
                StubHarness(),
                task_dict=task,
                workspace=workspace,
                trace_dir=tmp_path / f"trace-{thread_id[-6:]}",
                thread_id=thread_id,
                checkpointer=saver,
            )
        counts = []
        for thread_id in ids:
            counts.append(
                len(
                    [
                        s
                        async for s in saver.alist(
                            config={"configurable": {"thread_id": thread_id}}
                        )
                    ]
                )
            )

    # Each thread holds its own history rather than one merged pile.
    assert counts[0] >= 5 and counts[1] >= 5
    assert counts[0] == counts[1]


async def test_benchmark_core_threads_the_checkpointer_into_every_trial(
    tmp_path: Path,
):
    """The real benchmark path — not just run_inner_loop — checkpoints."""
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "stub-task"
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "task.json").write_text(json.dumps(_task(task_dir / "workspace")))

    run_id = unique_name("run-b")
    prefix = f"inner::{run_id}::{run_id}::cand"
    async with persistence_layer() as saver:
        result = await bench.benchmark_harness(
            harness_factory=StubHarness,
            tasks_dir=tasks_dir,
            trials=2,
            workers=1,
            trace_root=tmp_path / "traces",
            thread_prefix=prefix,
            checkpointer=saver,
        )
        assert result["accuracy"] == 1.0
        assert result["metrics_source"] == "measured"
        # Real usage was recorded, not a placeholder zero block.
        assert result["tokens"]["total_tokens"] > 0
        assert result["total_llm_calls"] > 0

        for trial in (1, 2):
            thread_id = f"{prefix}::stub-task::trial-{trial}"
            history = [
                s
                async for s in saver.alist(
                    config={"configurable": {"thread_id": thread_id}}
                )
            ]
            assert history, f"no inner checkpoints for {thread_id}"


def test_inner_graph_compiles_with_and_without_a_checkpointer():
    assert build_inner_graph(StubHarness(), checkpointer=None) is not None
    assert build_inner_graph(StubHarness()) is not None
