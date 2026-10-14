"""The shared trial runner: isolation of failures, timing, and scheduling.

The inner graph is replaced by a scripted stand-in, so these check the
runner itself — what a trial row says, what a raising trial does to its
siblings, what the clock measures, how two arms are scheduled — without
a model, a sandboxed agent, or a real task's tests.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from app.meta_harness import benchmark as bench
from app.meta_harness import experiment as exp
from app.meta_harness import inner as inner_mod
from app.meta_harness import metrics as met


class _Harness:
    MODEL = "declared-model"
    MAX_ACT_TURNS = 25
    MAX_VERIFY_RETRIES = 3

    async def _call_llm(self, *args: Any, **kwargs: Any) -> Any:
        return None


class _Graph:
    """Stands in for the compiled inner graph."""

    def __init__(self, script: "_Script") -> None:
        self.script = script

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        thread_id = config["configurable"]["thread_id"]
        self.script.started.append(thread_id)
        self.script.states.append(state)
        self.script.configs.append(config)
        try:
            await asyncio.sleep(self.script.sleep_s)
        except asyncio.CancelledError:
            self.script.cancelled.append(thread_id)
            raise
        if any(thread_id.endswith(suffix) for suffix in self.script.fail_on):
            raise RuntimeError("529 overloaded")
        self.script.finished.append(thread_id)
        return dict(self.script.final)


class _Script:
    def __init__(
        self,
        *,
        sleep_s: float = 0.0,
        fail_on: tuple[str, ...] = (),
        final: dict[str, Any] | None = None,
    ) -> None:
        self.sleep_s = sleep_s
        self.fail_on = fail_on
        self.final = final if final is not None else {"score": 1.0}
        self.started: list[str] = []
        self.finished: list[str] = []
        self.cancelled: list[str] = []
        self.states: list[dict[str, Any]] = []
        self.configs: list[dict[str, Any]] = []


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> _Script:
    current = _Script()

    def _build(harness: Any, *, checkpointer: Any = None, effects: Any = None) -> _Graph:
        return _Graph(current)

    monkeypatch.setattr(bench, "build_inner_graph", _build)
    return current


def _tasks(root: Path, *names: str) -> list[Path]:
    out = []
    for name in names:
        task = root / name
        (task / "workspace").mkdir(parents=True)
        (task / "workspace" / "a.py").write_text("x = 1\n")
        (task / "task.json").write_text(
            json.dumps({"id": name, "instruction": "do it", "test_command": "true"})
        )
        out.append(task)
    return out


async def _run(task_dirs: list[Path], *, trials: int, workers: int, **kwargs: Any):
    return await bench.run_trials(
        harness_factory=kwargs.pop("harness_factory", _Harness),
        task_dirs=task_dirs,
        trials=trials,
        workers=workers,
        trace_dir_for=lambda task_id, trial: None,
        inner_thread_id_for=lambda task_id, trial: f"{task_id}::trial-{trial}",
        **kwargs,
    )


async def test_a_raising_trial_becomes_an_error_row_and_the_run_finishes(
    tmp_path: Path, script: _Script
):
    script.fail_on = ("task-a::trial-1",)
    rows = await _run(_tasks(tmp_path, "task-a", "task-b"), trials=2, workers=2)

    assert len(rows) == 4
    errors = [r for r in rows if met.is_error_row(r)]
    assert len(errors) == 1
    error = errors[0]
    assert (error["task_id"], error["trial"]) == ("task-a", 1)
    assert error["passed"] is False
    assert "529 overloaded" in error["error"]
    assert error["total_tokens"] is None and error["cost_usd"] is None
    assert error["model"] == "declared-model"
    # Every other trial ran to completion.
    assert sorted(script.finished) == [
        "task-a::trial-2",
        "task-b::trial-1",
        "task-b::trial-2",
    ]
    summary = bench.summarize(rows, task_ids=["task-a", "task-b"], trials=2)
    assert summary["errored_trials"] == 1
    assert summary["per_task"]["task-a"]["errored_trials"] == 1


async def test_a_harness_that_cannot_be_built_is_an_error_row(tmp_path: Path, script: _Script):
    def _broken() -> Any:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    rows = await _run(_tasks(tmp_path, "task-a"), trials=1, workers=1, harness_factory=_broken)
    assert rows[0]["status"] == met.TRIAL_ERROR
    assert rows[0]["model"] is None


async def test_wall_time_measures_the_trial_not_the_queue(tmp_path: Path, script: _Script):
    script.sleep_s = 0.25
    started = time.monotonic()
    rows = await _run(_tasks(tmp_path, "task-a"), trials=3, workers=1)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.7, "with one worker the trials ran one after another"
    for row in rows:
        # Queued behind two others, the last trial would read ~0.75s if
        # the clock started before the semaphore.
        assert row["wall_time_s"] < 0.5, row


async def test_cancelling_the_run_cancels_every_sibling(tmp_path: Path, script: _Script):
    script.sleep_s = 30
    task = asyncio.create_task(_run(_tasks(tmp_path, "task-a", "task-b"), trials=2, workers=4))
    while len(script.started) < 4:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(script.cancelled) == sorted(script.started)
    assert script.finished == []


async def test_a_failure_no_trial_owns_stops_the_siblings_before_raising(
    tmp_path: Path, script: _Script
):
    script.sleep_s = 0.05

    def _progress(row: dict[str, Any]) -> None:
        raise ValueError("progress sink broke")

    with pytest.raises(ValueError, match="progress sink broke"):
        await _run(
            _tasks(tmp_path, "task-a"), trials=6, workers=1, on_trial=_progress
        )
    # One trial finished and its callback raised; the queued ones were
    # cancelled rather than left running unattended.
    await asyncio.sleep(0.2)
    assert len(script.finished) == 1


async def test_each_trial_is_given_its_pristine_source_workspace(
    tmp_path: Path, script: _Script
):
    task_dirs = _tasks(tmp_path, "task-a")
    await _run(task_dirs, trials=1, workers=1)
    task = script.states[0]["task"]
    assert task["_source_workspace"] == str(task_dirs[0] / "workspace")
    # The agent works in a copy, never in the source.
    assert script.states[0]["workspace_path"] != task["_source_workspace"]


async def test_grading_facts_are_carried_onto_the_row(tmp_path: Path, script: _Script):
    script.final = {
        "score": 0.0,
        "grade_result": {
            "passed": False,
            "output": "1 failed",
            "carried_files": ["a.py"],
            "out_of_plan_changes": ["tests/test_a.py"],
            "agent_verify_passed": True,
        },
    }
    rows = await _run(_tasks(tmp_path, "task-a"), trials=1, workers=1)
    assert rows[0]["passed"] is False
    assert rows[0]["agent_verify_passed"] is True
    assert rows[0]["out_of_plan_changes"] == ["tests/test_a.py"]


async def test_a_final_state_without_grading_reports_unknown(tmp_path: Path, script: _Script):
    rows = await _run(_tasks(tmp_path, "task-a"), trials=1, workers=1)
    assert rows[0]["agent_verify_passed"] is None
    assert rows[0]["out_of_plan_changes"] is None
    assert rows[0]["status"] == met.TRIAL_OK
    assert rows[0]["model"] == "declared-model"


async def test_the_step_budget_follows_the_harness(
    tmp_path: Path, script: _Script, monkeypatch: pytest.MonkeyPatch
):
    class _Patient(_Harness):
        MAX_ACT_TURNS = 60
        MAX_VERIFY_RETRIES = 5

    monkeypatch.delattr(inner_mod, "recursion_limit_for", raising=False)
    await _run(_tasks(tmp_path, "task-a"), trials=1, workers=1, harness_factory=_Patient)
    fallback = script.configs[-1]["recursion_limit"]
    assert fallback == bench.recursion_limit_for(_Patient())
    assert fallback > bench.recursion_limit_for(_Harness())
    # Enough for every retry to spend its full act budget.
    assert fallback >= (5 + 1) * 60

    monkeypatch.setattr(inner_mod, "recursion_limit_for", lambda harness: 777, raising=False)
    await _run(_tasks(tmp_path / "again", "task-a"), trials=1, workers=1)
    assert script.configs[-1]["recursion_limit"] == 777


async def test_asking_for_a_tape_without_a_checkpointer_fails_before_any_trial(
    tmp_path: Path, script: _Script
):
    with pytest.raises(ValueError, match="no checkpointer"):
        await _run(
            _tasks(tmp_path, "task-a"),
            trials=1,
            workers=1,
            recording_dir_for=lambda task_id, trial: tmp_path / "tape",
        )
    assert script.started == []


# ── the two-arm schedule ──────────────────────────────────────────────


def test_the_two_arms_are_interleaved_and_alternate_who_goes_first(tmp_path: Path):
    task_dirs = _tasks(tmp_path, "task-a", "task-b")
    specs = exp.interleaved_trial_specs(
        factories={"baseline": _Harness, "candidate": _Harness},
        task_dirs=task_dirs,
        trials=2,
        output_dir=tmp_path / "out",
        experiment_id="exp",
    )
    order = [(s.task_dir.name, s.trial, s.arm) for s in specs]
    assert order == [
        ("task-a", 1, "baseline"),
        ("task-a", 1, "candidate"),
        ("task-b", 1, "candidate"),
        ("task-b", 1, "baseline"),
        ("task-a", 2, "candidate"),
        ("task-a", 2, "baseline"),
        ("task-b", 2, "baseline"),
        ("task-b", 2, "candidate"),
    ]
    # Each arm sees every (task, trial) once, and goes first equally often.
    firsts = [order[i][2] for i in range(0, len(order), 2)]
    assert firsts.count("baseline") == firsts.count("candidate")


async def test_the_experiment_runs_both_arms_in_one_interleaved_schedule(
    tmp_path: Path, script: _Script
):
    tasks_dir = tmp_path / "tasks"
    _tasks(tasks_dir, "task-a", "task-b")
    seen: list[str] = []

    result = await exp.run_two_arm_experiment(
        repo_root=tmp_path,
        config={"experiment": "t"},
        experiment_id="exp-1",
        tasks_dir=tasks_dir,
        baseline_label="baseline",
        candidate_label="cand",
        baseline_factory=_Harness,
        candidate_factory=_Harness,
        arm_sources={},
        model="declared-model",
        trials=3,
        workers=1,
        output_dir=tmp_path / "out",
        on_trial=lambda arm, row: seen.append(arm),
    )
    # Never a run of more than two of the same arm in a row: the arms
    # share the clock rather than taking turns at it.
    runs = max(
        len(run) for run in "".join("b" if a == "baseline" else "c" for a in seen).replace(
            "bc", "b|c").replace("cb", "c|b").split("|")
    )
    assert runs <= 2, seen
    assert result["config"]["schedule"] == "interleaved"
    assert result["validation"]["identical_protocol"] is True
    assert result["validation"]["models"]["baseline"] == ["declared-model"]
    assert [r["arm"] for r in result["baseline_rows"]] == ["baseline"] * 6
