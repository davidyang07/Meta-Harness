"""Live inner-loop test on task-001 (BUILD_ORDER step 3 DoD).

Skipped automatically when ``ANTHROPIC_API_KEY`` is not set — the full
end-to-end test requires a real LLM call.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; live LLM test skipped",
)


def test_route_after_verify_uses_harness_retry_budget():
    from app.meta_harness.harness import CodingAgentHarness  # noqa: PLC0415
    from app.meta_harness.inner import _route_after_verify_for_harness  # noqa: PLC0415

    class _Harness(CodingAgentHarness):
        def __init__(self, retries: int) -> None:
            self.MAX_VERIFY_RETRIES = retries

    state = {"verify_result": {"tests_pass": False}, "verify_attempts": 4}
    assert _route_after_verify_for_harness(state, _Harness(5)) == "act"
    assert _route_after_verify_for_harness(state, _Harness(4)) == "submit"
    passed = {"verify_result": {"tests_pass": True}, "verify_attempts": 1}
    assert _route_after_verify_for_harness(passed, _Harness(5)) == "submit"


def test_the_dead_router_is_gone():
    """``_route_after_verify`` ignored the harness policy and nothing used it."""
    from app.meta_harness import inner  # noqa: PLC0415

    assert not hasattr(inner, "_route_after_verify")


def test_route_after_verify_delegates_harness_policy():
    from app.meta_harness.harness import CodingAgentHarness  # noqa: PLC0415
    from app.meta_harness.inner import _route_after_verify_for_harness  # noqa: PLC0415

    class NoRetryHarness(CodingAgentHarness):
        MAX_VERIFY_RETRIES = 5

        def __init__(self) -> None:
            pass

        def should_loop_back_to_act(self, verify_result: dict) -> bool:
            return False

    state = {"verify_result": {"tests_pass": False}, "verify_attempts": 1}

    assert _route_after_verify_for_harness(state, NoRetryHarness()) == "submit"


async def test_plan_uses_initial_context_and_harness_llm_call():
    from app.meta_harness.inner import plan  # noqa: PLC0415

    class _Block:
        type = "tool_use"
        name = "submit_plan"
        input = {"summary": "use custom context", "steps": []}

    class _Response:
        content = [_Block()]

    class _Harness:
        PLAN_PROMPT_TEMPLATE = "{instruction}|{tree}|{lang}|{test_runner}|{tests}"
        seen_messages = None
        seen_tools = None
        seen_tool_choice = None

        def _build_initial_context(self, orient_summary: dict) -> dict:
            return {
                "tree": "custom-tree",
                "project": {"lang": "custom-lang", "test_runner": "custom-test"},
                "tests": {"tests/test_contract.py": "assert True"},
            }

        async def _call_llm(self, messages, tools, *, tool_choice=None):
            self.seen_messages = messages
            self.seen_tools = tools
            self.seen_tool_choice = tool_choice
            return _Response()

    harness = _Harness()
    result = await plan(
        {
            "task": {"instruction": "solve it"},
            "orient_summary": {"tree": "raw-tree"},
        },
        harness,  # type: ignore[arg-type]
    )

    assert result["plan"]["summary"] == "use custom context"
    assert "custom-tree" in harness.seen_messages[0]["content"]
    assert "raw-tree" not in harness.seen_messages[0]["content"]
    assert harness.seen_tools[0]["name"] == "submit_plan"
    assert harness.seen_tool_choice == {"type": "tool", "name": "submit_plan"}


# ── the recursion limit follows the harness' budgets ──────────────────


def test_recursion_limit_is_derived_from_the_harness_budgets():
    from app.meta_harness.harness import CodingAgentHarness  # noqa: PLC0415
    from app.meta_harness.inner import recursion_limit_for  # noqa: PLC0415

    class _Budget(CodingAgentHarness):
        def __init__(self, turns: int, retries: int) -> None:
            self.MAX_ACT_TURNS = turns
            self.MAX_VERIFY_RETRIES = retries

    default = recursion_limit_for(_Budget(25, 3))
    assert default > 3 + 2 * 3  # orient, plan, submit + act/verify per attempt
    assert recursion_limit_for(_Budget(50, 3)) > default
    assert recursion_limit_for(_Budget(25, 6)) > default
    # A harness-like object without the attributes gets the defaults.
    assert recursion_limit_for(object()) == default


async def test_a_long_retry_loop_stays_under_the_derived_limit(tmp_path: Path):
    """60 act/verify rounds need ~123 super-steps; the old fixed limit of
    100 raised GraphRecursionError. The limit now grows with the budget."""
    from app.meta_harness.inner import run_inner_loop  # noqa: PLC0415
    from tests.harness_doubles import FailingHarness  # noqa: PLC0415

    class ManyRetries(FailingHarness):
        MAX_VERIFY_RETRIES = 60

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "calculator.py").write_text("x = 1\n", encoding="utf-8")
    task = {"id": "fast-fail", "instruction": "never passes", "test_command": "exit 1"}
    final = await run_inner_loop(
        ManyRetries(), task_dict=task, workspace=workspace, source_workspace=workspace
    )
    assert final["verify_attempts"] == 60
    assert final["score"] == 0.0


# ── override 11: the wiring of the five fixed phases ──────────────────


async def test_a_candidate_can_rewire_the_phases(tmp_path: Path):
    """Override 11 is reachable: a harness that skips ``plan``."""
    import shutil  # noqa: PLC0415

    from langgraph.graph import END, START  # noqa: PLC0415

    from app.meta_harness.inner import run_inner_loop  # noqa: PLC0415
    from tests.harness_doubles import (  # noqa: PLC0415
        FixTypoHarness,
        load_task,
        task_workspace,
    )

    class SkipPlan(FixTypoHarness):
        def _wire_inner_graph(self, graph, *, route_after_verify):
            graph.add_edge(START, "orient")
            graph.add_edge("orient", "act")
            graph.add_edge("act", "verify")
            graph.add_conditional_edges(
                "verify", route_after_verify, {"act": "act", "submit": "submit"}
            )
            graph.add_edge("submit", END)

    harness = SkipPlan()
    workspace = tmp_path / "ws"
    shutil.copytree(task_workspace(REPO_ROOT, "task-001-fix-typo"), workspace)
    final = await run_inner_loop(
        harness,
        task_dict=load_task(REPO_ROOT, "task-001-fix-typo"),
        workspace=workspace,
    )

    assert final["plan"] is None
    assert all("submit_plan" not in str(call) for call in harness.calls)
    assert final["score"] == 1.0


def test_a_harness_without_the_wiring_hook_gets_the_default_graph():
    """``evidence.graph_node_sets`` compiles the graph from a bare stub."""
    from app.meta_harness.inner import build_inner_graph  # noqa: PLC0415

    class _Stub:
        MAX_VERIFY_RETRIES = 3

        def should_loop_back_to_act(self, verify_result: dict) -> bool:
            return False

    graph = build_inner_graph(_Stub())  # type: ignore[arg-type]
    nodes = {n for n in graph.nodes if not n.startswith("__")}
    assert nodes == {"orient", "plan", "act", "verify", "submit"}


@requires_anthropic
async def test_inner_loop_runs_end_to_end_on_task_001(tmp_path: Path):
    """Run the baseline harness on task-001 and assert all trace files exist."""
    from agents.baseline import BaselineHarness  # noqa: PLC0415

    from app.meta_harness.inner import run_inner_loop  # noqa: PLC0415
    from app.meta_harness.sandbox import sandbox_for  # noqa: PLC0415

    task_dir = REPO_ROOT / "eval" / "tasks" / "task-001-fix-typo"
    task_spec = json.loads((task_dir / "task.json").read_text())

    harness = BaselineHarness()
    trace_dir = tmp_path / "traces"

    with sandbox_for(task_dir / "workspace") as sandbox:
        final_state = await run_inner_loop(
            harness,
            task_dict=task_spec,
            workspace=sandbox,
            trace_dir=trace_dir,
        )

    # Score is one of {0.0, 1.0} — this is a per-trial pass/fail.
    assert final_state.get("score") in (0.0, 1.0)

    # Every trace artifact from INTERFACES.md §2.7-2.11 is present.
    for fname in (
        "orient.json",
        "plan.json",
        "act-messages.jsonl",
        "act-tools.jsonl",
        "verify.json",
        "score.json",
        "summary.md",
        "final-files.json",
    ):
        assert (trace_dir / fname).exists(), f"missing trace artifact: {fname}"
