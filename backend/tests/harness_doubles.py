"""Deterministic inner-loop harnesses for offline tests.

The recording and replay suites need a real inner-loop run — the real
graph, the real tools, the real verify subprocess — without a model and
without an API key. A scripted harness supplies the one thing the
inner loop cannot do offline: the model's side of the conversation.

Everything else stays real, which is the point. A replay test that
stubbed the tools as well would only be testing the stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.meta_harness.harness import CodingAgentHarness


class Block:
    """Duck-types an Anthropic content block."""

    def __init__(self, **fields: Any) -> None:
        self.type = fields.get("type")
        self.text = fields.get("text")
        self.id = fields.get("id")
        self.name = fields.get("name")
        self.input = fields.get("input") or {}


class Usage:
    def __init__(self, input_tokens: int = 1200, output_tokens: int = 180) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class Response:
    def __init__(self, blocks: list[Block], model: str = "scripted-model") -> None:
        self.content = blocks
        self.model = model
        self.usage = Usage()
        self.stop_reason = "end_turn"


class ScriptedHarness(CodingAgentHarness):
    """A harness whose model turns are a fixed script.

    Constructs without ``ANTHROPIC_API_KEY`` and never builds a client:
    an inner-loop run driven by this harness issues no network call at
    all, recorded or replayed.
    """

    MODEL = "scripted-model"
    #: Ordered act-phase turns. Each entry is a list of blocks.
    SCRIPT: list[list[dict[str, Any]]] = []
    PLAN: dict[str, Any] = {"summary": "scripted plan", "steps": []}

    def __init__(self) -> None:  # noqa: D107 — deliberately skips the base __init__
        self.api_key = None
        self._client = None
        self._turn = 0
        self.calls: list[Any] = []

    async def _call_llm(
        self,
        messages: list[Any],
        tools: list[Any],
        *,
        tool_choice: dict[str, Any] | None = None,
    ) -> Response:
        self.calls.append(messages)
        if tool_choice is not None:
            return Response(
                [
                    Block(
                        type="tool_use",
                        id="plan-1",
                        name="submit_plan",
                        input=dict(self.PLAN),
                    )
                ]
            )
        index = min(self._turn, len(self.SCRIPT) - 1)
        self._turn += 1
        return Response([Block(**block) for block in self.SCRIPT[index]])


class FixTypoHarness(ScriptedHarness):
    """Solves ``task-001-fix-typo`` in two act turns, then completes."""

    SCRIPT = [
        [
            {"type": "text", "text": "Reading the calculator."},
            {
                "type": "tool_use",
                "id": "t1",
                "name": "read_file",
                "input": {"path": "calculator.py"},
            },
        ],
        [
            {
                "type": "tool_use",
                "id": "t2",
                "name": "apply_patch",
                "input": {
                    "path": "calculator.py",
                    "patch": (
                        "--- a/calculator.py\n"
                        "+++ b/calculator.py\n"
                        "@@\n"
                        "-    return a - b\n"
                        "+    return a + b\n"
                    ),
                },
            }
        ],
        [{"type": "tool_use", "id": "t3", "name": "task_complete", "input": {}}],
    ]
    PLAN = {
        "summary": "fix add() to return a + b",
        "steps": [{"action": "patch", "target": "calculator.py", "why": "bug"}],
    }


class FailingHarness(ScriptedHarness):
    """Never fixes anything, so the verify → act retry edge is exercised."""

    MAX_VERIFY_RETRIES = 2
    SCRIPT = [
        [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "read_file",
                "input": {"path": "calculator.py"},
            }
        ],
        [{"type": "tool_use", "id": "t2", "name": "task_complete", "input": {}}],
    ]


class DifferentPromptHarness(FixTypoHarness):
    """Composes a different act prompt, so its first request diverges.

    A replay cannot diverge in the *model's* output — that comes from the
    tape. It can diverge in what the harness asks for, which is exactly
    what an override point changes. This harness proves the tape checks
    the request rather than blindly handing back the next entry.
    """

    def _compose_act_prompt(self, plan: dict[str, Any]) -> str:
        return "a materially different instruction to the model"


class NoRetryHarness(FailingHarness):
    """Never loops back to act, so its node sequence diverges from the tape.

    Proves the replayed conditional edge is recomputed from state rather
    than replayed from a recorded list of transitions.
    """

    def should_loop_back_to_act(self, verify_result: dict[str, Any]) -> bool:
        return False


def load_task(repo_root: Path, task_id: str, *, holdout: bool = False) -> dict[str, Any]:
    root = repo_root / "eval" / ("holdout" if holdout else "tasks") / task_id
    return json.loads((root / "task.json").read_text())


def task_workspace(repo_root: Path, task_id: str, *, holdout: bool = False) -> Path:
    return (
        repo_root / "eval" / ("holdout" if holdout else "tasks") / task_id / "workspace"
    )
