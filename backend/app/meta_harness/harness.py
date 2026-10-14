"""``CodingAgentHarness`` — base class with the 11 override points (the
search space).

Per Appendix C §C.9 / INTERFACES.md §4. The 6 fixed inner-loop tools
and the 5 phase *nodes* (``orient``, ``plan``, ``act``, ``verify``,
``submit``) are NOT overridable; everything below is.

Candidates subclass ``CodingAgentHarness`` and override any subset of
the 11 marked points. Override 11 (structural) is
``_wire_inner_graph``: it decides how the five fixed phases are
connected — skip ``plan``, loop ``verify`` back to ``plan`` — while the
node bodies, and with them the effects boundary that exact replay rests
on, stay fixed. ``inner.build_inner_graph`` calls it on the candidate's
own harness, so a candidate reaches it by overriding the method.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import anthropic
from langgraph.graph import END, START

#: Shown to the model in place of the turns override 10 dropped.
ELIDED_NOTE = "[earlier turns elided to fit context]"

#: Most messages the default overflow strategy keeps, the first included.
_OVERFLOW_KEEP = 20


def wire_default_inner_graph(
    graph: Any, *, route_after_verify: Callable[[dict[str, Any]], str]
) -> None:
    """``orient → plan → act → verify → (act | submit)``, then END."""
    graph.add_edge(START, "orient")
    graph.add_edge("orient", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "verify")
    graph.add_conditional_edges(
        "verify", route_after_verify, {"act": "act", "submit": "submit"}
    )
    graph.add_edge("submit", END)


def _with_elided_note(message: dict[str, Any]) -> dict[str, Any]:
    """``message`` with the elision note appended to its content, once."""
    content = message.get("content")
    if isinstance(content, str):
        if content.endswith(ELIDED_NOTE):
            return message
        return {**message, "content": f"{content}\n\n{ELIDED_NOTE}"}
    if isinstance(content, list):
        last = content[-1] if content else None
        if isinstance(last, dict) and last.get("text") == ELIDED_NOTE:
            return message
        note = {"type": "text", "text": ELIDED_NOTE}
        return {**message, "content": [*content, note]}
    return message


_DEFAULT_SYSTEM_PROMPT = """\
You are a careful coding assistant. You have access to tools to read,
edit, and execute code in a sandboxed workspace. Solve the user's task
by:

1. Reading relevant files first — especially tests, when present.
2. Following the plan you were given.
3. Making targeted edits with apply_patch (preferred) or write_file
   (only for files that don't yet exist).
4. Running tests to verify your work.
5. Calling task_complete when you're confident the task is solved AND
   all tests pass.

Prefer minimal, surgical changes. Do not modify code unrelated to the
task. Use unified-diff patches that match the file's exact current
content; on context_mismatch, the tool returns the file's actual
content at the failed range — re-issue a corrected patch.
"""


_DEFAULT_PLAN_PROMPT_TEMPLATE = """\
You are about to solve a coding task. Build a structured plan first.

**Task:**
{instruction}

**Workspace tree (depth-limited):**
{tree}

**Project info:**
- Language: {lang}
- Test runner: {test_runner}

**Tests already in place (read these as a contract):**
{tests}

Call ``submit_plan`` now with:
- ``summary``: one-line description of what you'll do.
- ``steps``: ordered list of ``{{action, target, why}}`` entries.
- ``expected_files_changed``: which files you intend to touch.
- ``tests_to_run``: which tests to verify.
- ``risk_factors``: edge cases or gotchas to watch.
"""


PLAN_TOOL_SCHEMA: dict[str, Any] = {
    "name": "submit_plan",
    "description": "Submit your structured plan for the task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        "why": {"type": "string"},
                    },
                    "required": ["action", "target"],
                },
            },
            "expected_files_changed": {
                "type": "array",
                "items": {"type": "string"},
            },
            "tests_to_run": {
                "type": "array",
                "items": {"type": "string"},
            },
            "risk_factors": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["summary", "steps"],
    },
}


class CodingAgentHarness:
    """Base inner-loop harness. Override any of the marked points."""

    # Override 1 — system prompt
    SYSTEM_PROMPT: str = _DEFAULT_SYSTEM_PROMPT
    # Override 2 — plan prompt template
    PLAN_PROMPT_TEMPLATE: str = _DEFAULT_PLAN_PROMPT_TEMPLATE
    # Override 3 — turn budget for the act phase
    MAX_ACT_TURNS: int = 25
    # Override 4 — verify→act retry budget
    MAX_VERIFY_RETRIES: int = 3

    # Model knobs (not strict override points but candidates may tune).
    # Default = Haiku 4.5: rate-limit-friendly for parallel benchmarks,
    # ~10× cheaper than Sonnet, capable enough for the 5 calibration
    # tasks. Override with ``META_HARNESS_INNER_MODEL`` for Sonnet/Opus.
    MODEL: str = os.environ.get(
        "META_HARNESS_INNER_MODEL", "claude-haiku-4-5-20251001"
    )
    MAX_TOKENS: int = 4096

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env or export it "
                "before running the inner loop."
            )
        # Async client — required for use inside LangGraph async nodes
        # (see Appendix A §A.4 Gotcha 1: sync I/O inside async nodes
        # blocks the event loop). All node bodies in inner.py and
        # outer.py are async; ``_call_llm`` is awaited.
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)

    # ──────────────────────────────────────────────────────────────────
    # Override points 5–10 (methods).
    # ──────────────────────────────────────────────────────────────────

    # Override 5
    def _build_initial_context(self, orient_summary: dict[str, Any]) -> dict[str, Any]:
        """Project orient_summary into the structure the planner sees."""
        return orient_summary

    # Override 6
    def _format_tool_result(self, name: str, result: dict[str, Any]) -> str:
        """How tool outputs are rendered back to the model."""
        formatted = json.dumps(result, indent=2, default=str)
        if len(formatted) > 4000:
            formatted = (
                formatted[:1500]
                + f"\n[... truncated {len(formatted) - 3000} chars ...]\n"
                + formatted[-1500:]
            )
        return formatted

    # Override 7
    def _compose_act_prompt(self, plan: dict[str, Any]) -> str:
        """How the plan is injected into the act phase's first user turn."""
        return (
            "Execute this plan. Use the tools to read, edit, and verify. "
            "Call task_complete when all tests pass.\n\n"
            f"{json.dumps(plan, indent=2)}"
        )

    # Override 8 — the actual API call (now async; see Appendix A §A.4)
    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        """The Anthropic API call. Override for caching, ordering, etc."""
        kwargs: dict[str, Any] = {}
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return await self._client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            messages=messages,
            tools=tools,
            system=self.SYSTEM_PROMPT,
            **kwargs,
        )

    # Override 9 — control whether to retry act after verify failure
    def should_loop_back_to_act(self, verify_result: dict[str, Any]) -> bool:
        """Default: loop back if tests didn't pass."""
        return not verify_result.get("tests_pass", False)

    # Override 10 — context-overflow strategy
    def _summarize_for_overflow(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """When messages would exceed model context, trim.

        Default: keep the first message (the act prompt, with a note that
        turns were elided) and the most recent whole turns, at most 20
        messages in all. A cut is only ever made just before an
        ``assistant`` message, which is what keeps the result a valid
        request:

        - every ``tool_use`` keeps the ``tool_result`` that answers it in
          the next message, and no ``tool_result`` loses its ``tool_use``;
        - roles still alternate, because the note is folded into the
          first (user) message rather than sent as a second user turn.

        The old default — first 2 + a separate note + last 18 — orphaned
        the first turn's ``tool_use`` every time it fired (and, at even
        lengths, the window's leading ``tool_result``), so every trial
        that reached ~21 act turns got a 400 from the API.
        """
        if len(messages) <= _OVERFLOW_KEEP:
            return messages
        cuts = [
            index
            for index in range(1, len(messages))
            if messages[index].get("role") == "assistant"
            and messages[index - 1].get("role") == "user"
        ]
        if not cuts or messages[0].get("role") != "user":
            return messages
        fitting = [i for i in cuts if 1 + len(messages) - i <= _OVERFLOW_KEEP]
        cut = fitting[0] if fitting else cuts[-1]
        if cut == 1:
            return messages
        return [_with_elided_note(messages[0]), *messages[cut:]]

    # Override 11 — structural: how the five fixed phases are connected
    def _wire_inner_graph(
        self,
        graph: Any,
        *,
        route_after_verify: Callable[[dict[str, Any]], str],
    ) -> None:
        """Add the edges between the five phase nodes of ``graph``.

        ``graph`` is a ``StateGraph`` that already holds the nodes
        ``orient``, ``plan``, ``act``, ``verify`` and ``submit``; their
        bodies are fixed. ``route_after_verify`` is the default retry
        router (it honours ``MAX_VERIFY_RETRIES`` and override 9). A
        candidate may wire them differently — skip ``plan``, or send a
        failed verify back to ``plan`` — but only ``submit`` grades the
        trial: a wiring that never reaches it scores nothing.
        """
        wire_default_inner_graph(graph, route_after_verify=route_after_verify)
