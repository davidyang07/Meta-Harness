"""Message identity in ``act``, and why it has to be deterministic.

``act`` is the one node that can be re-entered — the ``verify → act``
edge sends it round again — and the one node that writes a reduced state
field. Both facts land on ``state["messages"]``:

- LangGraph's ``add_messages`` mints a random UUID for any message
  without an ``id``. Two executions of the same recorded run would then
  produce different state, and no exact-replay claim could survive that.
- ``add_messages`` merges by id and never removes, so an ``act`` that
  returns a shorter list (override 10's overflow strategy) would leave
  the dropped tail behind.

``act`` therefore stamps positional ids and prefixes a clear, making the
write a replace. That also makes the node idempotent, which the
interrupted-node invariant requires.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import RemoveMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages  # noqa: E402

from app.meta_harness import inner  # noqa: E402
from app.meta_harness.harness import ELIDED_NOTE, CodingAgentHarness  # noqa: E402
from tests.harness_doubles import (  # noqa: E402
    Block,
    CachingHarness,
    FixTypoHarness,
    Response,
    ScriptedHarness,
    load_task,
    task_workspace,
)

TASK_ID = "task-001-fix-typo"


def _apply(update: list[Any], existing: list[Any] | None = None) -> list[Any]:
    """Run a node's messages update through the real reducer."""
    return add_messages(existing or [], update)


# ── deterministic identity ────────────────────────────────────────────


def test_the_reducer_invents_a_uuid_when_a_message_has_no_id():
    """The failure mode being defended against, stated as a test."""
    first = _apply([{"role": "user", "content": "hi"}])
    second = _apply([{"role": "user", "content": "hi"}])
    assert first[0].id != second[0].id


def test_act_stamps_deterministic_positional_ids():
    messages = [
        {"role": "user", "content": "plan"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
    update = inner._messages_update(messages)
    stamped = [m for m in update if not isinstance(m, RemoveMessage)]
    assert [m["id"] for m in stamped] == ["m0000", "m0001"]


def test_the_same_trajectory_reduces_to_the_same_state_every_time():
    messages = [
        {"role": "user", "content": "plan"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
    first = _apply(inner._messages_update(messages))
    second = _apply(inner._messages_update(messages))
    assert [m.id for m in first] == [m.id for m in second] == ["m0000", "m0001"]
    assert [m.content for m in first] == [m.content for m in second]


def test_re_executing_act_replaces_rather_than_appends():
    """An interrupted node is re-executed on resume; it must be idempotent."""
    messages = [{"role": "user", "content": "plan"}]
    state = _apply(inner._messages_update(messages))
    again = _apply(inner._messages_update(messages), state)

    assert len(again) == 1
    assert [m.id for m in again] == ["m0000"]


def test_a_shorter_trajectory_does_not_leave_the_dropped_tail_behind():
    """Override 10 may trim the trajectory; the state must trim with it."""
    long_run = [{"role": "user", "content": f"turn-{i}"} for i in range(6)]
    state = _apply(inner._messages_update(long_run))
    assert len(state) == 6

    trimmed = long_run[:2] + [{"role": "user", "content": "[elided]"}]
    state = _apply(inner._messages_update(trimmed), state)

    assert len(state) == 3
    assert state[-1].content == "[elided]"


def test_the_update_clears_before_writing():
    update = inner._messages_update([{"role": "user", "content": "x"}])
    assert isinstance(update[0], RemoveMessage)
    assert update[0].id == REMOVE_ALL_MESSAGES


# ── the retry path ────────────────────────────────────────────────────


def test_state_messages_are_normalised_back_to_the_wire_shape():
    """On the second entry into act, state holds BaseMessage objects."""
    state = _apply(
        inner._messages_update(
            [
                {"role": "user", "content": "plan"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}
                    ],
                },
            ]
        )
    )

    normalised = inner._as_api_messages(state)

    assert [m["role"] for m in normalised] == ["user", "assistant"]
    assert normalised[0]["content"] == "plan"
    assert normalised[1]["content"][0]["name"] == "read_file"
    assert [m["id"] for m in normalised] == ["m0000", "m0001"]


def test_normalisation_is_a_no_op_for_messages_act_just_built():
    plain = [{"role": "user", "content": "plan"}]
    assert inner._as_api_messages(plain) == plain


def test_the_request_carries_no_bookkeeping_id():
    """``id`` is our state key, not part of the Anthropic request."""
    messages = [{"role": "user", "content": "plan", "id": "m0000"}]
    request = inner._request_messages(messages)
    assert request == [{"role": "user", "content": "plan"}]


def test_the_request_is_identical_on_the_first_and_second_entry_into_act():
    """A replayed request must key the same whichever path built it."""
    built_by_act = [{"role": "user", "content": "plan"}]
    round_tripped = inner._as_api_messages(
        _apply(inner._messages_update(built_by_act))
    )

    assert inner._request_messages(built_by_act) == inner._request_messages(
        round_tripped
    )


def test_the_request_is_a_deep_copy():
    """An override editing its request must not edit the conversation."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "x"}], "id": "m0000"}
    ]
    request = inner._request_messages(messages)
    request[0]["content"][0]["cache_control"] = {"type": "ephemeral"}
    assert "cache_control" not in messages[0]["content"][0]


# ── a valid request at every length (override 10's default) ──────────


def _conversation(
    turns: int, *, tools_per_turn: int = 1, trailing: str | None = None
) -> list[dict[str, Any]]:
    """An act-phase conversation shaped exactly like the one ``act`` builds."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Execute this plan."}]
    for turn in range(turns):
        uses = [
            {
                "type": "tool_use",
                "id": f"tu-{turn}-{k}",
                "name": "read_file",
                "input": {"path": "a.py"},
            }
            for k in range(tools_per_turn)
        ]
        text = {"type": "text", "text": f"t{turn}"}
        messages.append({"role": "assistant", "content": [text, *uses]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": u["id"], "content": "ok"}
                    for u in uses
                ],
            }
        )
    done = {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
    if trailing == "assistant":
        messages.append(done)
    elif trailing == "feedback":
        messages = inner._with_user_text(messages, "Verification attempt 1 failed")
    elif trailing == "assistant+feedback":
        messages.append(done)
        messages = inner._with_user_text(messages, "Verification attempt 1 failed")
    return messages


def _blocks(message: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == kind]


def assert_valid_request(messages: list[dict[str, Any]]) -> None:
    """What the Messages API checks that the old trim broke.

    The first message is the user's; roles alternate; every tool_use is
    answered by a tool_result in the very next message, and every
    tool_result answers a tool_use in the message just before it.
    """
    assert messages and messages[0]["role"] == "user"
    for previous, current in zip(messages, messages[1:]):
        assert previous["role"] != current["role"], "consecutive same-role turns"
    for index, message in enumerate(messages):
        uses = {b["id"] for b in _blocks(message, "tool_use")}
        results = {b["tool_use_id"] for b in _blocks(message, "tool_result")}
        if results:
            before = messages[index - 1] if index else {}
            assert results <= {b["id"] for b in _blocks(before, "tool_use")}, (
                f"orphaned tool_result at {index}"
            )
            kinds = [b.get("type") for b in message["content"]]
            first_other = next(
                (i for i, k in enumerate(kinds) if k != "tool_result"), len(kinds)
            )
            assert "tool_result" not in kinds[first_other:], "text before tool_result"
        if uses and index + 1 < len(messages):
            following = _blocks(messages[index + 1], "tool_result")
            assert uses == {b["tool_use_id"] for b in following}, (
                f"orphaned tool_use at {index}"
            )


def _trim(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    harness = CodingAgentHarness.__new__(CodingAgentHarness)
    return harness._summarize_for_overflow(messages)


@pytest.mark.parametrize("turns", [9, 10, 11, 19, 20, 21, 30, 45])
@pytest.mark.parametrize("tools_per_turn", [1, 3])
@pytest.mark.parametrize(
    "trailing", [None, "assistant", "feedback", "assistant+feedback"]
)
def test_the_default_trim_always_yields_a_valid_request(
    turns: int, tools_per_turn: int, trailing: str | None
):
    messages = _conversation(turns, tools_per_turn=tools_per_turn, trailing=trailing)
    assert_valid_request(messages)

    trimmed = _trim(messages)

    assert_valid_request(trimmed)
    assert len(trimmed) <= 20
    assert trimmed[-1] == messages[-1], "the newest turn is always kept"
    if len(messages) > 20:
        assert ELIDED_NOTE in str(trimmed[0]["content"])


def test_the_old_trim_was_not_a_valid_request():
    """Pin the failure this replaced: first 2 + a note + last 18."""
    messages = _conversation(20)
    old = messages[:2] + [{"role": "user", "content": ELIDED_NOTE}] + messages[-18:]
    with pytest.raises(AssertionError):
        assert_valid_request(old)


def test_a_short_conversation_is_left_alone():
    messages = _conversation(9)
    assert _trim(messages) is messages


def test_trimming_twice_does_not_stack_notes():
    once = _trim(_conversation(25))
    longer = once + _conversation(15)[1:]
    twice = _trim(longer)
    assert str(twice[0]["content"]).count(ELIDED_NOTE) == 1
    assert_valid_request(twice)


# ── the act loop end to end ───────────────────────────────────────────


class LoopingHarness(ScriptedHarness):
    """Never finishes: reads a file every turn with a fresh tool_use id."""

    MAX_ACT_TURNS = 25
    MAX_VERIFY_RETRIES = 2

    async def _call_llm(self, messages, tools, *, tool_choice=None):
        self.calls.append(messages)
        if tool_choice is not None:
            plan = {"summary": "s", "steps": []}
            return Response(
                [Block(type="tool_use", id="plan-1", name="submit_plan", input=plan)]
            )
        self._turn += 1
        return Response(
            [
                Block(type="text", text=f"turn {self._turn}"),
                Block(
                    type="tool_use",
                    id=f"tu-{self._turn}",
                    name="read_file",
                    input={"path": "calculator.py"},
                ),
            ]
        )


def _workspace(tmp_path: Path) -> Path:
    destination = tmp_path / "ws"
    shutil.copytree(task_workspace(REPO_ROOT, TASK_ID), destination)
    return destination


def _act_requests(harness: ScriptedHarness) -> list[list[dict[str, Any]]]:
    return [
        call
        for call in harness.calls
        if str(call[0]["content"]).startswith("Execute this plan")
    ]


async def test_every_request_the_act_loop_sends_is_valid(tmp_path: Path):
    harness = LoopingHarness()
    final = await inner.run_inner_loop(
        harness, task_dict=load_task(REPO_ROOT, TASK_ID), workspace=_workspace(tmp_path)
    )

    requests = _act_requests(harness)
    assert len(requests) == 50
    for request in requests:
        assert_valid_request(request)
    assert any(ELIDED_NOTE in str(r[0]["content"]) for r in requests), "trim never ran"
    assert final["turn_count"] == 50


async def test_each_verify_attempt_gets_its_own_act_budget(tmp_path: Path):
    """Retries used to inherit an exhausted budget and make zero calls."""
    harness = LoopingHarness()
    harness.MAX_ACT_TURNS = 3
    harness.MAX_VERIFY_RETRIES = 3
    final = await inner.run_inner_loop(
        harness, task_dict=load_task(REPO_ROOT, TASK_ID), workspace=_workspace(tmp_path)
    )
    assert len(_act_requests(harness)) == 9
    assert final["turn_count"] == 9
    assert final["verify_attempts"] == 3


async def test_a_retry_shows_the_model_what_verify_said(tmp_path: Path):
    harness = LoopingHarness()
    harness.MAX_ACT_TURNS = 2
    await inner.run_inner_loop(
        harness, task_dict=load_task(REPO_ROOT, TASK_ID), workspace=_workspace(tmp_path)
    )
    first_retry_request = _act_requests(harness)[2]
    last = first_retry_request[-1]

    assert last["role"] == "user"
    feedback = [b["text"] for b in _blocks(last, "text")]
    assert feedback, "the retry's request carries no verify output"
    assert "Verification attempt 1 of 2 failed" in feedback[-1]
    assert "assert -1 == 5" in feedback[-1]
    assert_valid_request(first_retry_request)


async def test_a_retry_after_a_turn_without_tools_adds_one_user_turn(tmp_path: Path):
    class TalkingHarness(LoopingHarness):
        MAX_VERIFY_RETRIES = 2

        async def _call_llm(self, messages, tools, *, tool_choice=None):
            response = await super()._call_llm(messages, tools, tool_choice=tool_choice)
            if tool_choice is None:
                response.content = [Block(type="text", text="I am done.")]
            return response

    harness = TalkingHarness()
    final = await inner.run_inner_loop(
        harness, task_dict=load_task(REPO_ROOT, TASK_ID), workspace=_workspace(tmp_path)
    )
    retry = _act_requests(harness)[1]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert "Verification attempt 1" in retry[-1]["content"]
    ids = [m.id for m in final["messages"]]
    assert ids == [f"m{i:04d}" for i in range(len(ids))]


async def test_an_override_that_edits_its_request_cannot_edit_state(tmp_path: Path):
    from app.meta_harness.tools import TOOL_SCHEMAS  # noqa: PLC0415

    final = await inner.run_inner_loop(
        CachingHarness(),
        task_dict=load_task(REPO_ROOT, TASK_ID),
        workspace=_workspace(tmp_path),
    )
    assert "cache_control" not in str([m.content for m in final["messages"]])
    assert "cache_control" not in str(TOOL_SCHEMAS)


# ── a reused thread starts clean ──────────────────────────────────────


async def test_a_reused_inner_thread_does_not_inherit_the_old_conversation(
    tmp_path: Path,
):
    """Inner thread ids are deterministic; a re-run trial reuses one."""
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "inner::same"}, "recursion_limit": 100}
    finals = []
    for attempt in (1, 2):
        harness = FixTypoHarness()
        graph = inner.build_inner_graph(harness, checkpointer=saver)
        finals.append(
            await graph.ainvoke(
                inner.initial_inner_state(
                    task_dict=load_task(REPO_ROOT, TASK_ID),
                    workspace=_workspace(tmp_path / f"run-{attempt}"),
                ),
                config=config,
            )
        )
        assert len(_act_requests(harness)[0]) == 1, f"run {attempt} inherited messages"

    first, second = ([m.id for m in f["messages"]] for f in finals)
    assert first == second
    assert finals[1]["score"] == 1.0


def test_the_entry_state_clears_the_message_channel():
    state = inner.initial_inner_state(task_dict={"id": "t"}, workspace="/tmp/x")
    assert _apply(state["messages"], _apply([{"role": "user", "content": "old"}])) == []
