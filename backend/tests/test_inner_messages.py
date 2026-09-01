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

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import RemoveMessage  # noqa: E402
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages  # noqa: E402

from app.meta_harness import inner  # noqa: E402


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
