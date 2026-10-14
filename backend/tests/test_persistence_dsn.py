"""Postgres DSN handling and checkpoint-thread deletion.

The healthcheck decides whether the API runs on Postgres or degrades to
in-memory checkpointing, so a healthcheck that misreads a valid DSN
silently disables checkpoint history, forking and branch recovery. It
did exactly that for key-value DSNs: ``?connect_timeout=5`` was appended
to the string, and ``user=meta_harness`` became the user
``meta_harness?connect_timeout=5``.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from app.meta_harness import persistence
from app.meta_harness.persistence import (
    delete_threads,
    get_dsn,
    healthcheck,
    healthcheck_conninfo,
    persistence_layer,
)
from tests.conftest import unique_name


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://meta_harness:pw@db.example:5432/meta_harness",
        "postgresql://meta_harness:pw@db.example:5432/meta_harness?sslmode=disable",
        "host=db.example port=5432 user=meta_harness password=pw dbname=meta_harness",
        "dbname=meta_harness user=meta_harness",
    ],
)
def test_healthcheck_conninfo_keeps_every_dsn_form_intact(dsn: str):
    parsed = conninfo_to_dict(healthcheck_conninfo(dsn))

    assert parsed["user"] == "meta_harness"
    assert parsed["dbname"] == "meta_harness"
    assert parsed["connect_timeout"] == "5"
    original = conninfo_to_dict(dsn)
    for key, value in original.items():
        assert parsed[key] == value


async def test_healthcheck_connects_with_the_parsed_conninfo(monkeypatch):
    """The conninfo actually handed to psycopg parses to the right user."""
    seen: dict[str, str] = {}

    async def fake_connect(*, conninfo: str, **_kwargs):
        seen["conninfo"] = conninfo
        raise OSError("not connecting in a unit test")

    monkeypatch.setattr(persistence.AsyncConnection, "connect", fake_connect)

    ok = await healthcheck("host=localhost user=meta_harness dbname=meta_harness")

    assert ok is False
    parsed = conninfo_to_dict(seen["conninfo"])
    assert parsed["user"] == "meta_harness"
    assert parsed["connect_timeout"] == "5"


_PG_OK = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
    healthcheck()
)
pg_only = pytest.mark.skipif(
    not _PG_OK,
    reason="Postgres not reachable at configured DSN; bring up via docker compose",
)


@pg_only
async def test_a_key_value_dsn_is_healthy_when_the_url_form_is():
    url = get_dsn()
    key_value = make_conninfo("", **conninfo_to_dict(url))
    assert "://" not in key_value

    assert await healthcheck(url) is True
    assert await healthcheck(key_value) is True


# ── delete_threads ────────────────────────────────────────────────────


class _S(TypedDict):
    x: int


async def _node(state: _S) -> dict:
    return {"x": state["x"] + 1}


def _graph(checkpointer):
    g = StateGraph(_S)
    g.add_node("n", _node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile(checkpointer=checkpointer)


async def _history(graph, thread_id: str) -> list:
    return [
        s async for s in graph.aget_state_history({"configurable": {"thread_id": thread_id}})
    ]


async def test_delete_threads_clears_only_the_named_threads():
    saver = MemorySaver()
    graph = _graph(saver)
    for thread in ("run-x", "run-x.fork.aa", "other"):
        await graph.ainvoke({"x": 0}, {"configurable": {"thread_id": thread}})

    deleted = await delete_threads(saver, ["run-x", "run-x.fork.aa", "run-x"])

    assert deleted == ["run-x", "run-x.fork.aa"]
    assert await _history(graph, "run-x") == []
    assert await _history(graph, "run-x.fork.aa") == []
    assert await _history(graph, "other") != []


async def test_delete_threads_without_a_checkpointer_is_a_no_op():
    assert await delete_threads(None, ["anything"]) == []


@pg_only
async def test_delete_threads_clears_postgres_history():
    thread = unique_name("delete-threads")
    async with persistence_layer() as saver:
        graph = _graph(saver)
        await graph.ainvoke({"x": 0}, {"configurable": {"thread_id": thread}})
        assert await _history(graph, thread) != []

        assert await delete_threads(saver, [thread]) == [thread]
        assert await _history(graph, thread) == []
