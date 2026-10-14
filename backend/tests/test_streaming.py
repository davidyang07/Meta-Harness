"""SSE event registry tests (BUILD_ORDER step 10)."""

from __future__ import annotations

import asyncio

import pytest

from app.streaming import (
    EventRegistry,
    REGISTERED_EVENT_TYPES,
    UnknownEventTypeError,
    channel_for_run,
)


def async_test(fn):
    """Run an async test body without relying on pytest-asyncio."""

    def _run():
        return asyncio.run(fn())

    return _run


def test_registered_event_types_are_closed_contract():
    assert REGISTERED_EVENT_TYPES == {
        "state-update",
        "checkpoint-written",
        "candidate-created",
        "validate-result",
        "eval-result",
        "frontier-updated",
        "iteration-complete",
        "fork-created",
        "branch-cancelled",
        "memory-pattern-stored",
        "error",
    }


def test_unknown_event_type_raises_500_class_error():
    registry = EventRegistry()

    with pytest.raises(UnknownEventTypeError) as excinfo:
        registry.emit(
            channel_for_run("run-a"),
            "not-registered",
            {"thread_id": "run-a"},
        )

    assert excinfo.value.status_code == 500


@async_test
async def test_subscribe_replays_channel_history_as_sse():
    registry = EventRegistry()
    event = registry.emit(
        channel_for_run("run-a"),
        "state-update",
        {
            "thread_id": "run-a",
            "node": "propose",
            "iteration": 1,
            "ts": "2026-04-25T00:00:00Z",
            "summary": {},
        },
    )

    stream = registry.subscribe(channel_for_run("run-a"), heartbeat_interval=0.01)
    chunk = await anext(stream)
    await stream.aclose()

    assert "event: state-update" in chunk
    assert f"id: {event.event_id}" in chunk
    assert '"thread_id":"run-a"' in chunk


def _payload(chunk: str) -> str:
    return chunk.split("data: ", 1)[1].strip() if "data: " in chunk else chunk.strip()


@async_test
async def test_an_event_emitted_during_backlog_replay_is_not_lost():
    """The subscriber is registered before the backlog is snapshotted.

    It used to register only after yielding the backlog, so an event
    published while the client was reading history was dropped.
    """
    registry = EventRegistry()
    channel = channel_for_run("run-a")
    registry.emit(channel, "state-update", {"thread_id": "run-a", "n": 1})
    registry.emit(channel, "state-update", {"thread_id": "run-a", "n": 2})

    stream = registry.subscribe(channel, heartbeat_interval=0.2)
    got = [await anext(stream)]
    registry.emit(channel, "state-update", {"thread_id": "run-a", "n": 3})
    for _ in range(2):
        got.append(await anext(stream))
    await stream.aclose()

    assert [_payload(c) for c in got] == [
        '{"thread_id":"run-a","n":1}',
        '{"thread_id":"run-a","n":2}',
        '{"thread_id":"run-a","n":3}',
    ]


@async_test
async def test_each_event_is_delivered_once():
    registry = EventRegistry()
    channel = channel_for_run("run-a")
    stream = registry.subscribe(channel, heartbeat_interval=0.1)
    pending = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    first = registry.emit(channel, "state-update", {"thread_id": "run-a", "n": 1})
    chunks = [await pending]
    registry.emit(channel, "state-update", {"thread_id": "run-a", "n": 2})
    chunks.append(await anext(stream))
    chunks.append(await anext(stream))  # nothing else queued: a heartbeat
    await stream.aclose()

    assert f"id: {first.event_id}" in chunks[0]
    assert '"n":2' in chunks[1]
    assert chunks[2] == ": heartbeat\n\n"


@async_test
async def test_last_event_id_resumes_after_that_event():
    registry = EventRegistry()
    channel = channel_for_run("run-a")
    first = registry.emit(channel, "state-update", {"thread_id": "run-a", "n": 1})
    registry.emit(channel, "state-update", {"thread_id": "run-a", "n": 2})

    stream = registry.subscribe(
        channel, last_event_id=first.event_id, heartbeat_interval=0.1
    )
    chunk = await anext(stream)
    await stream.aclose()

    assert '"n":2' in chunk


@async_test
async def test_a_slow_subscriber_is_disconnected_not_buffered_forever():
    registry = EventRegistry(subscriber_queue_limit=3)
    channel = channel_for_run("run-a")
    stream = registry.subscribe(channel, heartbeat_interval=5)
    pending = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    for n in range(10):  # the consumer reads none of these yet
        registry.emit(channel, "state-update", {"thread_id": "run-a", "n": n})

    delivered = [await pending]
    async for chunk in stream:
        delivered.append(chunk)

    # It gets what fit in its queue, then the stream ends so EventSource
    # reconnects with Last-Event-ID and resumes from history.
    assert [_payload(c) for c in delivered] == [
        '{"thread_id":"run-a","n":0}',
        '{"thread_id":"run-a","n":1}',
        '{"thread_id":"run-a","n":2}',
    ]
    assert registry.subscriber_count(channel) == 0
    assert len(registry.history(channel)) == 10


@async_test
async def test_evict_drops_history_but_keeps_live_subscribers():
    registry = EventRegistry()
    channel = channel_for_run("run-a")
    registry.emit(channel, "state-update", {"thread_id": "run-a", "n": "old"})
    stream = registry.subscribe(channel, heartbeat_interval=0.5)
    assert '"old"' in await anext(stream)

    registry.evict(channel)
    assert registry.history(channel) == []
    registry.emit(channel, "state-update", {"thread_id": "run-a", "n": "new"})
    assert '"new"' in await anext(stream)
    await stream.aclose()

    fresh = registry.subscribe(channel, heartbeat_interval=0.5)
    assert '"new"' in await anext(fresh)  # only the post-eviction event
    await fresh.aclose()


@async_test
async def test_an_event_emitted_from_a_worker_thread_reaches_subscribers():
    registry = EventRegistry()
    channel = channel_for_run("run-a")
    stream = registry.subscribe(channel, heartbeat_interval=2)
    pending = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)

    await asyncio.to_thread(
        registry.emit, channel, "state-update", {"thread_id": "run-a", "n": "thread"}
    )
    chunk = await asyncio.wait_for(pending, timeout=2)
    await stream.aclose()

    assert '"thread"' in chunk
