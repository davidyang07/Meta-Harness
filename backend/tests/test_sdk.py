"""Public SDK smoke tests."""

from __future__ import annotations

import asyncio

from meta_harness import TraceEvent, trace_run, wrap_graph


def test_trace_run_decorator_records_sync_events() -> None:
    seen: list[TraceEvent] = []

    @trace_run(run_id="sdk-test", sink=seen.append)
    def add_one(value: int) -> int:
        return value + 1

    assert add_one(1) == 2
    assert [event.event_type for event in seen] == ["run-start", "run-end"]
    assert add_one._meta_harness_trace == seen  # type: ignore[attr-defined]


def test_trace_run_decorator_records_async_events() -> None:
    seen: list[TraceEvent] = []

    @trace_run(run_id="sdk-async-test", sink=seen.append)
    async def add_two(value: int) -> int:
        return value + 2

    assert asyncio.run(add_two(1)) == 3
    assert [event.event_type for event in seen] == ["run-start", "run-end"]
    assert all(event.run_id == "sdk-async-test" for event in seen)


def test_trace_run_decorator_records_errors() -> None:
    seen: list[TraceEvent] = []

    @trace_run(run_id="sdk-error-test", sink=seen.append)
    def fail() -> None:
        raise RuntimeError("boom")

    try:
        fail()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    assert [event.event_type for event in seen] == ["run-start", "run-error"]
    assert seen[-1].payload["error"] == "boom"


def test_wrap_graph_delegates_sync_invoke() -> None:
    class Graph:
        def invoke(self, value: int) -> int:
            return value + 2

    wrapped = wrap_graph(Graph(), run_id="graph-test")

    assert wrapped.invoke(3) == 5
    assert [event.event_type for event in wrapped.trace_events] == [
        "graph-invoke-start",
        "graph-invoke-end",
    ]


def test_wrap_graph_delegates_async_ainvoke() -> None:
    class Graph:
        async def ainvoke(self, value: int) -> int:
            return value + 3

    wrapped = wrap_graph(Graph(), run_id="graph-test")

    assert asyncio.run(wrapped.ainvoke(4)) == 7
    assert [event.event_type for event in wrapped.trace_events] == [
        "graph-ainvoke-start",
        "graph-ainvoke-end",
    ]


# ── trace_run keeps each call's events separate ───────────────────────


def test_trace_run_events_are_per_call_not_accumulated() -> None:
    seen: list[TraceEvent] = []

    @trace_run(run_id="per-call", sink=seen.append)
    def noop() -> None:
        return None

    for _ in range(500):
        noop()

    trace = noop._meta_harness_trace  # type: ignore[attr-defined]
    assert [event.event_type for event in trace] == ["run-start", "run-end"]
    assert [event.sequence for event in trace] == [0, 1]
    # The sink still sees everything.
    assert len(seen) == 1000


def test_trace_run_async_events_are_per_call() -> None:
    @trace_run
    async def noop() -> None:
        return None

    async def _many() -> None:
        for _ in range(50):
            await noop()

    asyncio.run(_many())
    trace = noop._meta_harness_trace  # type: ignore[attr-defined]
    assert [event.sequence for event in trace] == [0, 1]


def test_trace_run_records_a_cancellation_as_an_error() -> None:
    seen: list[TraceEvent] = []

    @trace_run(sink=seen.append)
    async def forever() -> None:
        await asyncio.sleep(3600)

    async def _cancel() -> None:
        task = asyncio.create_task(forever())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_cancel())
    assert [e.event_type for e in seen] == ["run-start", "run-error"]
    assert seen[-1].payload["error_type"] == "CancelledError"


# ── wrap_graph traces every entry point and stays a good proxy ────────


class _Graph:
    name = "inner-graph"

    def stream(self, value: int):
        yield from range(value)

    async def astream(self, value: int):
        for item in range(value):
            yield item

    def batch(self, values: list[int]) -> list[int]:
        return [v + 1 for v in values]

    async def abatch(self, values: list[int]) -> list[int]:
        return [v + 2 for v in values]


def _types(wrapped) -> list[str]:
    return [event.event_type for event in wrapped.trace_events]


def test_wrap_graph_traces_stream() -> None:
    wrapped = wrap_graph(_Graph())
    assert list(wrapped.stream(3)) == [0, 1, 2]
    assert _types(wrapped) == ["graph-stream-start", "graph-stream-end"]
    assert wrapped.trace_events[-1].payload == {"chunks": 3, "completed": True}


def test_wrap_graph_traces_a_stream_closed_early() -> None:
    wrapped = wrap_graph(_Graph())
    stream = wrapped.stream(5)
    assert next(stream) == 0
    stream.close()
    assert wrapped.trace_events[-1].payload == {"chunks": 1, "completed": False}


def test_wrap_graph_traces_astream() -> None:
    wrapped = wrap_graph(_Graph())

    async def _collect() -> list[int]:
        return [item async for item in wrapped.astream(2)]

    assert asyncio.run(_collect()) == [0, 1]
    assert _types(wrapped) == ["graph-astream-start", "graph-astream-end"]


def test_wrap_graph_traces_batch_and_abatch() -> None:
    wrapped = wrap_graph(_Graph())
    assert wrapped.batch([1, 2]) == [2, 3]
    assert asyncio.run(wrapped.abatch([1])) == [3]
    assert _types(wrapped) == [
        "graph-batch-start",
        "graph-batch-end",
        "graph-abatch-start",
        "graph-abatch-end",
    ]


def test_wrap_graph_traces_a_failing_stream() -> None:
    class _Broken:
        def stream(self, value: int):
            yield 1
            raise RuntimeError("mid-stream")

    wrapped = wrap_graph(_Broken())
    try:
        list(wrapped.stream(1))
    except RuntimeError:
        pass
    assert _types(wrapped) == ["graph-stream-start", "graph-stream-error"]


def test_wrap_graph_still_delegates_other_attributes() -> None:
    wrapped = wrap_graph(_Graph())
    assert wrapped.name == "inner-graph"
    try:
        wrapped.no_such_attribute  # noqa: B018
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")


def test_wrap_graph_can_be_copied_and_pickled() -> None:
    """``__getattr__`` used to recurse forever on a half-built instance."""
    import copy
    import pickle

    wrapped = wrap_graph(_Graph(), run_id="copy-me")
    shallow = copy.copy(wrapped)
    deep = copy.deepcopy(wrapped)
    restored = pickle.loads(pickle.dumps(wrapped))

    for clone in (shallow, deep, restored):
        assert clone.run_id == "copy-me"
        assert clone.batch([0]) == [1]
