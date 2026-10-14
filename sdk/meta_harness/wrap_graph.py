"""Instrument existing graph-like objects with lightweight trace events."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from meta_harness.trace import TraceSink, _emit
from meta_harness.types import TraceEvent

#: The proxy's own attributes. ``__getattr__`` never forwards these, so a
#: half-built instance (``copy.copy`` and ``pickle`` create one without
#: running ``__init__``) raises AttributeError instead of recursing.
_OWN_ATTRIBUTES = frozenset({"graph", "run_id", "sink", "trace_events"})


class InstrumentedGraph:
    """Thin proxy that traces ``invoke``, ``ainvoke``, ``stream``,
    ``astream``, ``batch`` and ``abatch`` calls.

    Every other attribute is delegated to the wrapped graph. Events from
    all calls on one proxy accumulate on ``trace_events``.
    """

    def __init__(
        self,
        graph: Any,
        *,
        run_id: str | None = None,
        sink: TraceSink | None = None,
    ) -> None:
        self.graph = graph
        self.run_id = run_id or uuid.uuid4().hex
        self.sink = sink
        self.trace_events: list[TraceEvent] = []

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails. Dunder lookups (copy and
        # pickle probe ``__setstate__``, ``__deepcopy__`` …) and our own
        # attributes on an instance whose ``__init__`` never ran must not
        # reach ``self.graph``, which would call back in here forever.
        if name in _OWN_ATTRIBUTES or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        try:
            graph = self.__dict__["graph"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(graph, name)

    def _event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        _emit(self.trace_events, self.sink, self.run_id, event_type, payload)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._event(f"graph-{method}-start")
        try:
            result = getattr(self.graph, method)(*args, **kwargs)
        except BaseException as exc:
            self._event(f"graph-{method}-error", {"error": str(exc)})
            raise
        self._event(f"graph-{method}-end")
        return result

    async def _acall(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._event(f"graph-{method}-start")
        try:
            result = await getattr(self.graph, method)(*args, **kwargs)
        except BaseException as exc:
            self._event(f"graph-{method}-error", {"error": str(exc)})
            raise
        self._event(f"graph-{method}-end")
        return result

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Trace and delegate a synchronous graph invocation."""
        return self._call("invoke", *args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Trace and delegate an asynchronous graph invocation."""
        return await self._acall("ainvoke", *args, **kwargs)

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        """Trace and delegate a synchronous batch invocation."""
        return self._call("batch", *args, **kwargs)

    async def abatch(self, *args: Any, **kwargs: Any) -> Any:
        """Trace and delegate an asynchronous batch invocation."""
        return await self._acall("abatch", *args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Trace and delegate a synchronous stream, chunk by chunk.

        The end event carries the chunk count and whether the consumer
        read the stream to the end.
        """
        self._event("graph-stream-start")
        chunks = 0
        try:
            for chunk in self.graph.stream(*args, **kwargs):
                chunks += 1
                yield chunk
        except GeneratorExit:
            self._event("graph-stream-end", {"chunks": chunks, "completed": False})
            raise
        except BaseException as exc:
            self._event("graph-stream-error", {"error": str(exc), "chunks": chunks})
            raise
        self._event("graph-stream-end", {"chunks": chunks, "completed": True})

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Trace and delegate an asynchronous stream, chunk by chunk."""
        self._event("graph-astream-start")
        chunks = 0
        try:
            async for chunk in self.graph.astream(*args, **kwargs):
                chunks += 1
                yield chunk
        except GeneratorExit:
            self._event("graph-astream-end", {"chunks": chunks, "completed": False})
            raise
        except BaseException as exc:
            self._event("graph-astream-error", {"error": str(exc), "chunks": chunks})
            raise
        self._event("graph-astream-end", {"chunks": chunks, "completed": True})


def wrap_graph(
    graph: Any,
    *,
    run_id: str | None = None,
    sink: TraceSink | None = None,
) -> InstrumentedGraph:
    """Return a traced proxy around a graph-like object."""
    return InstrumentedGraph(graph, run_id=run_id, sink=sink)
