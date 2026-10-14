"""Generic run tracing decorator for public SDK users."""

from __future__ import annotations

import functools
import inspect
import uuid
from collections.abc import Callable
from typing import Any, TypeVar, overload

from meta_harness.types import TraceEvent

F = TypeVar("F", bound=Callable[..., Any])
TraceSink = Callable[[TraceEvent], None]


def _emit(
    events: list[TraceEvent],
    sink: TraceSink | None,
    run_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> None:
    event = TraceEvent(
        run_id=run_id,
        event_type=event_type,
        payload=payload or {},
        sequence=len(events),
    )
    events.append(event)
    if sink is not None:
        sink(event)


def _error_payload(function: str, exc: BaseException) -> dict[str, Any]:
    return {"function": function, "error": str(exc), "error_type": type(exc).__name__}


@overload
def trace_run(fn: F) -> F: ...


@overload
def trace_run(
    fn: None = None,
    *,
    run_id: str | None = None,
    sink: TraceSink | None = None,
) -> Callable[[F], F]: ...


def trace_run(
    fn: F | None = None,
    *,
    run_id: str | None = None,
    sink: TraceSink | None = None,
) -> F | Callable[[F], F]:
    """Trace start/end/error events around a sync or async callable.

    Every call gets its own event list, numbered from 0. The wrapper's
    ``_meta_harness_trace`` holds the list of the most recently *started*
    call; ``sink`` sees every event of every call. A single list shared by
    all calls grew without bound in a long-lived process and numbered a
    call's events after everything before it.

    An exception, including a cancellation, is recorded as ``run-error``
    and re-raised.
    """

    def decorate(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                events: list[TraceEvent] = []
                async_wrapper._meta_harness_trace = events  # type: ignore[attr-defined]
                rid = run_id or uuid.uuid4().hex
                _emit(events, sink, rid, "run-start", {"function": func.__name__})
                try:
                    result = await func(*args, **kwargs)
                except BaseException as exc:
                    payload = _error_payload(func.__name__, exc)
                    _emit(events, sink, rid, "run-error", payload)
                    raise
                _emit(events, sink, rid, "run-end", {"function": func.__name__})
                return result

            async_wrapper._meta_harness_trace = []  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            events: list[TraceEvent] = []
            wrapper._meta_harness_trace = events  # type: ignore[attr-defined]
            rid = run_id or uuid.uuid4().hex
            _emit(events, sink, rid, "run-start", {"function": func.__name__})
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                payload = _error_payload(func.__name__, exc)
                _emit(events, sink, rid, "run-error", payload)
                raise
            _emit(events, sink, rid, "run-end", {"function": func.__name__})
            return result

        wrapper._meta_harness_trace = []  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorate(fn)
    return decorate
