"""In-process counters and gauges.

Application code records metrics through the module-level functions
below. All of the state lives in this module's globals.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_counters: dict[str, int] = {}
_gauges: dict[str, float] = {}


def increment(name: str, amount: int = 1) -> int:
    """Add ``amount`` to counter ``name`` and return the new value.

    Counters only go up: a negative ``amount`` raises ``ValueError`` and
    leaves the counter unchanged.
    """
    if amount < 0:
        raise ValueError(f"cannot decrement counter {name!r}")
    _counters[name] = _counters.get(name, 0) + amount
    return _counters[name]


def counter(name: str) -> int:
    """Current value of counter ``name``; 0 if it was never incremented."""
    return _counters.get(name, 0)


def set_gauge(name: str, value: float) -> None:
    """Record ``value`` as the latest reading of gauge ``name``."""
    _gauges[name] = value


def gauge(name: str) -> float | None:
    """Latest reading of gauge ``name``, or None if it was never set."""
    return _gauges.get(name)


def snapshot() -> dict[str, dict[str, Any]]:
    """A copy of every value: ``{"counters": {...}, "gauges": {...}}``.

    Changing the returned dictionaries does not affect the metrics.
    """
    return {"counters": dict(_counters), "gauges": dict(_gauges)}


def reset() -> None:
    """Forget every counter and gauge."""
    _counters.clear()
    _gauges.clear()


def track_calls(name: str) -> Callable[[F], F]:
    """Decorator: increment counter ``name`` each time the function is called."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            increment(name)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
