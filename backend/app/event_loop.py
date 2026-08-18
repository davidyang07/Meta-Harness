"""Event-loop selection for the ASGI server.

psycopg's async driver cannot run on Windows' ``ProactorEventLoop``:

    InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in
    async mode.

uvicorn 0.46 picks ``ProactorEventLoop`` on Windows unless it needs
subprocesses, so ``uvicorn app.main:app`` (no ``--reload``) starts a
server on which every Postgres connection fails. The backend then fell
back to in-memory checkpointing *silently*, and the whole
checkpoint/fork/branch demo quietly stopped being Postgres-backed.

``selector_loop_factory`` is a uvicorn ``--loop`` target that always
yields a psycopg-compatible loop:

    uvicorn app.main:app --loop app.event_loop:selector_loop_factory

``meta-harness serve`` uses it by default.
"""

from __future__ import annotations

import asyncio
import sys


def selector_loop_factory(
    use_subprocess: bool = False,  # noqa: ARG001 — accepted for uvicorn parity
) -> asyncio.AbstractEventLoop:
    """Build a psycopg-compatible event loop.

    Returns an *instance*, not a class. uvicorn resolves a dotted
    ``--loop`` string to this callable and hands it straight to
    ``asyncio.Runner(loop_factory=...)``, which calls it with no
    arguments and expects a loop back. (Its built-in named loops go
    through a different branch that returns a class, which is why the
    two shapes differ.)
    """
    return asyncio.SelectorEventLoop()


def is_psycopg_compatible_loop(loop: asyncio.AbstractEventLoop | None = None) -> bool:
    """True if ``loop`` can host psycopg's async driver."""
    if sys.platform != "win32":
        return True
    try:
        loop = loop or asyncio.get_running_loop()
    except RuntimeError:
        return True
    return not isinstance(loop, asyncio.ProactorEventLoop)


def incompatible_loop_hint() -> str:
    """Actionable explanation for an incompatible running loop."""
    return (
        "the running event loop is ProactorEventLoop, which psycopg's async "
        "driver cannot use; start the server with "
        "'meta-harness serve', or pass "
        "'--loop app.event_loop:selector_loop_factory' to uvicorn"
    )
