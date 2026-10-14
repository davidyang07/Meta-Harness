"""AsyncPostgresSaver setup + connection pool (BUILD_ORDER step 7).

Per Appendix A §A.3:
- Sync ``PostgresSaver`` deadlocks under concurrent use; we use the
  async-native ``AsyncPostgresSaver`` exclusively.
- Pool sized ``max_size=20`` (≈2 connections per concurrent branch).
- ``saver.setup()`` creates the checkpoint tables on first call;
  idempotent on subsequent runs.

Usage:

    async with persistence_layer() as saver:
        graph = workflow.compile(checkpointer=saver)
        await graph.ainvoke(initial_state, config={...})
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

DEFAULT_DSN = "postgresql://meta_harness:meta_harness@localhost:5432/meta_harness"

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def get_dsn() -> str:
    """Return ``POSTGRES_DSN`` from env or the local-dev default."""
    return os.environ.get("POSTGRES_DSN", DEFAULT_DSN)


@asynccontextmanager
async def persistence_layer(
    dsn: str | None = None,
    *,
    min_size: int = 4,
    max_size: int = 20,
    setup: bool = True,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Async context manager that yields an ``AsyncPostgresSaver``
    backed by a sized connection pool.

    Per Appendix A §A.3 Piece 2 — ``max_size=20`` gives ~10 concurrent
    branches at 2 connections each. ``autocommit=True`` to keep
    LangGraph's checkpoint reads/writes from contending on locks.
    ``dict_row`` factory is what ``AsyncPostgresSaver`` expects.
    """
    dsn = dsn or get_dsn()
    async with AsyncConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        timeout=30,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=False,
    ) as pool:
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        if setup:
            await saver.setup()
        yield saver


def healthcheck_conninfo(dsn: str) -> str:
    """``dsn`` with a 5 s ``connect_timeout``, whatever form ``dsn`` takes.

    ``psycopg.AsyncConnection.connect`` rejects an unknown ``timeout``
    keyword; the libpq parameter is ``connect_timeout``. It used to be
    appended to the string as ``?connect_timeout=5``, which is only valid
    for a URL. A key-value DSN (``host=... user=meta_harness``) then
    parsed its last value as ``meta_harness?connect_timeout=5``, the
    check failed, and the API reported Postgres unreachable and dropped
    to in-memory checkpointing. ``make_conninfo`` merges the parameter
    into either form.
    """
    return make_conninfo(dsn, connect_timeout=5)


async def healthcheck(dsn: str | None = None) -> bool:
    """Return ``True`` iff Postgres is reachable at the configured DSN."""
    try:
        dsn = dsn or get_dsn()
        conn = await AsyncConnection.connect(
            conninfo=healthcheck_conninfo(dsn),
            row_factory=dict_row,
            autocommit=True,
        )
        async with conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - any connection error means unhealthy
        return False


async def delete_threads(checkpointer: Any, thread_ids: Iterable[str]) -> list[str]:
    """Delete every checkpoint of ``thread_ids`` from ``checkpointer``.

    A fresh re-run reuses its run id as its root thread id, and its forks'
    ids derive from it. Without this, the new run's checkpoint history
    starts with the old run's, and ``/checkpoints`` and the fork endpoint
    offer checkpoints from a run that no longer exists.

    Uses ``adelete_thread`` (or ``delete_thread``) where the saver has
    one. Returns the ids actually deleted; a saver with neither method
    deletes nothing, and the caller can see that. Errors propagate: a
    fresh run that silently kept stale history is the bug being fixed.
    """
    deleted: list[str] = []
    if checkpointer is None:
        return deleted
    adelete = getattr(checkpointer, "adelete_thread", None)
    delete = getattr(checkpointer, "delete_thread", None)
    for thread_id in dict.fromkeys(thread_ids):
        if adelete is not None:
            result = adelete(thread_id)
            if inspect.isawaitable(result):
                await result
        elif delete is not None:
            await asyncio.to_thread(delete, thread_id)
        else:
            break
        deleted.append(thread_id)
    return deleted
