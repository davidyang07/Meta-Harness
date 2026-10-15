"""SSE event registry: live delivery in-process, history on disk.

The event type allowlist is intentionally closed and mirrors
``docs/INTERFACES.md`` section 7.2. API routers and graph nodes emit into
per-run channels named ``run:{run_id}``; the dashboard consumes those
channels through ``GET /runs/{run_id}/stream``.

Live delivery is in-process. The replay history is durable once a
``RunEventLog`` is configured (``set_event_log_root``; the API and the
CLI both do): every event is appended to its thread's ``events.jsonl``,
so the decision log survives a backend restart, and a dashboard that
connects to a run driven by ``meta-harness loop`` in another process is
replayed what that process emitted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from app.meta_harness import runs as runs_mod

log = logging.getLogger("meta_harness.streaming")


REGISTERED_EVENT_TYPES: frozenset[str] = frozenset(
    {
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
)


class StreamingRegistryError(RuntimeError):
    """Base error for runtime-enforced SSE contract violations."""

    status_code = 500


class UnknownEventTypeError(StreamingRegistryError):
    """Raised when code attempts to emit an event outside the allowlist."""


class InvalidEventPayloadError(StreamingRegistryError):
    """Raised when an allowed event does not satisfy the shared payload shape."""


@dataclass(frozen=True)
class SSEEvent:
    """One event stored in the in-process backlog and written to SSE."""

    event_type: str
    event_id: str
    data: dict[str, Any]
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def channel_for_run(run_id: str) -> str:
    """Return the multiplexed SSE channel name for a run."""

    return f"run:{run_id}"


def _now() -> str:
    # Fixed precision, so timestamps from different processes sort as text.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def format_sse(event: SSEEvent) -> str:
    """Serialize one event according to the HTML SSE wire format."""

    payload = json.dumps(event.data, default=str, separators=(",", ":"))
    return f"event: {event.event_type}\nid: {event.event_id}\ndata: {payload}\n\n"


EVENTS_FILENAME = "events.jsonl"
_RUN_CHANNEL_PREFIX = "run:"


class RunEventLog:
    """Each run's event history, durable on disk.

    One ``events.jsonl`` per thread, under
    ``runs/<run>/threads/<thread>/``, appended by whichever process emits
    for that thread: execution artifacts are thread-scoped, so two
    branches never write one file. Read back merged across threads and
    ordered by time.

    Best-effort by design, like every other observability artifact: a
    write that fails is logged and the run carries on.
    """

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root)

    def run_dir(self, channel: str) -> Path | None:
        """The existing run directory behind ``channel``, or ``None``."""
        if not channel.startswith(_RUN_CHANNEL_PREFIX):
            return None
        run_id = channel[len(_RUN_CHANNEL_PREFIX) :]
        try:
            runs_mod.validate_artifact_name(run_id, kind="run")
        except ValueError:
            return None
        run_dir = self.runs_root / run_id
        return run_dir if run_dir.is_dir() else None

    def append(self, channel: str, event: SSEEvent) -> None:
        run_dir = self.run_dir(channel)
        # A run directory that is gone (deleted, or wiped by a fresh
        # re-run) is never recreated by a late event.
        if run_dir is None or not (run_dir / runs_mod.THREADS_DIRNAME).is_dir():
            return
        line = json.dumps(event.to_dict(), default=str, separators=(",", ":"))
        try:
            thread_dir = runs_mod.thread_dir(run_dir, str(event.data["thread_id"]))
            with open(thread_dir / EVENTS_FILENAME, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except (OSError, ValueError) as exc:
            log.warning(
                "could not persist %s event for %s (%s); it will not be replayed "
                "after a restart",
                event.event_type,
                channel,
                exc,
            )

    def load(self, channel: str, limit: int) -> list[SSEEvent]:
        """The newest ``limit`` events of ``channel``, oldest first."""
        run_dir = self.run_dir(channel)
        if run_dir is None:
            return []
        rows: list[tuple[str, str, int, SSEEvent]] = []
        for thread_dir in runs_mod.list_thread_dirs(run_dir):
            try:
                text = (thread_dir / EVENTS_FILENAME).read_text(encoding="utf-8")
            except OSError:
                continue
            for index, line in enumerate(text.splitlines()):
                event = _event_from_line(line)
                if event is not None:
                    rows.append((event.ts, thread_dir.name, index, event))
        rows.sort(key=lambda row: row[:3])
        seen: set[str] = set()
        events: list[SSEEvent] = []
        for *_, event in rows:
            if event.event_id not in seen:
                seen.add(event.event_id)
                events.append(event)
        return events[-limit:] if limit else events


def _event_from_line(line: str) -> SSEEvent | None:
    """Parse one persisted event; a torn or foreign line is skipped."""
    try:
        raw = json.loads(line)
        event = SSEEvent(
            event_type=str(raw["event_type"]),
            event_id=str(raw["event_id"]),
            data=raw["data"],
            ts=str(raw["ts"]),
        )
    except (ValueError, KeyError, TypeError):
        return None
    if event.event_type not in REGISTERED_EVENT_TYPES or not isinstance(event.data, dict):
        return None
    return event


class _Subscription:
    """One SSE consumer: a bounded queue on the event loop that reads it."""

    __slots__ = ("queue", "loop", "overflowed")

    def __init__(self, limit: int) -> None:
        self.queue: asyncio.Queue[SSEEvent] = asyncio.Queue(maxsize=limit)
        self.loop = asyncio.get_running_loop()
        self.overflowed = False

    def offer(self, event: SSEEvent) -> None:
        """Enqueue without blocking; a full queue marks the consumer slow."""
        if self.overflowed:
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.overflowed = True


class EventRegistry:
    """Closed-set registry with per-channel subscribers and replay history.

    Delivery is gap-free for a consumer that keeps up: a subscriber's
    queue is registered *before* the history it replays is snapshotted,
    so an event emitted while the backlog is being written is either in
    the snapshot or in the queue (or both, and then delivered once). With
    a durable log, ``emit`` writes the event to disk *before* it looks at
    the subscribers, which keeps that guarantee for a snapshot read from
    disk.

    An event id is emitted once per channel. A LangGraph node interrupted
    and re-executed on resume emits its events again; the outer loop
    gives them deterministic ids, so the repeat is dropped here instead
    of being appended to the history a second time.

    Delivery is bounded for a consumer that does not: each subscriber
    queue holds at most ``subscriber_queue_limit`` events. A consumer
    that falls that far behind is disconnected after what it already
    holds. ``EventSource`` reconnects with ``Last-Event-ID`` and resumes
    from history, so a slow client loses nothing that is still in the
    backlog, and a stalled one cannot grow server memory without limit.
    """

    def __init__(
        self,
        allowed_event_types: set[str] | frozenset[str] | None = None,
        *,
        history_limit: int = 1000,
        subscriber_queue_limit: int = 1000,
    ) -> None:
        self.allowed_event_types = frozenset(
            allowed_event_types or REGISTERED_EVENT_TYPES
        )
        self.history_limit = history_limit
        self.subscriber_queue_limit = subscriber_queue_limit
        self._history: dict[str, list[SSEEvent]] = {}
        self._ids: dict[str, set[str]] = {}
        self._subscribers: dict[str, set[_Subscription]] = {}
        self._event_log: RunEventLog | None = None
        #: Channels whose durable history has been read into ``_history``.
        self._loaded: set[str] = set()
        # Emitters may run on a worker thread; history and the subscriber
        # sets are only ever touched under this lock.
        self._lock = threading.Lock()

    def set_event_log(self, event_log: RunEventLog | None) -> None:
        """Persist history through ``event_log`` (``None``: memory only)."""
        with self._lock:
            self._event_log = event_log
            self._loaded.clear()

    def _durable(self, channel: str) -> RunEventLog | None:
        """The log that holds ``channel``'s history, if it is on disk."""
        event_log = self._event_log
        if event_log is None or event_log.run_dir(channel) is None:
            return None
        return event_log

    def _ensure_loaded(self, channel: str, event_log: RunEventLog) -> None:
        """Read the channel's durable history once, so ids already on disk
        (from before a restart) are known to ``emit``."""
        if channel in self._loaded:
            return
        persisted = event_log.load(channel, self.history_limit)
        with self._lock:
            if channel in self._loaded:
                return
            persisted_ids = {event.event_id for event in persisted}
            merged = persisted + [
                event
                for event in self._history.get(channel, [])
                if event.event_id not in persisted_ids
            ]
            self._set_history(channel, merged)
            self._loaded.add(channel)

    def _set_history(self, channel: str, events: list[SSEEvent]) -> None:
        """Replace a channel's in-memory history. Caller holds the lock."""
        events = events[-self.history_limit :] if self.history_limit else events
        self._history[channel] = events
        self._ids[channel] = {event.event_id for event in events}

    def emit(
        self,
        channel: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        event_id: str | None = None,
    ) -> SSEEvent:
        """Emit one event.

        Unknown event types raise a 500-class error by design; this is the
        runtime enforcement for the frontend/backend SSE contract.
        """

        if event_type not in self.allowed_event_types:
            raise UnknownEventTypeError(
                f"unregistered SSE event type: {event_type!r}"
            )
        if not isinstance(payload, dict):
            raise InvalidEventPayloadError("SSE payload must be a JSON object")
        if "thread_id" not in payload:
            raise InvalidEventPayloadError(
                f"SSE event {event_type!r} missing required thread_id"
            )

        event = SSEEvent(
            event_type=event_type,
            event_id=event_id or uuid.uuid4().hex,
            data=payload,
            ts=_now(),
        )
        event_log = self._durable(channel)
        if event_log is not None:
            self._ensure_loaded(channel, event_log)
        with self._lock:
            ids = self._ids.setdefault(channel, set())
            if event.event_id in ids:
                return event
            ids.add(event.event_id)
        if event_log is not None:
            # Before the subscriber snapshot below: see the class docstring.
            event_log.append(channel, event)
        with self._lock:
            history = self._history.setdefault(channel, [])
            history.append(event)
            if len(history) > self.history_limit:
                for dropped in history[: len(history) - self.history_limit]:
                    self._ids.get(channel, set()).discard(dropped.event_id)
                del history[: len(history) - self.history_limit]
            subscribers = list(self._subscribers.get(channel, ()))

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        for subscription in subscribers:
            if subscription.loop is running:
                subscription.offer(event)
            else:
                try:
                    subscription.loop.call_soon_threadsafe(subscription.offer, event)
                except RuntimeError:  # that loop is closed; nobody is reading
                    pass
        return event

    def history(self, channel: str) -> list[SSEEvent]:
        """Return a copy of a channel's replay backlog.

        Read from disk when the channel is durable: that includes events
        other processes emitted, and nothing a fresh re-run wiped.
        """

        event_log = self._durable(channel)
        if event_log is not None:
            return event_log.load(channel, self.history_limit)
        with self._lock:
            return list(self._history.get(channel, []))

    def evict(self, channel: str) -> None:
        """Forget a channel's in-memory backlog.

        Called when a run is deleted or re-created fresh. A fresh re-run
        also wipes the run directory, so a client of the new run is not
        replayed the old one's events; a cancelled run keeps its durable
        history, which is that run's record. Connected subscribers stay
        connected and receive the new run's events.
        """

        with self._lock:
            self._history.pop(channel, None)
            self._ids.pop(channel, None)
            self._loaded.discard(channel)

    def clear(self) -> None:
        """Clear subscribers and history. Intended for tests."""

        with self._lock:
            self._history.clear()
            self._ids.clear()
            self._loaded.clear()
            self._subscribers.clear()

    async def subscribe(
        self,
        channel: str,
        *,
        last_event_id: str | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[str]:
        """Yield formatted SSE chunks for one channel.

        Existing backlog is replayed first. If ``last_event_id`` is present
        and found, replay starts after that event.
        """

        subscription = _Subscription(self.subscriber_queue_limit)
        event_log = self._durable(channel)
        with self._lock:
            # Register first, snapshot second: nothing emitted in between
            # can fall into the gap between the backlog and the live queue.
            self._subscribers.setdefault(channel, set()).add(subscription)
            replay = list(self._history.get(channel, []))
        try:
            if event_log is not None:
                replay = await asyncio.to_thread(
                    event_log.load, channel, self.history_limit
                )
            if last_event_id:
                for idx, event in enumerate(replay):
                    if event.event_id == last_event_id:
                        replay = replay[idx + 1 :]
                        break
            # Events emitted after registration but before the snapshot
            # are in both; the queue's copy is skipped. Each is queued at
            # most once, so its id is forgotten once it has been skipped.
            replayed = {event.event_id for event in replay}
            for event in replay:
                yield format_sse(event)

            while True:
                if subscription.overflowed and subscription.queue.empty():
                    # Too slow: everything it could still be handed has
                    # been. Close; the client resumes via Last-Event-ID.
                    return
                try:
                    event = await asyncio.wait_for(
                        subscription.queue.get(),
                        timeout=heartbeat_interval,
                    )
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if event.event_id in replayed:
                    replayed.discard(event.event_id)
                    continue
                yield format_sse(event)
        finally:
            with self._lock:
                subscribers = self._subscribers.get(channel)
                if subscribers is not None:
                    subscribers.discard(subscription)
                    if not subscribers:
                        self._subscribers.pop(channel, None)

    def subscriber_count(self, channel: str) -> int:
        """How many consumers are attached to ``channel``."""

        with self._lock:
            return len(self._subscribers.get(channel, ()))


event_registry = EventRegistry()


def set_event_log_root(runs_root: Path | None) -> None:
    """Persist run event history under ``runs_root`` (``None``: memory only)."""

    event_registry.set_event_log(RunEventLog(runs_root) if runs_root else None)


def emit_run_event(
    run_id: str,
    event_type: str,
    payload: dict[str, Any],
    *,
    event_id: str | None = None,
) -> SSEEvent:
    """Emit one event to the multiplexed channel for ``run_id``."""

    return event_registry.emit(
        channel_for_run(run_id),
        event_type,
        payload,
        event_id=event_id,
    )
