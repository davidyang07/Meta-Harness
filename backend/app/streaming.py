"""In-process SSE event registry.

The event type allowlist is intentionally closed and mirrors
``docs/INTERFACES.md`` section 7.2. API routers and graph nodes emit into
per-run channels named ``run:{run_id}``; the dashboard consumes those
channels through ``GET /runs/{run_id}/stream``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator


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
    return datetime.now(timezone.utc).isoformat()


def format_sse(event: SSEEvent) -> str:
    """Serialize one event according to the HTML SSE wire format."""

    payload = json.dumps(event.data, default=str, separators=(",", ":"))
    return f"event: {event.event_type}\nid: {event.event_id}\ndata: {payload}\n\n"


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
    the snapshot or in the queue (or both, and then delivered once).

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
        self._subscribers: dict[str, set[_Subscription]] = {}
        # Emitters may run on a worker thread; history and the subscriber
        # sets are only ever touched under this lock.
        self._lock = threading.Lock()

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
        with self._lock:
            history = self._history.setdefault(channel, [])
            history.append(event)
            if len(history) > self.history_limit:
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
        """Return a copy of a channel's replay backlog."""

        with self._lock:
            return list(self._history.get(channel, []))

    def evict(self, channel: str) -> None:
        """Forget a channel's replay backlog.

        Called when a run is deleted or re-created fresh, so a client of
        the new run is not replayed the old one's events. Connected
        subscribers stay connected and receive the new run's events.
        """

        with self._lock:
            self._history.pop(channel, None)

    def clear(self) -> None:
        """Clear subscribers and history. Intended for tests."""

        with self._lock:
            self._history.clear()
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
        with self._lock:
            # Register first, snapshot second: nothing emitted in between
            # can fall into the gap between the backlog and the live queue.
            self._subscribers.setdefault(channel, set()).add(subscription)
            replay = list(self._history.get(channel, []))
        try:
            if last_event_id:
                for idx, event in enumerate(replay):
                    if event.event_id == last_event_id:
                        replay = replay[idx + 1 :]
                        break
            # Events emitted after registration but before the snapshot
            # are in both; the queue's copy is skipped. Once a queued
            # event is not in the snapshot, every later one is newer.
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
                if replayed:
                    if event.event_id in replayed:
                        continue
                    replayed = set()
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
