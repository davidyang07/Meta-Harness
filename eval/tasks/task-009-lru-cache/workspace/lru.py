"""A fixed-capacity cache that evicts the least recently used entry.

Specification
=============

``LRUCache(capacity, on_evict=None)``
    ``capacity`` is the maximum number of entries. It must be an ``int``
    of at least 1; anything else raises ``ValueError``. ``on_evict``, if
    given, is called as ``on_evict(key, value)`` once for every entry the
    cache evicts. It is *not* called for entries removed with ``pop()``,
    nor for a value replaced by ``put()`` on a key that is already present.

Recency
    Entries are kept in recency order. A ``get()`` that finds its key, and
    every ``put()`` (insert or update), make that key the most recently
    used. Nothing else changes the order: ``peek()``, ``in``, ``len()``,
    ``keys()`` and a ``get()`` that misses all leave it alone.

``get(key, default=None)``
    Return the value stored under ``key`` and make it the most recently
    used, or return ``default`` if ``key`` is absent. ``None`` is a valid
    stored value. Every call counts towards ``hits`` (key present) or
    ``misses`` (key absent).

``peek(key, default=None)``
    Like ``get()``, but changes neither the recency order nor
    ``hits``/``misses``.

``put(key, value)``
    Store ``value`` under ``key``. Updating a key that is already present
    never evicts anything. Inserting a new key into a full cache first
    evicts the least recently used entry.

``pop(key)``
    Remove ``key`` and return its value. Raises ``KeyError`` if absent.

``resize(capacity)``
    Change the capacity, validated exactly as in the constructor. If the
    cache holds more entries than the new capacity, least recently used
    entries are evicted, oldest first, until it fits; each eviction goes
    through ``on_evict``. An invalid capacity leaves the cache unchanged.

``keys()``
    A new list of the keys, least recently used first.

``len(cache)`` and ``key in cache``
    The number of entries, and membership.

``hits`` and ``misses``
    Integer counters, both starting at 0.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable


class LRUCache:
    def __init__(
        self,
        capacity: int,
        on_evict: Callable[[Hashable, Any], None] | None = None,
    ) -> None:
        raise NotImplementedError

    def get(self, key: Hashable, default: Any = None) -> Any:
        raise NotImplementedError

    def peek(self, key: Hashable, default: Any = None) -> Any:
        raise NotImplementedError

    def put(self, key: Hashable, value: Any) -> None:
        raise NotImplementedError

    def pop(self, key: Hashable) -> Any:
        raise NotImplementedError

    def resize(self, capacity: int) -> None:
        raise NotImplementedError

    def keys(self) -> list:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, key: object) -> bool:
        raise NotImplementedError

    @property
    def hits(self) -> int:
        raise NotImplementedError

    @property
    def misses(self) -> int:
        raise NotImplementedError
