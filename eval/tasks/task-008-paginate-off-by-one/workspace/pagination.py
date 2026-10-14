"""Split a sequence into numbered pages for display.

Conventions used throughout this module:

* Pages are numbered from 1.
* ``per_page`` must be at least 1; anything smaller raises ``ValueError``.
* Every collection has at least one page. An empty collection has exactly
  one page, page 1, and it holds no items.
* Asking for a page number outside ``1..total_pages`` raises ``ValueError``.
* Item positions shown to users (``Page.first_item`` and ``Page.last_item``)
  are 1-based and inclusive. Both are 0 on a page with no items.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence


@dataclass(frozen=True)
class Page:
    """One page of results."""

    items: list
    number: int
    per_page: int
    total_items: int
    total_pages: int

    @property
    def has_previous(self) -> bool:
        """True when there is a page before this one."""
        return self.number > 1

    @property
    def has_next(self) -> bool:
        """True when there is a page after this one."""
        return self.number <= self.total_pages

    @property
    def first_item(self) -> int:
        """Position of this page's first item in the whole collection."""
        if not self.items:
            return 0
        return (self.number - 1) * self.per_page

    @property
    def last_item(self) -> int:
        """Position of this page's last item in the whole collection."""
        if not self.items:
            return 0
        return self.first_item + len(self.items)


def count_pages(total_items: int, per_page: int) -> int:
    """How many pages ``total_items`` items fill at ``per_page`` per page."""
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    return max(1, total_items // per_page)


def paginate(items: Sequence, page: int, per_page: int = 10) -> Page:
    """Return page number ``page`` of ``items``."""
    pages = count_pages(len(items), per_page)
    if page < 0 or page > pages:
        raise ValueError(f"page {page} is out of range 1..{pages}")
    start = page * per_page
    return Page(
        items=list(items[start : start + per_page]),
        number=page,
        per_page=per_page,
        total_items=len(items),
        total_pages=pages,
    )


def iter_pages(items: Sequence, per_page: int = 10) -> Iterator[Page]:
    """Yield every page of ``items`` in order, starting with page 1."""
    for number in range(1, count_pages(len(items), per_page)):
        yield paginate(items, number, per_page)


def page_window(current: int, total_pages: int, radius: int = 2) -> list[int]:
    """Page numbers a pager shows around ``current``, in ascending order.

    The window holds every page within ``radius`` of ``current``, both
    ends included, clipped to ``1..total_pages``.
    """
    low = max(1, current - radius)
    high = min(total_pages, current + radius)
    return list(range(low, high))
