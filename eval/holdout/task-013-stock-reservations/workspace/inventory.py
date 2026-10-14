"""Stock levels for one warehouse.

For every SKU the inventory tracks two numbers:

* **on hand** - units physically in the warehouse;
* **reserved** - units promised to placed orders but not yet shipped.

``available(sku)`` is on hand minus reserved. At all times
``0 <= reserved <= on hand`` for every SKU, so available is never
negative and never more than what is on hand.

Every quantity passed to a method must be a positive ``int``; anything
else raises ``ValueError``. A method that raises leaves the inventory
unchanged. An ``Inventory`` owns its numbers: it never modifies a
mapping it was given, and later changes to that mapping do not affect it.
"""

from __future__ import annotations

from typing import Mapping


class OutOfStock(Exception):
    """Not enough available units to reserve."""


def _check_qty(qty: int) -> None:
    if not isinstance(qty, int) or qty <= 0:
        raise ValueError(f"quantity must be a positive int, got {qty!r}")


class Inventory:
    def __init__(self, on_hand: Mapping[str, int] | None = None) -> None:
        self._on_hand = on_hand or {}
        self._reserved: dict[str, int] = {}

    def on_hand(self, sku: str) -> int:
        return self._on_hand.get(sku, 0)

    def reserved(self, sku: str) -> int:
        return self._reserved.get(sku, 0)

    def available(self, sku: str) -> int:
        return self.on_hand(sku) - self.reserved(sku)

    def receive(self, sku: str, qty: int) -> None:
        """Add ``qty`` newly delivered units to stock on hand."""
        _check_qty(qty)
        self._on_hand[sku] = self.on_hand(sku) + qty

    def reserve(self, sku: str, qty: int) -> None:
        """Set aside ``qty`` available units. Raises ``OutOfStock`` if short."""
        _check_qty(qty)
        if qty > self.available(sku):
            raise OutOfStock(f"{sku}: wanted {qty}, {self.available(sku)} available")
        self._reserved[sku] = self.reserved(sku) + qty

    def release(self, sku: str, qty: int) -> None:
        """Return ``qty`` reserved units to the available pool.

        Releasing more than is reserved raises ``ValueError``.
        """
        _check_qty(qty)
        self._reserved[sku] = self.reserved(sku) - qty

    def ship(self, sku: str, qty: int) -> None:
        """``qty`` reserved units leave the warehouse.

        Shipping more than is reserved raises ``ValueError``.
        """
        _check_qty(qty)
        self._on_hand[sku] = self.on_hand(sku) - qty
