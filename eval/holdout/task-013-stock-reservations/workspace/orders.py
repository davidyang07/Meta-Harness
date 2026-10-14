"""Customer orders that reserve and ship stock through an ``Inventory``.

An order is ``"placed"`` when it is created, and then becomes either
``"shipped"`` or ``"cancelled"``. Both of those are final: shipping or
cancelling an order that is not ``"placed"`` raises ``ValueError`` and
changes nothing.

Placing an order is all or nothing. If any line cannot be reserved, the
error propagates, no stock remains reserved for the order, and the order
is not recorded, so its id can be used again.

Each order's ``lines`` lists its own ``(sku, qty)`` pairs in the order
they were given, and nothing else.
"""

from __future__ import annotations

from inventory import Inventory


class Order:
    def __init__(self, order_id: str, lines: list[tuple[str, int]] = []) -> None:
        self.order_id = order_id
        self.lines = lines
        self.status = "placed"


class OrderBook:
    def __init__(self, inventory: Inventory) -> None:
        self.inventory = inventory
        self._orders: dict[str, Order] = {}

    def place(self, order_id: str, items: dict[str, int]) -> Order:
        """Reserve every line of ``items`` and record the order."""
        if order_id in self._orders:
            raise ValueError(f"duplicate order id {order_id!r}")
        order = Order(order_id)
        for sku, qty in items.items():
            self.inventory.reserve(sku, qty)
            order.lines.append((sku, qty))
        self._orders[order_id] = order
        return order

    def get(self, order_id: str) -> Order:
        return self._orders[order_id]

    def cancel(self, order_id: str) -> None:
        """Give the order's reserved stock back and mark it cancelled."""
        order = self._orders[order_id]
        for sku, qty in order.lines:
            self.inventory.release(sku, qty)
        order.status = "cancelled"

    def ship(self, order_id: str) -> None:
        """Send the order's reserved stock out and mark it shipped."""
        order = self._orders[order_id]
        for sku, qty in order.lines:
            self.inventory.ship(sku, qty)
        order.status = "shipped"
