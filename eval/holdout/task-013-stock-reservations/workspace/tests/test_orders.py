import pytest

from inventory import Inventory, OutOfStock
from orders import OrderBook


def setup():
    inventory = Inventory({"apple": 10, "pear": 5})
    return inventory, OrderBook(inventory)


def levels(inventory, sku):
    return (inventory.on_hand(sku), inventory.reserved(sku), inventory.available(sku))


# ── inventory on its own ──────────────────────────────────────────────


def test_receive_and_reserve():
    inventory = Inventory()
    inventory.receive("fig", 4)
    inventory.reserve("fig", 3)
    assert levels(inventory, "fig") == (4, 3, 1)
    with pytest.raises(OutOfStock):
        inventory.reserve("fig", 2)
    assert levels(inventory, "fig") == (4, 3, 1)


def test_unknown_skus_have_nothing():
    inventory = Inventory()
    assert levels(inventory, "ghost") == (0, 0, 0)
    with pytest.raises(OutOfStock):
        inventory.reserve("ghost", 1)


@pytest.mark.parametrize("qty", [0, -1, 1.5, "2"])
@pytest.mark.parametrize("method", ["receive", "reserve", "release", "ship"])
def test_quantities_must_be_positive_ints(method, qty):
    inventory = Inventory({"apple": 10})
    inventory.reserve("apple", 4)
    with pytest.raises(ValueError):
        getattr(inventory, method)("apple", qty)
    assert levels(inventory, "apple") == (10, 4, 6)


def test_release_returns_units_to_the_pool():
    inventory = Inventory({"apple": 10})
    inventory.reserve("apple", 4)
    inventory.release("apple", 3)
    assert levels(inventory, "apple") == (10, 1, 9)


def test_releasing_more_than_reserved_is_rejected():
    inventory = Inventory({"apple": 10})
    inventory.reserve("apple", 2)
    with pytest.raises(ValueError):
        inventory.release("apple", 3)
    assert levels(inventory, "apple") == (10, 2, 8)
    with pytest.raises(ValueError):
        inventory.release("pear", 1)
    assert levels(inventory, "pear") == (0, 0, 0)


def test_ship_takes_units_off_the_shelf_and_out_of_the_reservation():
    inventory = Inventory({"apple": 10})
    inventory.reserve("apple", 4)
    inventory.ship("apple", 3)
    assert levels(inventory, "apple") == (7, 1, 6)


def test_shipping_more_than_reserved_is_rejected():
    inventory = Inventory({"apple": 10})
    inventory.reserve("apple", 2)
    with pytest.raises(ValueError):
        inventory.ship("apple", 3)
    assert levels(inventory, "apple") == (10, 2, 8)


def test_the_initial_mapping_belongs_to_the_caller():
    stock = {"apple": 10}
    first = Inventory(stock)
    second = Inventory(stock)
    first.receive("apple", 5)
    first.reserve("apple", 2)
    first.ship("apple", 2)
    assert stock == {"apple": 10}
    assert second.on_hand("apple") == 10
    stock["apple"] = 0
    assert first.on_hand("apple") == 13
    assert second.on_hand("apple") == 10


# ── orders through the inventory ──────────────────────────────────────


def test_placing_reserves_every_line():
    inventory, book = setup()
    order = book.place("o1", {"apple": 3, "pear": 1})
    assert order.status == "placed"
    assert order.lines == [("apple", 3), ("pear", 1)]
    assert levels(inventory, "apple") == (10, 3, 7)
    assert levels(inventory, "pear") == (5, 1, 4)
    assert book.get("o1") is order


def test_orders_only_hold_their_own_lines():
    inventory, book = setup()
    first = book.place("o1", {"apple": 1})
    second = book.place("o2", {"pear": 2})
    assert first.lines == [("apple", 1)]
    assert second.lines == [("pear", 2)]
    assert book.get("o1").lines == [("apple", 1)]


def test_cancel_releases_only_that_orders_stock():
    inventory, book = setup()
    book.place("o1", {"apple": 3})
    book.place("o2", {"apple": 2, "pear": 1})
    book.cancel("o2")
    assert book.get("o2").status == "cancelled"
    assert book.get("o1").status == "placed"
    assert levels(inventory, "apple") == (10, 3, 7)
    assert levels(inventory, "pear") == (5, 0, 5)


def test_ship_moves_only_that_orders_stock():
    inventory, book = setup()
    book.place("o1", {"apple": 3})
    book.place("o2", {"apple": 4, "pear": 2})
    book.ship("o1")
    assert book.get("o1").status == "shipped"
    assert levels(inventory, "apple") == (7, 4, 3)
    assert levels(inventory, "pear") == (5, 2, 3)
    book.cancel("o2")
    assert levels(inventory, "apple") == (7, 0, 7)
    assert levels(inventory, "pear") == (5, 0, 5)


def test_stock_freed_by_shipping_is_counted_correctly():
    inventory, book = setup()
    book.place("o1", {"apple": 6})
    book.ship("o1")
    assert levels(inventory, "apple") == (4, 0, 4)
    book.place("o2", {"apple": 4})
    with pytest.raises(OutOfStock):
        book.place("o3", {"apple": 1})
    inventory.receive("apple", 1)
    book.place("o3", {"apple": 1})
    assert levels(inventory, "apple") == (5, 5, 0)


def test_a_failed_placement_reserves_nothing_and_records_nothing():
    inventory, book = setup()
    with pytest.raises(OutOfStock):
        book.place("o1", {"apple": 4, "pear": 6})
    assert levels(inventory, "apple") == (10, 0, 10)
    assert levels(inventory, "pear") == (5, 0, 5)
    with pytest.raises(KeyError):
        book.get("o1")
    order = book.place("o1", {"apple": 4, "pear": 5})
    assert order.lines == [("apple", 4), ("pear", 5)]
    assert levels(inventory, "pear") == (5, 5, 0)


def test_an_invalid_quantity_rolls_the_placement_back():
    inventory, book = setup()
    with pytest.raises(ValueError):
        book.place("o1", {"apple": 2, "pear": 0})
    assert levels(inventory, "apple") == (10, 0, 10)
    with pytest.raises(KeyError):
        book.get("o1")


def test_a_failed_placement_leaves_earlier_orders_alone():
    inventory, book = setup()
    book.place("o1", {"apple": 5})
    with pytest.raises(OutOfStock):
        book.place("o2", {"apple": 2, "pear": 1, "plum": 1})
    assert levels(inventory, "apple") == (10, 5, 5)
    assert levels(inventory, "pear") == (5, 0, 5)
    assert book.get("o1").lines == [("apple", 5)]


def test_duplicate_order_ids_are_rejected():
    inventory, book = setup()
    book.place("o1", {"apple": 1})
    with pytest.raises(ValueError):
        book.place("o1", {"pear": 1})
    assert levels(inventory, "pear") == (5, 0, 5)


def test_cancelling_twice_is_rejected_and_changes_nothing():
    inventory, book = setup()
    book.place("o1", {"apple": 3})
    book.place("o2", {"apple": 2})
    book.cancel("o1")
    with pytest.raises(ValueError):
        book.cancel("o1")
    assert levels(inventory, "apple") == (10, 2, 8)


def test_a_shipped_order_cannot_be_cancelled_or_shipped_again():
    inventory, book = setup()
    book.place("o1", {"apple": 3})
    book.place("o2", {"apple": 3})
    book.ship("o1")
    with pytest.raises(ValueError):
        book.cancel("o1")
    with pytest.raises(ValueError):
        book.ship("o1")
    assert book.get("o1").status == "shipped"
    assert levels(inventory, "apple") == (7, 3, 4)


def test_a_cancelled_order_cannot_be_shipped():
    inventory, book = setup()
    book.place("o1", {"apple": 3})
    book.cancel("o1")
    with pytest.raises(ValueError):
        book.ship("o1")
    assert book.get("o1").status == "cancelled"
    assert levels(inventory, "apple") == (10, 0, 10)


def test_a_long_session_keeps_the_books_balanced():
    inventory = Inventory({"apple": 20, "pear": 8, "plum": 3})
    book = OrderBook(inventory)
    book.place("a", {"apple": 5, "pear": 2})
    book.place("b", {"apple": 7})
    book.place("c", {"plum": 3, "pear": 1})
    with pytest.raises(OutOfStock):
        book.place("d", {"pear": 2, "plum": 1})
    book.ship("a")
    book.cancel("c")
    book.place("d", {"pear": 2, "plum": 1})
    inventory.receive("apple", 4)
    book.place("e", {"apple": 12})
    book.ship("b")
    with pytest.raises(ValueError):
        book.cancel("b")
    book.cancel("e")
    book.ship("d")
    for sku in ("apple", "pear", "plum"):
        on_hand, reserved, available = levels(inventory, sku)
        assert 0 <= reserved <= on_hand
        assert available == on_hand - reserved
    assert levels(inventory, "apple") == (12, 0, 12)
    assert levels(inventory, "pear") == (4, 0, 4)
    assert levels(inventory, "plum") == (2, 0, 2)
    assert [book.get(o).status for o in "abcde"] == [
        "shipped", "shipped", "cancelled", "shipped", "cancelled",
    ]
