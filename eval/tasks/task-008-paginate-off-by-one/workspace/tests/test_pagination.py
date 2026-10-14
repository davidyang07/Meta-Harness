import pytest

from pagination import Page, count_pages, iter_pages, page_window, paginate

ITEMS = list(range(1, 26))


@pytest.mark.parametrize(
    ("total", "per_page", "pages"),
    [
        (0, 10, 1),
        (1, 10, 1),
        (9, 10, 1),
        (10, 10, 1),
        (11, 10, 2),
        (25, 10, 3),
        (30, 10, 3),
        (31, 10, 4),
        (7, 1, 7),
        (7, 3, 3),
        (6, 3, 2),
    ],
)
def test_count_pages(total, per_page, pages):
    assert count_pages(total, per_page) == pages


@pytest.mark.parametrize("per_page", [0, -1, -10])
def test_count_pages_rejects_non_positive_page_size(per_page):
    with pytest.raises(ValueError):
        count_pages(10, per_page)


def test_first_page():
    page = paginate(ITEMS, 1, per_page=10)
    assert isinstance(page, Page)
    assert page.items == list(range(1, 11))
    assert page.number == 1
    assert page.per_page == 10
    assert page.total_items == 25
    assert page.total_pages == 3
    assert page.has_previous is False
    assert page.has_next is True


def test_middle_page():
    page = paginate(ITEMS, 2, per_page=10)
    assert page.items == list(range(11, 21))
    assert page.has_previous is True
    assert page.has_next is True


def test_last_partial_page():
    page = paginate(ITEMS, 3, per_page=10)
    assert page.items == [21, 22, 23, 24, 25]
    assert page.has_previous is True
    assert page.has_next is False


def test_last_full_page_has_no_next():
    page = paginate(list(range(20)), 2, per_page=10)
    assert page.items == list(range(10, 20))
    assert page.total_pages == 2
    assert page.has_next is False


def test_default_page_size_is_ten():
    page = paginate(ITEMS, 3)
    assert page.items == [21, 22, 23, 24, 25]


@pytest.mark.parametrize("number", [0, -1, 4, 100])
def test_out_of_range_page_numbers_raise(number):
    with pytest.raises(ValueError):
        paginate(ITEMS, number, per_page=10)


def test_paginate_rejects_non_positive_page_size():
    with pytest.raises(ValueError):
        paginate(ITEMS, 1, per_page=0)


def test_empty_collection_has_one_empty_page():
    page = paginate([], 1)
    assert page.items == []
    assert page.total_items == 0
    assert page.total_pages == 1
    assert page.has_previous is False
    assert page.has_next is False
    assert page.first_item == 0
    assert page.last_item == 0
    with pytest.raises(ValueError):
        paginate([], 2)
    with pytest.raises(ValueError):
        paginate([], 0)


@pytest.mark.parametrize(
    ("number", "first", "last"),
    [(1, 1, 10), (2, 11, 20), (3, 21, 25)],
)
def test_item_positions_are_one_based_and_inclusive(number, first, last):
    page = paginate(ITEMS, number, per_page=10)
    assert (page.first_item, page.last_item) == (first, last)


def test_item_positions_on_a_single_item_page():
    page = paginate(["only"], 1, per_page=5)
    assert (page.first_item, page.last_item) == (1, 1)
    page = paginate(list("abcdefg"), 7, per_page=1)
    assert page.items == ["g"]
    assert (page.first_item, page.last_item) == (7, 7)


def test_works_on_any_sequence():
    assert paginate("abcdefg", 2, per_page=3).items == ["d", "e", "f"]
    assert paginate("abcdefg", 3, per_page=3).items == ["g"]
    assert paginate(tuple(range(5)), 2, per_page=3).items == [3, 4]
    assert paginate(range(100, 110), 4, per_page=3).items == [109]


def test_does_not_modify_the_input():
    data = [5, 4, 3, 2, 1]
    paginate(data, 2, per_page=2)
    list(iter_pages(data, per_page=2))
    assert data == [5, 4, 3, 2, 1]


def test_iter_pages_visits_every_item_exactly_once():
    for size in range(0, 24):
        data = [f"item-{i}" for i in range(size)]
        for per_page in range(1, 8):
            pages = list(iter_pages(data, per_page))
            assert [p.number for p in pages] == list(
                range(1, count_pages(size, per_page) + 1)
            ), (size, per_page)
            assert [x for p in pages for x in p.items] == data, (size, per_page)
            assert [p.has_next for p in pages] == [True] * (len(pages) - 1) + [False]
            assert [p.has_previous for p in pages] == [False] + [True] * (len(pages) - 1)
            assert all(len(p.items) == per_page for p in pages[:-1])


def test_iter_pages_default_page_size():
    pages = list(iter_pages(list(range(21))))
    assert [len(p.items) for p in pages] == [10, 10, 1]


def test_item_positions_cover_the_collection_contiguously():
    for size in range(1, 20):
        for per_page in range(1, 6):
            pages = list(iter_pages(list(range(size)), per_page))
            assert pages[0].first_item == 1
            assert pages[-1].last_item == size
            for page in pages:
                assert page.last_item - page.first_item + 1 == len(page.items)
            for before, after in zip(pages, pages[1:]):
                assert after.first_item == before.last_item + 1


@pytest.mark.parametrize(
    ("current", "total", "radius", "window"),
    [
        (1, 10, 2, [1, 2, 3]),
        (2, 10, 2, [1, 2, 3, 4]),
        (5, 10, 2, [3, 4, 5, 6, 7]),
        (9, 10, 2, [7, 8, 9, 10]),
        (10, 10, 2, [8, 9, 10]),
        (1, 1, 2, [1]),
        (3, 5, 0, [3]),
        (3, 5, 1, [2, 3, 4]),
        (3, 5, 10, [1, 2, 3, 4, 5]),
    ],
)
def test_page_window(current, total, radius, window):
    assert page_window(current, total, radius) == window


def test_page_window_default_radius():
    assert page_window(6, 20) == [4, 5, 6, 7, 8]
