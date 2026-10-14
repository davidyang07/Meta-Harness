import pytest

from stats import mean, median


def test_mean_basic():
    assert mean([1, 2, 3]) == 2


def test_mean_empty_raises():
    with pytest.raises(ValueError):
        mean([])


def test_median_odd_length():
    assert median([1, 2, 3]) == 2


def test_median_even_length():
    assert median([1, 2, 3, 4]) == 2.5


def test_median_single_element():
    assert median([5]) == 5


def test_median_empty_raises():
    with pytest.raises(ValueError):
        median([])


def test_median_unsorted():
    assert median([3, 1, 2]) == 2


def test_median_negative():
    assert median([-3, -1, -2]) == -2


def test_median_does_not_mutate_input():
    """median() leaves its argument unchanged."""
    xs = [3, 1, 2]
    median(xs)
    assert xs == [3, 1, 2], (
        f"median() mutated its input; expected [3, 1, 2], got {xs}."
    )


def test_median_with_duplicates():
    """Repeated values each count."""
    assert median([1, 1, 1, 2]) == 1.0
    assert median([2, 2, 3, 3]) == 2.5


def test_median_two_element_average():
    """The median of two values is their mean."""
    assert median([1, 3]) == 2.0
    assert median([10, 20]) == 15.0
