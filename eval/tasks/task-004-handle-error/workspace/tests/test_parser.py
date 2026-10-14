from parser import parse_ages


def test_parse_basic():
    assert parse_ages("12, 34, 56") == [12, 34, 56]


def test_parse_single():
    assert parse_ages("42") == [42]


def test_parse_empty_returns_empty():
    assert parse_ages("") == []


def test_parse_skips_invalid():
    assert parse_ages("12, bad, 34") == [12, 34]


def test_parse_skips_only_invalid():
    assert parse_ages("a, b, c") == []


def test_parse_with_spaces():
    assert parse_ages("  10  ,  20  ") == [10, 20]


def test_parse_negative_integers():
    """Negative integers are kept."""
    assert parse_ages("-5, -10, 5") == [-5, -10, 5]


def test_parse_only_whitespace_returns_empty():
    """Whitespace-only input yields []."""
    assert parse_ages("   ") == []
    assert parse_ages(" \t \n ") == []


def test_parse_decimals_skipped_not_converted():
    """Decimal tokens are not whole integers, so they are skipped."""
    result = parse_ages("1.5, 2, 3.0")
    assert result == [2], (
        f"Expected [2]; got {result}."
    )
