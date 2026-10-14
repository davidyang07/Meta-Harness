import pytest

from durations import format_duration, parse_duration


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("0s", 0),
        ("45s", 45),
        ("5m", 300),
        ("1h", 3600),
        ("1d", 86400),
        ("1h30m", 5400),
        ("2d 4h", 187200),
        ("1d2h3m4s", 93784),
        ("1d 2h 3m 4s", 93784),
        ("100m", 6000),
        ("48h", 172800),
        ("1d0h", 86400),
        ("007s", 7),
        ("2h 15s", 7215),
    ],
)
def test_valid_durations(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("1H", 3600), ("1D 2H 3M 4S", 93784), ("1h30M", 5400), ("10S", 10)],
)
def test_units_are_case_insensitive(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("  10m  ", 600), ("\t1h\n", 3600), ("1h\t5s", 3605), ("1h   5m", 3900)],
)
def test_surrounding_and_separating_whitespace(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize(("text", "seconds"), [("90", 90), ("0", 0), (" 3600 ", 3600)])
def test_a_bare_integer_is_seconds(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "h",
        "m5",
        "10x",
        "1w",
        "foo",
        "1h foo",
        "foo 1h",
        "1h,30m",
        "1h+30m",
        "1h1h",
        "1d 2d",
        "1h 30m 1m",
        "30m1h",
        "1s1m",
        "4h 1d",
        "1.5h",
        "1.5",
        "-5m",
        "+5m",
        "-5",
        "1 h",
        "1h30",
        "30 1h",
        "5 5",
        "1hh",
        "1h 2",
        "0x10s",
    ],
)
def test_malformed_input_raises_value_error(text):
    with pytest.raises(ValueError):
        parse_duration(text)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0s"),
        (1, "1s"),
        (59, "59s"),
        (60, "1m"),
        (61, "1m1s"),
        (3600, "1h"),
        (3605, "1h5s"),
        (5400, "1h30m"),
        (86399, "23h59m59s"),
        (86400, "1d"),
        (90061, "1d1h1m1s"),
        (93784, "1d2h3m4s"),
        (172800, "2d"),
        (8640000, "100d"),
    ],
)
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


@pytest.mark.parametrize("seconds", [-1, -60, -86400])
def test_format_rejects_negative_numbers(seconds):
    with pytest.raises(ValueError):
        format_duration(seconds)


def test_format_then_parse_round_trips():
    samples = list(range(0, 400)) + [
        3599, 3600, 3601, 7199, 86399, 86400, 86401, 90061, 1_000_000, 987_654_321,
    ]
    for seconds in samples:
        assert parse_duration(format_duration(seconds)) == seconds, seconds


@pytest.mark.parametrize(
    ("text", "canonical"),
    [("1H 30M", "1h30m"), ("90", "1m30s"), ("120m", "2h"), ("0d 0h", "0s"), ("25h", "1d1h")],
)
def test_parse_then_format_canonicalises(text, canonical):
    assert format_duration(parse_duration(text)) == canonical
