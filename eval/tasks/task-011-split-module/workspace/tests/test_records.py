import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import formatting
import parsing
import records

WORKSPACE = Path(__file__).resolve().parents[1]

PARSING_NAMES = ["FIELD_SEPARATOR", "Contact", "ParseError", "parse_line", "parse_lines"]
FORMATTING_NAMES = ["format_contact", "format_lines", "register_style", "available_styles"]

SAMPLE = """\
# name;email;phone
Ada Lovelace; ADA@Example.org; (555) 123-4567

Grace Hopper;grace@navy.mil
   # an indented comment
Alan Turing;alan@bletchley.uk;+44 20 7946 0958
"""


def _ada():
    return records.Contact("Ada Lovelace", "ada@example.org", "5551234567")


def _grace():
    return records.Contact("Grace Hopper", "grace@navy.mil")


# ── behaviour through the original import path ──────────────────────


def test_parse_lines():
    assert records.parse_lines(SAMPLE) == [
        records.Contact("Ada Lovelace", "ada@example.org", "5551234567"),
        records.Contact("Grace Hopper", "grace@navy.mil", ""),
        records.Contact("Alan Turing", "alan@bletchley.uk", "442079460958"),
    ]


def test_parse_line():
    assert records.parse_line("  Bob ;BOB@Example.com") == records.Contact(
        "Bob", "bob@example.com", ""
    )
    assert records.FIELD_SEPARATOR == ";"


@pytest.mark.parametrize(
    ("line", "message_part"),
    [
        ("just-a-name", "expected 2 or 3 fields"),
        ("a;b@c.d;e;f", "expected 2 or 3 fields"),
        (" ;x@y.z", "missing name"),
        ("Bob;bob.example.org", "invalid email"),
        ("Bob;@example.org", "invalid email"),
        ("Bob;bob@x@y", "invalid email"),
        ("Bob;bob@x.org;12-34", "7-15 digits"),
        ("Bob;bob@x.org;1234567890123456", "7-15 digits"),
    ],
)
def test_parse_line_errors(line, message_part):
    with pytest.raises(records.ParseError) as info:
        records.parse_line(line)
    assert message_part in str(info.value)
    assert info.value.line_no is None


def test_parse_lines_reports_the_failing_line_number():
    text = "A;a@x.org\n\nB;not-an-email\n"
    with pytest.raises(records.ParseError) as info:
        records.parse_lines(text)
    assert info.value.line_no == 3
    assert str(info.value).startswith("line 3: ")
    assert isinstance(info.value, ValueError)


def test_format_styles():
    alan = records.Contact("Alan Turing", "alan@bletchley.uk", "442079460958")
    assert records.format_contact(_ada()) == "Ada Lovelace;ada@example.org;5551234567"
    assert records.format_contact(_grace(), "line") == "Grace Hopper;grace@navy.mil"
    assert records.format_contact(_ada(), "card") == (
        "Ada Lovelace\n  email: ada@example.org\n  phone: 555-123-4567"
    )
    assert records.format_contact(alan, "card") == (
        "Alan Turing\n  email: alan@bletchley.uk\n  phone: 442079460958"
    )
    assert records.format_contact(_grace(), "card") == "Grace Hopper\n  email: grace@navy.mil"
    assert records.format_contact(_ada(), "mailto") == "Ada Lovelace <ada@example.org>"


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError):
        records.format_contact(_ada(), "no-such-style")


def test_format_lines():
    assert records.format_lines([_ada(), _grace()], "mailto") == (
        "Ada Lovelace <ada@example.org>\nGrace Hopper <grace@navy.mil>"
    )
    assert records.format_lines([]) == ""


def test_round_trip():
    contacts = records.parse_lines(SAMPLE)
    assert records.parse_lines(records.format_lines(contacts)) == contacts


def test_builtin_styles_are_listed_sorted():
    styles = records.available_styles()
    assert {"card", "line", "mailto"} <= set(styles)
    assert styles == sorted(styles)


# ── the same behaviour from the new modules ──────────────────────────


def test_new_modules_parse_and_format():
    contacts = parsing.parse_lines(SAMPLE)
    assert contacts == records.parse_lines(SAMPLE)
    assert parsing.parse_line("Ada Lovelace;ada@example.org;555 123 4567") == _ada()
    assert formatting.format_contact(_ada(), "card").endswith("phone: 555-123-4567")
    assert formatting.format_lines(contacts[:1]) == "Ada Lovelace;ada@example.org;5551234567"
    with pytest.raises(parsing.ParseError):
        parsing.parse_lines("ok;ok@x.org\nbroken\n")


def test_a_style_registered_anywhere_is_available_everywhere():
    @records.register_style("test-upper")
    def upper(contact):
        return contact.name.upper()

    assert upper(_ada()) == "ADA LOVELACE"
    assert "test-upper" in formatting.available_styles()
    assert formatting.format_contact(_ada(), "test-upper") == "ADA LOVELACE"

    @formatting.register_style("test-initials")
    def initials(contact):
        return "".join(word[0] for word in contact.name.split())

    assert "test-initials" in records.available_styles()
    assert records.format_contact(_ada(), "test-initials") == "AL"


# ── structure ─────────────────────────────────────────────────────────


def test_records_public_api_is_unchanged():
    assert sorted(records.__all__) == sorted(PARSING_NAMES + FORMATTING_NAMES)
    for name in records.__all__:
        assert hasattr(records, name), name


@pytest.mark.parametrize("name", PARSING_NAMES)
def test_parsing_side_lives_in_parsing(name):
    assert hasattr(parsing, name), f"parsing.{name} is missing"
    obj = getattr(parsing, name)
    assert getattr(records, name) is obj
    if inspect.isclass(obj) or inspect.isfunction(obj):
        assert obj.__module__ == "parsing"


@pytest.mark.parametrize("name", FORMATTING_NAMES)
def test_formatting_side_lives_in_formatting(name):
    assert hasattr(formatting, name), f"formatting.{name} is missing"
    obj = getattr(formatting, name)
    assert getattr(records, name) is obj
    assert obj.__module__ == "formatting"


def test_records_defines_no_functions_or_classes_of_its_own():
    own = [
        name
        for name, obj in vars(records).items()
        if (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__ == "records"
    ]
    assert own == []


def _imported_modules(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_parsing_imports_neither_formatting_nor_records():
    imported = _imported_modules(parsing)
    assert "formatting" not in imported
    assert "records" not in imported


def test_formatting_does_not_import_records():
    assert "records" not in _imported_modules(formatting)


@pytest.mark.parametrize(
    "script",
    [
        "import records; c = records.parse_line('A B;a@b.org;5551234567');"
        " assert records.format_contact(c, 'card').endswith('555-123-4567')",
        "import formatting, parsing; c = parsing.parse_line('A B;a@b.org');"
        " assert formatting.format_contact(c, 'mailto') == 'A B <a@b.org>'",
        "import parsing, sys; parsing.parse_lines('A;a@b.org');"
        " assert 'formatting' not in sys.modules and 'records' not in sys.modules",
    ],
)
def test_each_module_imports_cleanly_on_its_own(script):
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
