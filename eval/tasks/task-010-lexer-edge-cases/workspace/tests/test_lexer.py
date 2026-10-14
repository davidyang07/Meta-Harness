import pytest

from lexer import LexError, Token, tokenize


def kinds_and_texts(source):
    return [(token.kind, token.text) for token in tokenize(source)]


def error_position(source):
    with pytest.raises(LexError) as info:
        tokenize(source)
    return info.value.pos


def test_simple_expression():
    assert tokenize("a + 1") == [
        Token("NAME", "a", 0),
        Token("OP", "+", 2),
        Token("NUMBER", "1", 4),
        Token("EOF", "", 5),
    ]


def test_empty_and_blank_input():
    assert tokenize("") == [Token("EOF", "", 0)]
    assert tokenize("  \t\n ") == [Token("EOF", "", 5)]


@pytest.mark.parametrize("source", ["x", "x  ", "1 + 2\n", "f(a, b)", "# note", "a # note\n"])
def test_exactly_one_eof_token_at_the_end(source):
    tokens = tokenize(source)
    assert tokens[-1] == Token("EOF", "", len(source))
    assert [t.kind for t in tokens].count("EOF") == 1


@pytest.mark.parametrize("op", ["**", "//", "==", "!=", "<=", ">=", "->"])
def test_two_character_operators(op):
    assert tokenize(f"a{op}b") == [
        Token("NAME", "a", 0),
        Token("OP", op, 1),
        Token("NAME", "b", 3),
        Token("EOF", "", 4),
    ]


@pytest.mark.parametrize("op", list("+-*/%<>=(),:"))
def test_single_character_operators(op):
    assert tokenize(op) == [Token("OP", op, 0), Token("EOF", "", 1)]


def test_longest_operator_wins_left_to_right():
    assert [t.text for t in tokenize("a***b")] == ["a", "**", "*", "b", ""]
    assert [t.text for t in tokenize("x<==y")] == ["x", "<=", "=", "y", ""]
    assert [t.text for t in tokenize("p///q")] == ["p", "//", "/", "q", ""]
    assert [t.text for t in tokenize("f(-1)->-2")] == [
        "f", "(", "-", "1", ")", "->", "-", "2", "",
    ]
    assert [t.text for t in tokenize("a=-b")] == ["a", "=", "-", "b", ""]


def test_numbers():
    assert kinds_and_texts("7 42 3.25 .5 0.0 10.01") == [
        ("NUMBER", "7"),
        ("NUMBER", "42"),
        ("NUMBER", "3.25"),
        ("NUMBER", ".5"),
        ("NUMBER", "0.0"),
        ("NUMBER", "10.01"),
        ("EOF", ""),
    ]


def test_number_at_the_end_of_the_input():
    assert tokenize("x=10") == [
        Token("NAME", "x", 0),
        Token("OP", "=", 1),
        Token("NUMBER", "10", 2),
        Token("EOF", "", 4),
    ]


def test_leading_dot_number_after_an_operator():
    assert tokenize("1+.25") == [
        Token("NUMBER", "1", 0),
        Token("OP", "+", 1),
        Token("NUMBER", ".25", 2),
        Token("EOF", "", 5),
    ]


@pytest.mark.parametrize(
    ("source", "pos"),
    [("3.", 1), ("3. + 1", 1), ("a . b", 2), ("1 + .", 4), ("12.x", 2), (".", 0)],
)
def test_a_dot_without_a_digit_after_it_is_an_error(source, pos):
    assert error_position(source) == pos


def test_names_may_contain_digits_and_underscores():
    assert kinds_and_texts("rate2 _tmp x_1_y __ a1b2") == [
        ("NAME", "rate2"),
        ("NAME", "_tmp"),
        ("NAME", "x_1_y"),
        ("NAME", "__"),
        ("NAME", "a1b2"),
        ("EOF", ""),
    ]


def test_a_number_directly_followed_by_a_name():
    assert tokenize("2x") == [
        Token("NUMBER", "2", 0),
        Token("NAME", "x", 1),
        Token("EOF", "", 2),
    ]


def test_strings():
    assert tokenize('say("hi there")') == [
        Token("NAME", "say", 0),
        Token("OP", "(", 3),
        Token("STRING", '"hi there"', 4),
        Token("OP", ")", 14),
        Token("EOF", "", 15),
    ]


def test_empty_string_literal():
    assert tokenize('""') == [Token("STRING", '""', 0), Token("EOF", "", 2)]


def test_strings_with_escaped_quotes_and_backslashes():
    source = r'"a\"b" + "c\\" + "d"'
    assert tokenize(source) == [
        Token("STRING", r'"a\"b"', 0),
        Token("OP", "+", 7),
        Token("STRING", r'"c\\"', 9),
        Token("OP", "+", 15),
        Token("STRING", '"d"', 17),
        Token("EOF", "", 20),
    ]


def test_operators_and_hashes_inside_strings_are_part_of_the_string():
    assert tokenize('"a + b # c"') == [
        Token("STRING", '"a + b # c"', 0),
        Token("EOF", "", 11),
    ]


@pytest.mark.parametrize(
    ("source", "pos"),
    [
        ('"abc', 0),
        ('x = "abc', 4),
        ('"', 0),
        ('f("a", "b)', 7),
        ('"abc\\"', 0),
        ('"abc\\', 0),
        ('"a\nb', 0),
    ],
)
def test_an_unterminated_string_is_an_error_at_its_opening_quote(source, pos):
    assert error_position(source) == pos


def test_comments_are_skipped():
    source = "x = 1  # set x\ny = x ** 2"
    tokens = tokenize(source)
    assert [(t.kind, t.text) for t in tokens] == [
        ("NAME", "x"),
        ("OP", "="),
        ("NUMBER", "1"),
        ("NAME", "y"),
        ("OP", "="),
        ("NAME", "x"),
        ("OP", "**"),
        ("NUMBER", "2"),
        ("EOF", ""),
    ]
    assert tokens[3].pos == 15


def test_a_comment_can_end_the_input():
    assert tokenize("x # trailing") == [Token("NAME", "x", 0), Token("EOF", "", 12)]
    assert tokenize("#") == [Token("EOF", "", 1)]
    assert tokenize("a\n# one\n# two") == [Token("NAME", "a", 0), Token("EOF", "", 13)]


@pytest.mark.parametrize(
    ("source", "pos"),
    [("a $ b", 2), ("1 + ?", 4), ("!", 0), ("x ! y", 2), ("@", 0), ("a;b", 1), ("'s'", 0)],
)
def test_unknown_characters_are_errors_with_their_position(source, pos):
    assert error_position(source) == pos


def test_lex_error_is_a_value_error():
    assert issubclass(LexError, ValueError)


def test_every_token_text_is_found_at_its_position():
    source = (
        'total = price*qty  # first line\n'
        'label = "x\\"y" -> f(total, 2.5)//.5\n'
        "_n2 != 10 ** -3 # done"
    )
    tokens = tokenize(source)
    for token in tokens:
        assert source[token.pos : token.pos + len(token.text)] == token.text
    assert [t.kind for t in tokens] == [
        "NAME", "OP", "NAME", "OP", "NAME",
        "NAME", "OP", "STRING", "OP", "NAME", "OP", "NAME", "OP", "NUMBER", "OP", "OP", "NUMBER",
        "NAME", "OP", "NUMBER", "OP", "OP", "NUMBER",
        "EOF",
    ]
    assert [t.text for t in tokens if t.kind == "STRING"] == ['"x\\"y"']
    assert [t.text for t in tokens if t.kind == "OP"] == [
        "=", "*", "=", "->", "(", ",", ")", "//", "!=", "**", "-",
    ]
