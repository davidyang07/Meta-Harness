r"""Tokenizer for a small expression language.

``tokenize(source)`` returns a list of ``Token(kind, text, pos)``. ``text``
is the token exactly as written in ``source`` and ``pos`` is the 0-based
offset of its first character. The list always ends with exactly one
``Token("EOF", "", len(source))``.

Lexical rules, applied from left to right:

* Spaces, tabs and newlines separate tokens and are otherwise ignored.
* ``#`` starts a comment that runs to the end of the line, or to the end
  of the input if no newline follows. Comments produce no tokens.
* NUMBER: one or more digits, optionally followed by a ``.`` and one or
  more digits (``7``, ``42``, ``3.25``). A number may also begin with the
  dot (``.5``). A ``.`` that is not followed by a digit is an error.
* NAME: a letter or ``_``, followed by any number of letters, digits and
  underscores (``x``, ``_tmp``, ``rate2``).
* STRING: characters between double quotes. Inside a string, a backslash
  escapes the character after it, so ``\"`` does not end the string and
  ``\\`` is an escaped backslash. ``text`` is the whole literal as
  written, quotes and backslashes included. A string that is still open
  at the end of the input is an error.
* OP: the longest of the following that matches at the current position:
  ``**``, ``//``, ``==``, ``!=``, ``<=``, ``>=``, ``->``, or one of the
  single characters ``+ - * / % < > = ( ) , :``.

Any other character is an error. Every error raises ``LexError``, whose
``pos`` attribute is the offset of the offending character. For a string
that is never closed, that is the offset of its opening quote.
"""

from __future__ import annotations

from typing import NamedTuple


class Token(NamedTuple):
    kind: str
    text: str
    pos: int


class LexError(ValueError):
    """Malformed input. ``pos`` is the offset of the offending character."""

    def __init__(self, message: str, pos: int) -> None:
        super().__init__(f"{message} at position {pos}")
        self.pos = pos


OPERATORS = [
    "+", "-", "*", "/", "%", "<", ">", "=", "(", ")", ",", ":",
    "**", "//", "==", "!=", "<=", ">=", "->",
]


def tokenize(source: str) -> list[Token]:
    """Split ``source`` into tokens according to the module docstring."""
    tokens: list[Token] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in " \t\n":
            i += 1
            continue
        if ch == "#":
            i = source.index("\n", i)
            continue
        if ch.isdigit():
            start = i
            while i < n and source[i].isdigit():
                i += 1
            if i < n and source[i] == ".":
                i += 1
                while i < n and source[i].isdigit():
                    i += 1
            tokens.append(Token("NUMBER", source[start:i], start))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (source[i].isalpha() or source[i] == "_"):
                i += 1
            tokens.append(Token("NAME", source[start:i], start))
            continue
        if ch == '"':
            start = i
            i += 1
            while source[i] != '"':
                i += 1
            i += 1
            tokens.append(Token("STRING", source[start:i], start))
            continue
        for op in OPERATORS:
            if source.startswith(op, i):
                tokens.append(Token("OP", op, i))
                i += len(op)
                break
        else:
            raise LexError(f"unexpected character {ch!r}", i)
    tokens.append(Token("EOF", "", i))
    return tokens
