"""Contact records: parsing them from text and formatting them back.

A record is one line of ``name;email;phone`` (the phone is optional).
Callers import everything listed in ``__all__`` from this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

__all__ = [
    "FIELD_SEPARATOR",
    "Contact",
    "ParseError",
    "parse_line",
    "parse_lines",
    "format_contact",
    "format_lines",
    "register_style",
    "available_styles",
]

FIELD_SEPARATOR = ";"


class ParseError(ValueError):
    """A record could not be parsed. ``line_no`` is 1-based when known."""

    def __init__(self, message: str, line_no: int | None = None) -> None:
        self.message = message
        self.line_no = line_no
        where = f"line {line_no}: " if line_no is not None else ""
        super().__init__(f"{where}{message}")


@dataclass(frozen=True)
class Contact:
    name: str
    email: str
    phone: str = ""


def _normalize_phone(raw: str) -> str:
    """Keep only the digits of a phone number; 7 to 15 of them, or none."""
    digits = re.sub(r"\D", "", raw)
    if digits and not 7 <= len(digits) <= 15:
        raise ValueError(f"phone number must have 7-15 digits: {raw!r}")
    return digits


def parse_line(line: str) -> Contact:
    """Parse one ``name;email[;phone]`` record."""
    parts = [part.strip() for part in line.split(FIELD_SEPARATOR)]
    if len(parts) not in (2, 3):
        raise ParseError(f"expected 2 or 3 fields, got {len(parts)}")
    name, email = parts[0], parts[1].lower()
    if not name:
        raise ParseError("missing name")
    if email.count("@") != 1 or email.startswith("@") or email.endswith("@"):
        raise ParseError(f"invalid email: {parts[1]!r}")
    try:
        phone = _normalize_phone(parts[2]) if len(parts) == 3 else ""
    except ValueError as exc:
        raise ParseError(str(exc)) from None
    return Contact(name=name, email=email, phone=phone)


def parse_lines(text: str) -> list[Contact]:
    """Parse every record in ``text``, skipping blank lines and ``#`` comments."""
    contacts = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            contacts.append(parse_line(line))
        except ParseError as exc:
            raise ParseError(exc.message, line_no) from None
    return contacts


_STYLES: dict[str, Callable[[Contact], str]] = {}


def register_style(name: str) -> Callable[[Callable[[Contact], str]], Callable[[Contact], str]]:
    """Decorator: make ``fn(contact) -> str`` available as style ``name``."""

    def decorator(fn: Callable[[Contact], str]) -> Callable[[Contact], str]:
        _STYLES[name] = fn
        return fn

    return decorator


def available_styles() -> list[str]:
    """Names of every registered style, sorted."""
    return sorted(_STYLES)


def format_contact(contact: Contact, style: str = "line") -> str:
    """Render one contact in the given style."""
    try:
        formatter = _STYLES[style]
    except KeyError:
        raise ValueError(f"unknown style {style!r}") from None
    return formatter(contact)


def format_lines(contacts: Iterable[Contact], style: str = "line") -> str:
    """Render several contacts, one per line (or block) in order."""
    return "\n".join(format_contact(contact, style) for contact in contacts)


def _display_phone(digits: str) -> str:
    """Group a 10-digit number as ``555-123-4567``; leave others as is."""
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return digits


@register_style("line")
def _line_style(contact: Contact) -> str:
    fields = [contact.name, contact.email]
    if contact.phone:
        fields.append(contact.phone)
    return FIELD_SEPARATOR.join(fields)


@register_style("card")
def _card_style(contact: Contact) -> str:
    lines = [contact.name, f"  email: {contact.email}"]
    if contact.phone:
        lines.append(f"  phone: {_display_phone(contact.phone)}")
    return "\n".join(lines)


@register_style("mailto")
def _mailto_style(contact: Contact) -> str:
    return f"{contact.name} <{contact.email}>"
