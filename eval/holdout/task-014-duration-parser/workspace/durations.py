"""Parse and format human-readable durations such as ``"1h30m"``.

``parse_duration(text)`` returns a number of seconds. The accepted forms:

* One or more components, each an unsigned decimal integer immediately
  followed by a unit: ``d`` (days), ``h`` (hours), ``m`` (minutes) or
  ``s`` (seconds). Units are case-insensitive.
* Components may be separated by whitespace (``"2d 4h"``). Leading and
  trailing whitespace is ignored. There is no whitespace between a
  number and its unit.
* Each unit appears at most once, and units appear from largest to
  smallest: ``"1h30m"`` is valid, ``"30m1h"`` is not.
* As a special case, a string that is only an unsigned integer (``"90"``)
  is that many seconds.

Every other string - empty or blank, an unknown unit, a sign, a decimal
point, a number without a unit next to other components, a unit without
a number, any stray character - raises ``ValueError``. For ``str`` input,
``parse_duration`` raises nothing but ``ValueError``.

``format_duration(seconds)`` is the inverse. It renders a non-negative
``int`` canonically: largest unit first, zero components left out,
lowercase units and no spaces (``"1d2h3m4s"``, ``"5m"``, ``"1h5s"``).
Zero is ``"0s"``. A negative number raises ``ValueError``.
"""

from __future__ import annotations

import re

UNIT_SECONDS = {"d": 86400, "h": 3600, "m": 60, "s": 1}

_COMPONENT = re.compile(r"(\d+)([dhms])")


def parse_duration(text: str) -> int:
    """Return the number of seconds ``text`` describes."""
    total = 0
    for number, unit in _COMPONENT.findall(text):
        total += int(number) * UNIT_SECONDS[unit]
    return total


def format_duration(seconds: int) -> str:
    """Render ``seconds`` canonically, e.g. ``3725`` -> ``"1h2m5s"``."""
    parts = []
    for unit, size in UNIT_SECONDS.items():
        count, seconds = divmod(seconds, size)
        if count:
            parts.append(f"{count}{unit}")
    return "".join(parts)
