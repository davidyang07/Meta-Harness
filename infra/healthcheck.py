"""Container healthcheck for the Meta-Harness backend.

Exits 0 only when the API answers *and* reports a working persistence
backend. A backend that came up without Postgres still serves requests,
but with checkpointing degraded to in-memory — no checkpoint history, no
forking, no branch recovery. Calling that "healthy" would hide exactly
the failure this project cares most about, so it exits non-zero and says
which of the two went wrong.

Set ``META_HARNESS_HEALTHCHECK_REQUIRE_POSTGRES=0`` to accept in-memory
persistence, for running the image with no database attached.

Uses only the standard library, so the runtime image needs no curl.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

TIMEOUT_S = 4.0


def _require_postgres() -> bool:
    raw = os.environ.get("META_HARNESS_HEALTHCHECK_REQUIRE_POSTGRES", "1").lower()
    return raw not in {"0", "false", "no"}


def main() -> int:
    port = os.environ.get("META_HARNESS_PORT", "8000")
    url = f"http://127.0.0.1:{port}/health"

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:  # noqa: S310
            if response.status != 200:
                print(f"{url} returned HTTP {response.status}", file=sys.stderr)
                return 1
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"{url} unreachable: {exc}", file=sys.stderr)
        return 1

    if payload.get("status") != "ok":
        print(f"unhealthy: {payload}", file=sys.stderr)
        return 1

    backend = payload.get("persistence")
    if _require_postgres() and backend != "postgres":
        print(
            "API is up but persistence is "
            f"{backend!r}: {payload.get('persistence_error')}",
            file=sys.stderr,
        )
        return 1

    print(f"ok (persistence: {backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
