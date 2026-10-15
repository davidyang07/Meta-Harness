"""API credentials: the bearer token, stream tickets, and log redaction.

``EventSource`` cannot send an ``Authorization`` header, so the event
stream has to be authorised by its URL. The URL used to carry the API
token itself, and a URL ends up in access logs (uvicorn's, and any
proxy's in front of it). The dashboard now trades its token, over an
authenticated ``POST``, for a *stream ticket*: valid for one run's
stream, for ``STREAM_TICKET_TTL_S`` seconds, and useless for anything
else. The token never appears in a URL.

Tickets are stateless: ``<expiry>.<HMAC-SHA256(token, run, expiry)>``.
Only a holder of the token can mint one, and rotating the token voids
every ticket outstanding.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import time

#: Environment variable holding the API bearer token. Unset or empty
#: means the API is unauthenticated, which ``meta-harness serve`` only
#: allows on a loopback address.
API_TOKEN_ENV = "META_HARNESS_API_TOKEN"

#: How long a stream ticket can open a stream. It is checked when the
#: stream opens; an open stream is not cut off when it expires.
STREAM_TICKET_TTL_S = 60


def configured_api_token() -> str | None:
    """The API token from the environment, or ``None`` when auth is off."""
    token = os.environ.get(API_TOKEN_ENV, "").strip()
    return token or None


def _signature(token: str, run_id: str, expires: int) -> str:
    message = f"stream:{run_id}:{expires}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), message, hashlib.sha256).hexdigest()


def mint_stream_ticket(token: str, run_id: str, *, now: float | None = None) -> str:
    """A ticket that opens ``run_id``'s stream for ``STREAM_TICKET_TTL_S``."""
    expires = int((time.time() if now is None else now) + STREAM_TICKET_TTL_S)
    return f"{expires}.{_signature(token, run_id, expires)}"


def stream_ticket_valid(
    token: str, run_id: str, ticket: str, *, now: float | None = None
) -> bool:
    """True if ``ticket`` was minted with ``token`` for ``run_id`` and is
    unexpired."""
    expires_raw, dot, signature = ticket.partition(".")
    if not dot or not expires_raw.isdigit():
        return False
    expires = int(expires_raw)
    if expires < (time.time() if now is None else now):
        return False
    return hmac.compare_digest(
        signature.encode("utf-8"),
        _signature(token, run_id, expires).encode("utf-8"),
    )


# ── keeping credentials out of logs ───────────────────────────────────

_CREDENTIAL_QUERY_RE = re.compile(r"((?:^|[?&])(?:token|ticket)=)[^&\s\"]*")


def redact_credentials(text: str) -> str:
    """``text`` with the value of every ``token=`` / ``ticket=`` query
    parameter replaced. A client that still puts a credential in a URL
    must not get it written to the server's logs."""
    return _CREDENTIAL_QUERY_RE.sub(r"\1[redacted]", text)


class RedactCredentialsFilter(logging.Filter):
    """Redacts query-string credentials from a log record's arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                redact_credentials(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        elif isinstance(record.msg, str):
            record.msg = redact_credentials(record.msg)
        return True


def install_access_log_redaction(logger_name: str = "uvicorn.access") -> None:
    """Attach ``RedactCredentialsFilter`` to the access logger, once."""
    logger = logging.getLogger(logger_name)
    if not any(isinstance(f, RedactCredentialsFilter) for f in logger.filters):
        logger.addFilter(RedactCredentialsFilter())
