"""Bearer-token auth for the API (``META_HARNESS_API_TOKEN``).

The API starts runs that spend money and serves the repository's run
artifacts. With a token configured, every route but ``GET /health``
requires ``Authorization: Bearer <token>``. ``EventSource`` cannot send
headers, so the stream accepts a short-lived ticket for that one run
(``POST /runs/{id}/stream-ticket``) instead; the token itself is never
accepted in a URL, because a URL is logged. Without a token the API is
open, which ``meta-harness serve`` only allows on loopback (see
``test_cli.py``).
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.runs import clear_run_state
from app.auth import (
    STREAM_TICKET_TTL_S,
    RedactCredentialsFilter,
    mint_stream_ticket,
    redact_credentials,
)
from app.main import create_app
from app.meta_harness import runs as runs_mod
from app.meta_harness.branches import clear_branch_state
from app.streaming import event_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN = "test-token-" + uuid.uuid4().hex
ORIGIN = "http://localhost:3000"


@pytest.fixture
def client_factory(tmp_path: Path, monkeypatch):
    clear_run_state()
    clear_branch_state()
    event_registry.clear()
    repo_root = tmp_path / "repo"
    (repo_root / "runs").mkdir(parents=True)
    opened: list[TestClient] = []

    def _make(token: str | None) -> TestClient:
        if token is None:
            monkeypatch.delenv("META_HARNESS_API_TOKEN", raising=False)
        else:
            monkeypatch.setenv("META_HARNESS_API_TOKEN", token)
        app = create_app(
            repo_root=repo_root,
            eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
            use_persistence=False,
        )
        client = TestClient(app)
        client.__enter__()
        opened.append(client)
        return client

    try:
        yield _make
    finally:
        for client in opened:
            client.__exit__(None, None, None)
        clear_run_state()
        clear_branch_state()
        event_registry.clear()
        shutil.rmtree(repo_root, ignore_errors=True)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_every_route_but_health_requires_the_token(client_factory):
    client = client_factory(TOKEN)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["auth_required"] is True

    for method, path in (
        ("GET", "/runs"),
        ("GET", "/runs/some-run"),
        ("GET", "/runs/some-run/checkpoints"),
        ("GET", "/runs/some-run/branches"),
        ("DELETE", "/runs/some-run"),
        ("GET", "/memory/coding-agent"),
    ):
        response = client.request(method, path)
        assert response.status_code == 401, (method, path)
        assert response.headers["www-authenticate"] == "Bearer"

    for token in ("wrong", TOKEN + "x", TOKEN[:-1], ""):
        assert client.get("/runs", headers=_bearer(token)).status_code == 401
    assert client.get("/runs", headers={"Authorization": TOKEN}).status_code == 401

    assert client.get("/runs", headers=_bearer(TOKEN)).status_code == 200


def test_an_unauthenticated_post_creates_nothing(client_factory, tmp_path: Path):
    client = client_factory(TOKEN)

    response = client.post(
        "/runs",
        json={"run_name": "nope", "proposer": "mock", "mock_bench": True},
    )

    assert response.status_code == 401
    assert not (tmp_path / "repo" / "runs" / "nope").exists()


def test_the_token_is_never_accepted_in_a_url(client_factory):
    client = client_factory(TOKEN)

    assert client.get(f"/runs/missing/stream?token={TOKEN}").status_code == 401
    assert client.get(f"/runs?token={TOKEN}").status_code == 401


def test_a_stream_ticket_is_minted_only_for_a_token_holder(client_factory, tmp_path):
    client = client_factory(TOKEN)
    runs_mod.make_run_dir(tmp_path / "repo", "ticketed")
    runs_mod.write_manifest(tmp_path / "repo" / "runs" / "ticketed", run_id="ticketed")

    assert client.post("/runs/ticketed/stream-ticket").status_code == 401
    assert (
        client.post("/runs/missing/stream-ticket", headers=_bearer(TOKEN)).status_code
        == 404
    )
    minted = client.post("/runs/ticketed/stream-ticket", headers=_bearer(TOKEN))
    assert minted.status_code == 200
    assert minted.json()["expires_in"] == STREAM_TICKET_TTL_S
    assert TOKEN not in minted.json()["ticket"]


def test_a_ticket_opens_one_runs_stream_until_it_expires(client_factory):
    client = client_factory(TOKEN)

    # Auth passes (the run does not exist: 404)...
    ticket = mint_stream_ticket(TOKEN, "missing")
    assert client.get(f"/runs/missing/stream?ticket={ticket}").status_code == 404
    # ...but not for another run, once expired, forged, or anywhere else.
    assert client.get(f"/runs/other/stream?ticket={ticket}").status_code == 401
    expired = mint_stream_ticket(TOKEN, "missing", now=time.time() - 2 * STREAM_TICKET_TTL_S)
    assert client.get(f"/runs/missing/stream?ticket={expired}").status_code == 401
    forged = mint_stream_ticket("not-the-token", "missing")
    assert client.get(f"/runs/missing/stream?ticket={forged}").status_code == 401
    for junk in ("", "abc", "123.", ".abc", f"{int(time.time()) + 999}.00"):
        assert client.get(f"/runs/missing/stream?ticket={junk}").status_code == 401
    assert client.get(f"/runs?ticket={ticket}").status_code == 401
    assert client.get("/runs/missing/stream").status_code == 401


def test_credentials_in_a_url_never_reach_the_access_log():
    line = "GET /runs/r/stream?token=s3cret&last_event_id=7&ticket=123.abc HTTP/1.1"
    assert redact_credentials(line) == (
        "GET /runs/r/stream?token=[redacted]&last_event_id=7&ticket=[redacted] HTTP/1.1"
    )

    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", f"/runs/r/stream?ticket=123.{TOKEN}", "1.1", 200),
        None,
    )
    assert RedactCredentialsFilter().filter(record)
    assert TOKEN not in record.getMessage()

    create_app(repo_root=REPO_ROOT, use_persistence=False)
    assert any(
        isinstance(f, RedactCredentialsFilter)
        for f in logging.getLogger("uvicorn.access").filters
    )


def test_cors_origins_are_configurable(client_factory, monkeypatch):
    monkeypatch.setenv("META_HARNESS_CORS_ORIGINS", "http://localhost:4000, https://dash.example")
    client = client_factory(TOKEN)

    def _allowed(origin: str) -> bool:
        preflight = client.options(
            "/runs",
            headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
        )
        return preflight.headers.get("access-control-allow-origin") == origin

    assert _allowed("http://localhost:4000")
    assert _allowed("https://dash.example")
    assert not _allowed(ORIGIN)


def test_without_a_token_the_api_is_open(client_factory):
    client = client_factory(None)

    assert client.get("/health").json()["auth_required"] is False
    assert client.get("/runs").status_code == 200


def test_cors_lets_the_dashboard_send_the_token(client_factory):
    client = client_factory(TOKEN)

    preflight = client.options(
        "/runs",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert preflight.status_code == 200
    allowed = preflight.headers["access-control-allow-headers"].lower()
    assert "authorization" in allowed

    # A 401 still carries CORS headers, so the dashboard can read it.
    denied = client.get("/runs", headers={"Origin": ORIGIN})
    assert denied.status_code == 401
    assert denied.headers["access-control-allow-origin"] == ORIGIN
