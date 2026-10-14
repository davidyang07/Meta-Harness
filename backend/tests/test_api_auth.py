"""Bearer-token auth for the API (``META_HARNESS_API_TOKEN``).

The API starts runs that spend money and serves the repository's run
artifacts. With a token configured, every route but ``GET /health``
requires ``Authorization: Bearer <token>``; SSE routes also accept
``?token=``, because ``EventSource`` cannot send headers. Without a token
the API is open, which ``meta-harness serve`` only allows on loopback
(see ``test_cli.py``).
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.runs import clear_run_state
from app.main import create_app
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


def test_only_sse_routes_accept_a_query_token(client_factory):
    client = client_factory(TOKEN)

    # Auth passes on the stream route (the run does not exist: 404)...
    assert client.get(f"/runs/missing/stream?token={TOKEN}").status_code == 404
    # ...and fails without, or with a wrong, token.
    assert client.get("/runs/missing/stream").status_code == 401
    assert client.get("/runs/missing/stream?token=wrong").status_code == 401
    # A query token is not accepted anywhere else.
    assert client.get(f"/runs?token={TOKEN}").status_code == 401


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
