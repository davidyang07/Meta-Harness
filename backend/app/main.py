"""FastAPI application for the Meta-Harness backend."""

from __future__ import annotations

import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import parse_qs

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import branches, checkpoints, events, forks, memory, runs
from app.event_loop import incompatible_loop_hint, is_psycopg_compatible_loop
from app.api.runs import cancel_active_runs
from app.meta_harness.branches import cancel_all_branches, set_runs_root
from app.meta_harness.memory import memory_store as memory_store_cm
from app.meta_harness.persistence import healthcheck, persistence_layer
from app.streaming import StreamingRegistryError


REPO_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger("meta_harness.api")


#: Environment variable holding the API bearer token. Unset or empty
#: means the API is unauthenticated, which ``meta-harness serve`` only
#: allows on a loopback address.
API_TOKEN_ENV = "META_HARNESS_API_TOKEN"

#: Routes that accept ``?token=`` as well as the header: ``EventSource``
#: cannot send an ``Authorization`` header.
_SSE_PATH_RE = re.compile(r"^/runs/[^/]+/stream$")

#: Headers the dashboard may send cross-origin. ``Authorization`` has to
#: be named: a ``*`` wildcard does not cover it under the Fetch spec.
CORS_ALLOWED_HEADERS = ["Authorization", "Content-Type", "Last-Event-ID", "Cache-Control"]


def configured_api_token() -> str | None:
    """The API token from the environment, or ``None`` when auth is off."""
    token = os.environ.get(API_TOKEN_ENV, "").strip()
    return token or None


class APITokenMiddleware:
    """Require ``Authorization: Bearer <token>`` when a token is configured.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so it never buffers a
    streaming (SSE) response. Every route except ``GET /health`` is
    protected: the container healthcheck and load balancers probe
    ``/health`` without credentials, and it reveals nothing but liveness
    and the persistence backend. SSE routes also accept ``?token=``,
    because browsers' ``EventSource`` cannot set headers.

    The token is read per request, so a token in ``.env`` (loaded during
    lifespan startup) is honoured, and so tests can toggle it.
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        token_provider: Callable[[], str | None] = configured_api_token,
    ) -> None:
        self.app = app
        self.token_provider = token_provider

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = self.token_provider()
        if token is None or self._exempt(scope) or self._authorized(scope, token):
            await self.app(scope, receive, send)
            return
        response = JSONResponse(
            status_code=401,
            content={"detail": "missing or invalid API token"},
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)

    @staticmethod
    def _exempt(scope: dict[str, Any]) -> bool:
        return scope.get("path") == "/health" and scope.get("method") in {"GET", "HEAD"}

    @staticmethod
    def _authorized(scope: dict[str, Any], token: str) -> bool:
        expected = token.encode("utf-8")
        for name, value in scope.get("headers") or []:
            if name.lower() != b"authorization":
                continue
            scheme, _, presented = value.decode("latin-1").partition(" ")
            if scheme.lower() == "bearer" and hmac.compare_digest(
                presented.strip().encode("utf-8"), expected
            ):
                return True
        if scope.get("method") == "GET" and _SSE_PATH_RE.match(scope.get("path", "")):
            query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
            for presented in query.get("token", []):
                if hmac.compare_digest(presented.encode("utf-8"), expected):
                    return True
        return False


def _should_try_persistence(use_persistence: bool | None) -> bool:
    if use_persistence is not None:
        return use_persistence
    mode = os.environ.get("META_HARNESS_API_PERSISTENT", "auto").lower()
    return mode not in {"0", "false", "no", "memory"}


def create_app(
    *,
    repo_root: Path | None = None,
    eval_tasks_dir: Path | None = None,
    use_persistence: bool | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    ``use_persistence=False`` is mainly for tests. The default mode tries
    the shared AsyncPostgresSaver and falls back to per-run in-memory
    checkpointing when Postgres is not reachable.
    """

    env_repo_root = os.environ.get("META_HARNESS_REPO_ROOT")
    repo_root = (repo_root or Path(env_repo_root or REPO_ROOT)).resolve()
    env_eval_tasks_dir = os.environ.get("META_HARNESS_EVAL_TASKS_DIR")
    eval_tasks_dir = eval_tasks_dir or Path(
        env_eval_tasks_dir or (repo_root / "eval" / "tasks")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        load_dotenv(repo_root / ".env")
        app.state.repo_root = repo_root
        # Branch history is persisted under runs/<run_id>/branches.json so
        # the trajectory survives an API restart.
        set_runs_root(repo_root / "runs")
        app.state.eval_tasks_dir = eval_tasks_dir
        app.state.checkpointer = None
        app.state.memory_store = None
        app.state.persistence_backend = "memory"
        app.state.persistence_cm = None
        app.state.memory_store_cm = None
        app.state.persistence_error = None

        if _should_try_persistence(use_persistence):
            # Falling back to in-memory checkpointing is a real capability
            # loss: no checkpoint history, no forking, no branch recovery.
            # It used to happen silently. Say why, loudly.
            if not is_psycopg_compatible_loop():
                app.state.persistence_error = incompatible_loop_hint()
                log.error(
                    "Postgres persistence disabled: %s",
                    app.state.persistence_error,
                )
            elif not await healthcheck():
                app.state.persistence_error = (
                    "Postgres is not reachable at the configured POSTGRES_DSN; "
                    "start it with "
                    "'docker compose -f infra/docker-compose.yml up -d postgres'"
                )
                log.warning(
                    "Postgres persistence disabled: %s",
                    app.state.persistence_error,
                )
            else:
                cm = persistence_layer()
                app.state.checkpointer = await cm.__aenter__()
                app.state.persistence_cm = cm
                app.state.persistence_backend = "postgres"
                # Best-effort memory store (same Postgres instance)
                try:
                    mem_cm = memory_store_cm()
                    app.state.memory_store = await mem_cm.__aenter__()
                    app.state.memory_store_cm = mem_cm
                except Exception as exc:  # noqa: BLE001 — memory is optional
                    log.warning(
                        "cross-run memory store unavailable (%s); "
                        "the memory panel will show an empty state",
                        type(exc).__name__,
                    )

        try:
            yield
        finally:
            await cancel_active_runs()
            await cancel_all_branches()
            mem_cm = getattr(app.state, "memory_store_cm", None)
            if mem_cm is not None:
                try:
                    await mem_cm.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
            cm = getattr(app.state, "persistence_cm", None)
            if cm is not None:
                await cm.__aexit__(None, None, None)

    app = FastAPI(
        title="Meta-Harness Backend",
        version=__version__,
        lifespan=lifespan,
    )
    # Added first, so it runs *inside* CORS: a 401 still carries the CORS
    # headers the dashboard needs to read it, and preflights (which never
    # carry credentials) are answered by CORS before auth sees them.
    app.add_middleware(APITokenMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_methods=["*"],
        allow_headers=CORS_ALLOWED_HEADERS,
    )

    @app.exception_handler(StreamingRegistryError)
    async def _streaming_error_handler(_request, exc: StreamingRegistryError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc)},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness plus an honest report of which backend is in use.

        ``persistence: "memory"`` means checkpoint history, forking and
        branch recovery are unavailable; ``persistence_error`` says why.
        ``auth_required`` tells a client whether to send a token. This is
        the one unauthenticated route.
        """
        return {
            "status": "ok",
            "version": __version__,
            "persistence": app.state.persistence_backend,
            "persistence_error": getattr(app.state, "persistence_error", None),
            "memory_store": bool(getattr(app.state, "memory_store", None)),
            "auth_required": configured_api_token() is not None,
        }

    app.include_router(runs.router)
    app.include_router(checkpoints.router)
    app.include_router(forks.router)
    app.include_router(branches.router)
    app.include_router(events.router)
    app.include_router(memory.router)
    return app


app = create_app()
