# Meta-Harness backend — FastAPI (SSE + REST) on a Postgres-compatible
# event loop.
#
# Build from the repository root, because the uv workspace spans
# sdk/ and backend/ and the lockfile at the root is what pins both:
#
#   docker build -f infra/Dockerfile -t meta-harness-backend .
#
# Or, with Postgres, via compose:
#
#   docker compose -f infra/docker-compose.yml up -d --build
#
# No credentials are baked in. POSTGRES_DSN, ANTHROPIC_API_KEY and the
# META_HARNESS_* knobs are read from the environment at run time; see
# .env.example.

# ── builder ───────────────────────────────────────────────────────────
# Pinned by digest-free tag plus a frozen lockfile: `uv sync --frozen`
# fails rather than re-resolving, so the image's dependency set is
# exactly uv.lock and a drifted lockfile is a build error, not a silent
# difference between the image and the tests.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Manifests first, so a source-only edit does not re-resolve or
# re-download the dependency layer.
COPY pyproject.toml uv.lock ./
COPY sdk/pyproject.toml sdk/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml

# --no-dev: test-only packages have no business in a runtime image.
# --no-install-workspace: the workspace members are copied in next; this
# layer is the third-party dependency closure alone.
RUN uv sync --frozen --no-dev --no-install-workspace

COPY sdk/ sdk/
COPY backend/ backend/

RUN uv sync --frozen --no-dev

# ── runtime ───────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

# git is a runtime dependency, not a build tool: `tools.apply_patch`
# shells out to `git apply` for every inner-loop edit, and
# `experiment.capture_environment` records the commit a measurement was
# taken at. Without it patches fail as context_mismatch and provenance
# is null.
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The uid is fixed so a bind-mounted runs/ directory has
# predictable ownership on the host.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app sdk/ sdk/
COPY --chown=app:app backend/ backend/
COPY --chown=app:app agents/ agents/
COPY --chown=app:app eval/ eval/
COPY --chown=app:app benchmarks/ benchmarks/
COPY --chown=app:app infra/healthcheck.py infra/healthcheck.py

# Written to at run time by the outer loop and the benchmark runner.
RUN mkdir -p /app/runs && chown app:app /app/runs

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Bind all interfaces: the only reachable address inside a container.
    # Publishing the port is the compose file's decision, not the image's.
    META_HARNESS_HOST=0.0.0.0 \
    META_HARNESS_PORT=8000

USER app
EXPOSE 8000

# Reports unhealthy when the API is up but Postgres is not: a backend in
# that state answers requests with checkpointing silently degraded to
# in-memory, which disables checkpoint history, forking and branch
# recovery. "Running but cannot persist" is the failure this project most
# needs to see, so the check refuses to call it healthy.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "/app/infra/healthcheck.py"]

# `meta-harness serve`, never a bare `uvicorn app.main:app`: serve
# installs a selector event loop, which psycopg's async driver requires.
CMD ["sh", "-c", "exec meta-harness serve --host \"$META_HARNESS_HOST\" --port \"$META_HARNESS_PORT\""]
