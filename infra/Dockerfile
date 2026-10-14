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
# No credentials are baked in. POSTGRES_DSN, ANTHROPIC_API_KEY,
# META_HARNESS_API_TOKEN and the META_HARNESS_* knobs are read from the
# environment at run time; see .env.example.
#
# The container binds 0.0.0.0, so `meta-harness serve` refuses to start
# unless META_HARNESS_API_TOKEN is set: an unauthenticated API reachable
# from other machines can start runs that spend money.
#
# The real proposer shells out to the `claude` CLI, which is not in the
# image by default (proposer=mock and the whole API work without it).
# Opt in with:
#
#   docker build -f infra/Dockerfile --build-arg INSTALL_CLAUDE_CLI=1 \
#       -t meta-harness-backend .

# Global build args: usable in FROM lines below.
ARG NODE_IMAGE=node:22.23.3-bookworm-slim

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

# ── optional: the `claude` CLI ────────────────────────────────────────
# A pinned @anthropic-ai/claude-code. Its npm package installs a native
# binary for the build platform, so only that one self-contained file is
# carried into the runtime image: no Node.js there, no npm tree. With
# INSTALL_CLAUDE_CLI=0 (the default) /out/bin stays empty and the COPY
# below adds nothing.
FROM ${NODE_IMAGE} AS claude-cli

ARG INSTALL_CLAUDE_CLI=0
ARG CLAUDE_CODE_VERSION=2.1.292

RUN set -eu; \
    mkdir -p /out/bin; \
    case "$INSTALL_CLAUDE_CLI" in \
      0) ;; \
      1) npm install --global --no-fund --no-audit --prefix /tmp/claude-npm \
             "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"; \
         cp -L /tmp/claude-npm/bin/claude /out/bin/claude; \
         chmod 0755 /out/bin/claude; \
         /out/bin/claude --version ;; \
      *) echo "INSTALL_CLAUDE_CLI must be 0 or 1, got '$INSTALL_CLAUDE_CLI'" >&2; \
         exit 1 ;; \
    esac

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
# The proposer's system prompt: proposer=claude resolves
# skills/meta-harness-<domain>/SKILL.md and refuses to start without it.
COPY --chown=app:app skills/ skills/
COPY --chown=app:app infra/healthcheck.py infra/healthcheck.py

# Empty unless built with INSTALL_CLAUDE_CLI=1.
COPY --from=claude-cli /out/bin/ /usr/local/bin/

# Written to at run time by the outer loop and the benchmark runner.
RUN mkdir -p /app/runs && chown app:app /app/runs

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Bind all interfaces: the only reachable address inside a container.
    # Publishing the port is the compose file's decision, not the image's.
    META_HARNESS_HOST=0.0.0.0 \
    META_HARNESS_PORT=8000 \
    # A pinned CLI in an immutable image must not rewrite itself.
    DISABLE_AUTOUPDATER=1

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
