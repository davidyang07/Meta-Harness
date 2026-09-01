#!/usr/bin/env bash
# Docker smoke test — build the backend image and prove the container is
# actually a working backend, not just a green `docker build`.
#
# What it proves:
#   1. the image builds from the frozen lockfile;
#   2. the process runs as a non-root user;
#   3. no credential is baked into an image layer;
#   4. the container reaches Postgres and reports persistence: "postgres"
#      (not the in-memory fallback, which would disable checkpoint
#      history, forking and branch recovery);
#   5. the healthcheck refuses to call a Postgres-less backend healthy.
#
# What it deliberately does NOT do: run a model, run the outer loop, or
# spend anything. No API key is required.
#
# Usage:
#   bash scripts/docker_smoke.sh
#
# Requires a running Docker daemon. Exits non-zero on the first failure.

set -euo pipefail

# Git Bash rewrites /app/... arguments into Windows paths before docker
# sees them, which turns `docker exec ... python /app/infra/healthcheck.py`
# into a file-not-found. Harmless everywhere else.
export MSYS_NO_PATHCONV=1

IMAGE="${IMAGE:-meta-harness-backend:smoke}"
NET="mh-smoke-net-$$"
PG="mh-smoke-pg-$$"
APP="mh-smoke-app-$$"
NOPG="mh-smoke-nopg-$$"
# Docker on Windows is a native binary and does not understand Git Bash's
# /c/... paths, so hand it a Windows path when we are running under MSYS.
if command -v cygpath >/dev/null 2>&1; then
    REPO_ROOT="$(cygpath -w "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)")"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

pass() { printf '  [PASS] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1" >&2; exit 1; }

cleanup() {
    docker rm -f "$APP" "$NOPG" "$PG" >/dev/null 2>&1 || true
    docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== 1. build =================================================="
docker build -f "$REPO_ROOT/infra/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
pass "image built from uv.lock (--frozen; a drifted lockfile fails the build)"

echo "== 2. the image runs as a non-root user ======================"
uid="$(docker run --rm --entrypoint id "$IMAGE" -u)"
[ "$uid" != "0" ] || fail "container runs as root (uid 0)"
pass "runs as uid $uid"

echo "== 3. no credentials baked into the image ===================="
# The base image's own GPG_KEY is not ours; anything else matching a
# credential-shaped name would be.
baked="$(docker inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | grep -Ei '(api_key|password|secret|token|_dsn)=' || true)"
[ -z "$baked" ] || fail "credential-shaped env baked into the image: $baked"
if docker run --rm --entrypoint sh "$IMAGE" -c 'test -e /app/.env' 2>/dev/null; then
    fail "an .env file is present in the image"
fi
pass "no credential env and no .env in the image"

echo "== 4. the container reaches Postgres ========================="
docker network create "$NET" >/dev/null
docker run -d --name "$PG" --network "$NET" \
    -e POSTGRES_USER=meta_harness \
    -e POSTGRES_PASSWORD=meta_harness \
    -e POSTGRES_DB=meta_harness \
    --health-cmd "pg_isready -U meta_harness -d meta_harness" \
    --health-interval 3s --health-timeout 5s --health-retries 20 \
    postgres:16 >/dev/null

for _ in $(seq 1 40); do
    [ "$(docker inspect --format '{{.State.Health.Status}}' "$PG")" = "healthy" ] && break
    sleep 2
done
[ "$(docker inspect --format '{{.State.Health.Status}}' "$PG")" = "healthy" ] \
    || fail "the Postgres container never became healthy"

docker run -d --name "$APP" --network "$NET" \
    -e POSTGRES_DSN="postgresql://meta_harness:meta_harness@${PG}:5432/meta_harness" \
    "$IMAGE" >/dev/null

status=""
for _ in $(seq 1 40); do
    status="$(docker inspect --format '{{.State.Health.Status}}' "$APP")"
    [ "$status" = "healthy" ] && break
    [ "$status" = "unhealthy" ] && break
    sleep 3
done
[ "$status" = "healthy" ] || {
    docker logs "$APP" 2>&1 | tail -40 >&2
    fail "the backend container never became healthy (status: $status)"
}

health="$(docker exec "$APP" python -c \
    'import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen("http://127.0.0.1:8000/health"))))')"
echo "  /health -> $health"
case "$health" in
    *'"persistence": "postgres"'*|*'"persistence":"postgres"'*) ;;
    *) fail "the container did not connect to Postgres: $health" ;;
esac
pass "app -> Postgres connectivity, checkpointing on the real backend"

echo "== 5. a Postgres-less backend is NOT reported healthy ========"
# The failure this project most needs to see: the API answers, but
# checkpoint history, forking and branch recovery are silently gone.
docker run -d --name "$NOPG" \
    -e POSTGRES_DSN="postgresql://nobody:nobody@127.0.0.1:1/none" \
    "$IMAGE" >/dev/null
for _ in $(seq 1 20); do
    docker exec "$NOPG" python -c \
        'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health")' \
        >/dev/null 2>&1 && break
    sleep 2
done
if docker exec "$NOPG" python /app/infra/healthcheck.py >/dev/null 2>&1; then
    fail "the healthcheck called a Postgres-less backend healthy"
fi
pass "healthcheck exits non-zero when persistence degraded to in-memory"

# ...and the documented opt-out still works, for running with no database.
docker exec -e META_HARNESS_HEALTHCHECK_REQUIRE_POSTGRES=0 "$NOPG" \
    python /app/infra/healthcheck.py >/dev/null \
    || fail "META_HARNESS_HEALTHCHECK_REQUIRE_POSTGRES=0 did not relax the check"
pass "the documented opt-out accepts in-memory persistence"

echo
echo "DOCKER SMOKE OK"
