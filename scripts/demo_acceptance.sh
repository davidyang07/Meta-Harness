#!/usr/bin/env bash
# LEVEL 1 acceptance — offline / deterministic.
#
# Verifies everything that can be proven without paying for a single LLM
# call. It prints DEMO READY (LEVEL 1) only if every check below passes.
#
# What it deliberately does NOT prove: that a real model call works, that
# the claude-CLI proposer produces a valid candidate, or any measured
# benchmark number. Those are LEVEL 2 — see scripts/live_smoke.sh.
#
# Usage:
#   bash scripts/demo_acceptance.sh              # full ladder
#   SKIP_E2E=1 bash scripts/demo_acceptance.sh   # skip Playwright
#   SKIP_NPM_CI=1 bash scripts/demo_acceptance.sh
#
# NOTE: `npm ci` deletes and reinstalls frontend/dashboard/node_modules.
# Do not run this concurrently with a dev server, another acceptance run,
# or an `npx` invocation — they will race over node_modules and produce
# spurious "'next' is not recognized" failures. SKIP_NPM_CI=1 reuses the
# existing install (faster, but no longer proves a clean install works).
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; DIM='\033[2m'; NC='\033[0m'

LOG_DIR="$(mktemp -d)"
API_PORT="${API_PORT:-8765}"
API_URL="http://127.0.0.1:${API_PORT}"
RUN_NAME="acceptance-$(date +%s)"
SERVER_PID=""

pass=0; fail=0; skipped=0
declare -a FAILURES=()

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi
  rm -rf "runs/$RUN_NAME" 2>/dev/null || true
}
trap cleanup EXIT

section() { printf '\n%b── %s %b\n' "$DIM" "$1" "$NC"; }

check() {
  local label="$1"; shift
  local log="$LOG_DIR/$(echo "$label" | tr -c 'a-zA-Z0-9' '_').log"
  printf '  %-52s' "$label"
  if "$@" >"$log" 2>&1; then
    printf '%bOK%b\n' "$GREEN" "$NC"
    pass=$((pass + 1))
  else
    printf '%bFAIL%b\n' "$RED" "$NC"
    sed -n '1,12p' "$log" | sed 's/^/        /'
    printf '        %blog: %s%b\n' "$DIM" "$log" "$NC"
    fail=$((fail + 1))
    FAILURES+=("$label")
  fi
}

skip() {
  printf '  %-52s%bSKIPPED%b (%s)\n' "$1" "$YELLOW" "$NC" "$2"
  skipped=$((skipped + 1))
}

api() { curl -sS --max-time 30 "$@"; }

# ── 1. Toolchain ─────────────────────────────────────────────────────
section "1. Toolchain"
check "uv installed"            command -v uv
check "node installed"          command -v node
check "docker installed"        command -v docker
check "Postgres reachable"      bash -c '
  cd backend && uv run python -c "
import asyncio, sys
from app.meta_harness.persistence import healthcheck
sys.exit(0 if asyncio.run(healthcheck()) else 1)
"'

# ── 2. Backend test suite ────────────────────────────────────────────
section "2. Backend test suite"
check "pytest suite green"      bash -c 'cd backend && uv run pytest tests/ -q --no-header'
check "Postgres tests ran (not skipped)" bash -c '
  cd backend
  out=$(uv run pytest tests/test_persistence.py tests/test_branch_isolation.py \
        tests/test_inner_checkpoints.py tests/test_branch_persistence.py \
        -q --no-header 2>&1)
  echo "$out"
  # A silently skipped Postgres suite is the failure mode this guards.
  echo "$out" | grep -qE "^[0-9]+ passed" && ! echo "$out" | grep -q "skipped"
'

# ── 3. CLI surface ───────────────────────────────────────────────────
section "3. CLI"
check "meta-harness --help"     bash -c 'uv run meta-harness --help >/dev/null'
check "meta-harness version"    bash -c 'uv run meta-harness version | grep -q meta-harness'
check "all subcommands present" bash -c '
  out=$(uv run meta-harness --help 2>&1)
  for cmd in version inner benchmark loop fork serve experiment checkpoints replay init resume memory; do
    echo "$out" | grep -q "$cmd" || { echo "missing subcommand: $cmd"; exit 1; }
  done
'

# ── 4. Mock outer loop ───────────────────────────────────────────────
section "4. Mock outer loop (no LLM)"
check "loop produces thread-scoped artifacts" bash -c "
  uv run meta-harness loop --proposer mock --mock-bench --budget 2 --fresh \
      --run-name $RUN_NAME --no-persistent >/dev/null
  T=runs/$RUN_NAME/threads/$RUN_NAME
  test -f runs/$RUN_NAME/manifest.json
  test -f \$T/pending_eval.json
  test -f \$T/frontier_val.json
  test -f \$T/evolution_summary.jsonl
"
check "baseline is the measured search root" bash -c "
  uv run python -c \"
import json, sys
rows = [json.loads(l) for l in open('runs/$RUN_NAME/threads/$RUN_NAME/evolution_summary.jsonl') if l.strip()]
assert rows[0]['candidate'] == 'baseline', rows[0]
assert rows[0]['iteration'] == 0
assert rows[0]['scores']['accuracy'] > 0, 'baseline must be measured, not 0'
assert rows[1]['parent_candidate_name'] == 'baseline'
\"
"
check "mock metrics are labelled as mock" bash -c "
  uv run python -c \"
import json
fr = json.load(open('runs/$RUN_NAME/threads/$RUN_NAME/frontier_val.json'))
assert fr['metrics_source'] == 'mock', fr.get('metrics_source')
for c in fr['candidates']:
    assert 'dominated_by_names' in c
\"
"

# ── 5. Live API ──────────────────────────────────────────────────────
section "5. Live API + SSE + fork"
(cd backend && uv run meta-harness serve --port "$API_PORT" >"$LOG_DIR/server.log" 2>&1) &
SERVER_PID=$!
for _ in $(seq 1 60); do
  api "$API_URL/health" >/dev/null 2>&1 && break
  sleep 1
done

check "health endpoint"         bash -c "curl -sS --max-time 10 $API_URL/health | grep -q '\"status\":\"ok\"'"
check "persistence is postgres" bash -c "
  body=\$(curl -sS --max-time 10 $API_URL/health)
  echo \"\$body\"
  echo \"\$body\" | grep -q '\"persistence\":\"postgres\"'
"
check "API run + checkpoints + fork + branch isolation" bash -c "
  API_URL=$API_URL uv run python scripts/acceptance_api_flow.py
"
check "SSE stream emits events"  bash -c "
  API_URL=$API_URL uv run python scripts/acceptance_api_flow.py --sse-only
"

# ── 5b. Restart recovery ─────────────────────────────────────────────
# Branch history is persisted to runs/<run>/branches.json. Kill the
# backend, start a fresh one, and confirm the branch tree is still there
# — the asyncio tasks are gone, the history is not.
section "5b. Trajectory survives a backend restart"
RECOVER_RUN=""
if RECOVER_OUT="$(API_URL=$API_URL uv run python scripts/acceptance_api_flow.py --keep 2>&1)"; then
  RECOVER_RUN="$(echo "$RECOVER_OUT" | sed -n 's/^RUN_ID=//p' | tail -1)"
fi

if [[ -z "$RECOVER_RUN" ]]; then
  skip "restart trajectory recovery" "could not create a run to recover"
else
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  (cd backend && uv run meta-harness serve --port "$API_PORT" >"$LOG_DIR/server2.log" 2>&1) &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    api "$API_URL/health" >/dev/null 2>&1 && break
    sleep 1
  done
  check "branch tree recovered by a fresh process" bash -c "
    API_URL=$API_URL uv run python scripts/acceptance_api_flow.py --recover $RECOVER_RUN
  "
fi

# ── 6. Frontend ──────────────────────────────────────────────────────
section "6. Frontend"
if [[ "${SKIP_NPM_CI:-0}" == "1" ]]; then
  skip "npm ci" "SKIP_NPM_CI=1"
else
  # No --silent: a failure here must be diagnosable from the log.
  check "npm ci"                bash -c 'cd frontend/dashboard && npm ci --no-audit --no-fund'
fi
check "eslint"                  bash -c 'cd frontend/dashboard && npm run lint'
check "next build"              bash -c 'cd frontend/dashboard && npm run build'

# ── 7. Browser end-to-end ────────────────────────────────────────────
section "7. Browser end-to-end"
if [[ "${SKIP_E2E:-0}" == "1" ]]; then
  skip "playwright (mock)" "SKIP_E2E=1"
  skip "playwright (live backend)" "SKIP_E2E=1"
else
  check "playwright (mock fixtures)" bash -c '
    cd frontend/dashboard && npx playwright test --project=mock'
  check "playwright (live backend)"  bash -c "
    cd frontend/dashboard && NEXT_PUBLIC_API_BASE_URL=$API_URL \
      npx playwright test --project=live-backend"
fi

# ── 8. Repository hygiene ────────────────────────────────────────────
section "8. Repository hygiene"
check "no tracked .env"         bash -c '! git ls-files | grep -qE "(^|/)\.env$"'
check "no tracked run artifacts" bash -c '! git ls-files | grep -qE "^(runs|benchmark-results)/"'
check "no generated candidates tracked" bash -c '
  git ls-files agents/ | grep -vE "agents/(__init__|baseline)\.py" | grep -q . && exit 1 || exit 0'
check "no API key patterns tracked" bash -c '
  ! git grep -nE "sk-ant-[A-Za-z0-9_-]{20,}" -- . ":(exclude)scripts/demo_acceptance.sh" >/dev/null 2>&1'

# ── Summary ──────────────────────────────────────────────────────────
printf '\n%b── Summary %b\n' "$DIM" "$NC"
printf '  %bpass: %d%b   %bfail: %d%b   %bskipped: %d%b\n' \
  "$GREEN" "$pass" "$NC" "$RED" "$fail" "$NC" "$YELLOW" "$skipped" "$NC"

if (( fail > 0 )); then
  printf '\n  %bLEVEL 1 ACCEPTANCE FAILED%b\n' "$RED" "$NC"
  for f in "${FAILURES[@]}"; do printf '    - %s\n' "$f"; done
  exit 1
fi

if (( skipped > 0 )); then
  printf '\n  %bLEVEL 1 PARTIAL%b — %d check(s) skipped; re-run without SKIP_* to claim demo readiness.\n' \
    "$YELLOW" "$NC" "$skipped"
  exit 0
fi

printf '\n  %bDEMO READY (LEVEL 1: offline / deterministic)%b\n' "$GREEN" "$NC"
printf '  %bThis does NOT cover live model calls, the claude proposer, or any%b\n' "$DIM" "$NC"
printf '  %bmeasured benchmark number. For those run scripts/live_smoke.sh.%b\n' "$DIM" "$NC"
