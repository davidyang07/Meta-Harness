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

# Exported, not just assigned. Most checks run their body through
# `bash -c '...'` in single quotes, so $LOG_DIR is expanded by that inner
# shell rather than this one — unexported it arrived empty, and the three
# checks that build a path from it silently wrote to the filesystem root:
#
#   meta-harness report wandb-check --output /wandb.json
#   meta-harness report resume-evidence --json > /evidence.json
#
# which on Windows hung the ladder at "W&B offline probe" and meant the
# two evidence checks never read what they claimed to.
LOG_DIR="$(mktemp -d)"
# Git Bash's mktemp returns an MSYS path (/tmp/tmp.XXXX). Bash writes there
# happily, but the Windows Python these checks invoke resolves /tmp/... to
# C:\tmp\... and cannot find the file bash just wrote. Hand every
# consumer a native path.
if command -v cygpath >/dev/null 2>&1; then
  LOG_DIR="$(cygpath -w "$LOG_DIR")"
fi
export LOG_DIR
API_PORT="${API_PORT:-8765}"
API_URL="http://127.0.0.1:${API_PORT}"
RUN_NAME="acceptance-$(date +%s)"
SERVER_PID=""

pass=0; fail=0; skipped=0
declare -a FAILURES=()

cleanup() {
  # stop_server is defined below; guard so an early exit still works.
  if declare -f stop_server >/dev/null 2>&1; then
    stop_server >/dev/null 2>&1 || true
  elif [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "runs/$RUN_NAME" 2>/dev/null || true
}
trap cleanup EXIT

section() { printf '\n%b── %s %b\n' "$DIM" "$1" "$NC"; }

# ── backend lifecycle ────────────────────────────────────────────────
# `kill $SERVER_PID` reaps only the subshell. The real server is two
# levels down (uv -> meta-harness -> python) and survives, still holding
# the port. Two checks then quietly stopped meaning what they say:
#
#   - stage 5 verified whichever backend was already listening, which
#     after a previous run is one built from an older checkout;
#   - stage 5b's "survives a backend restart" never restarted anything —
#     the replacement could not bind, so the readiness probe succeeded
#     against the process the test believed it had killed.
#
# Terminating by *port* rather than by pid fixes both: it is what the
# next start actually depends on, and it works with an MSYS bash pid,
# which taskkill cannot use.

port_pids() {
  # Windows PIDs listening on $API_PORT, or POSIX pids, one per line.
  if command -v netstat >/dev/null 2>&1 && [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
    netstat -ano 2>/dev/null \
      | tr -d '\r' \
      | awk -v p=":$API_PORT" '$1 ~ /^TCP$/ && $2 ~ p"$" && $4 == "LISTENING" { print $5 }' \
      | sort -u
  elif command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:$API_PORT" -sTCP:LISTEN 2>/dev/null | sort -u
  fi
}

# PowerShell rather than taskkill: this script runs with
# MSYS_NO_PATHCONV=1, under which `taskkill //F` reaches the exe as the
# literal "//F" and is rejected ("Invalid argument/option - '//F'"), so
# the kill silently did nothing and the port stayed held.
ps_run() {
  command -v powershell >/dev/null 2>&1 || return 1
  powershell -NoProfile -NonInteractive -Command "$1" >/dev/null 2>&1 || true
}

stop_server() {
  for pid in $(port_pids); do
    ps_run "Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue" \
      || kill -9 "$pid" 2>/dev/null || true
  done
  # The uv / meta-harness wrappers do not hold the port but do pin the
  # virtualenv, and `kill $SERVER_PID` reaches only the subshell above them.
  ps_run "Get-CimInstance Win32_Process |
          Where-Object { \$_.CommandLine -like '*serve --port $API_PORT*' } |
          ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  # The port being free is the property the next start depends on.
  for _ in $(seq 1 30); do
    [[ -z "$(port_pids)" ]] && return 0
    sleep 1
  done
  return 1
}

start_server() {
  # A port already in use means an earlier run leaked its backend, and
  # every check below would silently test that one instead. Refuse.
  if [[ -n "$(port_pids)" ]]; then
    printf '  %bport %s is already in use; refusing to test someone else'"'"'s backend%b\n' \
      "$RED" "$API_PORT" "$NC"
    printf '  %bstop it, or re-run with API_PORT=<free port>%b\n' "$DIM" "$NC"
    exit 1
  fi
  (cd backend && uv run meta-harness serve --port "$API_PORT" >"$LOG_DIR/$1" 2>&1) &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    api "$API_URL/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}


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
  for cmd in version inner benchmark loop fork serve experiment checkpoints replay \
             init resume memory verify-replay resume-experiment report; do
    echo "$out" | grep -q "$cmd" || { echo "missing subcommand: $cmd"; exit 1; }
  done
'
check "report subcommands present" bash -c '
  out=$(uv run meta-harness report --help 2>&1)
  for cmd in resume-evidence version-graph wandb-check cost-estimate; do
    echo "$out" | grep -q "$cmd" || { echo "missing report subcommand: $cmd"; exit 1; }
  done
'
check "resume-experiment --dry-run spends nothing" bash -c '
  out=$(uv run meta-harness resume-experiment --dry-run 2>&1)
  echo "$out"
  echo "$out" | grep -q "nothing was executed and nothing was spent"
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
start_server server.log || true

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
  # Must actually be down before the replacement starts, or this check
  # queries the process it believed it had killed.
  if ! stop_server; then
    printf '  %bcould not stop the backend; restart recovery is unverifiable%b\n' \
      "$RED" "$NC"
    exit 1
  fi
  start_server server2.log || true
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

# ── 7b. Exact replay, tracking, evidence ─────────────────────────────
# All of this is provable offline: a scripted harness supplies the
# model's turns while the real graph, the real tools and the real pytest
# verify run underneath.
section "7b. Exact replay, tracking, evidence"
check "recording + exact replay suite" bash -c '
  cd backend && uv run pytest -q --no-header \
    tests/test_recording.py tests/test_exact_replay.py tests/test_inner_messages.py'
check "W&B adapter suite (no network, no account)" bash -c '
  cd backend && uv run pytest -q --no-header tests/test_tracking.py'
check "W&B offline probe" bash -c '
  out=$(WANDB_MODE=offline uv run meta-harness report wandb-check --output "$LOG_DIR/wandb.json" 2>&1)
  echo "$out"
  uv run python -c "
import json, sys
probe = json.load(open(sys.argv[1]))
sys.exit(0 if probe[\"ok\"] else 1)
" "$LOG_DIR/wandb.json"
'
check "evidence agrees with the artifacts" bash -c '
  uv run meta-harness report resume-evidence --check'
check "no claim is marked FAIL" bash -c '
  uv run meta-harness report resume-evidence --json > "$LOG_DIR/evidence.json"
  uv run python -c "
import json, sys
report = json.load(open(sys.argv[1]))
failed = [c for c in report[\"checks\"] if c[\"status\"] == \"FAIL\"]
for c in failed:
    print(c[\"claim\"], \"->\", c[\"detail\"])
counts = report[\"counts\"]
print(counts[\"PASS\"], \"PASS,\", counts[\"FAIL\"], \"FAIL,\", counts[\"UNSUPPORTED\"], \"UNSUPPORTED\")
sys.exit(1 if failed else 0)
" "$LOG_DIR/evidence.json"
'

# ── 8. Repository hygiene ────────────────────────────────────────────
section "8. Repository hygiene"
check "no tracked .env"         bash -c '! git ls-files | grep -qE "(^|/)\.env$"'
check "no tracked run artifacts" bash -c '! git ls-files | grep -qE "^(runs|benchmark-results)/"'
check "no generated candidates tracked" bash -c '
  git ls-files agents/ | grep -vE "agents/(__init__|baseline)\.py" | grep -q . && exit 1 || exit 0'
check "no API key patterns tracked" bash -c '
  ! git grep -nE "sk-ant-[A-Za-z0-9_-]{20,}" -- . ":(exclude)scripts/demo_acceptance.sh" >/dev/null 2>&1'
check "no W&B run directories tracked" bash -c '
  ! git ls-files | grep -qE "(^|/)wandb/"'
check "no execution tapes tracked" bash -c '
  ! git ls-files | grep -qE "(^|/)tape\.jsonl$"'

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
