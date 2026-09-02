#!/usr/bin/env bash
# `make verify` — everything this repository can prove without paying for
# a single model call.
#
# The point of one command is that a reviewer does not have to trust a
# document. Each stage below either passes or names what failed; nothing
# is skipped silently, and a stage that cannot run says so and fails
# rather than reporting green.
#
# What it does NOT prove, and cannot: that a real provider call works, or
# any measured pass-rate number. Those need credentials and money — see
# `scripts/live_smoke.sh` (LEVEL 2) and `meta-harness canonical-experiment`.
#
# Usage:
#   bash scripts/verify.sh              # everything below
#   SKIP_DOCKER=1 bash scripts/verify.sh
#   SKIP_FRONTEND=1 bash scripts/verify.sh
#   ONLY=tests bash scripts/verify.sh   # one stage by name
#
# Requires: uv, a reachable Postgres (POSTGRES_DSN or the compose default),
# and — unless SKIP_DOCKER=1 — a running Docker daemon.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; DIM='\033[2m'; NC='\033[0m'

pass=0
fail=0
declare -a FAILURES=()

stage() {
  local name="$1"; shift
  if [[ -n "${ONLY:-}" && "$ONLY" != "$name" ]]; then
    return 0
  fi
  printf '\n%b── %s %b\n' "$DIM" "$name" "$NC"
  if "$@"; then
    printf '  %bPASS%b  %s\n' "$GREEN" "$NC" "$name"
    pass=$((pass + 1))
  else
    printf '  %bFAIL%b  %s\n' "$RED" "$NC" "$name"
    fail=$((fail + 1))
    FAILURES+=("$name")
  fi
}

skip_note() {
  printf '  %bskipped%b  %s (%s)\n' "$YELLOW" "$NC" "$1" "$2"
}

# ── 1. the whole backend suite ────────────────────────────────────────

verify_tests() {
  ( cd backend && uv run pytest tests/ -q )
}

# ── 2. Postgres is real, and the suites that need it actually ran ─────
#
# Every Postgres-backed suite skips itself when the DSN is unreachable.
# That is the right behaviour for a laptop with no database, and exactly
# the wrong thing to let pass here: a verification run that silently
# tested none of the persistence layer is worse than one that failed.

verify_postgres() {
  ( cd backend && uv run python - <<'PY'
import asyncio
import sys

from app.meta_harness.persistence import healthcheck

if not asyncio.run(healthcheck()):
    print("Postgres unreachable. Start it with:", file=sys.stderr)
    print("  docker compose -f infra/docker-compose.yml up -d postgres", file=sys.stderr)
    sys.exit(1)
print("Postgres reachable")
PY
  ) || return 1

  local log
  log="$(mktemp)"
  ( cd backend && uv run pytest -q --no-header \
      tests/test_persistence.py \
      tests/test_branch_isolation.py \
      tests/test_branch_persistence.py \
      tests/test_inner_checkpoints.py \
      tests/test_replay.py \
      tests/test_versioning.py \
      tests/test_exact_replay.py ) 2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  if grep -qE '[0-9]+ skipped' "$log"; then
    echo "a Postgres-backed test skipped itself; persistence was not verified"
    rm -f "$log"
    return 1
  fi
  rm -f "$log"
  return "$status"
}

# ── 3. exact recorded-execution replay ────────────────────────────────

verify_replay() {
  ( cd backend && uv run pytest -q --no-header \
      tests/test_recording.py \
      tests/test_exact_replay.py \
      tests/test_inner_messages.py )
}

# ── 4. branch and version-graph integrity ─────────────────────────────

verify_branches() {
  ( cd backend && uv run pytest -q --no-header \
      tests/test_branches.py \
      tests/test_branch_isolation.py \
      tests/test_branch_persistence.py \
      tests/test_versioning.py \
      tests/test_runs_artifacts.py )
}

# ── 5. benchmark integrity + held-out isolation ───────────────────────

verify_benchmark() {
  ( cd backend && uv run pytest -q --no-header \
      tests/test_experiment.py \
      tests/test_holdout_isolation.py \
      tests/test_metrics.py \
      tests/test_frontier.py \
      tests/test_pipeline.py ) || return 1

  # Every published summary must fall out of its own committed rows.
  uv run python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "backend")
from app.meta_harness import experiment as exp  # noqa: E402

published = sorted(p for p in Path("benchmarks/results").glob("*") if p.is_dir())
if not published:
    print("no published results yet; nothing to re-derive")
    raise SystemExit(0)

failed = False
for directory in published:
    summary = json.loads((directory / "summary.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    recomputed = exp.summarize(
        baseline_rows=exp.read_rows(directory / "baseline-results.jsonl"),
        candidate_rows=exp.read_rows(directory / "candidate-results.jsonl"),
        task_ids=[t["task_id"] for t in environment["tasks"]],
        baseline_label=summary["baseline_label"],
        candidate_label=summary["candidate_label"],
    )
    for key in (
        "baseline_passes", "baseline_trials",
        "candidate_passes", "candidate_trials",
        "baseline_accuracy", "candidate_accuracy",
        "absolute_percentage_point_delta",
    ):
        if recomputed[key] != summary[key]:
            print(f"{directory}: {key} {summary[key]!r} != recomputed {recomputed[key]!r}")
            failed = True

    validation = directory / "validation.json"
    if not validation.exists():
        print(f"{directory}: no validation.json; the methodology is unrecorded")
        failed = True
    elif not json.loads(validation.read_text()).get("identical_protocol"):
        print(f"{directory}: the two arms did not run an identical protocol")
        failed = True
    print(f"{directory.name}: verified")

raise SystemExit(1 if failed else 0)
PY
}

# ── 6. W&B, offline and absent ────────────────────────────────────────

verify_wandb() {
  ( cd backend && uv run pytest -q --no-header tests/test_tracking.py ) || return 1
  local out
  out="$(mktemp)"
  WANDB_MODE=offline uv run meta-harness report wandb-check --output "$out" || {
    rm -f "$out"; return 1;
  }
  uv run python - "$out" <<'PY'
import json
import sys

probe = json.load(open(sys.argv[1]))
if not probe["ok"]:
    print("W&B probe not ok:", probe)
    raise SystemExit(1)
print("W&B:", probe["detail"])
PY
  local status=$?
  rm -f "$out"
  return "$status"
}

# ── 7. the evidence document agrees with its artifacts ────────────────

verify_evidence() {
  uv run meta-harness report capability-evidence --check || return 1
  local out
  out="$(mktemp)"
  uv run meta-harness report capability-evidence --json > "$out" || { rm -f "$out"; return 1; }
  uv run python - "$out" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
failed = [c for c in report["checks"] if c["status"] == "FAIL"]
for check in failed:
    print("FAIL:", check["claim"], "->", check["detail"])
counts = report["counts"]
print(f'{counts["PASS"]} PASS, {counts["FAIL"]} FAIL, {counts["UNSUPPORTED"]} UNSUPPORTED')
raise SystemExit(1 if failed else 0)
PY
  local status=$?
  rm -f "$out"
  return "$status"
}

# ── 8. the container is a working backend ─────────────────────────────

verify_docker() {
  if [[ -n "${SKIP_DOCKER:-}" ]]; then
    skip_note "docker" "SKIP_DOCKER is set"
    return 0
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "no reachable Docker daemon. Start Docker, or run with SKIP_DOCKER=1"
    return 1
  fi
  bash scripts/docker_smoke.sh
}

# ── 9. repository hygiene ─────────────────────────────────────────────

verify_hygiene() {
  local bad=0
  _no() {
    local label="$1" pattern="$2"
    if git ls-files | grep -qE "$pattern"; then
      echo "  $label"
      bad=1
    fi
  }
  _no "a .env file is tracked" '(^|/)\.env$'
  _no "generated run artifacts are tracked" '^(runs|benchmark-results)/'
  _no "a W&B run directory is tracked" '(^|/)wandb/'
  _no "an execution tape is tracked" '(^|/)tape\.jsonl$'
  _no "a tool cache or OS artifact is tracked" \
     '(^|/)(\.DS_Store|Thumbs\.db|__pycache__|\.pytest_cache|\.pytest-tmp|node_modules|\.next)(/|$)'

  if git ls-files agents/ | grep -vE 'agents/(__init__|baseline)\.py' | grep -q .; then
    echo "  proposer-generated candidates are tracked"
    bad=1
  fi
  if git grep -nE 'sk-ant-[A-Za-z0-9_-]{20,}' -- . ':(exclude)scripts/*' >/dev/null 2>&1; then
    echo "  an Anthropic API key pattern is committed"
    bad=1
  fi
  for path in infra/Dockerfile infra/healthcheck.py .dockerignore \
              scripts/docker_smoke.sh docs/CAPABILITY_EVIDENCE.md; do
    if ! git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
      echo "  $path is not tracked"
      bad=1
    fi
  done
  [[ $bad -eq 0 ]] && echo "clean"
  return $bad
}

# ── 10. the dashboard still builds ────────────────────────────────────

verify_frontend() {
  if [[ -n "${SKIP_FRONTEND:-}" ]]; then
    skip_note "frontend" "SKIP_FRONTEND is set"
    return 0
  fi
  if ! command -v npm >/dev/null 2>&1; then
    skip_note "frontend" "npm is not installed"
    return 0
  fi
  ( cd frontend/dashboard \
    && { [[ -d node_modules ]] || npm ci; } \
    && npm run lint \
    && npm run build )
}

printf '%bMeta-Harness verification — no credentials, no paid model calls%b\n' "$DIM" "$NC"

stage tests      verify_tests
stage postgres   verify_postgres
stage replay     verify_replay
stage branches   verify_branches
stage benchmark  verify_benchmark
stage wandb      verify_wandb
stage evidence   verify_evidence
stage docker     verify_docker
stage hygiene    verify_hygiene
stage frontend   verify_frontend

printf '\n%b── Summary %b\n' "$DIM" "$NC"
printf '  %bpassed: %d%b   %bfailed: %d%b\n' "$GREEN" "$pass" "$NC" "$RED" "$fail" "$NC"

if (( fail > 0 )); then
  printf '\n  %bVERIFICATION FAILED%b\n' "$RED" "$NC"
  for f in "${FAILURES[@]}"; do printf '    - %s\n' "$f"; done
  exit 1
fi

printf '\n  %bVERIFIED%b — everything provable without credentials passes.\n' "$GREEN" "$NC"
printf '  %bStill unproven here (needs credentials and money): a live provider\n'
printf '  call, and any measured pass-rate number. See scripts/live_smoke.sh\n'
printf '  and `meta-harness canonical-experiment --dry-run`.%b\n' "$NC"
