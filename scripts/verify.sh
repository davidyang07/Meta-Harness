#!/usr/bin/env bash
# `make verify` — everything this repository can prove without paying for
# a single model call.
#
# The point of one command is that a reviewer does not have to trust a
# document. Each stage below either passes or names what failed; nothing
# is skipped silently, and a stage that cannot run says so and fails
# rather than reporting green. The only skips are the ones asked for by
# name (SKIP_DOCKER=1, SKIP_FRONTEND=1), and the summary lists them.
#
# Usage:
#   bash scripts/verify.sh              # every stage
#   SKIP_DOCKER=1 bash scripts/verify.sh
#   SKIP_FRONTEND=1 bash scripts/verify.sh
#   ONLY=tests bash scripts/verify.sh   # one stage by name (see STAGES)
#
# Requires: uv, a reachable Postgres (POSTGRES_DSN or the compose default),
# npm (unless SKIP_FRONTEND=1) and — unless SKIP_DOCKER=1 — a running
# Docker daemon.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; DIM='\033[2m'; NC='\033[0m'

STAGES="tests postgres replay branches benchmark wandb evidence docker hygiene frontend"

# A typo in ONLY used to run nothing at all and still print VERIFIED.
if [[ -n "${ONLY:-}" ]]; then
  known=0
  for name in $STAGES; do
    if [[ "$name" == "$ONLY" ]]; then known=1; fi
  done
  if (( known == 0 )); then
    printf '%bunknown stage: %s%b\n' "$RED" "$ONLY" "$NC" >&2
    printf 'valid stages: %s\n' "$STAGES" >&2
    exit 2
  fi
fi

pass=0
fail=0
skipped=0
FAILURES=""
SKIPPED=""

# stage <name> <function> [<skip variable>]
#
# The function runs in a subshell with errexit live. Called as an `if`
# condition instead, bash silently disables `set -e` for the function's
# whole body, and a failing command in the middle of a stage would not
# fail it.
stage() {
  local name="$1" fn="$2" skip_var="${3:-}"
  if [[ -n "${ONLY:-}" && "$ONLY" != "$name" ]]; then
    return 0
  fi
  printf '\n%b── %s %b\n' "$DIM" "$name" "$NC"
  if [[ -n "$skip_var" && "${!skip_var:-0}" == "1" ]]; then
    printf '  %bSKIPPED%b  %s (%s=1)\n' "$YELLOW" "$NC" "$name" "$skip_var"
    skipped=$((skipped + 1))
    SKIPPED="$SKIPPED $name"
    return 0
  fi
  local rc=0
  set +e
  ( set -e; "$fn" )
  rc=$?
  set -e
  if (( rc == 0 )); then
    printf '  %bPASS%b  %s\n' "$GREEN" "$NC" "$name"
    pass=$((pass + 1))
  else
    printf '  %bFAIL%b  %s\n' "$RED" "$NC" "$name"
    fail=$((fail + 1))
    FAILURES="$FAILURES $name"
  fi
}

# pytest exits 0 having run nothing when every test skips itself, and 4
# (no "passed", no "skipped") when a listed file is missing. A stage that
# exists to exercise something must see a positive pass count.
assert_passed() {
  local log="$1"
  if ! grep -qE '(^|[^0-9])[1-9][0-9]* passed' "$log"; then
    echo "no test passed; nothing was verified"
    return 1
  fi
}

# ── 1. the whole backend suite ────────────────────────────────────────

verify_tests() {
  local log
  log="$(mktemp)"
  local rc=0
  ( cd backend && uv run pytest tests/ -q ) 2>&1 | tee "$log" || rc=$?
  if (( rc == 0 )); then assert_passed "$log" || rc=1; fi
  rm -f "$log"
  return "$rc"
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
  )

  # Every Postgres-backed test skips with "Postgres not reachable", so
  # that phrase finds them all — including files added later — where a
  # hand-kept list would silently miss them.
  local files=() file
  while IFS= read -r file; do
    files+=("$file")
  done < <(cd backend && grep -l 'Postgres not reachable' tests/test_*.py)
  if (( ${#files[@]} == 0 )); then
    echo "found no Postgres-backed test files; nothing to verify"
    return 1
  fi

  local log
  log="$(mktemp)"
  local rc=0
  ( cd backend && uv run pytest -q --no-header -rs "${files[@]}" ) 2>&1 | tee "$log" || rc=$?
  if (( rc == 0 )); then assert_passed "$log" || rc=1; fi
  if grep -q 'Postgres not reachable' "$log"; then
    echo "a Postgres-backed test skipped itself; persistence was not verified"
    rc=1
  fi
  rm -f "$log"
  return "$rc"
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
      tests/test_pipeline.py )

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
  ( cd backend && uv run pytest -q --no-header tests/test_tracking.py )
  local out
  out="$(mktemp)"
  local rc=0
  WANDB_MODE=offline uv run meta-harness report wandb-check --output "$out" || rc=$?
  if (( rc == 0 )); then
    uv run python - "$out" <<'PY' || rc=$?
import json
import sys

probe = json.load(open(sys.argv[1]))
if not probe["ok"]:
    print("W&B probe not ok:", probe)
    raise SystemExit(1)
print("W&B:", probe["detail"])
PY
  fi
  rm -f "$out"
  return "$rc"
}

# ── 7. capability evidence is derivable, and portable ─────────────────
#
# The evidence document is derived, never committed: regenerate it into a
# temporary file so this stage changes nothing in the checkout, and fail
# if it cannot be derived. Its inputs, docs/evidence/*.json, are
# committed, so they must not carry the absolute paths of the machine
# that wrote them.

verify_evidence() {
  local out
  out="$(mktemp -d)"
  local rc=0
  uv run meta-harness report capability-evidence \
    --output "$out/CAPABILITY_EVIDENCE.md" || rc=$?
  if (( rc != 0 )); then
    echo "the capability-evidence document could not be derived"
  fi
  rm -rf "$out"

  uv run python - <<'PY' || rc=1
import json
import re
from pathlib import Path

# A drive letter (C:\ or C:/), a UNC share, or a home/temp directory.
ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"
    r"|^\\\\"
    r"|(?:^|[\s\"'=(:])/(?:home|Users|tmp|root|private|var/folders)/"
)


def walk(value, where):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, f"{where}[{index}]")
    elif isinstance(value, str) and ABSOLUTE.search(value):
        yield where, value


artifacts = sorted(Path("docs/evidence").glob("*.json"))
bad = []
for path in artifacts:
    bad.extend((path, where, value) for where, value in walk(json.loads(path.read_text()), "$"))
for path, where, value in bad[:20]:
    print(f"{path}: {where} is an absolute path: {value}")
if len(bad) > 20:
    print(f"... and {len(bad) - 20} more")
if bad:
    print("docs/evidence/*.json must use repo-relative POSIX paths; regenerate them")
    raise SystemExit(1)
print(f"{len(artifacts)} evidence artifact(s) carry only repo-relative paths")
PY
  return "$rc"
}

# ── 8. the container is a working backend ─────────────────────────────

verify_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "no reachable Docker daemon. Start Docker, or run with SKIP_DOCKER=1"
    return 1
  fi
  bash scripts/docker_smoke.sh
}

# ── 9. repository hygiene ─────────────────────────────────────────────

verify_hygiene() {
  local bad=0 tracked
  # Read once into a variable: `git ls-files | grep -q` under pipefail
  # fails whenever grep exits early and git takes a SIGPIPE, so a match
  # could read as no match.
  tracked="$(git ls-files)"
  _no() {
    local label="$1" pattern="$2"
    if grep -qE "$pattern" <<<"$tracked"; then
      echo "  $label"
      bad=1
    fi
  }
  _no "a .env file is tracked" '(^|/)\.env$'
  _no "generated run artifacts are tracked" '^(runs|benchmark-results)/'
  _no "a W&B run directory is tracked" '(^|/)wandb/'
  _no "an execution tape is tracked" '(^|/)tape\.jsonl$'
  _no "the derived capability-evidence document is tracked" '^docs/CAPABILITY_EVIDENCE\.md$'
  _no "a tool cache or OS artifact is tracked" \
     '(^|/)(\.DS_Store|Thumbs\.db|__pycache__|\.pytest_cache|\.pytest-tmp|node_modules|\.next)(/|$)'

  local extra
  extra="$(grep -E '^agents/' <<<"$tracked" | grep -vE '^agents/(__init__|baseline)\.py$' || true)"
  if [[ -n "$extra" ]]; then
    echo "  proposer-generated candidates are tracked: $extra"
    bad=1
  fi
  if git grep -nE 'sk-ant-[A-Za-z0-9_-]{20,}' -- . ':(exclude)scripts/*' >/dev/null 2>&1; then
    echo "  an Anthropic API key pattern is committed"
    bad=1
  fi
  for path in infra/Dockerfile infra/healthcheck.py .dockerignore \
              scripts/docker_smoke.sh; do
    if ! git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
      echo "  $path is not tracked"
      bad=1
    fi
  done
  if (( bad == 0 )); then echo "clean"; fi
  return "$bad"
}

# ── 10. the dashboard type-checks, lints and builds ───────────────────

verify_frontend() {
  # Missing npm is a failure, not a pass: skip this stage only by asking
  # for it with SKIP_FRONTEND=1.
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is not installed. Install Node.js 20+, or run with SKIP_FRONTEND=1"
    return 1
  fi
  cd frontend/dashboard
  if [[ ! -d node_modules ]]; then npm ci; fi
  npm run lint
  npx tsc --noEmit
  npm run build
}

printf '%bMeta-Harness verification — no credentials, no paid model calls%b\n' "$DIM" "$NC"

stage tests      verify_tests
stage postgres   verify_postgres
stage replay     verify_replay
stage branches   verify_branches
stage benchmark  verify_benchmark
stage wandb      verify_wandb
stage evidence   verify_evidence
stage docker     verify_docker   SKIP_DOCKER
stage hygiene    verify_hygiene
stage frontend   verify_frontend SKIP_FRONTEND

printf '\n%b── Summary %b\n' "$DIM" "$NC"
printf '  %bpassed: %d%b   %bfailed: %d%b   %bskipped: %d%b\n' \
  "$GREEN" "$pass" "$NC" "$RED" "$fail" "$NC" "$YELLOW" "$skipped" "$NC"

if (( fail > 0 )); then
  printf '\n  %bVERIFICATION FAILED%b\n' "$RED" "$NC"
  for name in $FAILURES; do printf '    - %s\n' "$name"; done
  exit 1
fi

if [[ -n "${ONLY:-}" ]]; then
  if (( skipped > 0 )); then
    printf '\n  %bNOT VERIFIED%b — the only requested stage, %s, was skipped.\n' \
      "$YELLOW" "$NC" "$ONLY"
    exit 1
  fi
  printf '\n  %bVERIFIED%b — stage %s only (ONLY=%s); the other stages did not run.\n' \
    "$GREEN" "$NC" "$ONLY" "$ONLY"
  exit 0
fi

if (( skipped > 0 )); then
  printf '\n  %bVERIFIED, except%b stages skipped on request:%s\n' "$YELLOW" "$NC" "$SKIPPED"
  printf '  %bRe-run without SKIP_* to verify everything.%b\n' "$DIM" "$NC"
  exit 0
fi

printf '\n  %bVERIFIED%b — everything provable without credentials passes.\n' "$GREEN" "$NC"
