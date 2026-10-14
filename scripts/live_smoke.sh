#!/usr/bin/env bash
# LEVEL 2 acceptance — live model.
#
# Verifies the parts LEVEL 1 cannot: a real Anthropic API call through
# the inner loop, the real claude-CLI proposer, importing and
# benchmarking the candidate it wrote, and measured token/cost metrics
# landing in the run artifacts.
#
# Deliberately small. This is a smoke test, not the 200-trial
# experiment — that is `meta-harness experiment` and it costs real money.
#
# Requires:
#   ANTHROPIC_API_KEY   for the inner loop (in .env or the environment)
#   claude CLI          for the proposer (subscription auth is fine)
#   Postgres            for the checkpoint-history check
#
# Exit status:
#   0  DEMO READY (LEVEL 2) — every check ran and passed; or
#      SKIPPED — no ANTHROPIC_API_KEY, so nothing was attempted. That is
#      exit 0 by design, so a credential-free CI job can call this script,
#      and it says plainly that nothing was verified.
#   1  a check failed
#   2  PARTIAL — the checks that ran passed, but some could not run (no
#      claude CLI). Not a pass.
#
# Usage:
#   bash scripts/live_smoke.sh
#   MH_SMOKE_TASK=task-002-add-function bash scripts/live_smoke.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; DIM='\033[2m'; NC='\033[0m'

TASK="${MH_SMOKE_TASK:-task-001-fix-typo}"
RUN_NAME="live-smoke-$(date +%s)"
LOG_DIR="$(mktemp -d)"
# Check bodies run in `bash -c '...'` and read these from the environment.
export TASK RUN_NAME LOG_DIR

pass=0; fail=0; skipped=0
FAILURES=()

section() { printf '\n%b── %s %b\n' "$DIM" "$1" "$NC"; }

check() {
  local label="$1"; shift
  local log
  log="$LOG_DIR/$(echo "$label" | tr -c 'a-zA-Z0-9' '_').log"
  printf '  %-52s' "$label"
  if "$@" >"$log" 2>&1; then
    printf '%bOK%b\n' "$GREEN" "$NC"; pass=$((pass + 1))
  else
    printf '%bFAIL%b\n' "$RED" "$NC"
    sed -n '1,20p' "$log" | sed 's/^/        /'
    printf '        %blog: %s%b\n' "$DIM" "$log" "$NC"
    fail=$((fail + 1)); FAILURES+=("$label")
  fi
}

skip_all() {
  printf '\n  %bSKIPPED%b — %s\n' "$YELLOW" "$NC" "$1"
  printf '  %bNothing was verified. This is not a pass.%b\n' "$DIM" "$NC"
  printf '  %bExit status 0 by design: a missing credential is not a failure of the code.%b\n' "$DIM" "$NC"
  printf '\n  To run it:\n'
  printf '    1. put ANTHROPIC_API_KEY in .env (or export it)\n'
  printf '    2. install the Claude Code CLI so `claude` is on PATH\n'
  printf '    3. bash scripts/live_smoke.sh\n'
  exit 0
}

# ── Credential gate ──────────────────────────────────────────────────
section "0. Credentials"

API_KEY="${ANTHROPIC_API_KEY:-}"
if [[ -z "$API_KEY" && -f .env ]]; then
  API_KEY="$(sed -n 's/^ANTHROPIC_API_KEY=//p' .env | head -1 | tr -d '\r\n')"
fi

if [[ -z "$API_KEY" ]]; then
  skip_all "ANTHROPIC_API_KEY is not set (checked the environment and .env)"
fi
printf '  %-52s%bOK%b\n' "ANTHROPIC_API_KEY present" "$GREEN" "$NC"

HAVE_CLAUDE_CLI=0
if command -v claude >/dev/null 2>&1; then
  HAVE_CLAUDE_CLI=1
  printf '  %-52s%bOK%b\n' "claude CLI on PATH" "$GREEN" "$NC"
else
  printf '  %-52s%bSKIPPED%b (not on PATH)\n' "claude CLI on PATH" "$YELLOW" "$NC"
fi

export ANTHROPIC_API_KEY="$API_KEY"
MODEL="${META_HARNESS_INNER_MODEL:-claude-haiku-4-5-20251001}"
printf '  %binner model: %s%b\n' "$DIM" "$MODEL" "$NC"

# ── 1. One real inner-loop trial ─────────────────────────────────────
section "1. Inner loop against a live model (1 trial)"
check "inner loop solves $TASK" bash -euo pipefail -c '
  uv run meta-harness inner --task "$TASK" --candidate baseline \
      --run-name "$RUN_NAME" > "$LOG_DIR/inner.json"
  cat "$LOG_DIR/inner.json"
  uv run python - "$LOG_DIR/inner.json" <<"PY"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["passed"] is True, "trial did not pass"
PY
'
check "measured token metrics were recorded" bash -euo pipefail -c '
  uv run python - "$LOG_DIR/inner.json" <<"PY"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["llm_calls"] > 0, f"no LLM calls recorded: {d}"
assert d["total_tokens"] > 0, f"no tokens recorded: {d}"
print("llm_calls", d["llm_calls"], "tokens", d["total_tokens"], "cost", d["cost_usd"])
PY
'
check "per-trial metrics.json written" bash -euo pipefail -c '
  uv run python - "runs/$RUN_NAME/threads/$RUN_NAME/candidates/baseline/traces" <<"PY"
import glob, json, sys
paths = glob.glob(sys.argv[1] + "/*/metrics.json")
assert paths, "no metrics.json under the trace dir"
row = json.load(open(paths[0]))
assert row["metrics_source"] == "measured", row["metrics_source"]
assert row["total_tokens"] > 0
PY
'

# ── 2. Measured benchmark ────────────────────────────────────────────
section "2. Measured benchmark (1 task x 2 trials)"
check "benchmark writes measured aggregates" bash -euo pipefail -c '
  uv run meta-harness benchmark --candidate baseline --trials 2 --workers 2 \
      --run-name "$RUN_NAME-bench" >/dev/null
  uv run python - "runs/$RUN_NAME-bench/threads/$RUN_NAME-bench/candidates/baseline/eval-result.json" <<"PY"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["metrics_source"] == "measured", d["metrics_source"]
assert d["tokens"]["total_tokens"] > 0, "tokens not measured"
assert d["total_llm_calls"] > 0, "llm calls not measured"
assert d["total_trials"] == len(d["trials"]), "summary must match raw rows"
print("accuracy", d["accuracy"], "tokens", d["tokens"]["total_tokens"], "cost", d["total_cost_usd"])
PY
'

# ── 3. Real proposer ─────────────────────────────────────────────────
section "3. Real claude-CLI proposer (budget 1)"
if (( HAVE_CLAUDE_CLI == 0 )); then
  printf '  %-52s%bSKIPPED%b (claude CLI unavailable)\n' "proposer writes a valid candidate" "$YELLOW" "$NC"
  skipped=$((skipped + 1))
else
  check "proposer writes an importable candidate" bash -euo pipefail -c '
    uv run meta-harness loop --proposer claude --budget 1 --fresh --mock-bench \
        --run-name "$RUN_NAME-prop" --no-persistent > "$LOG_DIR/loop.json"
    cat "$LOG_DIR/loop.json"
    uv run python - "$LOG_DIR/loop.json" <<"PY"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["n_candidates"] >= 2, f"expected baseline + 1 candidate: {d}"
PY
  '
  check "candidate source was snapshotted per branch" bash -euo pipefail -c '
    uv run python - "runs/$RUN_NAME-prop/threads/$RUN_NAME-prop/agents" <<"PY"
import glob, sys
paths = glob.glob(sys.argv[1] + "/*.py")
assert paths, "no per-branch candidate source snapshot"
print(paths)
PY
  '
  check "proposer session log captured" bash -euo pipefail -c '
    uv run python - "runs/$RUN_NAME-prop/threads/$RUN_NAME-prop/proposer-sessions/iter-1/session.json" <<"PY"
import json, os, sys
assert os.path.exists(sys.argv[1]), "no proposer session log"
s = json.load(open(sys.argv[1]))
assert s["mode"] == "claude", s["mode"]
assert s["exit_code"] == 0, s
print("cost_usd", s.get("cost_usd"), "tokens", s.get("token_usage"))
PY
  '
fi

# ── 4. Checkpoint history ────────────────────────────────────────────
# Not just "Postgres answers": run a persistent (mock, free) outer loop
# and read its checkpoint history back out of Postgres.
section "4. Checkpoint history"
check "checkpoints listed from Postgres" bash -euo pipefail -c '
  uv run meta-harness loop --proposer mock --mock-bench --budget 1 --trials 1 \
      --workers 1 --fresh --run-name "$RUN_NAME-ckpt" >/dev/null
  uv run meta-harness checkpoints "$RUN_NAME-ckpt" --limit 50 > "$LOG_DIR/checkpoints.json"
  uv run python - "$LOG_DIR/checkpoints.json" <<"PY"
import json, sys
rows = json.load(open(sys.argv[1]))
assert isinstance(rows, list) and rows, "no checkpoints listed"
assert all(r.get("checkpoint_id") for r in rows), "a checkpoint has no id"
print(len(rows), "checkpoints; newest after", rows[0].get("node"))
PY
'

# ── Summary ──────────────────────────────────────────────────────────
printf '\n%b── Summary %b\n' "$DIM" "$NC"
printf '  %bpass: %d%b   %bfail: %d%b   %bskipped: %d%b\n' \
  "$GREEN" "$pass" "$NC" "$RED" "$fail" "$NC" "$YELLOW" "$skipped" "$NC"
printf '  %brun artifacts: runs/%s*%b\n' "$DIM" "$RUN_NAME" "$NC"

if (( fail > 0 )); then
  printf '\n  %bLEVEL 2 ACCEPTANCE FAILED%b\n' "$RED" "$NC"
  for f in "${FAILURES[@]}"; do printf '    - %s\n' "$f"; done
  exit 1
fi

if (( skipped > 0 )); then
  printf '\n  %bLEVEL 2 PARTIAL%b — %d check(s) could not run. This is not a pass (exit 2).\n' \
    "$YELLOW" "$NC" "$skipped"
  exit 2
fi

printf '\n  %bDEMO READY (LEVEL 2: live model)%b\n' "$GREEN" "$NC"
printf '  %bThis still proves no benchmark number. Run%b\n' "$DIM" "$NC"
printf '  %b"uv run meta-harness experiment --candidate <name>" for that.%b\n' "$DIM" "$NC"
