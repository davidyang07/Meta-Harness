# INTERFACES.md — Meta-Harness cross-component contracts

*Every contract that crosses a process, file, or HTTP boundary, in one
document. Pulled verbatim from Appendices B and C wherever the appendices
specify a shape; **derivations** are flagged inline. The scope is contracts,
not implementation.*

Phase 1.2 (FE/BE protocol) is resolved: **SSE for events + REST for
commands.** Phase 1.3 (skill loading mechanism) is still open and is
called out where it intersects this document.

---

## 0. Amendments — read this first

The sections below were written against a run layout where every
artifact lived at `runs/{run_id}/`. That is no longer true, and two other
contracts changed. Where a later section contradicts this one, **this
section wins**.

### 0.1 Artifacts are thread-scoped

Execution state now lives under `runs/{run_id}/threads/{thread_slug}/`.

**Why.** Two branches forked from the same checkpoint reach the same
iteration number. With run-level paths, the fork's proposer overwrote
`runs/{run}/pending_eval.json` and the root branch then benchmarked a
candidate it never proposed. Every path in §2 that referenced
`runs/{run_id}/<artifact>` now reads
`runs/{run_id}/threads/{thread_slug}/<artifact>`:

| Old | New |
|---|---|
| `runs/{run}/pending_eval.json` | `runs/{run}/threads/{thread}/pending_eval.json` |
| `runs/{run}/frontier_val.json` | `runs/{run}/threads/{thread}/frontier_val.json` |
| `runs/{run}/evolution_summary.jsonl` | `runs/{run}/threads/{thread}/evolution_summary.jsonl` |
| `runs/{run}/proposer-sessions/iter-N/` | `runs/{run}/threads/{thread}/proposer-sessions/iter-N/` |
| `runs/{run}/candidates/{name}/` | `runs/{run}/threads/{thread}/candidates/{name}/` |
| — | `runs/{run}/threads/{thread}/agents/{name}.py` (new: per-branch source snapshot) |
| — | `runs/{run}/branches.json` (new: durable branch metadata) |

`{thread_slug}` is the LangGraph `thread_id` when it is a valid artifact
name, else `t-<sha256(thread_id)[:32]>` (see `runs.thread_slug`). The
run's root thread uses the run id.

Run-level views are produced by merging across threads
(`runs.aggregate_evolution_rows`, `runs.aggregate_frontiers`); every
merged row carries the `thread_id` that produced it.

### 0.2 Candidate names are run-unique; labels are human-readable

`Candidate.name` is the run-unique artifact key.
`Candidate.label` is what the proposer called it. On a forked branch the
name gains a stable `__<sha256(thread_id)[:8]>` suffix
(`runs.qualify_candidate_name`, idempotent), so two branches on the same
iteration cannot claim the same directory or source file.

New `Candidate` fields: `label`, `source_path` (branch-private snapshot
of the source that was actually benchmarked), `source_sha256`.
`axis` gains the value `"baseline"`.

### 0.3 Metrics are measured, and mock is labelled

Every result payload carries `metrics_source: "measured" | "mock"`.
Aggregation refuses to fold rows whose source disagrees.

`cost_usd` is `null` when the model has no configured price. **It is
never `0.0` to mean "unknown".** Aggregates carry `cost_complete: bool`.

On the Pareto cost axis, `avg_tokens: null` means "not measured": such a
candidate is compared on accuracy alone and can never dominate a
measured one.

### 0.4 Every number in the examples below is illustrative

The JSON examples throughout this document show *shape*, not results.
No accuracy, token or cost figure in this file is a measurement. See
[`docs/RESUME_CLAIMS.md`](RESUME_CLAIMS.md) and
[`benchmarks/resume-claim/README.md`](../benchmarks/resume-claim/README.md).

---

## 1. State schemas (LangGraph TypedDicts)

### 1.1 `MetaHarnessState` — outer state machine

*Verbatim from Appendix B §B.6.1.*

```python
from typing import TypedDict
from app.meta_harness.state import Candidate  # see §1.3 below

class MetaHarnessState(TypedDict):
    run_id: str                      # the tree identifier (= parent thread_id)
    iteration: int                   # 1-indexed; current outer-loop iteration
    budget_remaining: int            # iterations left before END
    candidates: list[Candidate]      # all candidates ever, append-only
    frontier: list[str]              # candidate names on the Pareto frontier
    best_candidate: str | None       # name of the highest-scoring accepted candidate
    proposer_prior: str              # editable via update_state at fork-time
```

### 1.2 `CodingAgentState` — inner state machine

*Verbatim from Appendix C §C.8.*

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class CodingAgentState(TypedDict):
    task: dict                                  # see §2.6 task.json schema
    workspace_path: str                         # /tmp/meta-harness-task-{uuid}/
    orient_summary: dict | None                 # see §2.7 orient.json
    plan: dict | None                           # see §2.8 plan.json
    messages: Annotated[list, add_messages]     # LangGraph reducer-merged
    turn_count: int                             # incremented per act-loop iteration
    verify_attempts: int                        # bounded by harness.MAX_VERIFY_RETRIES
    verify_result: dict | None                  # see §2.10 verify.json
    final_files: dict[str, str] | None          # see §2.11 final-files.json
    score: float | None                         # 0.0 or 1.0 (per-trial pytest pass)
```

### 1.3 `Candidate` — element of `MetaHarnessState.candidates`

*Locked. Synthesized from the union of fields used in pending_eval.json
(proposer-written), the validate/benchmark/update_frontier nodes
(graph-enriched), and the trace structure (Appendix C §C.10).*

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass
class Candidate:
    name: str                       # e.g. "few_shot_tool_results"
    import_path: str                # e.g. "agents.few_shot_tool_results:CodingAgentHarness"
    parent: str | None              # parent candidate name; None for baseline
    hypothesis: str                 # falsifiable claim from the proposer
    axis: Literal["exploration", "exploitation"]
    expected_score_delta: float | None
    iteration: int                  # outer-loop iteration that produced it
    traces_dir: Path                # runs/{run_id}/candidates/{name}/traces/ — created at propose-time, populated at benchmark-time, never None
    status: Literal["pending", "smoke_failed", "evaluated", "rejected", "accepted"]
    scores: dict | None             # eval-result.json content; None until benchmark
    delta: float | None             # score - parent.score; None until update_frontier
    cost_usd: float | None
```

---

## 2. Filesystem JSON / JSONL contracts

> **Paths in this section are superseded by §0.1** — execution artifacts
> live under `runs/{run_id}/threads/{thread_slug}/`. The JSON *shapes*
> below remain accurate except where §0 amends them.

All files live under `runs/{run_id}/`. The directory layout follows
Appendix C §C.10 (per-candidate trace structure) plus Stanford's
filesystem-first convention (Appendix B §B.2).

### 2.1 `pending_eval.json` — proposer→outer-graph handoff

*Verbatim from Appendix B §B.5.1.*

```json
{
  "iteration": 3,
  "candidates": [
    {
      "name": "few-shot-tool-results",
      "import_path": "agents.few_shot_tool_results:CodingAgentHarness",
      "parent": "more-specific-descriptions",
      "hypothesis": "Inlining 1 example tool result reduces patch-context misses",
      "axis": "exploitation",
      "expected_score_delta": 0.04
    }
  ]
}
```

The proposer writes ONE candidate per iteration (TB2 convention; we are
single-domain coding-agent). Class name is always `CodingAgentHarness`.

### 2.2 `frontier_val.json` — current Pareto frontier

*Locked. Pareto on (accuracy × tokens) per Appendix C §C.11. Each
candidate carries `dominated_by_names`: empty list for Pareto-optimal,
otherwise the names of candidates that dominate it. The dashboard's
frontier rendering filters on `dominated_by_names == []`.*

```json
{
  "iteration": 4,
  "thread_id": "run-x",
  "metrics_source": "measured",
  "candidates": [
    {"name": "baseline",   "accuracy": 0.62, "avg_tokens": 18200, "metrics_source": "measured", "dominated_by_names": []},
    {"name": "cand-a",     "accuracy": 0.80, "avg_tokens": 24800, "metrics_source": "measured", "dominated_by_names": []},
    {"name": "cand-b",     "accuracy": 0.66, "avg_tokens": 26000, "metrics_source": "measured", "dominated_by_names": ["cand-a"]},
    {"name": "unmeasured", "accuracy": 0.70, "avg_tokens": null,  "metrics_source": "measured", "dominated_by_names": ["cand-a"]}
  ],
  "_pareto_names": ["baseline", "cand-a"],
  "_best": {"name": "cand-a", "accuracy": 0.80, "avg_tokens": 24800},
  "per_task": {
    "task-001-fix-typo":     {"best_candidate": "cand-a", "pass_rate": 0.95},
    "task-002-add-function": {"best_candidate": "cand-a", "pass_rate": 0.85}
  }
}
```

*(Illustrative values — see §0.4.)* Notes:

- The measured `baseline` is a normal frontier member and stays
  non-dominated when it is cheaper than everything more accurate.
- `avg_tokens: null` means the candidate's tokens were never measured.
  It is compared on accuracy alone: `unmeasured` is dominated by the
  strictly more accurate `cand-a`, and dominates nothing.
- `metrics_source` at the top level is `"mixed"` when mock and measured
  candidates appear together, so no consumer can render a synthetic
  frontier as a measurement.

`_pareto_names` is a convenience derived from `dominated_by_names == []`;
both forms are present so the dashboard can choose by render path.

### 2.3 `evolution_summary.jsonl` — append-only candidate log

*Locked. One JSON per line, one line per evaluated candidate.
`parent_candidate_name` lets the dashboard reconstruct the trajectory
tree without scanning every status.json.*

```jsonl
{"iteration": 0, "candidate": "baseline", "label": "baseline", "thread_id": "run-x", "import_path": "agents.baseline:BaselineHarness", "source_sha256": null, "parent_candidate_name": null, "axis": "baseline", "hypothesis": "immutable starting harness (no overrides)", "scores": {"accuracy": 0.62, "per_task": {}}, "delta": 0.0, "outcome": "62.0% (+0.0%)", "tokens": 18200, "cost_usd": 0.31, "metrics_source": "measured"}
{"iteration": 1, "candidate": "retry-on-test-fail", "label": "retry-on-test-fail", "thread_id": "run-x", "import_path": "agents.retry_on_test_fail:CodingAgentHarness", "source_sha256": "9f2c...", "parent_candidate_name": "baseline", "axis": "exploration", "hypothesis": "...", "scores": {"accuracy": 0.70, "per_task": {}}, "delta": 0.08, "outcome": "70.0% (+8.0%)", "tokens": 23400, "cost_usd": 0.42, "metrics_source": "measured"}
{"iteration": 2, "candidate": "few-shot__a1b2c3d4", "label": "few-shot", "thread_id": "run-x.fork.a1b2c3d4", "import_path": "agents.few_shot:CodingAgentHarness", "source_sha256": "4e81...", "parent_candidate_name": "retry-on-test-fail", "axis": "exploitation", "hypothesis": "...", "scores": {"accuracy": 0.78, "per_task": {}}, "delta": 0.08, "outcome": "78.0% (+8.0%)", "tokens": 26100, "cost_usd": 0.47, "metrics_source": "measured"}
```

*(Illustrative values — see §0.4.)*

Fields added since the original contract:

- `thread_id` — which branch produced the row. Every branch writes to
  its own file; this field survives the run-level merge so a merged view
  stays attributable.
- `label` — the proposer's human-readable name (§0.2).
- `source_sha256` — hash of the branch-private source snapshot that was
  actually benchmarked.
- `metrics_source` — `"measured"` or `"mock"`.
- `tokens` — mean measured total tokens per trial, or `null` when
  unmeasured. `cost_usd` is `null` when unpriced (§0.3).

`timing_s` was never implemented; per-trial `wall_time_s` and
candidate-level `total_wall_time_s` live in `eval-result.json` instead.

**Iteration 0 is the measured baseline.** It is written before the first
propose, so every later `delta` is a comparison against a real
measurement rather than against zero. `parent_candidate_name` is `null`
only for that row.

### 2.4 `proposer-sessions/iter-{N}/session.json`

*DERIVED — schema-compatible with Stanford's `claude_wrapper.py`'s
`SessionResult` dataclass + `log_session()` output.*

```json
{
  "timestamp": "2026-04-25T14:32:11.421Z",
  "prompt": "Run iteration 3 of the evolution loop...",
  "model": "opus",
  "session_id": "<claude-session-id>",
  "exit_code": 0,
  "duration_seconds": 38.21,
  "cost_usd": 0.15,
  "token_usage": {"input_tokens": 187432, "output_tokens": 4128, "cache_read_input_tokens": 89000},
  "command": ["claude", "--dangerously-skip-permissions", "-p", "...", "--model", "opus", "..."],
  "cwd": "runs/run-2026-04-25-1430/",
  "skill": [{"path": "skills/meta-harness-coding-agent/SKILL.md", "name": "meta-harness-coding-agent"}],
  "files_read":   {"agents/baseline.py": {"reads": 1, "lines": 152}, "evolution_summary.jsonl": {"reads": 2, "lines": 12}},
  "files_written": {"agents/few_shot_tool_results.py": {"lines_written": 168}, "pending_eval.json": {"lines_written": 12}},
  "tool_summary": ["Read(agents/baseline.py)", "Read(traces/...)", "Bash(python -c '...')", "Write(agents/few_shot_tool_results.py)", "Write(pending_eval.json)"]
}
```

Companion files in the same directory:

- `transcript.txt` — concatenated text events from the stream-json log.
- `system_prompt.txt` — the exact `--append-system-prompt` payload (SKILL.md + domain_spec.md + proposer_prior.md).
- `events.jsonl` — raw stream-json events, one per line.
- `tools/{NNN}_{ToolName}.txt` — one file per tool call, human-readable.

### 2.5 `runs/{run}/threads/{thread}/candidates/{N}/eval-result.json`

*DERIVED — referenced as "aggregate score across all 25 trials" in
Appendix C §C.10 but not given a verbatim shape. Produced by
`benchmark.summarize()` from raw trial rows; identical shape for the
measured and mock paths, differing only in `metrics_source`.*

```json
{
  "candidate": "few-shot-tool-results",
  "thread_id": "run-x",
  "n_tasks": 5,
  "n_trials_per_task": 5,
  "accuracy": 0.78,
  "per_task": {
    "task-001-fix-typo":     {"pass_rate": 0.95, "trials": [true, true, true, true, false]},
    "task-002-add-function": {"pass_rate": 0.80, "trials": [true, true, true, false, true]}
  },
  "trials": [
    {
      "task_id": "task-001-fix-typo", "trial": 1, "passed": true, "score": 1.0,
      "llm_calls": 7, "input_tokens": 18432, "output_tokens": 2104,
      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
      "total_tokens": 20536, "cost_usd": 0.0289, "wall_time_s": 23.71,
      "metrics_source": "measured",
      "inner_thread_id": "inner::run-x::run-x::few-shot-tool-results::task-001-fix-typo::trial-1",
      "calls": [{"index": 1, "model": "claude-haiku-4-5-20251001", "input_tokens": 2104,
                 "output_tokens": 318, "total_tokens": 2422, "latency_s": 1.83,
                 "cost_usd": 0.0037, "has_usage": true}]
    }
  ],
  "total_trials": 25,
  "passed_trials": 20,
  "tokens": {"input_tokens": 124200, "output_tokens": 19800,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "total_tokens": 144000},
  "total_llm_calls": 168,
  "total_wall_time_s": 184.6,
  "mean_total_tokens_per_trial": 5760.0,
  "median_total_tokens_per_trial": 5612.0,
  "total_cost_usd": 0.42,
  "cost_complete": true,
  "metrics_source": "measured",
  "_mock_bench": false,
  "timestamp": "2026-04-25T14:35:49.218Z"
}
```

*(Illustrative values — see §0.4.)*

- `trials` holds the raw per-trial rows every aggregate is derived from,
  so a reader can recompute `accuracy`, the token totals and the
  mean/median independently. Each trial also writes its own row to
  `traces/{task-id}-trial-{T}/metrics.json`.
- `total_cost_usd` is `null` and `cost_complete` is `false` when any
  trial used a model with no configured price (§0.3).
- `cost_usd` (the pre-amendment field name) is gone; it was a
  hardcoded `0.0` on the real benchmark path.
- Holdout evaluation writes the same shape to
  `runs/{run}/threads/{thread}/holdout-result.json` with
  `task_set: "holdout"`.

### 2.6 `eval/tasks/<task-id>/task.json` — task specification

*DERIVED from Appendix C §C.11 task descriptions.*

```json
{
  "id": "task-002-add-function",
  "tier": "implement-spec",
  "instruction": "Add a `median()` function to stats.py. It must handle empty lists, single elements, and even-length lists.",
  "test_command": "pytest tests/test_stats.py -q",
  "expected_files_changed": ["stats.py"],
  "baseline_pass_rate": 0.50,
  "best_known_pass_rate": 0.85
}
```

### 2.7 `runs/{run_id}/candidates/{N}/traces/{task-id}-trial-{T}/orient.json`

*DERIVED from Appendix C §C.3.*

```json
{
  "tree": "<output of `tree --gitignore -L 3` or equivalent>",
  "project": {
    "lang": "python",
    "test_runner": "pytest",
    "entry_points": ["src/calc.py"],
    "test_files": ["tests/test_calc.py"]
  },
  "configs": {"README.md": "...", "pyproject.toml": "...", ".relay/AGENTS.md": null},
  "tests": {"tests/test_calc.py": "<file content>"}
}
```

### 2.8 `traces/{task-id}-trial-{T}/plan.json`

*Verbatim from Appendix C §C.4.*

```json
{
  "summary": "Add median() function to stats.py",
  "steps": [
    {"action": "read",      "target": "stats.py",            "why": "see existing patterns"},
    {"action": "read",      "target": "tests/test_stats.py", "why": "see test contract"},
    {"action": "implement", "target": "stats.py",            "why": "add median fn"},
    {"action": "verify",    "target": "pytest tests/test_stats.py", "why": "confirm green"}
  ],
  "expected_files_changed": ["stats.py"],
  "tests_to_run": ["tests/test_stats.py"],
  "risk_factors": ["edge case: empty list"]
}
```

### 2.9 `traces/{task-id}-trial-{T}/act-messages.jsonl` and `act-tools.jsonl`

`act-messages.jsonl` — full Anthropic message history; one JSON per line:

```jsonl
{"role": "user",      "content": "<task instruction + plan>"}
{"role": "assistant", "content": [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "tu_01", "name": "read_file", "input": {"path": "stats.py"}}]}
{"role": "tool",      "tool_use_id": "tu_01", "content": "1→def mean(...)\n..."}
```

`act-tools.jsonl` — *verbatim from Appendix C §C.10*:

```jsonl
{"turn": 7, "tool": "apply_patch", "input": {"path": "stats.py", "patch": "..."}, "output_summary": "applied 14 lines", "duration_ms": 142, "is_error": false}
```

### 2.10 `traces/{task-id}-trial-{T}/verify.json`

*Verbatim from Appendix C §C.7.*

```json
{
  "tests_pass": false,
  "tests_failed": ["test_median_with_empty_list"],
  "test_output": "<truncated to 2000 chars>",
  "lint_pass": true,
  "lint_errors": [],
  "out_of_plan_changes": []
}
```

### 2.11 `traces/{task-id}-trial-{T}/{score.json, summary.md, final-files.json}`

`score.json` — per-trial outcome:

```json
{"passed": true, "score": 1.0, "why": "all tests green; no out-of-plan changes; lint clean"}
```

`summary.md` — *Appendix C §C.10*: 5-line auto-generated summary
("Agent read calc.py, planned to add median(), implemented correctly, but
missed empty-list edge case → test_median_empty failed.")

`final-files.json` — workspace state at end of trial:

```json
{"stats.py": "<full file content>", "tests/test_stats.py": "<unchanged>"}
```

### 2.12 `runs/{run}/threads/{thread}/candidates/{N}/status.json`

*DERIVED from Appendix C §C.10.*

```json
{
  "candidate": "few-shot-tool-results",
  "thread_id": "run-x",
  "accepted": true,
  "parent": "more-specific-descriptions",
  "compared_against": "more-specific-descriptions",
  "compared_against_accuracy": 0.73,
  "delta": 0.05,
  "reason": "accepted"
}
```

`reason` ∈ {`accepted`, `smoke_failed`, `regression`, `failed_holdout`,
`measured baseline (search root)`}.

`compared_against` / `compared_against_accuracy` name the prior best this
candidate's `delta` was computed against, so the comparison is auditable
rather than implied. On iteration 1 that is the measured `baseline`;
both are `null` only on the baseline's own row.

---

## 3. Inner-loop tool I/O schemas (the FIXED contract)

These six tools are the contract with the evaluator. The proposer cannot
modify them. Tool schemas for read_file, apply_patch, run_bash,
grep_search, and task_complete are *verbatim from Appendix C §C.6*;
write_file (§3.6) expands the §C.6.2 fallback into the formal tool list.

### 3.1 `read_file`

```json
{
  "name": "read_file",
  "description": "Read a file from the workspace, with optional line range.",
  "input_schema": {
    "type": "object",
    "properties": {
      "path":       {"type": "string"},
      "start_line": {"type": "integer", "default": 1},
      "end_line":   {"type": "integer", "description": "Inclusive; -1 = EOF"}
    },
    "required": ["path"]
  }
}
```
Returns: line-numbered content. Files >2000 lines must specify a range or get an error directing them to use `grep_search`.

### 3.2 `apply_patch`

```json
{
  "name": "apply_patch",
  "description": "Apply a unified-diff patch to a file. Patches are surgical and preserve unchanged lines exactly. Use this to make targeted edits rather than rewriting whole files. The patch must apply cleanly; fuzz matching is disabled.",
  "input_schema": {
    "type": "object",
    "properties": {
      "path":  {"type": "string"},
      "patch": {"type": "string", "description": "Unified diff format (the same format as `git diff`)."}
    },
    "required": ["path", "patch"]
  }
}
```
**Failure response** (structured error):

```json
{
  "status": "error",
  "error_type": "context_mismatch",
  "error_message": "Patch context did not match at lines 42-46. The file currently reads:\n<content>\nEdit the patch to match this and retry.",
  "context_echo": {
    "path": "stats.py",
    "start_line": 42,
    "end_line": 46,
    "content": "<actual current content of those lines>"
  }
}
```

`error_type` ∈ {`context_mismatch`, `file_not_found`, `invalid_patch`,
`invalid_patch_path`, `path_mismatch`}. The patch must be a single-file
diff whose `---` / `+++` headers target the same workspace-relative file
as the `path` argument; otherwise the tool returns `path_mismatch` and
does not invoke `git apply`. On `context_mismatch`, `context_echo` is
populated with the current file content at the failed range so the model
can fix the patch without re-reading the file (per Appendix C §C.6.2's
"this tells the model exactly what to fix" rule). On other failures,
`context_echo` is `null`.

### 3.3 `run_bash`

```json
{
  "name": "run_bash",
  "description": "Run a bash command in the sandboxed workspace. Returns stdout, stderr, exit_code, and duration_ms. Commands run with a 30s default timeout (max 120s). The workspace is reset between tasks.",
  "input_schema": {
    "type": "object",
    "properties": {
      "command":     {"type": "string"},
      "timeout_sec": {"type": "integer", "default": 30, "maximum": 120}
    },
    "required": ["command"]
  }
}
```
Sandbox: `/tmp/meta-harness-task-{uuid}/`, no network, rlimit 512MB RAM /
60s CPU. Allowed binaries: `python3, pip, pytest, git, bash, ls, cat, grep,
sed, head, tail, find, diff, make`. **No** `curl, wget, ssh`.

### 3.4 `grep_search`

```json
{
  "name": "grep_search",
  "description": "Search files in the workspace using ripgrep. Returns file paths and matching lines with line numbers. Prefer this over reading many files individually.",
  "input_schema": {
    "type": "object",
    "properties": {
      "pattern":       {"type": "string", "description": "Regex pattern."},
      "path":          {"type": "string", "default": "."},
      "file_glob":     {"type": "string", "description": "e.g. '*.py'"},
      "context_lines": {"type": "integer", "default": 2, "maximum": 10}
    },
    "required": ["pattern"]
  }
}
```

### 3.5 `task_complete`

```json
{
  "name": "task_complete",
  "description": "Signal that the task is done. Call this when you believe the task is solved AND tests pass. The harness will run final verification.",
  "input_schema": {"type": "object", "properties": {}}
}
```

### 3.6 `write_file`

```json
{
  "name": "write_file",
  "description": "Create a new file. Errors if the file already exists — use apply_patch to modify existing files. Use this only for files that do not yet exist in the workspace.",
  "input_schema": {
    "type": "object",
    "properties": {
      "path":    {"type": "string"},
      "content": {"type": "string"}
    },
    "required": ["path", "content"]
  }
}
```

**Return shape** (mirrors apply_patch's structured pattern):

```json
{
  "status": "ok",
  "path": "geometry/point.py",
  "bytes_written": 412
}
```

```json
{
  "status": "error",
  "error_type": "file_exists",
  "error_message": "File 'geometry/point.py' already exists. Use apply_patch to modify it."
}
```

`status` ∈ {`ok`, `error`}. On `ok`, `bytes_written` is set. On `error`,
`error_type` ∈ {`file_exists`, `invalid_path`, `write_failed`} and
`error_message` is set.

This expands Appendix C §C.6.2's `write_file` fallback into the formal
tool list; both `INTERFACES.md` and `ARCHITECTURE_SECTION_1.md` are now
the canonical source for the tool count (six). `write_file`, like the
other five tools, is part of the FIXED contract — not overridable by
candidates.

---

## 4. The 11 inner-loop override points (the SEARCH SPACE)

*Verbatim from Appendix C §C.9.* Candidates may override any of these on
their `CodingAgentHarness` subclass. Tools (§3) and phase boundaries are
NOT overridable.

```python
class CodingAgentHarness:
    SYSTEM_PROMPT: str                   # 1. tone, framing, few-shot examples
    PLAN_PROMPT_TEMPLATE: str            # 2. plan structure, signals to highlight
    MAX_ACT_TURNS: int = 25              # 3. tighter (10) or looser (40)
    MAX_VERIFY_RETRIES: int = 3          # 4. retry budget on verify failure

    def _build_initial_context(self, orient_summary: dict) -> dict: ...   # 5
    def _format_tool_result(self, name: str, result: ToolResult) -> str: ... # 6
    def _compose_act_prompt(self, plan: dict) -> str: ...                 # 7
    async def _call_llm(self, messages: list, tools: list, *, tool_choice: dict | None = None) -> Response: ...  # 8
    def should_loop_back_to_act(self, verify_result: dict) -> bool: ...   # 9
    def _summarize_for_overflow(self, messages: list) -> list: ...        # 10
    # 11. (Structural) Reordering phases: skip plan for simple tasks, add
    #     a re-plan phase after first verify failure, etc. Implemented by
    #     overriding the graph-build hook `build_inner_graph(self) -> StateGraph`.
```

`ToolResult` and `Response` are SDK type stubs (`sdk/meta_harness/types.py`).

---

## 5. SKILL.md — frontmatter + body convention

Per Appendix B §B.3.1 and the two Stanford reference SKILL.md files
inspected during research.

### 5.1 Frontmatter (YAML)

```yaml
---
name: meta-harness-coding-agent       # max 64 chars; lowercase / digits / hyphens
description: Evolve the coding agent harness. Use when running meta_harness.py iterations to propose new candidate harnesses based on prior execution traces and scores.
---
```

Constraints (per Claude Code skill spec): `name` ≤64 chars, `description`
≤1024 chars, no XML tags in either, no reserved words.

### 5.2 Body sections (required, in order)

1. **What you are doing** — one paragraph framing the task.
2. **Hard rules (Anti-Overfitting)** — explicit forbidden behaviors
   (no task-specific hints, no hardcoded fixes, generalize-only).
3. **Hard rules (Anti-Parameter-Tuning)** — mechanism-first design,
   self-critique step before write, no combinatorial sweeps.
4. **Workflow** — numbered steps: Analyze → (Pick hypothesis) →
   Prototype → Implement → Register.
5. **Interface contract** — Python class signature the candidate must
   implement; here `CodingAgentHarness`.
6. **The pending_eval.json schema** — exact JSON shape (per §2.1).

Body length: 100–200 lines. Total file: ~5 KB Markdown.

### 5.3 Skill loading mechanism (PHASE 1.3 — RESOLVED)

`meta-harness loop` resolves `skill_path` in this precedence order:

1. **Absolute path** (starts with `/`) — used directly.
2. **Relative path** — resolved against repo root (the directory holding
   the workspace `pyproject.toml`).
3. **Omitted** — defaults to `skills/meta-harness-coding-agent/SKILL.md`
   (relative to repo root).

**Validation at run-start** (before the propose node executes):

- File exists and is readable.
- Parses as YAML frontmatter + Markdown body.
- Frontmatter has required `name` (≤64 chars) and `description`
  (≤1024 chars) fields.
- Body has all 6 required sections (per §5.2).
- On any failure: refuse to start the run with a clear error; do not
  partial-init.

**CLI surface:**

- `meta-harness loop` (no flag) → uses default `skill_path`.
- `meta-harness loop --skill <path>` → explicit override.
- `meta-harness loop --skill-dir <dir>` → resolves to `<dir>/SKILL.md`
  (convenience for users with multiple domain skills).

The HTTP `POST /runs` body's `domain` field continues to map a known
domain name to its default `skill_path` via these same rules; explicit
`skill_path` may also be passed in the request body to override the
domain-default lookup.

---

## 6. REST endpoints (FastAPI / `backend/app/api/`)

All paths relative to `http://localhost:8000`. Bodies are JSON unless
noted. Status codes are conventional (200 OK, 201 Created, 202 Accepted,
404 Not Found, 409 Conflict).

### 6.1 Runs (`api/runs.py`)

| Method | Path | Request | Response | Status |
|---|---|---|---|---|
| `POST` | `/runs` | `{"domain": "coding-agent", "skill_path": "<optional>", "budget": 5, "model": "opus", "fresh": true, "run_name": "demo-2026-04-25", "proposer": "claude", "mock_bench": false, "trials": 5, "workers": 3}` | **201 Created** with header `Location: /runs/{run_id}`. Body: full Run object: `{"run_id", "thread_id", "status", "started_at", "domain", "skill_path", "budget", "model", "current_iteration": 0}` | **201** |
| `GET`  | `/runs` | — | `{"runs": [{"run_id", "thread_id", "status", "started_at", "current_iteration", "best_score"}]}` | 200 |
| `GET`  | `/runs/{run_id}` | — | full `RunInfo`: manifest, root-branch `frontier_val`, `branch_frontiers` (thread_id → frontier), **all** `summary_rows` across every branch each tagged with `thread_id`, run-wide `best_score`, and `metrics_source` | 200 |
| `GET`  | `/runs/{run_id}/candidates/{candidate_name}/diff` | — | `{"candidate", "parent", "thread_id", "from_path", "to_path", "diff"}`. The diff is taken against the **branch-private source snapshot** (§0.1), not the shared repo-root file a concurrent branch may have rewritten. **404** for an unknown candidate. | 200 |
| `GET`  | `/runs/{run_id}/candidates/{candidate_name}/test-output` | — | `{"candidate", "output"}` summarizing eval-result and available verify trace output | 200 |
| `DELETE` | `/runs/{run_id}` | — | `{"status": "cancelled"}` (cascades to all branches via `branch_registry`) | 200 |

`proposer`, `mock_bench`, `trials`, and `workers` are backend run-control
fields used by the CLI-equivalent API path and smoke tests. Omitted
values preserve the real proposer path: `proposer="claude"`,
`mock_bench=false`, `trials=5`, `workers=3`.

`POST /runs` returns as soon as the run task is scheduled. The task
benchmarks `agents/baseline.py` **before** the first propose, so an
API-started run has the same measured search root as a CLI run.

`metrics_source` on `GET /runs/{run_id}` is `"measured"` or `"mock"`.
The dashboard must render the two distinguishably; a mock run may never
be presented as an experiment.

### 6.1.1 Health (`app/main.py`)

| Method | Path | Response | Status |
|---|---|---|---|
| `GET` | `/health` | `{"status": "ok", "version", "persistence": "postgres" \| "memory", "persistence_error": <string \| null>, "memory_store": <bool>}` | 200 |

`persistence: "memory"` means checkpoint history, forking and branch
recovery are **unavailable**, and `persistence_error` says why (Postgres
unreachable, or an event loop psycopg cannot use). Degradation is never
silent — clients and acceptance scripts are expected to check this.

### 6.2 Checkpoints (`api/checkpoints.py`)

| Method | Path | Request | Response | Status |
|---|---|---|---|---|
| `GET` | `/runs/{run_id}/checkpoints` | — | `{"checkpoints": [{"checkpoint_id", "thread_id", "ts", "node", "iteration", "values_summary", "parent_checkpoint_id", "next", "metadata"}]}` (output of `graph.aget_state_history` projected, newest first) | 200 |
| `GET` | `/runs/{run_id}/checkpoints/{checkpoint_id}` | — | `{"checkpoint_id", "thread_id", "state": <full MetaHarnessState>, "ts", "node"}` | 200 |

### 6.3 Forks (`api/forks.py`)

| Method | Path | Request | Response | Status |
|---|---|---|---|---|
| `POST` | `/runs/{run_id}/fork` | `{"parent_checkpoint_id": "<id>", "mods": {"proposer_prior": "<new prior>"}, "parent_thread_id": "<optional>", "name": "<optional>"}` | `{"thread_id": "<run-id>.fork.<8hex>", "status": "running", "parent_checkpoint_id": "<id>", "branch_id": "<8hex>"}` | 202 |

`parent_thread_id` defaults to `run_id`, which preserves the original
fork-from-root behavior. Supplying it allows a future dashboard to fork
from an existing branch thread without changing the route shape.

### 6.4 Branches (`api/branches.py`)

| Method | Path | Request | Response | Status |
|---|---|---|---|---|
| `GET` | `/runs/{run_id}/branches` | — | `{"branches": [<BranchMetadata>]}` | 200 |
| `GET` | `/runs/{run_id}/trajectory` | — | `{"trajectory": {"run_id", "threads", "edges"}}` | 200 |
| `POST` | `/runs/{run_id}/branches/{thread_id}/cancel` | — | `{"status": "cancelled"}` (calls `task.cancel()` through Step 9 branch metadata) | 200 |

**Branch history is durable; branch execution is not.** Metadata is
mirrored to `runs/{run_id}/branches.json` on every transition, so both
endpoints reconstruct the full tree after an API restart. An asyncio
task does not survive that restart: a branch persisted as `running`
reloads with `status: "interrupted"` and an explanatory `error`.

`BranchMetadata` fields: `branch_id`, `run_id`, `thread_id`,
`parent_thread_id`, `parent_checkpoint_id`, `parent_candidate`, `name`,
`mods`, `status`, `created_at`, `started_at`, `finished_at`,
`cancelled_at`, `error`, `result`, and `live`.

`status` ∈ {`created`, `running`, `completed`, `failed`, `cancelled`,
`interrupted`}. `live` is `true` only while *this* backend process is
driving the branch, and is never persisted as `true`.

Each `edges` entry is
`{"source", "target", "parent_checkpoint_id", "parent_candidate"}`.
`threads` includes the run's root thread with `status: "root"`.

### 6.5 Memory (`api/memory.py`)

| Method | Path | Request | Response | Status |
|---|---|---|---|---|
| `GET` | `/memory/{namespace}` | — (query: `?limit=50`) | `{"namespace": ["learned_patterns","coding"], "entries": [{"key", "value", "score_delta", "evidence_run_ids"}], "implemented": false}` | 200 |
| `POST` | `/memory/{namespace}/search` | `{"query": "schema drift retry", "limit": 5}` | `{"results": [...], "implemented": false}` | 200 |

`namespace` is URL-encoded; the conventional shape is
`("learned_patterns", "<domain>")`. When the Postgres store is unavailable,
the backend returns empty result sets with `implemented=false`; callers
must treat that as a valid degraded mode, not an error.

### 6.6 Events / SSE (`api/events.py`)

| Method | Path | Response |
|---|---|---|
| `GET` | `/runs/{run_id}/stream` | `text/event-stream` — see §7 below for event shape |

A single SSE stream per run multiplexes events from all branches; each
event carries its `thread_id` so the dashboard can route events to the
right branch in the trajectory tree.

---

## 7. SSE event format

*Channel naming convention (internal): `run:{run_id}` (multiplexed across
all threads of that run). Per Appendix A §A.5 the alternative is one
EventSource per branch (`run:{thread_id}`); we go with the multiplexed
form to limit browser connection count, with `thread_id` in every event.*

### 7.1 Wire format (HTML SSE spec)

```
event: state-update
id: <checkpoint-id>
data: {"thread_id": "run-2026-04-25-1430", "node": "propose", "iteration": 3, "ts": "2026-04-25T14:32:11Z", "summary": {"candidates_count": 3, "budget_remaining": 2}}

event: candidate-created
id: <checkpoint-id>
data: {"thread_id": "...", "candidate": "few-shot-tool-results", "import_path": "agents.few_shot_tool_results:CodingAgentHarness", "parent": "more-specific-descriptions"}

```

Blank line terminates each event. Reconnect via `Last-Event-ID` header
(checkpoint-id is monotonic per thread).

### 7.2 Event types (closed set)

| Event | Fired by | Data shape |
|---|---|---|
| `state-update` | every LangGraph node transition | `{thread_id, node, iteration, ts, summary}` |
| `checkpoint-written` | AsyncPostgresSaver post-write | `{thread_id, checkpoint_id, parent_checkpoint_id, ts, node}` |
| `candidate-created` | `propose` node after parsing pending_eval.json | `{thread_id, candidate, label, parent_candidate_name, import_path, parent, iteration, status, scores, delta, hypothesis, axis}` |
| `validate-result` | `validate` node | `{thread_id, candidate, valid, error?}` |
| `eval-result` | `benchmark` node | `{thread_id, candidate, parent_candidate_name, iteration, status, accuracy, scores, per_task, tokens, cost_usd, metrics_source, hypothesis, axis}` |
| `frontier-updated` | `update_frontier` node | `{thread_id, candidate, parent_candidate_name, iteration, frontier, best_candidate, best_score, status, accepted, delta, scores, hypothesis, axis}` |
| `iteration-complete` | end of `update_frontier` | `{thread_id, iteration, status: "improved"\|"no_improvement"}` |
| `fork-created` | `worktree_add` | `{thread_id, parent_thread_id, parent_checkpoint_id, mods_summary}` |
| `branch-cancelled` | cancel endpoint | `{thread_id, reason}` |
| `memory-pattern-stored` | end-of-run memory write | `{thread_id, namespace, key, score_delta}` |
| `error` | any node exception | `{thread_id, node, message, traceback}` |

**`thread_id` is mandatory on every event** and the registry rejects a
payload without one (`InvalidEventPayloadError`). Consumers must key
tree state on `(thread_id, candidate)`, never on candidate name or
iteration number alone: two branches forked from one checkpoint reach
the same iteration, and collapsing them merges one branch's result into
the other's node.

`eval-result.cost_usd` is `null` when the model has no configured price
(§0.3); `metrics_source` tells the consumer whether the numbers in the
event are measurements or mock data.

### 7.3 Closed-set enforcement rule

The 11 event types in §7.2 are a **closed set, enforced at runtime**.

- To add a new SSE event type post-launch, both `INTERFACES.md` and
  `frontend/lib/sse.ts` must update in the same commit.
- The backend SSE channel registry (`backend/app/streaming.py`) maintains
  a registered allowlist of event-type strings derived from §7.2.
  `emit(event_type, payload)` for an unregistered type raises a
  500-class error rather than silently dropping; this surfaces drift in
  CI and at runtime instead of degrading the dashboard.
- Closed-ness is enforced, not just documented.

### 7.4 Subscription model — per-run multiplex (rationale)

The dashboard subscribes once per run via `GET /runs/{run_id}/stream`,
which multiplexes events from all branches. `thread_id` is on every
event payload; the frontend filters by it for view-specific updates.

Rationale: a user viewing a run wants to see all branches simultaneously
— the demo-day beat is two branches growing on screen at once. The
per-thread alternative (one EventSource per branch, per Appendix A §A.5)
would force the frontend to subscribe / unsubscribe as forks happen,
which is fragile during a live demo. Multiplexing costs one extra field
(`thread_id`) per event; cheap.

---

## 8. Resolutions of formerly-open questions

All 8 items previously surfaced here are now resolved. The locking pass
above pulled each into the relevant section.

| # | Topic | Resolution |
|---|---|---|
| 1 | Skill loading mechanism (Phase 1.3) | §5.3 — `skill_path` precedence rules + `--skill` / `--skill-dir` CLI |
| 2 | `Candidate` dataclass shape | §1.3 — `traces_dir: Path` is non-optional |
| 3 | `frontier_val.json` shape | §2.2 — `dominated_by_names: list[str]` per candidate; `_pareto_names` convenience |
| 4 | `evolution_summary.jsonl` row shape | §2.3 — `parent_candidate_name: str \| null` added |
| 5 | `write_file` status | §3.6 — formal 6th fixed tool with structured return shape |
| 6 | REST endpoint shapes | §6 — `POST /runs` returns **201 Created** + `Location` header |
| 7 | SSE event-types closed-set | §7.3 — runtime-enforced allowlist; 500 on unknown type |
| 8 | SSE subscription model | §7.4 — per-run multiplex with `thread_id` per event |

If a genuinely new ambiguity surfaces during implementation, it goes back
into this section as a 9th item rather than being silently designed.
