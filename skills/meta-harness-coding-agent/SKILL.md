---
name: meta-harness-coding-agent
description: Evolve the coding-agent harness. Use this skill when invoked by `meta-harness loop` to propose a new candidate harness based on prior execution traces and scores. Read the filesystem first; form falsifiable hypotheses; produce ONE new agents/<name>.py file and register it in your branch's pending_eval.json.
---

# Meta-Harness Coding Agent Evolution

You are evolving the source code of a 5-phase coding-agent harness. Your
job is to read the full history of prior candidate harnesses, identify a
specific failure pattern, and write ONE new candidate harness file that
addresses it.

## Where you are

Your working directory is a private copy of the repository made for this
one iteration. Repo-relative paths work as usual. When you finish, exactly
two things are kept:

- the candidate file you write at `agents/<name>.py`, and
- `<branch-dir>/pending_eval.json`.

Every other change is discarded. Do not try to reach outside the copy:
this session is audited, and a tool call that names a path outside it, or
anything held out from you, voids the iteration.

**`<branch-dir>`** is the directory the prompt names under "Your branch
directory" — `runs/<run_id>/threads/<thread>/`. Every run artifact you read
or write is under it. There is no run-level `pending_eval.json`,
`frontier_val.json` or `evolution_summary.jsonl`: each branch of a run has
its own, and another branch's are not yours to read.

## What gets evolved

The harness is a single Python file in `agents/<name>.py` that subclasses
`CodingAgentHarness` from `app.meta_harness.harness`. You may override
any of the **11 search-space methods** (Appendix C §C.9):

- `SYSTEM_PROMPT` — how the agent is instructed.
- `PLAN_PROMPT_TEMPLATE` — how planning is framed.
- `MAX_ACT_TURNS` — turn budget for the act phase (default 25).
- `MAX_VERIFY_RETRIES` — verify→act retry budget (default 3).
- `_build_initial_context(orient_summary)` — what the planner sees.
- `_format_tool_result(name, result)` — how tool outputs render.
- `_compose_act_prompt(plan)` — plan injection into act phase.
- `_call_llm(messages, tools, *, tool_choice=None)` — Anthropic API call mechanics.
- `should_loop_back_to_act(verify_result)` — retry decision logic.
- `_summarize_for_overflow(messages)` — context overflow strategy.
- (Structural) `_wire_inner_graph(graph, *, route_after_verify)` — how the five
  phase nodes are connected (skip `plan`, send a failed verify back to `plan`, …).
  The node bodies are fixed, and only `submit` grades the trial.

How a trial is scored is not part of the search space. Each trial is graded
in a fresh copy of the task's original workspace into which only the files
the task lists in `expected_files_changed` are carried over from the agent's
workspace. Edits to tests, test configuration or any other file never reach
the grade, so a harness that steers the agent toward them gains nothing.

You may **NOT** override the 6 fixed inner-loop tools (`read_file`,
`apply_patch`, `write_file`, `run_bash`, `grep_search`, `task_complete`)
or the phase boundaries (`orient`, `plan`, `act`, `verify`, `submit`) —
those are the contract with the evaluator. This is checked before your
file is executed: a candidate that imports any `app.meta_harness` module
other than `app.meta_harness.harness`, imports `sys`, `importlib`,
`builtins`, `ctypes` or `gc`, assigns into an imported module (including
`os.environ`), calls `setattr`/`delattr` on anything but `self`/`cls`, or
calls `exec`, `eval`, `compile` or `__import__` is rejected as
`smoke_failed` and wastes the iteration.

## Hard rules (Anti-Overfitting)

1. **No task-specific knowledge.** Never reference specific tasks like
   "calculator.py" or "the typo bug" in your code or comments. Your
   improvements must generalize.
2. **No hard-coded fixes.** Don't write code that detects the eval tasks
   by name and special-cases them. The evaluator will reject candidates
   with string-leakage from task names.
3. **General principles only.** Frame every change as a hypothesis about
   *coding agents in general*, not "what would have worked on task 003."

## Hard rules (Anti-Parameter-Tuning)

1. **Mechanism, not constants.** If your only change is `MAX_ACT_TURNS = 30`
   instead of `MAX_ACT_TURNS = 25`, that is a parameter tweak, not an
   evolution. Reject it.
2. **Self-critique before writing.** Before writing the candidate file,
   verify in a comment block at the top:
   ```
   # STRUCTURAL CHANGE: this candidate differs from {parent} by {mechanism}.
   # The mechanism is genuinely new, not a constant change.
   ```
3. **No combinatorial sweeps.** Don't propose 3 candidates that vary one
   constant. Propose 1 candidate that introduces a new mechanism.

## Workflow (mandatory order)

### Step 1 — Analyze (read the filesystem)

Read these files in this order:

1. `<branch-dir>/evolution_summary.jsonl` — every prior candidate on this
   branch, its hypothesis and score. A row with `"status": "smoke_failed"`
   is a candidate that never loaded; it has no score.
2. `<branch-dir>/frontier_val.json` — current Pareto frontier (which
   candidates are non-dominated on accuracy × tokens).
3. The parent candidate's source, at the path the prompt gives under
   "Parent". That is the exact source that was benchmarked: for the
   baseline it is `agents/baseline.py`; for an evolved parent it is the
   snapshot under the `runs/.../threads/<thread>/agents/` directory of the
   branch that measured it, which may be an ancestor of yours.
4. The 2-3 lowest-scoring candidates' sources (their snapshots under
   `<branch-dir>/agents/` or an ancestor branch's `agents/`) AND their
   `<branch-dir>/candidates/<name>/traces/` execution traces.

Then form THREE falsifiable hypotheses about why the best candidate
fails on specific tasks. Briefly note them in your reasoning.

### Step 2 — Pick one hypothesis

From the three, pick the most likely to produce a >5% improvement.

### Step 3 — Prototype (test the mechanism in isolation)

Write a small `/tmp/prototype-iter-{N}.py` that exercises the new
mechanism on 1–2 trace examples WITHOUT the full harness. Verify the
mechanism does what you think it does before committing it to a candidate.

### Step 4 — Implement (write the candidate)

1. Copy the parent's source (the path from the prompt) to
   `agents/<descriptive-snake-case-name>.py`. Pick a name no earlier
   candidate of this run has used.
2. Apply the targeted modification (override at most 2-3 of the 11
   search-space methods).
3. Add the self-critique comment block at the top.
4. Verify the file imports cleanly, from the working directory:
   ```bash
   python -c "from agents.<name> import *; print('OK')"
   ```
   Use `python` directly. `uv run`/`uv sync` would try to build a fresh
   environment inside the copy.

### Step 5 — Register (write pending_eval.json)

Write to `<branch-dir>/pending_eval.json` (the exact path is in the
prompt):

```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "<descriptive-snake-case-name>",
      "import_path": "agents.<descriptive-snake-case-name>:<ClassName>",
      "parent": "<parent-candidate-name>",
      "hypothesis": "<one-sentence falsifiable claim>",
      "axis": "exploration | exploitation",
      "expected_score_delta": <float between -0.2 and +0.2>
    }
  ]
}
```

The handoff is checked before anything is benchmarked, and the iteration
fails if any of these does not hold:

- `iteration` is the iteration number in the prompt;
- `candidates` holds exactly one entry;
- `name` is new in this run, and `agents/<name>.py` exists;
- `import_path` is `agents.<name>:<ClassName>`, and the class in that file
  is a subclass of `CodingAgentHarness`.

## Interface contract

Your new candidate must:

```python
from app.meta_harness.harness import CodingAgentHarness


class YourCandidateName(CodingAgentHarness):
    """One-sentence hypothesis."""

    # ... overrides ...
```

The candidate is loaded from its branch snapshot of `agents/<name>.py`
and instantiated with `YourCandidateName()`. The `__init__` must accept
no required args (it inherits from `CodingAgentHarness.__init__` which
reads the Anthropic API key from the env).

## What you may NOT do

- Write anything you expect to keep outside `agents/<name>.py` and
  `<branch-dir>/pending_eval.json` (scratch work belongs in `/tmp/`).
- Modify the eval tasks in `eval/` or anything in `benchmarks/`.
- Modify any baseline file (`agents/baseline.py`).
- Look for, or refer to, evaluation tasks or results that are not in your
  copy. They are held out from you on purpose, and a search for them —
  including a path, a task id, or a string in the candidate's own code —
  fails the iteration.
- Propose more than ONE candidate per iteration.
