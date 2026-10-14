"""Record an inner-loop run against Postgres, without a provider.

Produces the recording that `meta-harness verify-replay` turns into
`docs/evidence/replay-verification.json`. Everything in the run is real —
the inner graph, the six tools, the pytest verify subprocess, the
grading, the workspace, the Postgres checkpoints — except the model's
turns, which come from a script.

**That is stated in the artifact, not hidden by it.** The recording's
manifest records `model: "scripted-offline"` and a `recorded_with` note,
and both ride into the verification report and from there into the
evidence document. What this demonstrates is the replay machinery, which
is what the claim is about; it is not a substitute for the pass-rate
measurement, which needs a provider and appears as UNSUPPORTED until one
is run.

With credentials, `meta-harness canonical-experiment --record-trials N`
overwrites this artifact with recordings of real provider calls.

The scripted trial is graded like any other: the submit node re-runs the
task's tests in a fresh copy of the pristine workspace that carries over
only the task's ``expected_files_changed``. The script's fix touches
exactly that file, and the script refuses to leave behind a recording of
a trial that did not pass.

The scripted model is a harness *class*, ``ScriptedBaselineHarness``,
with its model name at class level. Replay rebuilds the recorded class
from this file (checked against the recorded SHA-256) and compares its
configuration with the recording's, so this file must not change between
recording and verifying; re-record after editing it.

Usage:
    docker compose -f infra/docker-compose.yml up -d postgres
    uv run python scripts/record_replay_evidence.py
    uv run meta-harness verify-replay runs/replay-evidence
"""

from __future__ import annotations

import asyncio
import difflib
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents.baseline import BaselineHarness  # noqa: E402
from app.meta_harness import replay as replay_mod  # noqa: E402
from app.meta_harness import runs as runs_mod  # noqa: E402
from app.meta_harness.persistence import healthcheck, persistence_layer  # noqa: E402
from app.meta_harness.sandbox import sandbox_for  # noqa: E402

TASK_ID = "task-001-fix-typo"
RUN_NAME = "replay-evidence"
TASK_DIR = REPO_ROOT / "eval" / "tasks" / TASK_ID


class _Block:
    def __init__(self, **fields: Any) -> None:
        self.type = fields.get("type")
        self.text = fields.get("text")
        self.id = fields.get("id")
        self.name = fields.get("name")
        self.input = fields.get("input") or {}


class _Usage:
    input_tokens = 1200
    output_tokens = 180
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _Response:
    def __init__(self, blocks: list[_Block]) -> None:
        self.content = blocks
        self.model = ScriptedBaselineHarness.MODEL
        self.usage = _Usage()
        self.stop_reason = "tool_use"


# add() and sub() both read `return a - b`, so the fix is located by the
# function it belongs to, never by the line alone.
BUGGY_ADD = "def add(a: int, b: int) -> int:\n    return a - b\n"
FIXED_ADD = "def add(a: int, b: int) -> int:\n    return a + b\n"


def repo_relative(path: Path) -> str:
    """A path as it may appear in anything that reaches docs/evidence/."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fix_patch() -> str:
    """The unified diff the scripted model 'writes', built from the real file.

    Generated rather than hand-written: ``git apply`` runs underneath and
    is strict about hunk ranges and context, and the committed task file
    is the only reliable source for both.
    """
    source = (TASK_DIR / "workspace" / "calculator.py").read_text(encoding="utf-8")
    if source.count(BUGGY_ADD) != 1:
        raise SystemExit(
            "the task workspace no longer contains exactly one add() returning "
            "`a - b`; update BUGGY_ADD"
        )
    fixed = source.replace(BUGGY_ADD, FIXED_ADD)
    return "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile="a/calculator.py",
            tofile="b/calculator.py",
            n=3,
        )
    )


#: The act-phase turns, in order. The last entry repeats if the graph
#: needs more turns than the script provides.
SCRIPT: list[list[dict[str, Any]]] = [
    [
        {"type": "text", "text": "Reading calculator.py before editing it."},
        {
            "type": "tool_use",
            "id": "t1",
            "name": "read_file",
            "input": {"path": "calculator.py"},
        },
    ],
    [
        {
            "type": "tool_use",
            "id": "t2",
            "name": "apply_patch",
            "input": {"path": "calculator.py", "patch": fix_patch()},
        }
    ],
    [{"type": "tool_use", "id": "t3", "name": "task_complete", "input": {}}],
]

PLAN = {
    "summary": "fix add() to return a + b; leave sub() alone",
    "steps": [{"action": "patch", "target": "calculator.py", "why": "add() subtracts"}],
    "expected_files_changed": ["calculator.py"],
    "tests_to_run": ["tests/test_calculator.py"],
}


class ScriptedBaselineHarness(BaselineHarness):
    """The real baseline with its model turns taken from ``SCRIPT``.

    A class rather than a patched instance: replay rebuilds the recorded
    class and checks its configuration (``MODEL`` among it) against the
    recording, so everything that distinguishes this harness lives on the
    class. Every override point other than ``_call_llm`` is the
    baseline's own, which is what makes the replay meaningful.

    ``__init__`` skips the base constructor's API-key requirement and
    builds no client: this issues no network call of any kind.
    """

    MODEL = "scripted-offline"

    def __init__(self) -> None:
        self.api_key = None
        self._client = None
        self._turn = 0

    async def _call_llm(
        self,
        messages: list[Any],
        tools: list[Any],
        *,
        tool_choice: dict[str, Any] | None = None,
    ) -> _Response:
        if tool_choice is not None:
            return _Response(
                [_Block(type="tool_use", id="plan-1", name="submit_plan", input=dict(PLAN))]
            )
        index = min(self._turn, len(SCRIPT) - 1)
        self._turn += 1
        return _Response([_Block(**block) for block in SCRIPT[index]])


async def main() -> int:
    if not await healthcheck():
        print(
            "Postgres is not reachable. A recording without a checkpoint "
            "history cannot be replayed from a checkpoint.\n"
            "  docker compose -f infra/docker-compose.yml up -d postgres",
            file=sys.stderr,
        )
        return 1

    source_workspace = TASK_DIR / "workspace"
    task_spec = json.loads((TASK_DIR / "task.json").read_text())
    # The submit node grades in a fresh copy of the pristine workspace and
    # refuses a task dict that does not say where that is.
    task_spec["_source_workspace"] = str(source_workspace)
    if PLAN["expected_files_changed"] != task_spec.get("expected_files_changed"):
        raise SystemExit(
            "the scripted plan no longer matches the task's "
            "expected_files_changed; the graded copy would not carry the fix"
        )

    run_dir = runs_mod.make_run_dir(REPO_ROOT, RUN_NAME, fresh=True)
    runs_mod.write_manifest(
        run_dir,
        run_id=RUN_NAME,
        thread_id=RUN_NAME,
        budget=0,
        trials=1,
        mock_proposer=True,
        mock_bench=False,
        metrics_source="measured",
    )
    recording_dir = (
        runs_mod.thread_dir(run_dir, RUN_NAME) / "recordings" / f"{TASK_ID}-trial-1"
    )
    shutil.rmtree(recording_dir, ignore_errors=True)

    # Checkpoint history outlives the process, so a fixed thread id would
    # accumulate across runs of this script and leave the tape unable to
    # line up with it. ``assert_thread_unused`` catches that; a fresh id
    # avoids it.
    thread_id = f"record::{RUN_NAME}::{TASK_ID}::trial-1::{uuid.uuid4().hex[:8]}"

    async with persistence_layer() as saver:
        with sandbox_for(source_workspace) as sandbox:
            result = await replay_mod.record_inner_execution(
                harness_factory=ScriptedBaselineHarness,
                task_dict=task_spec,
                workspace=sandbox,
                source_workspace=source_workspace,
                thread_id=thread_id,
                checkpointer=saver,
                recording_dir=recording_dir,
                manifest_extra={
                    "recorded_with": (
                        "scripted model turns; no provider call was made. The "
                        "graph, the six tools, the pytest verify subprocess, "
                        "the grading and the Postgres checkpoints are real."
                    )
                },
            )

    summary = {k: v for k, v in result.items() if k != "final_state"}
    summary["recording_dir"] = repo_relative(recording_dir)
    print(json.dumps(summary, indent=2))
    score = result["final_state"].get("score")
    print(f"\nscore: {score}")
    print(f"recording: {repo_relative(recording_dir)}")
    if (score or 0.0) < 1.0:
        # Replaying a failed trial exactly would still "verify", and the
        # evidence would then rest on a run that did not do its job.
        shutil.rmtree(recording_dir, ignore_errors=True)
        print(
            "the scripted trial did not pass under grading; recording removed",
            file=sys.stderr,
        )
        return 1
    print(f"\nnext: uv run meta-harness verify-replay runs/{RUN_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
