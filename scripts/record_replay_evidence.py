"""Record an inner-loop run against Postgres, without a provider.

Produces the recording that `meta-harness verify-replay` turns into
`docs/evidence/replay-verification.json`. Everything in the run is real —
the inner graph, the six tools, the pytest verify subprocess, the
workspace, the Postgres checkpoints — except the model's turns, which
come from a script.

**That is stated in the artifact, not hidden by it.** The recording's
manifest records `model: "scripted-offline"` and a `recorded_with` note,
and both ride into the verification report and from there into the
evidence document. What this demonstrates is the replay machinery, which
is what the claim is about; it is not a substitute for the pass-rate
measurement, which needs a provider and appears as UNSUPPORTED until one
is run.

With credentials, `meta-harness resume-experiment --record-trials N`
overwrites this artifact with recordings of real provider calls.

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
        self.model = "scripted-offline"
        self.usage = _Usage()
        self.stop_reason = "tool_use"


BUGGY_LINE = "    return a - b  # BUG: should be a + b"
FIXED_LINE = "    return a + b"


def fix_patch() -> str:
    """The unified diff the scripted model 'writes', built from the real file.

    Generated rather than hand-written: ``git apply`` runs underneath and
    is strict about hunk ranges and context, and the committed task file
    is the only reliable source for both.
    """
    source = (
        REPO_ROOT / "eval" / "tasks" / TASK_ID / "workspace" / "calculator.py"
    ).read_text(encoding="utf-8")
    fixed = source.replace(BUGGY_LINE, FIXED_LINE)
    if fixed == source:
        raise SystemExit(
            "the task workspace no longer contains the line this script "
            "patches; update BUGGY_LINE"
        )
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
    "summary": "fix add() to return a + b",
    "steps": [{"action": "patch", "target": "calculator.py", "why": "add() subtracts"}],
    "expected_files_changed": ["calculator.py"],
    "tests_to_run": ["tests/test_calculator.py"],
}


def scripted_harness() -> BaselineHarness:
    """A real ``BaselineHarness`` whose model turns come from ``SCRIPT``.

    Built with ``__new__`` so the base ``__init__``'s API-key requirement
    is bypassed and no client is ever constructed: this issues no network
    call of any kind. The instance keeps every override point the real
    baseline has, which is what makes the replay meaningful — replay
    reconstructs the same class and recomputes the same routing.
    """
    harness = BaselineHarness.__new__(BaselineHarness)
    harness.api_key = None
    harness._client = None
    harness.MODEL = "scripted-offline"
    turn = {"n": 0}

    async def _call_llm(
        messages: list[Any],
        tools: list[Any],
        *,
        tool_choice: dict[str, Any] | None = None,
    ) -> _Response:
        if tool_choice is not None:
            return _Response(
                [
                    _Block(
                        type="tool_use",
                        id="plan-1",
                        name="submit_plan",
                        input=dict(PLAN),
                    )
                ]
            )
        index = min(turn["n"], len(SCRIPT) - 1)
        turn["n"] += 1
        return _Response([_Block(**block) for block in SCRIPT[index]])

    harness._call_llm = _call_llm
    return harness


async def main() -> int:
    if not await healthcheck():
        print(
            "Postgres is not reachable. A recording without a checkpoint "
            "history cannot be replayed from a checkpoint.\n"
            "  docker compose -f infra/docker-compose.yml up -d postgres",
            file=sys.stderr,
        )
        return 1

    task_dir = REPO_ROOT / "eval" / "tasks" / TASK_ID
    task_spec = json.loads((task_dir / "task.json").read_text())

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
        with sandbox_for(task_dir / "workspace") as sandbox:
            result = await replay_mod.record_inner_execution(
                harness_factory=scripted_harness,
                task_dict=task_spec,
                workspace=sandbox,
                thread_id=thread_id,
                checkpointer=saver,
                recording_dir=recording_dir,
                manifest_extra={
                    "recorded_with": (
                        "scripted model turns; no provider call was made. The "
                        "graph, the six tools, the pytest verify subprocess "
                        "and the Postgres checkpoints are real."
                    )
                },
            )

    print(json.dumps({k: v for k, v in result.items() if k != "final_state"}, indent=2))
    print(f"\nscore: {result['final_state'].get('score')}")
    print(f"recording: {recording_dir}")
    print(f"\nnext: uv run meta-harness verify-replay runs/{RUN_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
