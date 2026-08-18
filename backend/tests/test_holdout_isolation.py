"""The proposer must never see the holdout tasks.

Holdout results are the only honest signal about whether an evolved
harness generalises or just fits the five search tasks. That signal is
worth exactly nothing if a holdout task's content reaches the proposer,
so these tests check every channel by which it could:

- the prompt and system prompt handed to the `claude` subprocess;
- the branch artifact directory the prompt points that subprocess at;
- the search task set the outer loop benchmarks against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import benchmark as bench  # noqa: E402
from app.meta_harness import proposer as prp  # noqa: E402
from app.meta_harness import runs as runs_mod  # noqa: E402

HOLDOUT_DIR = REPO_ROOT / "eval" / "holdout"
TASKS_DIR = REPO_ROOT / "eval" / "tasks"


def _holdout_task_ids() -> list[str]:
    return [d.name for d in bench.discover_tasks(HOLDOUT_DIR)]


def _holdout_instructions() -> list[str]:
    return [
        json.loads((d / "task.json").read_text())["instruction"]
        for d in bench.discover_tasks(HOLDOUT_DIR)
    ]


def test_the_holdout_set_is_disjoint_from_the_search_set():
    search = {d.name for d in bench.discover_tasks(TASKS_DIR)}
    holdout = set(_holdout_task_ids())
    assert search and holdout
    assert not (search & holdout), "a task appears in both sets"


def test_proposer_prompt_never_mentions_a_holdout_task(tmp_path: Path):
    run_dir = runs_mod.make_run_dir(tmp_path, "leak-check", fresh=True)
    prompt = prp._render_proposer_prompt(
        iteration=3,
        run_dir=run_dir,
        repo_root=tmp_path,
        parent_name="some-parent",
        thread_id="leak-check",
    )

    for task_id in _holdout_task_ids():
        assert task_id not in prompt, f"prompt leaks holdout task {task_id}"
    for instruction in _holdout_instructions():
        assert instruction[:60] not in prompt
    assert "holdout" not in prompt.lower()
    # It does point at the search set's artifacts, which is the point.
    assert "evolution_summary.jsonl" in prompt


def test_proposer_prompt_scopes_the_proposer_to_its_own_branch(tmp_path: Path):
    """The prompt must not hand a branch another branch's artifacts."""
    run_dir = runs_mod.make_run_dir(tmp_path, "leak-check", fresh=True)
    fork_prompt = prp._render_proposer_prompt(
        iteration=3,
        run_dir=run_dir,
        repo_root=tmp_path,
        parent_name="p",
        thread_id="leak-check.fork.beef",
    )
    root_prompt = prp._render_proposer_prompt(
        iteration=3,
        run_dir=run_dir,
        repo_root=tmp_path,
        parent_name="p",
        thread_id="leak-check",
    )
    assert "leak-check.fork.beef" in fork_prompt
    assert "leak-check.fork.beef" not in root_prompt
    assert fork_prompt != root_prompt


def test_the_branch_artifact_dir_the_proposer_reads_holds_no_holdout_data(
    tmp_path: Path,
):
    """The prompt points the subprocess at one directory. Check its contents.

    A mock run is enough: the question is whether the outer loop ever
    writes holdout material into the directory the proposer is told to
    read.
    """
    run_dir = runs_mod.make_run_dir(tmp_path, "leak-artifacts", fresh=True)
    thread = "leak-artifacts"
    runs_mod.write_pending_eval(run_dir, thread, {"iteration": 1, "candidates": []})
    runs_mod.write_frontier(run_dir, thread, {"iteration": 1, "candidates": []})
    runs_mod.record_evolution_row(
        run_dir, thread, {"iteration": 1, "candidate": "c"}
    )

    thread_root = runs_mod.thread_dir(run_dir, thread)
    blob = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in thread_root.rglob("*")
        if p.is_file()
    )
    for task_id in _holdout_task_ids():
        assert task_id not in blob


def test_the_outer_loop_benchmarks_only_the_search_set():
    """`--holdout` is a post-hoc evaluation, never part of the search."""
    from app.meta_harness.outer import OuterLoopRunner

    runner = OuterLoopRunner(
        run_dir=REPO_ROOT / "runs" / "unused",
        repo_root=REPO_ROOT,
        eval_tasks_dir=TASKS_DIR,
        mock_proposer=True,
        mock_bench=True,
        trials=1,
        bench_workers=1,
    )
    benchmarked = {d.name for d in runner._task_dirs()}
    assert benchmarked == {d.name for d in bench.discover_tasks(TASKS_DIR)}
    assert not (benchmarked & set(_holdout_task_ids()))


def test_holdout_results_are_labelled_so_they_cannot_be_read_as_search_scores():
    """Guard the tagging that keeps the two numbers apart."""
    rows = [
        {
            "task_id": "task-006-fix-recursion",
            "trial": 1,
            "passed": True,
            "score": 1.0,
            "wall_time_s": 1.0,
            "metrics_source": "measured",
            "llm_calls": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_tokens": 15,
            "cost_usd": 0.001,
        }
    ]
    result = bench.summarize(
        rows, task_ids=["task-006-fix-recursion"], trials=1
    )
    result["task_set"] = "holdout"
    assert result["task_set"] == "holdout"
    assert result["metrics_source"] == "measured"
    # And the search-set default is explicit rather than absent.
    assert bench.summarize(rows, task_ids=["t"], trials=1).get("task_set") is None


def test_published_results_pin_the_task_definitions_they_measured():
    """A result is only comparable against the tasks it actually ran.

    capture_environment() hashes every file of every task, so changing a
    task after publishing is detectable rather than silent.
    """
    from app.meta_harness import experiment as exp

    env = exp.capture_environment(
        repo_root=REPO_ROOT,
        model="test-model",
        tasks=bench.discover_tasks(TASKS_DIR),
        arm_sources={},
    )
    hashed = {t["task_id"]: t for t in env["tasks"]}
    assert set(hashed) == {d.name for d in bench.discover_tasks(TASKS_DIR)}
    for task in hashed.values():
        assert task["task_json_sha256"]
        assert task["task_sha256"]
        # Tests are part of the definition: a task whose tests changed is
        # a different task.
        assert any(
            "/tests/" in k for k in task["files_sha256"]
        ), f"{task['task_id']} hashed no test files"

    # Holdout tasks are not part of a search-set experiment's provenance.
    assert not (set(hashed) & set(_holdout_task_ids()))
