"""The run version graph: commits, refs, isolation, immutability.

"Git-style versioning in PostgreSQL" is four separate claims, and these
tests take them one at a time:

- checkpoint identity is immutable, and re-reading proves it;
- a checkpoint records its parent, so the history is a DAG rather than a
  list;
- a branch is a ref with a fork point, and its history survives a
  process restart;
- each branch has its own working tree, so two branches forked from one
  checkpoint cannot overwrite each other's candidate source.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import branches as br  # noqa: E402
from app.meta_harness import replay as replay_mod  # noqa: E402
from app.meta_harness import runs as runs_mod  # noqa: E402
from app.meta_harness import versioning as ver  # noqa: E402
from app.meta_harness.outer import OuterLoopRunner, initial_state  # noqa: E402
from app.meta_harness.persistence import healthcheck, persistence_layer  # noqa: E402
from tests.conftest import unique_name as unique  # noqa: E402


def _forkable(versions: list[ver.VersionNode]) -> list[ver.VersionNode]:
    """Checkpoints that carry run state.

    A thread's very first checkpoint is written before any node has run
    and holds no state; forking from it would start a branch with an
    empty state. Every checkpoint a node produced is a valid ref.
    """
    return [
        v
        for v in versions
        if v.node not in (None, "fork") and v.values_summary.get("iteration") is not None
    ]


# ── pure diffing (no Postgres needed) ─────────────────────────────────


def test_diff_reports_added_removed_and_changed_keys():
    before = {"iteration": 1, "best_candidate": "a", "gone": True}
    after = {"iteration": 2, "best_candidate": "a", "added": 1}

    diff = ver.diff_states(before, after)

    assert diff["added"] == ["added"]
    assert diff["removed"] == ["gone"]
    assert [c["key"] for c in diff["changed"]] == ["iteration"]
    assert diff["identical"] is False


def test_diff_of_identical_states_is_empty():
    state = {"iteration": 3, "candidates": [{"name": "a"}]}
    diff = ver.diff_states(dict(state), dict(state))
    assert diff["identical"] is True
    assert diff["changed"] == []
    assert diff["before_sha256"] == diff["after_sha256"]


def test_diff_detects_a_nested_change_without_dumping_the_value():
    before = {"candidates": [{"name": "a", "scores": {"accuracy": 0.5}}]}
    after = {"candidates": [{"name": "a", "scores": {"accuracy": 0.6}}]}

    diff = ver.diff_states(before, after)

    changed = diff["changed"][0]
    assert changed["key"] == "candidates"
    assert changed["before_sha256"] != changed["after_sha256"]
    # The diff carries hashes, not the nested payload.
    assert set(changed) == {"key", "before_sha256", "after_sha256"}


# ── branch working trees ──────────────────────────────────────────────


def test_branch_artifacts_report_each_thread_separately(tmp_path: Path):
    run_dir = runs_mod.make_run_dir(tmp_path, "vg-artifacts", fresh=True)
    for thread_id, source in (
        ("vg-artifacts", "class A: pass\n"),
        ("vg-artifacts.fork.aaaa", "class A: pass  # forked\n"),
    ):
        path = runs_mod.candidate_source_path(run_dir, thread_id, "iter_1")
        path.write_text(source, encoding="utf-8")

    artifacts = ver.branch_artifacts(run_dir)

    assert set(artifacts) == {"vg-artifacts", "vg-artifacts.fork.aaaa"}
    root = artifacts["vg-artifacts"]["candidate_sources"]["iter_1"]
    fork = artifacts["vg-artifacts.fork.aaaa"]["candidate_sources"]["iter_1"]
    assert root["path"] != fork["path"]
    assert root["sha256"] != fork["sha256"], (
        "two branches proposing the same label must keep different source"
    )


def test_a_branch_cannot_write_into_another_branchs_source_path(tmp_path: Path):
    """The isolation is structural: the path itself is thread-scoped."""
    run_dir = runs_mod.make_run_dir(tmp_path, "vg-paths", fresh=True)
    a = runs_mod.candidate_source_path(run_dir, "thread-a", "same_label")
    b = runs_mod.candidate_source_path(run_dir, "thread-b", "same_label")

    assert a != b
    assert a.parent != b.parent
    a.write_text("A\n", encoding="utf-8")
    b.write_text("B\n", encoding="utf-8")
    assert a.read_text() == "A\n"
    assert b.read_text() == "B\n"


# ── Postgres-backed version graph ─────────────────────────────────────

_PG_OK = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
    healthcheck()
)

pg_only = pytest.mark.skipif(
    not _PG_OK,
    reason="Postgres not reachable at configured DSN; bring up via docker compose",
)


async def _seeded_run(tmp_path: Path, name: str, saver, *, budget: int = 2):
    run_dir = runs_mod.make_run_dir(tmp_path, name, fresh=True)
    runner = OuterLoopRunner(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=2,
        bench_workers=1,
        checkpointer=saver,
    )
    runs_mod.write_manifest(
        run_dir,
        run_id=name,
        thread_id=name,
        budget=budget,
        trials=2,
        mock_proposer=True,
        mock_bench=True,
    )
    seeds = [await runner.benchmark_baseline(run_id=name)]
    graph = runner.build()
    await graph.ainvoke(
        initial_state(run_id=name, budget=budget, seed_candidates=seeds),
        config={"configurable": {"thread_id": name}, "recursion_limit": 200},
    )
    return run_dir, graph


@pg_only
async def test_every_checkpoint_records_its_parent(tmp_path: Path):
    name = unique("vg-dag")
    async with persistence_layer() as saver:
        _, graph = await _seeded_run(tmp_path, name, saver)
        versions = await ver.thread_versions(graph, thread_id=name)

    assert len(versions) >= 4
    ids = [v.checkpoint_id for v in versions]
    assert len(set(ids)) == len(ids), "checkpoint ids must be unique"
    # Exactly one root; every other checkpoint points at a known parent.
    roots = [v for v in versions if v.parent_checkpoint_id is None]
    assert len(roots) == 1
    known = set(ids)
    for version in versions:
        if version.parent_checkpoint_id is not None:
            assert version.parent_checkpoint_id in known


@pg_only
async def test_a_stored_checkpoint_still_hashes_to_what_it_hashed_to(
    tmp_path: Path,
):
    """Immutability is the whole basis for replay and forking."""
    name = unique("vg-immutable")
    async with persistence_layer() as saver:
        _, graph = await _seeded_run(tmp_path, name, saver)
        versions = await ver.thread_versions(graph, thread_id=name)
        expected = {}
        for version in versions:
            state = await br.get_checkpoint_state(
                graph, thread_id=name, checkpoint_id=version.checkpoint_id
            )
            expected[version.checkpoint_id] = replay_mod.state_hash(state)

        result = await ver.verify_immutability(
            graph, thread_id=name, expected=expected
        )

    assert result["immutable"] is True
    assert result["checked"] == len(versions)
    assert all(r["ok"] for r in result["results"])


@pg_only
async def test_immutability_check_reports_a_mismatch(tmp_path: Path):
    """The check must be able to fail, or it proves nothing."""
    name = unique("vg-mismatch")
    async with persistence_layer() as saver:
        _, graph = await _seeded_run(tmp_path, name, saver)
        versions = await ver.thread_versions(graph, thread_id=name)
        result = await ver.verify_immutability(
            graph,
            thread_id=name,
            expected={versions[0].checkpoint_id: "0" * 64},
        )

    assert result["immutable"] is False
    assert result["results"][0]["ok"] is False


@pg_only
async def test_consecutive_checkpoints_differ_and_diff_explains_how(
    tmp_path: Path,
):
    name = unique("vg-diff")
    async with persistence_layer() as saver:
        _, graph = await _seeded_run(tmp_path, name, saver)
        versions = await ver.thread_versions(graph, thread_id=name)

        diff = await ver.diff_checkpoints(
            graph,
            thread_a=name,
            checkpoint_a=versions[0].checkpoint_id,
            thread_b=name,
            checkpoint_b=versions[-1].checkpoint_id,
        )
        same = await ver.diff_checkpoints(
            graph,
            thread_a=name,
            checkpoint_a=versions[-1].checkpoint_id,
            thread_b=name,
            checkpoint_b=versions[-1].checkpoint_id,
        )

    assert diff["identical"] is False
    assert diff["before_sha256"] != diff["after_sha256"]
    assert same["identical"] is True


@pg_only
async def test_the_version_graph_includes_branch_refs_and_fork_edges(
    tmp_path: Path,
):
    name = unique("vg-branches")
    br.clear_branch_state()
    br.set_runs_root(tmp_path / "runs")
    try:
        async with persistence_layer() as saver:
            run_dir, graph = await _seeded_run(tmp_path, name, saver)
            versions = await ver.thread_versions(graph, thread_id=name)
            candidates = _forkable(versions)
            fork_point = candidates[len(candidates) // 2].checkpoint_id

            metadata, task = await br.worktree_add(
                graph,
                run_id=name,
                parent_thread_id=name,
                parent_checkpoint_id=fork_point,
                name="vg-fork",
            )
            await task

            version = await ver.version_graph(graph, run_id=name)
    finally:
        await br.cancel_all_branches()
        br.clear_branch_state()
        br.set_runs_root(None)

    assert version["branch_count"] == 1
    assert metadata.thread_id in version["threads"]
    fork_edges = [e for e in version["edges"] if e["kind"] == "fork"]
    assert fork_edges and fork_edges[0]["from"] == fork_point
    assert fork_edges[0]["to"] == metadata.thread_id
    # The fork's own checkpoints are in the graph too.
    assert version["threads"][metadata.thread_id]


@pg_only
async def test_branch_history_survives_a_registry_reset(tmp_path: Path):
    """A restart loses the asyncio task, not the version history."""
    name = unique("vg-restart")
    fork_point: str | None = None
    br.clear_branch_state()
    br.set_runs_root(tmp_path / "runs")
    try:
        async with persistence_layer() as saver:
            _, graph = await _seeded_run(tmp_path, name, saver)
            versions = await ver.thread_versions(graph, thread_id=name)
            fork_point = _forkable(versions)[0].checkpoint_id
            metadata, task = await br.worktree_add(
                graph,
                run_id=name,
                parent_thread_id=name,
                parent_checkpoint_id=fork_point,
            )
            await task

            # Simulate a backend restart: in-process state is gone.
            br.branch_registry.clear()
            br.branch_metadata.clear()

            reloaded = await ver.version_graph(graph, run_id=name)
    finally:
        await br.cancel_all_branches()
        br.clear_branch_state()
        br.set_runs_root(None)

    threads = [b["thread_id"] for b in reloaded["branches"]]
    assert metadata.thread_id in threads
    restored = next(b for b in reloaded["branches"] if b["thread_id"] == metadata.thread_id)
    assert restored["parent_checkpoint_id"] == fork_point
    # A branch is never reported as live after a restart.
    assert restored["live"] is False


@pg_only
async def test_any_persisted_checkpoint_can_be_forked_from(tmp_path: Path):
    """Not just the newest one — every checkpoint a node wrote is a valid ref."""
    name = unique("vg-anyfork")
    br.clear_branch_state()
    br.set_runs_root(tmp_path / "runs")
    try:
        async with persistence_layer() as saver:
            _, graph = await _seeded_run(tmp_path, name, saver)
            versions = await ver.thread_versions(graph, thread_id=name)
            forkable = _forkable(versions)
            assert len(forkable) >= 3

            forked = []
            for version in forkable[:3]:
                metadata, task = await br.worktree_add(
                    graph,
                    run_id=name,
                    parent_thread_id=name,
                    parent_checkpoint_id=version.checkpoint_id,
                )
                await task
                forked.append((version.checkpoint_id, metadata))
    finally:
        await br.cancel_all_branches()
        br.clear_branch_state()
        br.set_runs_root(None)

    assert len(forked) == 3
    assert len({m.thread_id for _, m in forked}) == 3
    for checkpoint_id, metadata in forked:
        assert metadata.parent_checkpoint_id == checkpoint_id
