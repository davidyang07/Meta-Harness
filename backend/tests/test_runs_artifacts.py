"""Thread-scoped run artifact helpers.

These cover the storage layer that makes concurrent branches safe: two
threads must be able to write the same *logical* artifact (pending eval,
frontier, evolution row, candidate traces) without touching each other's
files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.meta_harness import runs as runs_mod


def _run(tmp_path: Path, name: str = "run-a") -> Path:
    return runs_mod.make_run_dir(tmp_path, name, fresh=True)


def test_make_run_dir_creates_root_thread(tmp_path: Path):
    run_dir = _run(tmp_path)
    assert (run_dir / "threads" / "run-a" / "thread.json").exists()
    assert runs_mod.thread_id_for_dir(run_dir / "threads" / "run-a") == "run-a"


def test_thread_slug_is_stable_and_filesystem_safe():
    assert runs_mod.thread_slug("run-a") == "run-a"
    # Fork thread ids are dotted but still name-safe.
    assert runs_mod.thread_slug("run-a.fork.ab12cd34") == "run-a.fork.ab12cd34"
    # Over-long ids collapse to a stable hash slug.
    long_id = "run-" + ("x" * 300)
    slug = runs_mod.thread_slug(long_id)
    assert slug.startswith("t-") and len(slug) == 34
    assert slug == runs_mod.thread_slug(long_id)
    assert runs_mod.thread_slug(long_id + "!") != slug


def test_pending_eval_is_isolated_per_thread(tmp_path: Path):
    run_dir = _run(tmp_path)
    runs_mod.write_pending_eval(run_dir, "run-a", {"iteration": 3, "who": "root"})
    runs_mod.write_pending_eval(
        run_dir, "run-a.fork.beef", {"iteration": 3, "who": "fork"}
    )

    assert runs_mod.read_pending_eval(run_dir, "run-a")["who"] == "root"
    assert runs_mod.read_pending_eval(run_dir, "run-a.fork.beef")["who"] == "fork"


def test_frontier_is_isolated_per_thread(tmp_path: Path):
    run_dir = _run(tmp_path)
    runs_mod.write_frontier(run_dir, "run-a", {"iteration": 1, "_best": {"name": "a"}})
    runs_mod.write_frontier(
        run_dir, "run-a.fork.beef", {"iteration": 1, "_best": {"name": "b"}}
    )

    assert runs_mod.read_frontier(run_dir, "run-a")["_best"]["name"] == "a"
    assert runs_mod.read_frontier(run_dir, "run-a.fork.beef")["_best"]["name"] == "b"

    frontiers = runs_mod.aggregate_frontiers(run_dir)
    assert set(frontiers) == {"run-a", "run-a.fork.beef"}


def test_evolution_rows_are_isolated_and_tagged(tmp_path: Path):
    run_dir = _run(tmp_path)
    runs_mod.record_evolution_row(run_dir, "run-a", {"iteration": 2, "candidate": "x"})
    runs_mod.record_evolution_row(
        run_dir, "run-a.fork.beef", {"iteration": 2, "candidate": "y"}
    )

    root = runs_mod.read_evolution_summary(run_dir, "run-a")
    fork = runs_mod.read_evolution_summary(run_dir, "run-a.fork.beef")
    assert [r["candidate"] for r in root] == ["x"]
    assert [r["candidate"] for r in fork] == ["y"]
    assert root[0]["thread_id"] == "run-a"
    assert fork[0]["thread_id"] == "run-a.fork.beef"

    merged = runs_mod.aggregate_evolution_rows(run_dir)
    assert {(r["candidate"], r["thread_id"]) for r in merged} == {
        ("x", "run-a"),
        ("y", "run-a.fork.beef"),
    }


def test_proposer_sessions_do_not_collide_on_same_iteration(tmp_path: Path):
    run_dir = _run(tmp_path)
    a = runs_mod.proposer_session_dir(run_dir, "run-a", 3)
    b = runs_mod.proposer_session_dir(run_dir, "run-a.fork.beef", 3)
    assert a != b
    (a / "session.json").write_text('{"who": "root"}')
    (b / "session.json").write_text('{"who": "fork"}')
    assert json.loads((a / "session.json").read_text())["who"] == "root"


def test_candidate_dirs_and_sources_do_not_collide(tmp_path: Path):
    run_dir = _run(tmp_path)
    a = runs_mod.candidate_dir(run_dir, "run-a", "cand")
    b = runs_mod.candidate_dir(run_dir, "run-a.fork.beef", "cand")
    assert a != b

    sa = runs_mod.candidate_source_path(run_dir, "run-a", "cand")
    sb = runs_mod.candidate_source_path(run_dir, "run-a.fork.beef", "cand")
    assert sa != sb
    sa.write_text("ROOT")
    sb.write_text("FORK")
    assert sa.read_text() == "ROOT"
    assert sb.read_text() == "FORK"


def test_qualify_candidate_name_is_unique_per_branch():
    root = runs_mod.qualify_candidate_name("_mock_iter_2", "run-a", "run-a")
    fork = runs_mod.qualify_candidate_name("_mock_iter_2", "run-a.fork.beef", "run-a")
    other = runs_mod.qualify_candidate_name("_mock_iter_2", "run-a.fork.cafe", "run-a")

    assert root == "_mock_iter_2"
    assert fork != root and other != root and fork != other
    assert fork.startswith("_mock_iter_2__")
    # Stable across calls so re-reads resolve the same artifacts.
    assert fork == runs_mod.qualify_candidate_name(
        "_mock_iter_2", "run-a.fork.beef", "run-a"
    )


def test_qualify_candidate_name_is_idempotent():
    """A proposer that already branch-qualified its label is left alone.

    The mock proposer suffixes its authored agents/<label>.py so two
    forks do not clobber each other's file. Re-qualifying that label
    produced names like _mock_iter_2__abc12345__abc12345.
    """
    once = runs_mod.qualify_candidate_name("_mock_iter_2", "run-a.fork.beef", "run-a")
    twice = runs_mod.qualify_candidate_name(once, "run-a.fork.beef", "run-a")
    assert twice == once
    assert once.count("__") == 1

    # A different branch still qualifies it, since the suffix differs.
    other = runs_mod.qualify_candidate_name(once, "run-a.fork.cafe", "run-a")
    assert other != once and other.startswith(once + "__")


def test_find_candidate_dir_searches_all_threads(tmp_path: Path):
    run_dir = _run(tmp_path)
    runs_mod.candidate_dir(run_dir, "run-a.fork.beef", "only-on-fork")
    found = runs_mod.find_candidate_dir(run_dir, "only-on-fork")
    assert found is not None
    assert found.parent.parent.name == "run-a.fork.beef"
    assert runs_mod.find_candidate_dir(run_dir, "nope") is None


def test_artifact_names_reject_path_traversal(tmp_path: Path):
    run_dir = _run(tmp_path)
    with pytest.raises(ValueError):
        runs_mod.candidate_dir(run_dir, "run-a", "../escape")
    with pytest.raises(ValueError):
        runs_mod.make_run_path(tmp_path, "../escape")
    # Traversal in a thread id is neutralised by slugging, not an error.
    d = runs_mod.thread_dir(run_dir, "../escape")
    assert d.parent == (run_dir / "threads")


def test_write_json_atomic_leaves_no_partial_file(tmp_path: Path):
    target = tmp_path / "nested" / "out.json"
    runs_mod.write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    runs_mod.write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    # No temp leftovers next to the target.
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]


def test_evolution_rows_are_idempotent_on_reexecution(tmp_path: Path):
    """A re-run LangGraph node replaces its row instead of duplicating it.

    A node interrupted after its side effects but before its checkpoint
    commits is re-executed on resume. Plain appending produced a second
    row for the same iteration, which surfaced as
    `duplicate iterations in summary: [0, 1, 2, 3, 3]`.
    """
    run_dir = _run(tmp_path)
    runs_mod.record_evolution_row(
        run_dir, "run-a", {"iteration": 3, "candidate": "c", "delta": 0.1}
    )
    runs_mod.record_evolution_row(
        run_dir, "run-a", {"iteration": 3, "candidate": "c", "delta": 0.2}
    )

    rows = runs_mod.read_evolution_summary(run_dir, "run-a")
    assert len(rows) == 1
    # The rerun is authoritative, so its value wins.
    assert rows[0]["delta"] == 0.2


def test_evolution_rows_keep_distinct_candidates_on_one_iteration(tmp_path: Path):
    """Idempotency keys on (iteration, candidate), not iteration alone."""
    run_dir = _run(tmp_path)
    for name in ("a", "b"):
        runs_mod.record_evolution_row(
            run_dir, "run-a", {"iteration": 2, "candidate": name}
        )
    rows = runs_mod.read_evolution_summary(run_dir, "run-a")
    assert [r["candidate"] for r in rows] == ["a", "b"]


def test_replacing_a_row_preserves_order_of_the_others(tmp_path: Path):
    run_dir = _run(tmp_path)
    for i in (0, 1, 2):
        runs_mod.record_evolution_row(
            run_dir, "run-a", {"iteration": i, "candidate": f"c{i}"}
        )
    runs_mod.record_evolution_row(
        run_dir, "run-a", {"iteration": 1, "candidate": "c1", "delta": 9}
    )
    rows = runs_mod.read_evolution_summary(run_dir, "run-a")
    assert [r["iteration"] for r in rows] == [0, 2, 1]
    assert rows[-1]["delta"] == 9
    # No temp files left behind by the atomic rewrite.
    td = runs_mod.thread_dir(run_dir, "run-a")
    assert not [p for p in td.iterdir() if p.name.startswith(".")]
