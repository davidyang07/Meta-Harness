"""Candidate source snapshotting and isolated harness loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.meta_harness import candidates as cand
from app.meta_harness import runs as runs_mod

HARNESS_SRC = textwrap.dedent(
    '''
    from app.meta_harness.harness import CodingAgentHarness


    class {name}(CodingAgentHarness):
        MARKER = {marker!r}
        MAX_ACT_TURNS = {turns}
    '''
).lstrip()


def _authored(repo_root: Path, label: str, *, marker: str, turns: int = 25) -> Path:
    agents = repo_root / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    path = agents / f"{label}.py"
    path.write_text(HARNESS_SRC.format(name="Cand", marker=marker, turns=turns))
    return path


def test_snapshot_captures_the_source_at_propose_time(tmp_path: Path):
    repo_root = tmp_path / "repo"
    run_dir = runs_mod.make_run_dir(repo_root, "run-a", fresh=True)
    _authored(repo_root, "cand", marker="original")

    snapshot = cand.snapshot_candidate_source(
        repo_root=repo_root,
        run_dir=run_dir,
        thread_id="run-a",
        candidate_name="cand",
        label="cand",
    )
    assert "original" in snapshot.read_text()

    # A concurrent branch rewriting the authored file cannot change what
    # this branch already captured.
    _authored(repo_root, "cand", marker="clobbered")
    assert "original" in snapshot.read_text()
    assert cand.source_sha256(snapshot)


def test_two_branches_snapshot_to_separate_files(tmp_path: Path):
    repo_root = tmp_path / "repo"
    run_dir = runs_mod.make_run_dir(repo_root, "run-a", fresh=True)
    _authored(repo_root, "cand", marker="left")
    left = cand.snapshot_candidate_source(
        repo_root=repo_root,
        run_dir=run_dir,
        thread_id="run-a",
        candidate_name="cand__left",
        label="cand",
    )
    _authored(repo_root, "cand", marker="right")
    right = cand.snapshot_candidate_source(
        repo_root=repo_root,
        run_dir=run_dir,
        thread_id="run-a.fork.beef",
        candidate_name="cand__right",
        label="cand",
    )
    assert left != right
    assert "left" in left.read_text()
    assert "right" in right.read_text()


def test_snapshot_reports_a_candidate_the_proposer_never_wrote(tmp_path: Path):
    repo_root = tmp_path / "repo"
    run_dir = runs_mod.make_run_dir(repo_root, "run-a", fresh=True)
    with pytest.raises(cand.CandidateSourceError, match="never-written"):
        cand.snapshot_candidate_source(
            repo_root=repo_root,
            run_dir=run_dir,
            thread_id="run-a",
            candidate_name="never-written",
            label="never-written",
        )


def test_snapshot_tolerates_a_briefly_unreadable_file(tmp_path: Path, monkeypatch):
    """A just-written file can momentarily fail to stat on synced volumes."""
    repo_root = tmp_path / "repo"
    run_dir = runs_mod.make_run_dir(repo_root, "run-a", fresh=True)
    src = _authored(repo_root, "cand", marker="ok")

    real_read = Path.read_text
    calls = {"n": 0}

    def flaky_read(self, *args, **kwargs):
        if self == src and calls["n"] < 2:
            calls["n"] += 1
            raise OSError("transient")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read)
    snapshot = cand.snapshot_candidate_source(
        repo_root=repo_root,
        run_dir=run_dir,
        thread_id="run-a",
        candidate_name="cand",
        label="cand",
    )
    monkeypatch.undo()
    assert calls["n"] == 2, "expected the transient failures to be retried"
    assert "ok" in snapshot.read_text()


def test_loading_two_snapshots_with_the_same_class_name_does_not_collide(
    tmp_path: Path,
):
    """Branch A's harness must never shadow branch B's in sys.modules."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text(HARNESS_SRC.format(name="Cand", marker="a", turns=11))
    b.write_text(HARNESS_SRC.format(name="Cand", marker="b", turns=22))

    cls_a = cand.load_harness_from_source(a, "Cand")
    cls_b = cand.load_harness_from_source(b, "Cand")

    assert cls_a is not cls_b
    assert (cls_a.MARKER, cls_a.MAX_ACT_TURNS) == ("a", 11)
    assert (cls_b.MARKER, cls_b.MAX_ACT_TURNS) == ("b", 22)
    cand.assert_is_harness(cls_a, import_path="a.py:Cand")


def test_load_harness_class_prefers_the_branch_snapshot(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    snapshot = tmp_path / "snap.py"
    snapshot.write_text(HARNESS_SRC.format(name="Snap", marker="snap", turns=7))

    cls = cand.load_harness_class(
        {"import_path": "agents.whatever:Snap", "source_path": str(snapshot)},
        repo_root=repo_root,
    )
    assert cls.MAX_ACT_TURNS == 7


def test_load_harness_class_falls_back_to_the_committed_module():
    repo_root = Path(__file__).resolve().parents[2]
    cls = cand.load_harness_class(
        {"import_path": "agents.baseline:BaselineHarness", "source_path": None},
        repo_root=repo_root,
    )
    assert cls.__name__ == "BaselineHarness"
    cand.assert_is_harness(cls, import_path="agents.baseline:BaselineHarness")


def test_missing_class_in_a_snapshot_is_reported(tmp_path: Path):
    src = tmp_path / "s.py"
    src.write_text("X = 1\n")
    with pytest.raises(cand.CandidateSourceError, match="does not define"):
        cand.load_harness_from_source(src, "Nope")


def test_non_harness_class_is_rejected():
    class NotAHarness:
        pass

    with pytest.raises(TypeError, match="not a CodingAgentHarness"):
        cand.assert_is_harness(NotAHarness, import_path="x:NotAHarness")
