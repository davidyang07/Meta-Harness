"""Candidate source snapshotting + isolated harness loading.

The proposer authors a candidate at ``agents/<label>.py`` in the repo
root — that is what ``skills/meta-harness-coding-agent/SKILL.md`` tells
it to do, and keeping the file there is what makes candidate diffs
inspectable with ordinary tooling.

That shared location is not safe to *benchmark* from. Two branches
forked from the same checkpoint reach the same iteration with the same
default label, and whichever proposer finishes last wins the file. The
loser would then be benchmarked against source it never wrote.

So: immediately after propose, snapshot the authored file into the
branch's own ``runs/<run>/threads/<thread>/agents/<candidate>.py`` and
load the harness class from *that* path. The snapshot is the branch's
private, immutable record of what it actually evaluated.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

from app.meta_harness import runs as runs_mod
from app.meta_harness.harness import CodingAgentHarness


class CandidateSourceError(RuntimeError):
    """Raised when a candidate's source cannot be located or loaded."""


def authored_source_path(repo_root: Path, label: str) -> Path:
    """Where the proposer is told to author a candidate."""
    safe = runs_mod.validate_artifact_name(label, kind="candidate")
    return repo_root / "agents" / f"{safe}.py"


def snapshot_candidate_source(
    *,
    repo_root: Path,
    run_dir: Path,
    thread_id: str,
    candidate_name: str,
    label: str,
) -> Path:
    """Copy the authored candidate file into the branch's private dir.

    Returns the snapshot path. Raises ``CandidateSourceError`` if the
    proposer claimed a candidate it never wrote.
    """
    src = authored_source_path(repo_root, label)
    if not src.is_file():
        raise CandidateSourceError(
            f"proposer registered candidate {label!r} but {src} does not exist"
        )
    dest = runs_mod.candidate_source_path(run_dir, thread_id, candidate_name)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def source_sha256(path: Path) -> str:
    """Hex SHA-256 of a candidate source file, for benchmark provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_harness_from_source(source_path: Path, class_name: str) -> type:
    """Import ``class_name`` from an arbitrary .py file, in isolation.

    Each load gets a unique synthetic module name, so two branches whose
    candidates share a class name never collide in ``sys.modules`` and a
    stale cache entry can never shadow a fresh snapshot.
    """
    if not source_path.is_file():
        raise CandidateSourceError(f"candidate source not found: {source_path}")
    module_name = f"_mh_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise CandidateSourceError(f"cannot build import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise CandidateSourceError(
            f"{source_path} does not define {class_name!r}"
        ) from exc


def load_harness_class(candidate: dict[str, Any], *, repo_root: Path) -> type:
    """Resolve a candidate dict to its harness class.

    Prefers the branch-private ``source_path`` snapshot. Falls back to
    the declared ``import_path`` module only when no snapshot was taken
    (e.g. the committed ``agents.baseline`` root candidate).
    """
    _, _, class_name = str(candidate["import_path"]).partition(":")
    # Candidates subclass BaselineHarness / CodingAgentHarness by
    # ordinary import, so the repo root has to be importable regardless
    # of which branch below we take.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    source_path = candidate.get("source_path")
    if source_path:
        return load_harness_from_source(Path(source_path), class_name)

    module_path = str(candidate["import_path"]).partition(":")[0]
    sys.modules.pop(module_path, None)
    importlib.invalidate_caches()
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def assert_is_harness(cls: type, *, import_path: str) -> None:
    """Validate that a loaded class is usable as an inner-loop harness."""
    if not isinstance(cls, type) or not issubclass(cls, CodingAgentHarness):
        raise TypeError(f"{import_path} is not a CodingAgentHarness subclass")
