"""Candidate source snapshotting + isolated harness loading.

The proposer authors a candidate at ``agents/<label>.py`` — that is what
``skills/meta-harness-coding-agent/SKILL.md`` tells it to do. Where that
``agents/`` directory is depends on the proposer:

- the real proposer runs in its own private staged copy of the
  repository (see ``proposer.stage_repository``), so its file is
  authored there and never touches the shared repository;
- the mock proposer writes into the repository's ``agents/`` under a
  branch-unique label.

Neither location is what gets *benchmarked*. Two branches forked from
the same checkpoint reach the same iteration, often with the same label,
and a shared file goes to whichever proposer finished last. So
immediately after propose, the authored file is snapshotted into the
branch's own ``runs/<run>/threads/<thread>/agents/<candidate>.py`` and
the harness class is loaded from *that* path. The snapshot is the
branch's private, immutable record of what it actually evaluated.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from app.meta_harness import runs as runs_mod
from app.meta_harness.harness import CodingAgentHarness


class CandidateSourceError(RuntimeError):
    """Raised when a candidate's source cannot be located or loaded."""


class CandidateContractError(CandidateSourceError):
    """Raised when a candidate's source reaches past its override points."""


#: Framework modules a candidate may import. Everything else under
#: ``app.meta_harness`` — the tools, the grader, the sandbox, the effects
#: boundary, the graph builders — is the evaluator's side of the contract.
_ALLOWED_FRAMEWORK_MODULES = frozenset({"app.meta_harness.harness"})

#: Modules whose only use in a harness would be to reach into the running
#: process: rebinding modules, importing by string, patching builtins.
_FORBIDDEN_MODULES = frozenset({"sys", "importlib", "builtins", "ctypes", "gc"})

#: Builtins that execute or rebind code by name.
_FORBIDDEN_CALLS = frozenset({"exec", "eval", "compile", "__import__"})


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def check_candidate_source(source: str, *, origin: str = "<candidate>") -> None:
    """Reject candidate source that reaches past the 11 override points.

    A candidate is Python loaded into the backend process, so the tool
    contract and the grader are only fixed if a candidate cannot rebind
    them. This is a static check of the realistic ways to do that —
    importing the evaluator's modules, assigning into any imported
    module, ``setattr`` on something other than ``self``/``cls``,
    executing code by string — run before the module is executed. It is
    a guard against a proposer that writes a monkeypatch, not a sandbox
    against an adversary; grading in a pristine copy is what keeps the
    agent itself from moving the score.
    """
    try:
        tree = ast.parse(source, filename=origin)
    except SyntaxError as exc:
        raise CandidateSourceError(f"{origin}: {exc}") from exc

    imported: set[str] = set()
    problems: list[str] = []

    def _module_allowed(name: str) -> bool:
        top = name.split(".", 1)[0]
        if top in _FORBIDDEN_MODULES:
            return False
        if name == "app" or name.startswith("app."):
            return name in _ALLOWED_FRAMEWORK_MODULES
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _module_allowed(alias.name):
                    problems.append(f"line {node.lineno}: imports {alias.name}")
                imported.add((alias.asname or alias.name).split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and not _module_allowed(module):
                problems.append(f"line {node.lineno}: imports from {module}")
            for alias in node.names:
                if node.level == 0 and module == "app.meta_harness" and (
                    f"app.meta_harness.{alias.name}" not in _ALLOWED_FRAMEWORK_MODULES
                ):
                    problems.append(
                        f"line {node.lineno}: imports app.meta_harness.{alias.name}"
                    )
                imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _FORBIDDEN_CALLS:
                problems.append(f"line {node.lineno}: calls {name}()")
            elif name in {"setattr", "delattr"} and node.args:
                if _root_name(node.args[0]) not in {"self", "cls"}:
                    problems.append(
                        f"line {node.lineno}: {name}() on something other than self/cls"
                    )

    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, (ast.Attribute, ast.Subscript)):
                    root = _root_name(sub)
                    if root in imported:
                        problems.append(
                            f"line {node.lineno}: assigns into imported name {root!r}"
                        )
                        break

    if problems:
        raise CandidateContractError(
            f"{origin} reaches past the harness override points: "
            + "; ".join(sorted(set(problems)))
        )


def authored_source_path(repo_root: Path, label: str) -> Path:
    """Where the proposer is told to author a candidate."""
    safe = runs_mod.validate_artifact_name(label, kind="candidate")
    return repo_root / "agents" / f"{safe}.py"


#: How long to keep retrying a read of a just-authored candidate file.
#: The proposer writes into the repo's ``agents/`` directory, which on a
#: developer machine is often inside a syncing folder (OneDrive, Dropbox)
#: and on CI is shared with concurrently running branches. A file written
#: microseconds ago can briefly fail to stat. Losing a candidate to that
#: would waste a whole proposer invocation.
_SNAPSHOT_READ_TIMEOUT_S = 2.0
_SNAPSHOT_READ_INTERVAL_S = 0.05


def _read_authored_source(src: Path, label: str) -> str:
    """Read a just-authored candidate file, tolerating a brief delay.

    Retries only transient failures. If the file is still unreadable
    when the budget runs out, the error names the path and the label so
    "the proposer never wrote it" stays diagnosable.
    """
    deadline = time.monotonic() + _SNAPSHOT_READ_TIMEOUT_S
    last_error: OSError | None = None
    while True:
        try:
            return src.read_text(encoding="utf-8")
        except OSError as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(_SNAPSHOT_READ_INTERVAL_S)
    raise CandidateSourceError(
        f"proposer registered candidate {label!r} but {src} was not readable "
        f"after {_SNAPSHOT_READ_TIMEOUT_S}s ({type(last_error).__name__}: "
        f"{last_error})"
    )


def snapshot_candidate_source(
    *,
    repo_root: Path,
    run_dir: Path,
    thread_id: str,
    candidate_name: str,
    label: str,
) -> Path:
    """Copy the authored candidate file into the branch's private dir.

    ``repo_root`` is the root the proposer authored into: the real
    proposer's staged copy, or the repository for the mock. Returns the
    snapshot path. Raises ``CandidateSourceError`` if the proposer
    claimed a candidate it never wrote.
    """
    src = authored_source_path(repo_root, label)
    source = _read_authored_source(src, label)
    dest = runs_mod.candidate_source_path(run_dir, thread_id, candidate_name)
    dest.write_text(source, encoding="utf-8")
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
    check_candidate_source(
        source_path.read_text(encoding="utf-8"), origin=str(source_path)
    )
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
