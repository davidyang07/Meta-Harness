"""Proposer node body — claude_wrapper.py-shaped.

Two paths:
- ``mock`` mode (BUILD_ORDER step 5): generates a deterministic stub
  candidate file for fast outer-loop testing, no LLM calls.
- ``claude`` mode (BUILD_ORDER step 6): spawns the ``claude`` CLI
  subprocess with the SKILL.md ``--append-system-prompt``'d, parses
  stream-json, and reads back the ``agents/<name>.py`` and
  ``pending_eval.json`` the proposer wrote. Mirrors Stanford's reference
  ``claude_wrapper.py`` shape.

This module is the body of the outer state machine's ``propose`` node
(per Correction 1 — the proposer is graph-internal, not a separate
tier).

**The real proposer never runs in the repository.** It runs
``--dangerously-skip-permissions`` with Bash, so a prompt instruction
is not isolation. Each invocation instead gets a fresh scratch copy of
the repository (``stage_repository``) that physically lacks the holdout
tasks, the published results, the evidence artifacts and every other
run's and branch's artifacts. Around that copy:

- deny rules are passed to the CLI for the holdout and results paths,
  in the copy and in the real repository;
- ``eval/`` and ``benchmarks/`` in the real repository are hashed before
  and after the session, and any change fails the iteration;
- every tool call in the session's stream-json is audited for holdout
  paths, holdout task ids and the real repository's absolute path, and
  a hit fails the iteration;
- only two files come back: the candidate source, straight into this
  branch's snapshot location, and ``pending_eval.json``, into this
  branch's thread directory. The candidate never touches the shared
  ``agents/`` directory, so a concurrent branch cannot overwrite what
  this branch benchmarks (AGENTS.md invariant 2).

Failures of the last three raise ``experiment.LeakageError``.

**Cancellation stops the whole process tree.** The CLI is started in its
own process group (a new session on POSIX, ``CREATE_NEW_PROCESS_GROUP``
on Windows). A timeout, or cancelling the asyncio task awaiting
``run_claude_proposer``, kills the group (``killpg`` / ``taskkill /T
/F``), so neither the CLI nor anything it spawned outlives the node.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.meta_harness import candidates as cand_mod
from app.meta_harness import runs as runs_mod
from app.meta_harness.experiment import LeakageError
from app.meta_harness.state import BASELINE_CANDIDATE_NAME


_MOCK_HARNESS_TEMPLATE = '''"""Mock candidate harness for outer-loop testing (iteration {iteration}).

Subclasses ``BaselineHarness`` with a hypothetical override that mock
benchmarking interprets as a pre-determined accuracy bump. Real
benchmark runs would actually exercise the override.

Authored as ``{label}``.
"""

from agents.baseline import BaselineHarness


class {class_name}(BaselineHarness):
    """Mock candidate. Hypothesis: {hypothesis}"""

    HYPOTHESIS = {hypothesis_repr}
    EXPECTED_DELTA = {expected_delta}
    LABEL = {label_repr}
'''


class ProposalError(ValueError):
    """The proposer's handoff is not one valid candidate for this iteration."""


class ProposerCancelled(RuntimeError):
    """The proposer subprocess was stopped because its node was cancelled."""


def mock_label(iteration: int, thread_id: str, run_id: str) -> str:
    """Branch-unique label for the mock proposer's authored file.

    Two forks that reach the same iteration would otherwise both author
    ``agents/_mock_iter_3.py`` and clobber each other in the shared repo
    root. The root thread keeps the plain label so single-branch runs
    stay readable.
    """
    base = f"_mock_iter_{iteration}"
    if thread_id == run_id:
        return base
    suffix = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:8]
    return f"{base}__{suffix}"


def clear_pending_eval(run_dir: Path, thread_id: str) -> None:
    """Delete this thread's ``pending_eval.json`` before a proposer runs.

    A proposer that exits 0 without writing one must fail, not silently
    re-submit the previous iteration's payload.
    """
    path = runs_mod.thread_dir(run_dir, thread_id) / "pending_eval.json"
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def mock_propose(
    *,
    run_dir: Path,
    thread_id: str,
    iteration: int,
    parent_name: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    """Generate a deterministic mock candidate for one branch.

    Writes the harness file under ``agents/<label>.py`` and this
    thread's ``pending_eval.json``. Returns the pending_eval payload.
    """
    label = mock_label(iteration, thread_id, run_dir.name)
    hypothesis = f"mock hypothesis #{iteration}: pretend we tweaked something"
    expected_delta = 0.05
    class_name = f"MockHarness_iter_{iteration}"

    harness_src = _MOCK_HARNESS_TEMPLATE.format(
        iteration=iteration,
        class_name=class_name,
        hypothesis=hypothesis,
        hypothesis_repr=repr(hypothesis),
        expected_delta=expected_delta,
        label=label,
        label_repr=repr(label),
    )

    agents_dir = repo_root / "agents"
    agents_dir.mkdir(exist_ok=True)
    harness_path = agents_dir / f"{label}.py"
    harness_path.write_text(harness_src, encoding="utf-8")

    payload: dict[str, Any] = {
        "iteration": iteration,
        "candidates": [
            {
                "name": label,
                "import_path": f"agents.{label}:{class_name}",
                "parent": parent_name,
                "hypothesis": hypothesis,
                "axis": "exploitation",
                "expected_score_delta": expected_delta,
            }
        ],
    }
    runs_mod.write_pending_eval(run_dir, thread_id, payload)

    sess_dir = runs_mod.proposer_session_dir(run_dir, thread_id, iteration)
    runs_mod.write_json_atomic(
        sess_dir / "session.json",
        {
            "mode": "mock",
            "thread_id": thread_id,
            "iteration": iteration,
            "exit_code": 0,
            "duration_seconds": 0.0,
            # No model was called: the cost is not zero, it does not exist.
            "cost_usd": None,
            "metrics_source": "mock",
            "files_written": {
                f"agents/{label}.py": {
                    "lines_written": harness_src.count("\n"),
                    "sha256": hashlib.sha256(harness_src.encode("utf-8")).hexdigest(),
                }
            },
        },
    )
    return payload


# ──────────────────────────────────────────────────────────────────────
# The handoff: one candidate, this iteration, a unique name, a real file.
# ──────────────────────────────────────────────────────────────────────


def _claimed_elsewhere(run_dir: Path, thread_id: str, name: str) -> bool:
    """Whether another thread of this run already owns candidate ``name``.

    This thread's own directory is ignored on purpose: a ``propose`` node
    re-executed on resume re-proposes into its own thread, and that is
    the same proposal, not a collision (AGENTS.md invariant 6).
    """
    own = runs_mod.thread_dir(run_dir, thread_id).resolve()
    for td in runs_mod.list_thread_dirs(run_dir):
        if td.resolve() == own:
            continue
        if (td / "candidates" / name).is_dir() or (td / "agents" / f"{name}.py").is_file():
            return True
    return False


def accept_proposal(
    payload: Any,
    *,
    iteration: int,
    run_dir: Path,
    thread_id: str,
    existing_names: set[str] | frozenset[str],
    source_root: Path,
) -> dict[str, Any]:
    """Validate a proposer handoff and snapshot its candidate source.

    The outer loop benchmarks exactly one candidate per iteration, so the
    handoff must name exactly one, for *this* iteration, under a name no
    other candidate in the run has, with a source file that exists under
    ``source_root/agents/``. Anything else raises ``ProposalError``
    rather than being half-used.

    ``source_root`` is where the proposer authored the file: the
    proposer's private staged copy for the real proposer, the repository
    for the mock. The source is copied from there into this branch's
    snapshot location, which is what gets benchmarked.

    Returns a copy of ``payload`` whose single candidate gains
    ``qualified_name`` (the run-unique name), ``source_path`` (the
    snapshot) and ``source_sha256``.
    """
    if not isinstance(payload, dict):
        raise ProposalError(f"pending_eval.json must be a JSON object, got {type(payload).__name__}")
    if payload.get("iteration") != iteration:
        raise ProposalError(
            f"pending_eval.json is for iteration {payload.get('iteration')!r}, "
            f"but this is iteration {iteration}"
        )
    entries = payload.get("candidates")
    if not isinstance(entries, list) or len(entries) != 1:
        count = len(entries) if isinstance(entries, list) else "no"
        raise ProposalError(
            f"pending_eval.json must register exactly one candidate; it registers {count}"
        )
    entry = entries[0]
    if not isinstance(entry, dict):
        raise ProposalError("the registered candidate must be a JSON object")

    label = str(entry.get("name") or "")
    try:
        runs_mod.validate_artifact_name(label, kind="candidate")
        qualified = runs_mod.qualify_candidate_name(label, thread_id, run_dir.name)
    except ValueError as exc:
        raise ProposalError(f"invalid proposer candidate: {exc}") from exc

    import_path = str(entry.get("import_path") or "")
    module, _, class_name = import_path.partition(":")
    if module != f"agents.{label}" or not class_name.isidentifier():
        raise ProposalError(
            f"import_path {import_path!r} must be 'agents.{label}:<ClassName>'"
        )

    if qualified in existing_names or _claimed_elsewhere(run_dir, thread_id, qualified):
        raise ProposalError(
            f"candidate name {qualified!r} is already used in run {run_dir.name}; "
            "every candidate needs a new name"
        )

    if not cand_mod.authored_source_path(source_root, label).is_file():
        raise ProposalError(
            f"pending_eval.json registers {label!r} but agents/{label}.py was not written"
        )
    snapshot = cand_mod.snapshot_candidate_source(
        repo_root=source_root,
        run_dir=run_dir,
        thread_id=thread_id,
        candidate_name=qualified,
        label=label,
    )
    accepted_entry = {
        **entry,
        "qualified_name": qualified,
        "source_path": str(snapshot),
        "source_sha256": cand_mod.source_sha256(snapshot),
    }
    return {**payload, "candidates": [accepted_entry]}


# ──────────────────────────────────────────────────────────────────────
# Real proposer: ``claude`` CLI subprocess, Stanford-shape.
# ──────────────────────────────────────────────────────────────────────


_PROPOSER_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]

#: Repo-relative paths the proposer must never read. They are absent
#: from its staged copy; these also feed the deny rules and the audit.
_HOLDOUT_PATHS = ("eval/holdout", "benchmarks/holdout")
_RESULTS_PATHS = ("benchmarks/results",)

#: Top-level entries never copied into the proposer's stage. ``runs`` is
#: rebuilt separately from just the directories this branch may read.
_STAGE_EXCLUDED_TOP = frozenset(
    {
        ".git",
        ".venv",
        ".claude",
        ".env",
        "frontend",
        "benchmark-results",
        "runs",
        "wandb",
        "graphify-out",
    }
)
#: Repo-relative directories never copied.
_STAGE_EXCLUDED_PATHS = frozenset(
    {*_HOLDOUT_PATHS, *_RESULTS_PATHS, "docs/evidence"}
)
#: Names never copied at any depth: dependency trees, caches, and the
#: pytest fallback temp root (which holds copies of tasks).
_STAGE_EXCLUDED_NAMES = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".pytest-tmp",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        ".git",
    }
)
#: Inside the thread directories that *are* staged: holdout post-evals
#: (``meta-harness loop --holdout`` writes them into the thread), tapes,
#: and old handoffs stay behind. Matched exactly, so a candidate whose
#: own name happens to contain "holdout" is still staged.
_THREAD_EXCLUDED_NAMES = frozenset(
    {"recordings", "pending_eval.json", "holdout-result.json", "holdout-traces"}
)


def _is_holdout_name(name: str) -> bool:
    return "holdout" in name.lower()


def _norm(text: str) -> str:
    """Case- and separator-insensitive form for path matching."""
    return text.replace("\\", "/").lower()


def holdout_task_ids(repo_root: Path) -> list[str]:
    """Every holdout task id: directories of the holdout set, plus any
    the committed holdout protocol names."""
    ids: set[str] = set()
    holdout_dir = repo_root / "eval" / "holdout"
    if holdout_dir.is_dir():
        ids.update(d.name for d in holdout_dir.iterdir() if d.is_dir())
    config_path = repo_root / "benchmarks" / "holdout" / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        ids.update(str(t) for t in config.get("tasks") or [])
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return sorted(i for i in ids if i)


def _thread_lineage(thread_id: str) -> list[str]:
    """``thread_id`` and every thread it was forked from, nearest first."""
    lineage = [thread_id]
    current = thread_id
    while ".fork." in current:
        current = current.rpartition(".fork.")[0]
        lineage.append(current)
    return lineage


def stage_repository(
    *,
    repo_root: Path,
    run_dir: Path,
    thread_id: str,
    extra_thread_dirs: list[Path] | None = None,
) -> Path:
    """Build the proposer's private scratch copy of the repository.

    Copied with ``shutil.copytree`` and an ignore filter, never links, so
    it behaves the same on Windows. Symlinks are skipped outright: a link
    could point back into the holdout set.

    Excluded: version control, environments, dependency trees, the
    frontend, the holdout tasks and protocol, published results, the
    evidence artifacts, ``.env``, and all of ``runs/`` except this run's
    lineage — this branch's thread directory and the directories of the
    threads it was forked from (where its parent's snapshot lives) —
    minus their holdout post-evaluations and tapes. Repo-relative paths
    in the prompt resolve the same way inside the copy.
    """
    real = repo_root.resolve()
    # Wherever the run directory lives, it is never copied wholesale:
    # only this branch's lineage is, below.
    run_rel = run_dir.resolve().relative_to(real).as_posix()
    stage = Path(tempfile.mkdtemp(prefix="mh-proposer-")).resolve()
    if real == stage or real in stage.parents:
        shutil.rmtree(stage, ignore_errors=True)
        raise RuntimeError(
            f"the proposer stage {stage} would sit inside the repository {real}; "
            "point TMPDIR (TEMP on Windows) outside the repository"
        )

    def _ignore(directory: str, names: list[str]) -> set[str]:
        here = Path(directory)
        try:
            rel = here.resolve().relative_to(real).as_posix()
        except ValueError:
            rel = ""
        rel = "" if rel == "." else rel
        ignored: set[str] = set()
        for name in names:
            child_rel = f"{rel}/{name}" if rel else name
            if (
                (not rel and name in _STAGE_EXCLUDED_TOP)
                or (not rel and name.startswith(".env"))
                or child_rel == run_rel
                or child_rel in _STAGE_EXCLUDED_PATHS
                or name in _STAGE_EXCLUDED_NAMES
                or name.endswith((".pyc", ".pyo"))
                or _is_holdout_name(name)
                or (here / name).is_symlink()
            ):
                ignored.add(name)
        return ignored

    try:
        shutil.copytree(real, stage, ignore=_ignore, dirs_exist_ok=True)

        def _thread_ignore(directory: str, names: list[str]) -> set[str]:
            return {
                name
                for name in names
                if name in _THREAD_EXCLUDED_NAMES
                or name in _STAGE_EXCLUDED_NAMES
                or (Path(directory) / name).is_symlink()
            }

        allowed = [runs_mod.thread_dir(run_dir, t) for t in _thread_lineage(thread_id)]
        allowed.extend(extra_thread_dirs or [])
        for source in dict.fromkeys(d.resolve() for d in allowed):
            if not source.is_dir():
                continue
            target = stage / source.relative_to(real)
            shutil.copytree(source, target, ignore=_thread_ignore, dirs_exist_ok=True)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return stage


def assert_stage_isolated(stage: Path, *, holdout_ids: list[str]) -> None:
    """Fail if anything holdout-shaped made it into the stage.

    The copy filter is the mechanism; this is the check that it held. It
    looks at path names — a directory named after a holdout task, or any
    holdout path — rather than file contents.
    """
    needles = {_norm(i) for i in holdout_ids}
    for path in stage.rglob("*"):
        rel = _norm(path.relative_to(stage).as_posix())
        parts = set(rel.split("/"))
        if needles & parts or any(rel.startswith(p) for p in (*_HOLDOUT_PATHS, *_RESULTS_PATHS)):
            raise LeakageError(f"holdout material reached the proposer stage: {rel}")


def _tree_digests(root: Path) -> dict[str, str]:
    """``relative path -> sha256`` for every file under ``root``.

    Interpreter and test-runner caches are skipped: running a task's
    tests in place may create them, and they are not part of the task.
    """
    digests: dict[str, str] = {}
    if not root.is_dir():
        return digests
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if not path.is_file() or _STAGE_EXCLUDED_NAMES.intersection(rel_parts):
            continue
        if path.name.endswith((".pyc", ".pyo")):
            continue
        try:
            digests[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        except OSError:
            digests[path.relative_to(root).as_posix()] = "<unreadable>"
    return digests


def frozen_tree_digests(repo_root: Path) -> dict[str, str]:
    """Digests of ``eval/`` and ``benchmarks/`` in the real repository."""
    out: dict[str, str] = {}
    for top in ("eval", "benchmarks"):
        for rel, digest in _tree_digests(repo_root / top).items():
            out[f"{top}/{rel}"] = digest
    return out


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Paths added, removed or modified between two digest maps."""
    return sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def audit_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    repo_root: Path,
    stage: Path | None,
    holdout_ids: list[str],
) -> list[dict[str, Any]]:
    """Tool calls whose input names holdout material or the real repo.

    Matching is case- and separator-insensitive. The stage's own path is
    removed before looking for the repository's, so a stage that happens
    to share a prefix with it is not a hit.
    """
    real_forms = {_norm(str(repo_root)), _norm(str(repo_root.resolve()))}
    stage_forms = (
        {_norm(str(stage)), _norm(str(stage.resolve()))} if stage is not None else set()
    )
    needles = [
        *(_norm(p) for p in (*_HOLDOUT_PATHS, *_RESULTS_PATHS)),
        *(_norm(i) for i in holdout_ids),
    ]
    hits: list[dict[str, Any]] = []
    for call in tool_calls:
        for text in _strings(call.get("input") or {}):
            haystack = _norm(text)
            for form in sorted(stage_forms, key=len, reverse=True):
                haystack = haystack.replace(form, "<stage>")
            matched = [n for n in needles if n and n in haystack]
            matched.extend(
                "<repository path>" for form in real_forms if form and form in haystack
            )
            if matched:
                hits.append(
                    {
                        "tool": call.get("name"),
                        "matched": sorted(set(matched)),
                        "input": text[:300],
                    }
                )
    return hits


def _deny_rules(repo_root: Path) -> list[str]:
    """CLI deny rules for the holdout and results paths.

    Stated for the staged copy (relative) and the real repository
    (absolute). Path rules for the file tools are gitignore-style; Bash
    rules are prefix/wildcard patterns and are best-effort only, which is
    why the transcript audit exists.
    """
    real = repo_root.resolve().as_posix().lstrip("/")
    rules: list[str] = []
    for rel in (*_HOLDOUT_PATHS, *_RESULTS_PATHS):
        for tool in ("Read", "Edit", "Write", "Grep", "Glob"):
            rules.append(f"{tool}(./{rel}/**)")
            rules.append(f"{tool}(//{real}/{rel}/**)")
        rules.append(f"Bash(*{rel}*)")
    for tool in ("Read", "Edit", "Write", "Grep", "Glob"):
        rules.append(f"{tool}(//{real}/**)")
    rules.append(f"Bash(*{real}*)")
    return rules


def _repo_relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def resolve_parent_source(
    *,
    run_dir: Path,
    repo_root: Path,
    parent_name: str | None,
    parent_candidate: dict[str, Any] | None = None,
) -> str:
    """Repo-relative path of the source the parent candidate was measured with.

    The measured baseline is the committed ``agents/baseline.py``. Any
    other parent is the branch snapshot its own thread benchmarked —
    which, for a fork, lives in an ancestor thread's directory, not this
    one's. ``parent_candidate`` (the candidate dict from the outer
    state) is preferred; without it the snapshot is looked up by name.
    Raises ``ValueError`` rather than pointing the proposer at a file
    that does not exist.
    """
    candidate = parent_candidate or {}
    source = candidate.get("source_path")
    if source:
        return _repo_relative(Path(source), repo_root)
    if parent_name is None or parent_name == BASELINE_CANDIDATE_NAME:
        return "agents/baseline.py"
    if candidate.get("import_path"):
        module = str(candidate["import_path"]).partition(":")[0]
        module_file = repo_root / Path(*module.split(".")).with_suffix(".py")
        if module_file.is_file():
            return _repo_relative(module_file, repo_root)
    found = runs_mod.find_candidate_source(run_dir, str(parent_name))
    if found is None:
        raise ValueError(
            f"no source snapshot for parent candidate {parent_name!r} in {run_dir}"
        )
    return _repo_relative(found, repo_root)


def _render_proposer_prompt(
    iteration: int,
    run_dir: Path,
    repo_root: Path,
    parent_name: str | None,
    thread_id: str,
    parent_source: str | None = None,
) -> str:
    """Render the user-message prompt for the proposer subprocess.

    All paths handed to the proposer are repo-relative and scoped to
    *its own* branch, so a concurrently running fork can neither read nor
    overwrite this branch's handoff file. They resolve identically in the
    staged copy the proposer actually runs in.
    """
    thread_root = runs_mod.thread_dir(run_dir, thread_id)
    rel_run = _repo_relative(thread_root, repo_root)
    if parent_source is None:
        parent_source = resolve_parent_source(
            run_dir=run_dir, repo_root=repo_root, parent_name=parent_name
        )
    parent_label = parent_name or BASELINE_CANDIDATE_NAME
    return (
        f"Run iteration {iteration} of the meta-harness coding-agent evolution loop.\n\n"
        f"## Your branch directory\n"
        f"All logs/results for this branch are under `{rel_run}/`.\n"
        f"- `{rel_run}/evolution_summary.jsonl` — past candidates and scores.\n"
        f"- `{rel_run}/frontier_val.json` — current Pareto frontier.\n"
        f"- `{rel_run}/candidates/<name>/traces/` — per-trial inner-loop traces.\n"
        f"- `{rel_run}/agents/<name>.py` — the exact source each candidate was "
        f"benchmarked with.\n"
        f"- Write `pending_eval.json` to: `{rel_run}/pending_eval.json`, with "
        f'`"iteration": {iteration}` and exactly one candidate.\n\n'
        f"## Parent\n"
        f"Parent candidate `{parent_label}`: `{parent_source}` (the source that was "
        f"actually benchmarked). Read it, then evolve from it. `agents/baseline.py` "
        f"is the immutable starting point.\n\n"
        f"## Your candidate\n"
        f"Write it to `agents/<descriptive-snake-case-name>.py`, using a name no "
        f"earlier candidate has used. It must subclass `CodingAgentHarness` from "
        f"`app.meta_harness.harness`.\n\n"
        f"Your working directory is a private copy of the repository. Only "
        f"`agents/<name>.py` and `{rel_run}/pending_eval.json` are kept; every "
        f"other change is discarded.\n\n"
        f"Follow the meta-harness-coding-agent skill workflow exactly. Produce ONE\n"
        f"candidate. Self-critique before writing."
    )


def _build_claude_command(
    *,
    prompt: str,
    system_prompt: str,
    model: str = "opus",
    tools: list[str] | None = None,
    plugin_dir: Path,
    deny_rules: list[str] | None = None,
) -> list[str]:
    """Build the ``claude`` CLI command list, mirroring Stanford's
    claude_wrapper.build_command()."""
    tools = tools or _PROPOSER_ALLOWED_TOOLS
    command = [
        "claude",
        "--dangerously-skip-permissions",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--setting-sources",
        "",
        "--allowedTools",
        *tools,
    ]
    if deny_rules:
        command += ["--disallowedTools", *deny_rules]
    command += [
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--plugin-dir",
        str(plugin_dir),
        "--append-system-prompt",
        system_prompt,
    ]
    return command


def _enqueue_lines(pipe, q: queue.Queue, stream_name: str) -> None:
    """Reader thread: push (stream_name, line) tuples to the queue."""
    try:
        for line in iter(pipe.readline, ""):
            q.put((stream_name, line))
    finally:
        pipe.close()


def _process_group_kwargs() -> dict[str, Any]:
    """Popen arguments that put the child in a process group of its own."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    return {"start_new_session": True}


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Stop ``proc`` and everything it started. Idempotent, never raises.

    POSIX: the child leads its own session, so its process group is
    every descendant that has not deliberately left it; ``killpg`` sends
    SIGKILL to all of them. Windows: ``taskkill /T /F`` walks the tree.
    """
    if os.name == "nt":
        try:
            subprocess.run(  # noqa: S603 — fixed argv
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def claude_propose(
    *,
    run_dir: Path,
    thread_id: str,
    iteration: int,
    parent_name: str | None,
    repo_root: Path,
    skill_path: Path,
    proposer_prior: str = "",
    timeout_seconds: int = 2400,
    model: str = "opus",
    parent_source: str | None = None,
    existing_names: set[str] | frozenset[str] = frozenset(),
    cancel_event: threading.Event | None = None,
    on_spawn: Callable[[subprocess.Popen], None] | None = None,
) -> dict[str, Any]:
    """Run the ``claude`` CLI proposer for one iteration of one branch.

    Stages a private copy of the repository, runs the CLI there with the
    SKILL.md ``--append-system-prompt``'d and the deny rules attached,
    parses stream-json, and logs session.json / transcript.txt /
    system_prompt.txt / events.jsonl. Then it checks isolation — the real
    ``eval/`` and ``benchmarks/`` are unchanged and no tool call named
    holdout material or the real repository — and only then accepts the
    handoff (see ``accept_proposal``) and copies back the candidate
    source and ``pending_eval.json``. Returns the accepted payload.

    Mirrors Stanford's reference ``claude_wrapper.run`` shape. The
    environment is passed through as-is apart from ``CLAUDECODE``:
    ``ANTHROPIC_API_KEY`` is kept, so the CLI bills that key when it is
    set and falls back to its own subscription login when it is not.

    Blocking. ``run_claude_proposer`` is the async entry point; it passes
    ``cancel_event`` and ``on_spawn`` so cancelling the awaiting task
    stops the process tree.
    """
    sess_dir = runs_mod.proposer_session_dir(run_dir, thread_id, iteration)
    clear_pending_eval(run_dir, thread_id)

    # 1) Build the system prompt: SKILL.md + (optional) proposer_prior.
    skill_text = skill_path.read_text(encoding="utf-8")
    system_prompt_parts = [f"## Skill: {skill_path.parent.name}\n{skill_text}"]
    if proposer_prior:
        system_prompt_parts.append(f"## Proposer prior\n{proposer_prior}")
    system_prompt = "Follow these skill instructions:\n\n" + "\n\n".join(
        system_prompt_parts
    )

    # 2) Build the user-message prompt.
    if parent_source is None:
        parent_source = resolve_parent_source(
            run_dir=run_dir, repo_root=repo_root, parent_name=parent_name
        )
    prompt = _render_proposer_prompt(
        iteration, run_dir, repo_root, parent_name, thread_id, parent_source
    )

    # 3) Empty plugin dir for hermeticity, outside the stage.
    empty_plugin_dir = run_dir / ".empty_plugins"
    empty_plugin_dir.mkdir(exist_ok=True)

    cmd = _build_claude_command(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        plugin_dir=empty_plugin_dir,
        deny_rules=_deny_rules(repo_root),
    )

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    # Persist the exact prompt + system prompt for debugging.
    (sess_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (sess_dir / "user_prompt.txt").write_text(prompt, encoding="utf-8")

    holdout_ids = holdout_task_ids(repo_root)
    parent_dir: list[Path] = []
    parent_path = repo_root / parent_source
    if parent_path.parent.name == "agents" and runs_mod.THREADS_DIRNAME in parent_path.parts:
        parent_dir.append(parent_path.parent.parent)

    started = time.monotonic()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    raw_events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_call_map: dict[str, dict[str, Any]] = {}
    files_read: dict[str, dict[str, int]] = {}
    files_written: dict[str, dict[str, int]] = {}
    token_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    # ``None`` until the CLI reports a cost: a session that never emitted
    # its result event has an unknown cost, not a zero one.
    cost_usd: float | None = None
    session_id = ""
    exit_code = 0
    cancelled = False
    tampered: list[str] = []
    audit_hits: list[dict[str, Any]] = []
    stage: Path | None = None

    before = frozen_tree_digests(repo_root)
    try:
        stage = stage_repository(
            repo_root=repo_root,
            run_dir=run_dir,
            thread_id=thread_id,
            extra_thread_dirs=parent_dir,
        )
        assert_stage_isolated(stage, holdout_ids=holdout_ids)
        try:
            proc = subprocess.Popen(  # noqa: S603 — controlled command
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(stage),
                env=env,
                **_process_group_kwargs(),
            )
        except FileNotFoundError as exc:
            stderr_lines.append(str(exc))
            exit_code = 127
            proc = None

        if proc is not None:
            if on_spawn is not None:
                on_spawn(proc)
            try:
                deadline = started + timeout_seconds
                q: queue.Queue = queue.Queue()
                readers = [
                    threading.Thread(
                        target=_enqueue_lines, args=(proc.stdout, q, "stdout"), daemon=True
                    ),
                    threading.Thread(
                        target=_enqueue_lines, args=(proc.stderr, q, "stderr"), daemon=True
                    ),
                ]
                for reader in readers:
                    reader.start()

                def _handle(stream: str, line: str) -> None:
                    nonlocal session_id, cost_usd
                    if stream != "stdout":
                        stderr_lines.append(line)
                        return
                    stdout_lines.append(line)
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        return
                    if not isinstance(event, dict):
                        return
                    raw_events.append(event)
                    _accumulate_event(
                        event,
                        text_parts,
                        tool_calls,
                        tool_call_map,
                        token_usage,
                        files_read,
                        files_written,
                    )
                    if event.get("type") == "result":
                        session_id = event.get("session_id", session_id)
                        reported = event.get("total_cost_usd")
                        if reported is not None:
                            cost_usd = float(reported)

                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        kill_process_tree(proc)
                        stderr_lines.append("\n[cancelled]\n")
                        exit_code = 130
                        cancelled = True
                        break
                    if time.monotonic() > deadline:
                        kill_process_tree(proc)
                        stderr_lines.append(f"\n[timed out after {timeout_seconds}s]\n")
                        exit_code = 124
                        break
                    try:
                        stream, line = q.get(timeout=0.2)
                    except queue.Empty:
                        if proc.poll() is None:
                            continue
                        # The CLI exited. Anything it left running would
                        # hold the pipes open; stop it, then drain what the
                        # readers already have.
                        kill_process_tree(proc)
                        for reader in readers:
                            reader.join(timeout=5)
                        while True:
                            try:
                                _handle(*q.get_nowait())
                            except queue.Empty:
                                break
                        break
                    _handle(stream, line)
                proc.wait()
                if cancel_event is not None and cancel_event.is_set() and not cancelled:
                    # The canceller killed the tree before this loop saw
                    # the flag; the exit is still a cancellation.
                    cancelled = True
                    exit_code = 130
                if exit_code == 0:
                    exit_code = proc.returncode
            finally:
                # Whatever happened, nothing the CLI started outlives it.
                kill_process_tree(proc)

        if not cancelled:
            tampered = changed_paths(before, frozen_tree_digests(repo_root))
            audit_hits = audit_tool_calls(
                tool_calls, repo_root=repo_root, stage=stage, holdout_ids=holdout_ids
            )
        duration = time.monotonic() - started

        # 4) Persist logs.
        (sess_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e, default=str) for e in raw_events) + "\n",
            encoding="utf-8",
        )
        (sess_dir / "transcript.txt").write_text("".join(text_parts), encoding="utf-8")
        runs_mod.write_json_atomic(
            sess_dir / "session.json",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "claude",
                "thread_id": thread_id,
                "iteration": iteration,
                "metrics_source": "measured",
                "model": model,
                "session_id": session_id,
                "exit_code": exit_code,
                "cancelled": cancelled,
                "duration_seconds": round(duration, 2),
                "cost_usd": round(cost_usd, 4) if cost_usd is not None else None,
                "token_usage": token_usage,
                "command": cmd[:8] + ["...", "--append-system-prompt", "<skill>"],
                "cwd": "<staged copy of the repository>",
                "parent_source": parent_source,
                "skill": [{"path": str(skill_path), "name": skill_path.parent.name}],
                "files_read": files_read,
                "files_written": files_written,
                "tool_summary": [
                    f"{tc['name']}({_brief_tool_arg(tc.get('input', {}))})"
                    for tc in tool_calls
                ],
                "isolation": {
                    "staged": True,
                    "frozen_tree_changes": tampered,
                    "audit_hits": audit_hits,
                },
                "stderr": "".join(stderr_lines)[-2000:] if stderr_lines else "",
            },
        )

        if cancelled:
            raise ProposerCancelled(
                f"proposer for {thread_id} iteration {iteration} was cancelled"
            )
        if tampered:
            raise LeakageError(
                f"eval/ or benchmarks/ changed while the proposer ran "
                f"({len(tampered)} path(s), e.g. {tampered[:5]}); the iteration is "
                f"void. See {sess_dir}/session.json"
            )
        if audit_hits:
            raise LeakageError(
                f"the proposer session referenced held-out material or the real "
                f"repository in {len(audit_hits)} tool call(s), e.g. "
                f"{audit_hits[0]['tool']}: {audit_hits[0]['matched']}; the iteration "
                f"is void. See {sess_dir}/session.json"
            )
        if exit_code != 0:
            reason = " ".join(text_parts).strip() or "".join(stderr_lines).strip()
            reason_suffix = f": {reason[:300]}" if reason else ""
            raise RuntimeError(
                f"proposer subprocess failed (exit_code={exit_code}){reason_suffix}; "
                f"see {sess_dir}/session.json"
            )

        # 5) Read the handoff the proposer wrote in its own copy of this
        # branch's thread directory — never the shared run root.
        rel_thread = runs_mod.thread_dir(run_dir, thread_id).resolve().relative_to(
            repo_root.resolve()
        )
        staged_pending = stage / rel_thread / "pending_eval.json"
        if not staged_pending.is_file():
            raise RuntimeError(
                f"proposer exited 0 but did not write {rel_thread.as_posix()}/"
                f"pending_eval.json; see {sess_dir}/transcript.txt"
            )
        try:
            payload = json.loads(staged_pending.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProposalError(f"pending_eval.json is not valid JSON: {exc}") from exc

        accepted = accept_proposal(
            payload,
            iteration=iteration,
            run_dir=run_dir,
            thread_id=thread_id,
            existing_names=existing_names,
            source_root=stage,
        )
        # The two files that come back. The candidate already landed in
        # this branch's snapshot location inside ``accept_proposal``.
        runs_mod.write_pending_eval(run_dir, thread_id, payload)
        return accepted
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


async def run_claude_proposer(**kwargs: Any) -> dict[str, Any]:
    """Async entry point for ``claude_propose``; cancellation stops the CLI.

    The blocking proposer runs on a worker thread. If the awaiting task is
    cancelled (a branch cancelled through the API, a shutdown, a timeout
    around the node), the process tree is killed at once, the worker is
    told to stop, and this coroutine waits for it to finish cleaning up
    before re-raising ``CancelledError``. A plain ``task.cancel()`` is
    therefore enough to stop a proposer — no thread or CLI keeps running
    after it.
    """
    cancel = threading.Event()
    spawned: list[subprocess.Popen] = []
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        None,
        functools.partial(
            claude_propose, **kwargs, cancel_event=cancel, on_spawn=spawned.append
        ),
    )
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        cancel.set()
        for proc in spawned:
            kill_process_tree(proc)
        await asyncio.wait({future}, timeout=60)
        if future.done() and not future.cancelled():
            future.exception()  # consumed: the cancellation is what propagates
        raise


def _accumulate_event(
    event: dict[str, Any],
    text_parts: list[str],
    tool_calls: list[dict[str, Any]],
    tool_call_map: dict[str, dict[str, Any]],
    token_usage: dict[str, int],
    files_read: dict[str, dict[str, int]],
    files_written: dict[str, dict[str, int]],
) -> None:
    """Update accumulators from one stream-json event."""
    etype = event.get("type")
    if etype == "assistant":
        msg = event.get("message", {})
        usage = msg.get("usage", {}) or {}
        token_usage["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        token_usage["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        for block in msg.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", "") + "\n")
            elif btype == "tool_use":
                tc = {
                    "name": block.get("name", ""),
                    "id": block.get("id", ""),
                    "input": block.get("input", {}) or {},
                }
                tool_calls.append(tc)
                tool_call_map[tc["id"]] = tc
                _track_file_op(tc, files_read, files_written)
    elif etype == "user":
        msg = event.get("message", {})
        for block in msg.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tid = block.get("tool_use_id", "")
                tc = tool_call_map.get(tid)
                if tc:
                    tc["output"] = str(block.get("content", ""))


def _track_file_op(
    tc: dict[str, Any],
    files_read: dict[str, dict[str, int]],
    files_written: dict[str, dict[str, int]],
) -> None:
    """Record file-read/write counts for the session log."""
    name = tc["name"]
    inp = tc.get("input", {}) or {}
    path = inp.get("file_path") or inp.get("path") or ""
    if not path:
        return
    if name in {"Read", "read_file"}:
        e = files_read.setdefault(path, {"reads": 0})
        e["reads"] += 1
    elif name in {"Write", "Edit", "write_file", "apply_patch"}:
        e = files_written.setdefault(path, {"writes": 0})
        e["writes"] += 1


def _brief_tool_arg(inp: dict[str, Any]) -> str:
    for key in ("file_path", "path", "command", "pattern", "description"):
        if key in inp:
            return f"{key}={str(inp[key])[:80]}"
    return ""
