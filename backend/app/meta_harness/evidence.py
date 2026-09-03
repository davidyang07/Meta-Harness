"""Derive the capability-evidence document from committed artifacts.

Every row in `docs/CAPABILITY_EVIDENCE.md` is produced here, and every row
is computed rather than written down:

- **Measured rows** (pass rates, the percentage-point delta, trial
  counts) are recomputed from the raw ``*-results.jsonl`` rows of a
  published experiment. The published ``summary.json`` is compared
  against that recomputation; if they disagree the row FAILs, because a
  summary that cannot be re-derived is not evidence.
- **Structural rows** (the two state machines, the stack) introspect the
  actual code objects and the actual dependency declarations — the graph
  is compiled and its node set read, not described.
- **Artifact rows** (exact replay, branching) read verification reports
  that a command produced, and FAIL when there is none.

There is no constant in this module holding a pass rate, a delta, or a
target of any kind, and no row asserts that a measurement cleared a
threshold. A threshold row is how a document starts steering the
measurement it is supposed to report: once a number has a bar to clear,
every later decision about tasks, trials and selection is made in its
shadow. The rows here state what was measured and how it can be
re-derived, and nothing else. A claim with no supporting artifact
reports UNSUPPORTED — a correct outcome, not a bug to work around.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import tokenize
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.meta_harness import experiment as exp

#: Repository-relative path of the generated document. One definition, so
#: the CLI, the tests and CI cannot drift apart on where it lives.
DOCUMENT_PATH = "docs/CAPABILITY_EVIDENCE.md"

PASS = "PASS"
FAIL = "FAIL"
UNSUPPORTED = "UNSUPPORTED"


@dataclass
class Check:
    """One capability claim, its verdict, and where it came from."""

    key: str
    claim: str
    status: str
    value: Any = None
    detail: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "claim": self.claim,
            "status": self.status,
            "value": self.value,
            "detail": self.detail,
            "evidence": self.evidence,
        }


# ── published experiment discovery ────────────────────────────────────


def published_results(repo_root: Path) -> list[Path]:
    """Committed experiment directories, newest last."""
    root = repo_root / "benchmarks" / "results"
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if (d / "summary.json").is_file())


def load_experiment(directory: Path) -> dict[str, Any]:
    """Load a published experiment and recompute its summary from raw rows.

    The recomputation is the point: a published number is evidence only
    while it still falls out of the trials it claims to summarise.
    """
    summary = json.loads((directory / "summary.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    baseline_rows = exp.read_rows(directory / "baseline-results.jsonl")
    candidate_rows = exp.read_rows(directory / "candidate-results.jsonl")
    recomputed = exp.summarize(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        task_ids=[t["task_id"] for t in environment.get("tasks", [])],
        baseline_label=summary.get("baseline_label", "baseline"),
        candidate_label=summary.get("candidate_label", "candidate"),
    )
    mismatches = [
        key
        for key in (
            "baseline_passes",
            "baseline_trials",
            "candidate_passes",
            "candidate_trials",
            "baseline_accuracy",
            "candidate_accuracy",
            "absolute_percentage_point_delta",
        )
        if summary.get(key) != recomputed.get(key)
    ]
    validation_path = directory / "validation.json"
    validation = (
        json.loads(validation_path.read_text()) if validation_path.is_file() else None
    )
    return {
        "dir": directory,
        "experiment": config.get("experiment"),
        "config": config,
        "summary": summary,
        "recomputed": recomputed,
        "environment": environment,
        "validation": validation,
        "mismatched_keys": mismatches,
        "reproducible": not mismatches,
        "raw_rows": len(baseline_rows) + len(candidate_rows),
    }


def latest_experiment(
    repo_root: Path, *, experiment: str = "pass-rate"
) -> dict[str, Any] | None:
    """The newest published experiment of a given name, fully re-derived."""
    for directory in reversed(published_results(repo_root)):
        loaded = load_experiment(directory)
        if loaded["experiment"] == experiment:
            return loaded
    return None


# ── structural introspection ──────────────────────────────────────────


def graph_node_sets(repo_root: Path) -> dict[str, list[str]]:
    """Compile both state machines and read back their node names.

    Compiling is what makes this evidence rather than description: if a
    node is renamed or a machine is collapsed into the other, the row
    changes without anyone editing this file.
    """
    from app.meta_harness.inner import build_inner_graph  # noqa: PLC0415
    from app.meta_harness.outer import OuterLoopRunner  # noqa: PLC0415

    class _StubHarness:
        MAX_VERIFY_RETRIES = 3

        def should_loop_back_to_act(self, verify_result: dict[str, Any]) -> bool:
            return False

    inner = build_inner_graph(_StubHarness())  # type: ignore[arg-type]
    outer = OuterLoopRunner(
        run_dir=repo_root / "runs" / "_evidence",
        repo_root=repo_root,
        eval_tasks_dir=repo_root / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=1,
        bench_workers=1,
    ).build()
    return {
        "inner": sorted(_node_names(inner)),
        "outer": sorted(_node_names(outer)),
    }


def _node_names(graph: Any) -> list[str]:
    names = list(getattr(graph, "nodes", {}) or {})
    return [n for n in names if not str(n).startswith("__")]


def declared_dependencies(repo_root: Path) -> dict[str, str]:
    """Every dependency declared across the uv workspace, name → spec."""
    found: dict[str, str] = {}
    for pyproject in (
        repo_root / "pyproject.toml",
        repo_root / "backend" / "pyproject.toml",
        repo_root / "sdk" / "pyproject.toml",
    ):
        if not pyproject.is_file():
            continue
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project", {})
        specs = list(project.get("dependencies", []))
        for group in (project.get("optional-dependencies", {}) or {}).values():
            specs.extend(group)
        for spec in specs:
            name = re.split(r"[<>=!\[;\s]", str(spec).strip(), maxsplit=1)[0]
            found.setdefault(name.lower(), str(spec))
    return found


def executable_code(source: str) -> str:
    """``source`` with comments and string literals removed.

    ``source_uses`` searches for evidence that a dependency is *used*, and
    a name inside a string or a docstring is a mention, not a use. This
    module is the case that forces the distinction: it holds every one of
    those patterns as a string literal, so a plain text search counts
    the scanner as its own evidence.

    Skipping this file by name was the previous answer, and it drifted —
    the committed document claimed nine LangGraph modules against a
    scanner that had started reporting eight. Dropping literals instead
    is self-maintaining: no file is special-cased, and a module that
    genuinely imports a dependency counts even if it is this one.

    Tokens are rejoined one space apart, so a pattern must be written
    with single spaces between tokens (``from langgraph``,
    ``class WandbTracker``). A file that cannot be tokenised is returned
    unchanged rather than silently scanning as empty.
    """
    rows: dict[int, list[str]] = {}
    dropped = {tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE}
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fstring_middle is not None:
        dropped.add(fstring_middle)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    for token in tokens:
        if token.type in dropped:
            continue
        rows.setdefault(token.start[0], []).append(token.string)
    return "\n".join(" ".join(rows[line]) for line in sorted(rows))


def source_uses(repo_root: Path, pattern: str) -> list[str]:
    """Backend modules whose *code* matches ``pattern``. Evidence of real use."""
    hits: list[str] = []
    regex = re.compile(pattern)
    for path in sorted((repo_root / "backend" / "app").rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if regex.search(executable_code(source)):
            hits.append(str(path.relative_to(repo_root)).replace("\\", "/"))
    return hits


#: Trace artifacts every inner-loop trial writes, and the branch-scoped
#: inputs the proposer conditions on. "Trace-driven" is the conjunction of
#: the two: traces are produced *and* they are what the optimizer reads.
TRACE_ARTIFACTS = (
    "orient.json",
    "plan.json",
    "act-tools.jsonl",
    "act-messages.jsonl",
    "verify.json",
    "score.json",
)
PROPOSER_TRACE_INPUTS = ("evolution_summary", "frontier_val", "traces")


def trace_driven_signals(repo_root: Path) -> dict[str, Any]:
    """Check that traces are written by the executor and read by the optimizer."""
    inner_src = (repo_root / "backend/app/meta_harness/inner.py").read_text(
        encoding="utf-8"
    )
    proposer_src = (repo_root / "backend/app/meta_harness/proposer.py").read_text(
        encoding="utf-8"
    )
    written = [name for name in TRACE_ARTIFACTS if name in inner_src]
    read = [name for name in PROPOSER_TRACE_INPUTS if name in proposer_src]
    missing = [
        *(f"inner.py does not write {n}" for n in TRACE_ARTIFACTS if n not in written),
        *(
            f"proposer.py does not read {n}"
            for n in PROPOSER_TRACE_INPUTS
            if n not in read
        ),
    ]
    return {
        "trace_artifacts": written,
        "proposer_inputs": read,
        "missing": missing,
        "trace_driven": not missing,
    }


def git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


EVIDENCE_DIRNAME = Path("docs") / "evidence"


def evidence_path(repo_root: Path, name: str) -> Path:
    return repo_root / EVIDENCE_DIRNAME / name


# ── the checks ────────────────────────────────────────────────────────


def _stack_check(
    *,
    key: str,
    claim: str,
    repo_root: Path,
    dependency: str | None,
    dependencies: dict[str, str],
    use_pattern: str | None = None,
    files: list[str] | None = None,
    optional_note: str | None = None,
) -> Check:
    evidence: list[str] = []
    reasons: list[str] = []
    ok = True

    if dependency is not None:
        spec = dependencies.get(dependency.lower())
        if spec:
            evidence.append(f"declared dependency `{spec}`")
        else:
            ok = False
            reasons.append(f"`{dependency}` is not a declared dependency")

    if use_pattern is not None:
        hits = source_uses(repo_root, use_pattern)
        if hits:
            evidence.append(f"used in {len(hits)} module(s): {', '.join(hits[:3])}")
        else:
            ok = False
            reasons.append(f"no backend module matches /{use_pattern}/")

    for relative in files or []:
        path = repo_root / relative
        if path.exists():
            evidence.append(f"`{relative}`")
        else:
            ok = False
            reasons.append(f"missing `{relative}`")

    return Check(
        key=key,
        claim=claim,
        status=PASS if ok else FAIL,
        value=ok,
        detail="; ".join(reasons) or (optional_note or "declared and used"),
        evidence=evidence,
    )


def build_checks(repo_root: Path) -> list[Check]:
    """Derive every capability claim's verdict. The document is this list."""
    checks: list[Check] = []
    dependencies = declared_dependencies(repo_root)
    nodes = graph_node_sets(repo_root)
    search = latest_experiment(repo_root, experiment="pass-rate")
    holdout = latest_experiment(repo_root, experiment="holdout")

    # 1. trace-driven optimizer
    trace_signals = trace_driven_signals(repo_root)
    checks.append(
        Check(
            key="trace_driven_optimizer",
            claim="Trace-driven agent optimizer",
            status=PASS if trace_signals["trace_driven"] else FAIL,
            value={
                "trace_artifacts_written": trace_signals["trace_artifacts"],
                "proposer_inputs": trace_signals["proposer_inputs"],
            },
            detail=(
                "every inner-loop trial writes orient/plan/act/verify/score "
                "traces, and the proposer's prior is assembled from the "
                "branch's own traces, evolution log and Pareto frontier — so "
                "each proposal is conditioned on recorded execution, not on "
                "a scalar score"
                if trace_signals["trace_driven"]
                else f"missing signals: {trace_signals['missing']}"
            ),
            evidence=[
                "backend/app/meta_harness/inner.py",
                "backend/app/meta_harness/proposer.py",
                "backend/app/meta_harness/runs.py",
            ],
        )
    )

    # 2. two separate LangGraph state machines
    inner_ok = nodes["inner"] == ["act", "orient", "plan", "submit", "verify"]
    outer_ok = nodes["outer"] == [
        "benchmark",
        "propose",
        "update_frontier",
        "validate",
    ]
    checks.append(
        Check(
            key="dual_state_machines",
            claim="Separate execution and self-improvement LangGraph state machines",
            status=PASS if inner_ok and outer_ok and not set(nodes["inner"]) & set(nodes["outer"]) else FAIL,
            value={"inner": nodes["inner"], "outer": nodes["outer"]},
            detail=(
                "both graphs compiled and their node sets read back; the sets "
                "are disjoint, so these are two machines and not one"
            ),
            evidence=[
                "backend/app/meta_harness/inner.py",
                "backend/app/meta_harness/outer.py",
            ],
        )
    )

    checks.extend(_measurement_checks(search))
    checks.extend(_holdout_checks(holdout))

    # PostgreSQL checkpointing / version graph
    version_evidence = read_json(evidence_path(repo_root, "version-graph.json"))
    checks.append(_version_graph_check(version_evidence))

    # Exact recorded-execution replay
    checks.append(_replay_check(repo_root))

    checks.append(_branching_check(version_evidence))

    # Stack
    checks.append(
        _stack_check(
            key="stack_langgraph",
            claim="Stack: LangGraph",
            repo_root=repo_root,
            dependency="langgraph",
            dependencies=dependencies,
            use_pattern=r"from langgraph",
        )
    )
    wandb_evidence = read_json(evidence_path(repo_root, "wandb-offline.json"))
    wandb_check = _stack_check(
        key="stack_wandb",
        claim="Stack: Weights & Biases",
        repo_root=repo_root,
        dependency="wandb",
        dependencies=dependencies,
        use_pattern=r"class WandbTracker",
        optional_note=(
            "optional integration: the repository, its tests and every CLI "
            "command run with no W&B account and no network"
        ),
    )
    if wandb_evidence:
        wandb_check.value = {
            "mode": wandb_evidence.get("mode"),
            "logged": wandb_evidence.get("logged"),
        }
        wandb_check.evidence.append(
            str(EVIDENCE_DIRNAME / "wandb-offline.json").replace("\\", "/")
        )
        if not wandb_evidence.get("ok"):
            wandb_check.status = FAIL
            wandb_check.detail = str(
                wandb_evidence.get("detail") or "offline logging probe failed"
            )
    checks.append(wandb_check)

    checks.append(
        _stack_check(
            key="stack_fastapi",
            claim="Stack: FastAPI",
            repo_root=repo_root,
            dependency="fastapi",
            dependencies=dependencies,
            use_pattern=r"from fastapi",
        )
    )
    checks.append(
        _stack_check(
            key="stack_docker",
            claim="Stack: Docker",
            repo_root=repo_root,
            dependency=None,
            dependencies=dependencies,
            # A compose file that provisions a database is not "Docker as
            # a project technology" -- it is a dev dependency. The claim
            # is only supported while the application itself has an image,
            # a container healthcheck, a build context that excludes
            # secrets, and something that checks the container actually
            # serves.
            files=[
                "infra/docker-compose.yml",
                "infra/Dockerfile",
                "infra/healthcheck.py",
                ".dockerignore",
                "scripts/docker_smoke.sh",
            ],
        )
    )
    checks.append(
        _stack_check(
            key="stack_postgres",
            claim="Stack: PostgreSQL",
            repo_root=repo_root,
            dependency="langgraph-checkpoint-postgres",
            dependencies=dependencies,
            use_pattern=r"AsyncPostgresSaver",
        )
    )
    checks.append(
        _stack_check(
            key="stack_pydantic",
            claim="Stack: Pydantic",
            repo_root=repo_root,
            dependency="pydantic",
            dependencies=dependencies,
            use_pattern=r"from pydantic",
        )
    )
    return checks


def _version_graph_check(version_evidence: dict[str, Any] | None) -> Check:
    """The checkpoint DAG, its refs, and whether stored state is immutable."""
    claim = "Git-style run versioning in PostgreSQL"
    artifact = str(EVIDENCE_DIRNAME / "version-graph.json").replace("\\", "/")

    if version_evidence is None:
        return Check(
            key="postgres_version_graph",
            claim=claim,
            status=UNSUPPORTED,
            value=None,
            detail=(
                "no version-graph artifact. Immutable checkpoint identity, "
                "parent references, branch refs and per-branch working trees "
                "are covered by tests/test_versioning.py against real "
                "Postgres; what is missing is a captured graph from a run. "
                "Produce one with `docker compose -f infra/docker-compose.yml "
                "up -d postgres` then `meta-harness report version-graph "
                "<run>`."
            ),
            evidence=["backend/tests/test_versioning.py"],
        )

    immutable = bool(version_evidence.get("immutable"))
    value = {
        "checkpoints": version_evidence.get("checkpoint_count"),
        "branches": version_evidence.get("branch_count"),
        "immutable": immutable,
    }
    return Check(
        key="postgres_version_graph",
        claim=claim,
        status=PASS if immutable else FAIL,
        value=value,
        detail=(
            "checkpoint DAG with immutable ids and parent references, branch "
            "refs persisted in branches.json, per-branch working trees under "
            "runs/<run>/threads/<thread>/; every stored checkpoint re-read "
            "and confirmed to still hash to what it hashed to"
            if immutable
            else "a stored checkpoint no longer hashes to what it hashed to, "
            "which invalidates every replay and fork claim built on it"
        ),
        evidence=[artifact],
    )


def _branching_check(version_evidence: dict[str, Any] | None) -> Check:
    """Branching from a persisted checkpoint, read from the version graph."""
    claim = "Branching from any persisted checkpoint"
    artifact = str(EVIDENCE_DIRNAME / "version-graph.json").replace("\\", "/")

    if version_evidence is None:
        return Check(
            key="branching_from_checkpoints",
            claim=claim,
            status=UNSUPPORTED,
            value=None,
            detail=(
                "no version-graph artifact. Forking from any persisted "
                "checkpoint, branch isolation and history survival across a "
                "restart are covered by tests/test_versioning.py and "
                "tests/test_branch_isolation.py against real Postgres; what "
                "is missing is a captured graph from a run. Produce one with "
                "`meta-harness report version-graph <run>`."
            ),
            evidence=[
                "backend/tests/test_versioning.py",
                "backend/tests/test_branch_isolation.py",
            ],
        )

    branches = version_evidence.get("branches") or []
    forked = [b for b in branches if b.get("parent_checkpoint_id")]
    return Check(
        key="branching_from_checkpoints",
        claim=claim,
        status=PASS if forked else FAIL,
        value=len(forked),
        detail=(
            "each branch is recorded with the checkpoint it forked from and "
            "its own working tree under runs/<run>/threads/<thread>/; the "
            "registry reloads from branches.json after a restart"
            if forked
            else "the captured version graph contains no branch with a fork "
            "point, so nothing here demonstrates branching"
        ),
        evidence=[artifact],
    )


#: Replay is only as sound as the boundary it replays through. The
#: verification artifact shows that *the recorded graph* replayed
#: exactly; it cannot show that a later edit kept every world-crossing
#: behind ``effects``. This static check does, so it is cited with the
#: artifact rather than in place of it.
BOUNDARY_TEST = "backend/tests/test_effects_boundary.py"


def _replay_check(repo_root: Path) -> Check:
    """Exact recorded-execution replay, from a verification artifact.

    The claim has two halves — "the tape reproduces the run" and "any
    stored checkpoint is a valid entry point" — so a report containing
    only whole-run replays supports the first and not the second, and
    says so rather than passing on the strength of the easier half.
    """
    report = read_json(evidence_path(repo_root, "replay-verification.json"))
    artifact = str(EVIDENCE_DIRNAME / "replay-verification.json").replace("\\", "/")

    if report is None:
        return Check(
            key="exact_recorded_replay",
            claim="Exact replay of a recorded execution from any checkpoint",
            status=UNSUPPORTED,
            value=None,
            detail=(
                "no replay-verification artifact. The mechanism is covered "
                "offline by tests/test_exact_replay.py, which records a real "
                "inner-loop run and replays it from every stored checkpoint; "
                "what is missing is a verification report from a recorded "
                "production run. Produce one with `meta-harness inner "
                "--record` then `meta-harness verify-replay <dir>`, or as "
                "part of `meta-harness canonical-experiment`."
            ),
            evidence=["backend/tests/test_exact_replay.py", BOUNDARY_TEST],
        )

    from_checkpoint = int(report.get("replays_from_checkpoint") or 0)
    value = {
        "replays": report.get("replays"),
        "from_checkpoint": from_checkpoint,
        "all_verified": report.get("all_verified"),
        "model_calls_issued": report.get("model_calls_issued"),
    }
    verified = bool(report.get("all_verified"))
    no_model_calls = report.get("model_calls_issued") == 0

    if not (verified and no_model_calls):
        return Check(
            key="exact_recorded_replay",
            claim="Exact replay of a recorded execution from any checkpoint",
            status=FAIL,
            value=value,
            detail=(
                "a replay did not reproduce its recording"
                if not verified
                else "a replay issued a model call, which it must never do"
            ),
            evidence=[artifact],
        )

    if from_checkpoint == 0:
        skipped = report.get("skipped") or []
        return Check(
            key="exact_recorded_replay",
            claim="Exact replay of a recorded execution from any checkpoint",
            status=UNSUPPORTED,
            value=value,
            detail=(
                "whole-run replays verified, but no replay started from a "
                "stored checkpoint, so the 'from any checkpoint' half of the "
                "claim is untested here"
                + (f": {skipped[0].get('reason')}" if skipped else "")
            ),
            evidence=[artifact],
        )

    models = report.get("recorded_models") or []
    value["recorded_models"] = models
    return Check(
        key="exact_recorded_replay",
        claim="Exact replay of a recorded execution from any checkpoint",
        status=PASS,
        value=value,
        detail=(
            "each recorded trial re-executed against its tape, from the start "
            "of the run and from a mid-run checkpoint; node sequence, per-step "
            "state hashes and final state hash all matched, the tape was "
            "consumed exactly, and no model call was issued. Recorded model"
            + (f"(s): {', '.join(models)}" if models else ": unrecorded")
            + "."
        ),
        evidence=[artifact, BOUNDARY_TEST],
    )


def _measurement_checks(search: dict[str, Any] | None) -> list[Check]:
    """The measured rows, each re-derived from the raw trial rows.

    There is deliberately no "did it clear X points" row. The delta is
    reported with the interval that describes its precision, and the
    reader draws the conclusion.
    """
    if search is None:
        missing = (
            "no published pass-rate experiment under benchmarks/results/. "
            "Run `uv run meta-harness canonical-experiment` with credentials, "
            "then commit the result directory."
        )
        return [
            Check(
                key=key,
                claim=claim,
                status=UNSUPPORTED,
                value=None,
                detail=missing,
                evidence=[],
            )
            for key, claim in (
                ("canonical_200_trials", "Canonical 200 task trials executed"),
                ("baseline_pass_rate", "Baseline pass rate"),
                ("evolved_pass_rate", "Evolved pass rate"),
                ("absolute_improvement_pp", "Absolute percentage-point improvement"),
            )
        ]

    summary = search["recomputed"]
    directory = str(search["dir"].name)
    evidence = [
        f"benchmarks/results/{directory}/summary.json",
        f"benchmarks/results/{directory}/baseline-results.jsonl",
        f"benchmarks/results/{directory}/candidate-results.jsonl",
    ]
    validation = search.get("validation") or {}
    protocol_ok = bool(validation.get("identical_protocol"))
    total = summary.get("total_trials") or 0
    delta = summary.get("absolute_percentage_point_delta")
    ci = summary.get("difference_ci") or {}

    checks = [
        Check(
            key="canonical_200_trials",
            claim="Canonical 200 task trials executed",
            status=PASS
            if total >= 200 and search["reproducible"] and protocol_ok
            else FAIL,
            value=total,
            detail=(
                f"{search['raw_rows']} raw trial rows on disk; the published "
                f"summary re-derives from them exactly"
                + ("" if protocol_ok else "; the two arms did NOT run an identical protocol")
                + (
                    ""
                    if search["reproducible"]
                    else f"; summary disagrees with raw rows on {search['mismatched_keys']}"
                )
            ),
            evidence=evidence,
        ),
        Check(
            key="baseline_pass_rate",
            claim="Baseline pass rate",
            status=PASS if summary.get("baseline_accuracy") is not None else UNSUPPORTED,
            value=summary.get("baseline_accuracy"),
            detail=(
                f"{summary.get('baseline_passes')}/{summary.get('baseline_trials')} "
                f"trials of `{search['summary'].get('baseline_label')}`"
            ),
            evidence=evidence,
        ),
        Check(
            key="evolved_pass_rate",
            claim="Evolved pass rate",
            status=PASS if summary.get("candidate_accuracy") is not None else UNSUPPORTED,
            value=summary.get("candidate_accuracy"),
            detail=(
                f"{summary.get('candidate_passes')}/{summary.get('candidate_trials')} "
                f"trials of `{search['summary'].get('candidate_label')}`"
            ),
            evidence=evidence,
        ),
        Check(
            key="absolute_improvement_pp",
            claim="Absolute percentage-point improvement",
            status=PASS if delta is not None else UNSUPPORTED,
            value=delta,
            detail=(
                _delta_detail(delta, ci, summary.get("cluster_bootstrap_ci"))
                if delta is not None
                else "no measured delta"
            ),
            evidence=evidence,
        ),
    ]
    return checks


def _delta_detail(
    delta: float,
    wald: dict[str, Any],
    cluster: dict[str, Any] | None,
) -> str:
    """State the delta with the interval that describes its precision.

    Both intervals are reported. The Wald one assumes 200 independent
    Bernoulli trials, which the design does not have — trials are
    clustered within tasks — so the cluster-aware interval is the one to
    read, and its own limitation travels with it.
    """
    parts = [
        f"measured {delta:+.1f} pp; 95% Wald interval on the difference "
        f"[{_pp(wald.get('lower'))}, {_pp(wald.get('upper'))}] pp, which "
        f"assumes independent trials the design does not have"
    ]
    if cluster and cluster.get("lower") is not None:
        parts.append(
            f"95% task-cluster bootstrap interval "
            f"[{_pp(cluster.get('lower'))}, {_pp(cluster.get('upper'))}] pp "
            f"over {cluster.get('clusters')} task clusters "
            f"({cluster.get('resamples')} resamples, seed {cluster.get('seed')})"
        )
        if not cluster.get("informative", True):
            parts.append(str(cluster.get("limitation", "")).strip())
    else:
        parts.append("no cluster-aware interval exists for this design")
    return ". ".join(part.rstrip(".") for part in parts if part) + "."


def _holdout_checks(holdout: dict[str, Any] | None) -> list[Check]:
    if holdout is None:
        return [
            Check(
                key="holdout_generalisation",
                claim="Generalisation on unseen holdout tasks",
                status=UNSUPPORTED,
                value=None,
                detail=(
                    "no published holdout experiment. Generalisation to "
                    "unseen tasks is a separate claim from the search-set "
                    "number, not a substitute for it."
                ),
                evidence=[],
            )
        ]
    summary = holdout["recomputed"]
    delta = summary.get("absolute_percentage_point_delta")
    directory = holdout["dir"].name
    return [
        Check(
            key="holdout_generalisation",
            claim="Generalisation on unseen holdout tasks",
            status=PASS if delta is not None and holdout["reproducible"] else FAIL,
            value=delta,
            detail=(
                f"{summary.get('baseline_passes')}/{summary.get('baseline_trials')} "
                f"baseline vs {summary.get('candidate_passes')}/"
                f"{summary.get('candidate_trials')} evolved on tasks the "
                f"proposer never saw"
            ),
            evidence=[f"benchmarks/results/{directory}/summary.json"],
        )
    ]


def _pp(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}"


# ── rendering ─────────────────────────────────────────────────────────


def _fmt_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, separators=(',', ':'))}`"
    return str(value)


def render_markdown(
    checks: list[Check], *, repo_root: Path, generated_at: str | None = None
) -> str:
    """The whole of ``docs/CAPABILITY_EVIDENCE.md``.

    Deterministic apart from the header timestamp, which CI strips before
    comparing so a regeneration check does not fail on the clock.
    """
    search = latest_experiment(repo_root, experiment="pass-rate")
    headline = (
        exp.reported_metric_sentence(search["recomputed"])
        if search
        else "No measured result: the canonical experiment has not been run."
    )
    counts = {status: sum(1 for c in checks if c.status == status) for status in (PASS, FAIL, UNSUPPORTED)}

    lines = [
        "# Capability evidence",
        "",
        f"<!-- generated-at: {generated_at or datetime.now(timezone.utc).isoformat()} -->",
        f"<!-- commit: {git_commit(repo_root)} -->",
        "",
        "Generated by `uv run meta-harness report capability-evidence`. Every",
        "row below is derived from committed artifacts — raw trial rows,",
        "compiled graphs, declared dependencies, verification reports — at the",
        "moment the command ran. Nothing here is hand-entered, and a claim with",
        "no supporting artifact reports `UNSUPPORTED` rather than being",
        "softened. No row asserts that a measurement cleared a target.",
        "",
        "CI regenerates this file and fails if it disagrees with the raw",
        "results, so a stale or edited number cannot survive a push.",
        "",
        "## Headline",
        "",
        f"> {headline}",
        "",
        f"**{counts[PASS]} PASS · {counts[FAIL]} FAIL · {counts[UNSUPPORTED]} UNSUPPORTED**",
        "",
        "## Claims",
        "",
        "| Claim | Status | Value | Basis |",
        "|---|---|---|---|",
    ]
    for check in checks:
        detail = check.detail.replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {check.claim} | **{check.status}** | {_fmt_value(check.value)} | {detail} |"
        )

    lines += ["", "## Evidence paths", ""]
    for check in checks:
        if not check.evidence:
            continue
        paths = ", ".join(f"`{p}`" for p in check.evidence)
        lines.append(f"- **{check.claim}** — {paths}")

    lines += [
        "",
        "## How to reproduce every row",
        "",
        "```bash",
        "# the measured rows: recompute the published summary from raw trials",
        "cd backend && uv run pytest tests/test_capability_evidence.py -q",
        "",
        "# the structural rows: compile both graphs and read their node sets",
        "uv run meta-harness report capability-evidence --json",
        "",
        "# the replay row: re-execute a recorded trial against its tape",
        "uv run meta-harness verify-replay <recordings-dir>",
        "",
        "# the version-graph row: read the checkpoint DAG out of Postgres",
        "uv run meta-harness report version-graph <run-name>",
        "```",
        "",
        "## What `UNSUPPORTED` means here",
        "",
        "The measurement machinery exists and is tested, but the artifact that",
        "would settle the claim has not been produced in this repository. For",
        "the pass-rate rows that means the canonical experiment has not been",
        "run against a real model — it costs money and needs credentials. The",
        "one command that produces every missing artifact is",
        "`uv run meta-harness canonical-experiment`.",
        "",
    ]
    return "\n".join(lines)


TIMESTAMP_RE = re.compile(r"<!-- generated-at: .*? -->|<!-- commit: .*? -->")


def comparable(markdown: str) -> str:
    """Strip the volatile header so two generations can be compared."""
    return TIMESTAMP_RE.sub("", markdown).strip()


def build_report(repo_root: Path) -> dict[str, Any]:
    """Everything the CLI needs: the checks, the markdown, and the verdict."""
    checks = build_checks(repo_root)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(repo_root),
        "checks": [c.to_dict() for c in checks],
        "counts": {
            status: sum(1 for c in checks if c.status == status)
            for status in (PASS, FAIL, UNSUPPORTED)
        },
        "markdown": render_markdown(checks, repo_root=repo_root),
        "any_failed": any(c.status == FAIL for c in checks),
    }


__all__ = [
    "Check",
    "DOCUMENT_PATH",
    "FAIL",
    "PASS",
    "UNSUPPORTED",
    "build_checks",
    "build_report",
    "comparable",
    "declared_dependencies",
    "graph_node_sets",
    "latest_experiment",
    "load_experiment",
    "published_results",
    "render_markdown",
]
