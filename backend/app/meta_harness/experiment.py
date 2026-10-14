"""The canonical two-arm pass-rate experiment.

Protocol (``benchmarks/pass-rate/config.json``): every frozen search
task is run ``trials_per_task`` independent times by the baseline arm and
the same number of times by the evolved arm. The task list and the trial
count are read from the committed config; nothing here hard-codes them.

Design rules this module enforces, because the whole point is that the
resulting number is defensible:

1. **The summary is derived, never entered.** ``summarize()`` takes raw
   trial rows and computes everything. There is no code path that
   accepts a pass rate, a delta, or a target.
2. **Both arms run the identical protocol** — same tasks, same trial
   count, same worker pool, same model — so the difference is
   attributable to the harness and nothing else. The arms are
   interleaved under one scheduler (alternating per task and trial), so
   drift over the run's wall-clock time — provider load, rate limits,
   a model update — lands on both arms rather than biasing the delta.
   Which model served each trial is read from the trial rows, not
   asserted by the caller.
3. **Provenance is captured, secrets are not.** Commit SHA, dirty flag,
   model id, per-task definition and test hashes, harness source hashes,
   interpreter and platform. No environment variables are copied.
4. **The statistics are stated with their limitations.** Each arm's pass
   rate carries a Wilson score interval and the difference carries
   Newcombe's hybrid score interval — both stay sensible at 0% and 100%,
   where a Wald interval collapses to zero width. They assume
   independent Bernoulli trials; trials are clustered within tasks, so
   a task-cluster bootstrap is reported beside them and neither is
   turned into a p-value.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.meta_harness import benchmark as bench
from app.meta_harness import metrics as met
from app.meta_harness import runs as runs_mod

#: Bump when the on-disk result schema changes incompatibly.
RESULT_SCHEMA_VERSION = "1.0.0"

#: 95% two-sided normal quantile, for the score intervals.
Z_95 = 1.959963984540054

#: Files a task directory can pick up from being run in place (a pytest
#: cache, byte-compiled modules). They are not part of the task's
#: definition, so they must not change its hash.
_HASH_IGNORED_DIRS = frozenset({"__pycache__", ".pytest_cache"})
_HASH_IGNORED_SUFFIXES = (".pyc", ".pyo")


# ── provenance ────────────────────────────────────────────────────────


def _git(*args: str, repo_root: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def file_sha256(path: Path) -> str | None:
    """Hex SHA-256 of a file, or ``None`` if it is not readable."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def hash_task(task_dir: Path) -> dict[str, Any]:
    """Content hashes that pin one task's definition and its tests.

    A published result is only meaningful if the tasks it was measured
    against can be shown not to have changed since.
    """
    spec_path = task_dir / "task.json"
    files: dict[str, str] = {}
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(task_dir).parts
        if _HASH_IGNORED_DIRS.intersection(relative_parts) or path.name.endswith(
            _HASH_IGNORED_SUFFIXES
        ):
            continue
        digest = file_sha256(path)
        if digest:
            files[str(path.relative_to(task_dir)).replace("\\", "/")] = digest
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "task_id": task_dir.name,
        "task_json_sha256": file_sha256(spec_path),
        "files_sha256": files,
        "task_sha256": combined,
    }


def capture_environment(
    *,
    repo_root: Path,
    model: str,
    tasks: list[Path],
    arm_sources: dict[str, Path | None],
    arm_models: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Everything needed to judge, and re-run, a published result.

    Deliberately records no environment variables: the point is
    reproducibility, and copying the environment is how API keys end up
    in committed artifacts.

    ``model`` and ``arm_models`` are what each arm's harness *declared*.
    What actually served the trials is in the rows, and that is what the
    protocol check compares.
    """
    dirty = _git("status", "--porcelain", repo_root=repo_root)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git("rev-parse", "HEAD", repo_root=repo_root),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo_root=repo_root),
            "dirty": bool(dirty) if dirty is not None else None,
            "dirty_paths": sorted(
                line[3:] for line in (dirty or "").splitlines() if line[3:]
            )[:50],
        },
        "model": model,
        "arm_models": dict(arm_models or {}),
        "pricing_source": "META_HARNESS_PRICING" if _pricing_overridden() else "builtin",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tasks": [hash_task(t) for t in tasks],
        "arm_sources": {
            name: {
                "path": repo_relative(path, repo_root) if path else None,
                "sha256": file_sha256(path) if path else None,
            }
            for name, path in arm_sources.items()
        },
    }


def repo_relative(path: Path, repo_root: Path) -> str:
    """``path`` as a repo-relative POSIX string when it is inside the repo.

    Published artifacts are committed and read on other machines; an
    absolute path names the machine that wrote it, not the file.
    """
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _pricing_overridden() -> bool:
    import os

    return bool(os.environ.get("META_HARNESS_PRICING"))


# ── statistics ────────────────────────────────────────────────────────


def _wilson_bounds(passes: int, n: int, z: float) -> tuple[float, float]:
    """Unrounded Wilson score bounds for ``passes`` out of ``n``."""
    p = passes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_interval(passes: int, n: int, *, z: float = Z_95) -> dict[str, Any]:
    """95% Wilson score interval for one arm's pass rate.

    Unlike a Wald interval it does not collapse to zero width when every
    trial passed or every trial failed, which is exactly where a small
    task set is most likely to put an arm.
    """
    if n <= 0:
        return {"lower": None, "upper": None, "method": "wilson-score-95"}
    lower, upper = _wilson_bounds(passes, n, z)
    return {
        "estimate": round(passes / n, 6),
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "confidence": 0.95,
        "method": "wilson-score-95",
    }


def newcombe_diff_ci(
    passes_a: int, n_a: int, passes_b: int, n_b: int, *, z: float = Z_95
) -> dict[str, Any]:
    """95% Newcombe hybrid score interval for ``p_b - p_a``.

    Composed from each arm's Wilson interval (Newcombe 1998, method 10),
    so it keeps a sensible width when either arm sits at 0% or 100%.

    Reported with its assumptions rather than as a significance verdict:
    trials are clustered within tasks, so this interval is optimistic
    about independence.
    """
    if n_a <= 0 or n_b <= 0:
        return {
            "lower": None,
            "upper": None,
            "method": "newcombe-hybrid-score-95",
            "note": "no trials",
        }
    p_a, p_b = passes_a / n_a, passes_b / n_b
    l_a, u_a = _wilson_bounds(passes_a, n_a, z)
    l_b, u_b = _wilson_bounds(passes_b, n_b, z)
    diff = p_b - p_a
    lower = diff - math.sqrt((p_b - l_b) ** 2 + (u_a - p_a) ** 2)
    upper = diff + math.sqrt((u_b - p_b) ** 2 + (p_a - l_a) ** 2)
    return {
        "difference": round(diff, 6),
        "lower": round(max(-1.0, lower), 6),
        "upper": round(min(1.0, upper), 6),
        "confidence": 0.95,
        "method": "newcombe-hybrid-score-95",
        "assumptions": (
            "independent Bernoulli trials; trials are clustered within tasks, "
            "so this interval mis-states the design's precision. The direction "
            "is not guaranteed: it is usually too narrow, but a per-task effect "
            "that is consistent across tasks can make the cluster-aware "
            "interval the narrower of the two. Report both."
        ),
    }


#: Task-cluster bootstrap configuration. Both values are recorded in every
#: summary, because an interval nobody can recompute is not evidence.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260901

#: Cluster-robust inference is conventionally regarded as unreliable below
#: roughly this many clusters. It is a rule of thumb about when an interval
#: stops describing a population of tasks and starts describing the handful
#: of tasks in hand -- NOT a threshold above which the interval becomes
#: valid, and nothing in this module changes behaviour when it is crossed.
#: It exists so the limitation is reported rather than left to the reader.
MIN_INFORMATIVE_CLUSTERS = 30


def _percentile(ordered: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not ordered:
        raise ValueError("no values")
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def task_clusters(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_ids: list[str],
) -> dict[str, dict[str, list[bool]]]:
    """Group both arms' trials by ``task_id``.

    The cluster is the task, not the trial: repeated trials of one task
    are repeated looks at the same problem, and treating them as
    independent observations is what makes a trial-level interval look
    far more precise than the design can support.
    """
    clusters: dict[str, dict[str, list[bool]]] = {}
    for task_id in task_ids:
        baseline = [bool(r["passed"]) for r in baseline_rows if r["task_id"] == task_id]
        candidate = [
            bool(r["passed"]) for r in candidate_rows if r["task_id"] == task_id
        ]
        if baseline and candidate:
            clusters[task_id] = {"baseline": baseline, "candidate": candidate}
    return clusters


def cluster_bootstrap_diff_ci(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_ids: list[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile interval for ``p_candidate - p_baseline``, clustering by task.

    Tasks are the independent unit. Each resample draws ``len(clusters)``
    task ids with replacement and takes **every** trial of a drawn task,
    from both arms, keeping the within-task correlation the design
    actually has. The paired draw -- the same sampled task contributes its
    baseline *and* its candidate trials -- matters because both arms ran
    the identical task set, so the arms are not independent samples.

    What this fixes: a trial-level interval assumes every trial is an
    independent Bernoulli observation and states a precision the design
    cannot support. What it does **not** fix: with a handful of tasks the
    resampling distribution is coarse and driven by which of a few tasks
    happened to be drawn. Nor does it guarantee a wider interval than the
    trial-level one -- a per-task effect that is consistent across tasks can make
    this the narrower of the two, which is information, not reassurance.
    The interval is reported with ``clusters`` and ``informative`` so that
    limit travels with the number instead of being lost.

    Deterministic: seeded ``random.Random``, clusters iterated in sorted
    order, so the same rows and the same ``seed`` give the same interval on
    any machine.

    Raises ``ValueError`` on any row that is not ``metrics_source ==
    "measured"``. A scripted or mock trial must never reach a published
    interval.
    """
    for row in (*baseline_rows, *candidate_rows):
        source = row.get("metrics_source")
        if source != met.MEASURED:
            raise ValueError(
                f"cannot bootstrap a {source!r} trial into a measured interval"
            )

    clusters = task_clusters(baseline_rows, candidate_rows, task_ids)
    names = sorted(clusters)
    base: dict[str, Any] = {
        "method": "task-cluster-bootstrap-percentile",
        "cluster_unit": "task_id",
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "clusters": len(names),
        "cluster_sizes": {
            name: {
                "baseline_trials": len(clusters[name]["baseline"]),
                "candidate_trials": len(clusters[name]["candidate"]),
            }
            for name in names
        },
    }

    if len(names) < 2:
        return {
            **base,
            "difference": None,
            "lower": None,
            "upper": None,
            "informative": False,
            "note": (
                "fewer than two task clusters: there is nothing to resample "
                "over, so no cluster-aware interval exists for this design"
            ),
        }

    observed_b = sum(sum(clusters[n]["baseline"]) for n in names)
    observed_bn = sum(len(clusters[n]["baseline"]) for n in names)
    observed_c = sum(sum(clusters[n]["candidate"]) for n in names)
    observed_cn = sum(len(clusters[n]["candidate"]) for n in names)
    observed = observed_c / observed_cn - observed_b / observed_bn

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        b_pass = b_n = c_pass = c_n = 0
        for _ in names:
            picked = clusters[names[rng.randrange(len(names))]]
            b_pass += sum(picked["baseline"])
            b_n += len(picked["baseline"])
            c_pass += sum(picked["candidate"])
            c_n += len(picked["candidate"])
        if b_n and c_n:
            draws.append(c_pass / c_n - b_pass / b_n)

    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    informative = len(names) >= MIN_INFORMATIVE_CLUSTERS
    result = {
        **base,
        "difference": round(observed, 6),
        "lower": round(_percentile(draws, alpha), 6),
        "upper": round(_percentile(draws, 1.0 - alpha), 6),
        "informative": informative,
        "assumptions": (
            "tasks are resampled with replacement as the independent unit "
            "and every trial of a drawn task travels with it; both arms ran "
            "the same task set, so a drawn task contributes to both"
        ),
    }
    if not informative:
        result["limitation"] = (
            f"{len(names)} task clusters. Cluster-robust intervals are "
            f"conventionally regarded as unreliable below ~{MIN_INFORMATIVE_CLUSTERS} "
            "clusters: this interval describes the tasks in hand, and should "
            "not be read as an estimate for coding tasks in general. Widening "
            "it with easy tasks would raise the cluster count without adding "
            "evidence, so the task set is left as it is and the limit stated."
        )
    return result


def _arm_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trials = len(rows)
    passes = sum(1 for r in rows if r.get("passed"))
    aggregate = met.aggregate_trials(rows, metrics_source=met.MEASURED) if rows else {}
    return {
        "trials": trials,
        "passes": passes,
        "errored_trials": sum(1 for r in rows if met.is_error_row(r)),
        "accuracy": round(passes / trials, 6) if trials else None,
        "accuracy_ci": wilson_interval(passes, trials),
        "models": _row_models(rows),
        "tokens": aggregate.get("tokens"),
        "mean_total_tokens_per_trial": aggregate.get("mean_total_tokens_per_trial"),
        "median_total_tokens_per_trial": aggregate.get(
            "median_total_tokens_per_trial"
        ),
        "total_llm_calls": aggregate.get("total_llm_calls"),
        "total_wall_time_s": aggregate.get("total_wall_time_s"),
        "total_cost_usd": aggregate.get("total_cost_usd"),
        "cost_complete": aggregate.get("cost_complete", False),
    }


def _per_task(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_ids: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task_id in task_ids:
        b = [r for r in baseline_rows if r["task_id"] == task_id]
        c = [r for r in candidate_rows if r["task_id"] == task_id]
        b_pass = sum(1 for r in b if r["passed"])
        c_pass = sum(1 for r in c if r["passed"])
        b_acc = b_pass / len(b) if b else None
        c_acc = c_pass / len(c) if c else None
        out[task_id] = {
            "baseline_passes": b_pass,
            "baseline_trials": len(b),
            "baseline_accuracy": round(b_acc, 6) if b_acc is not None else None,
            "candidate_passes": c_pass,
            "candidate_trials": len(c),
            "candidate_accuracy": round(c_acc, 6) if c_acc is not None else None,
            "percentage_point_delta": (
                round((c_acc - b_acc) * 100, 4)
                if b_acc is not None and c_acc is not None
                else None
            ),
        }
    return out


def summarize(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_ids: list[str],
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    """Derive the whole summary from raw trial rows.

    This is the only place the headline number is produced, and it has
    no input other than the rows themselves.
    """
    baseline = _arm_stats(baseline_rows)
    candidate = _arm_stats(candidate_rows)
    clusters = task_clusters(baseline_rows, candidate_rows, task_ids)
    delta_pp = (
        round((candidate["accuracy"] - baseline["accuracy"]) * 100, 4)
        if baseline["accuracy"] is not None and candidate["accuracy"] is not None
        else None
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_passes": baseline["passes"],
        "baseline_trials": baseline["trials"],
        "baseline_accuracy": baseline["accuracy"],
        "candidate_passes": candidate["passes"],
        "candidate_trials": candidate["trials"],
        "candidate_accuracy": candidate["accuracy"],
        "absolute_percentage_point_delta": delta_pp,
        "total_trials": baseline["trials"] + candidate["trials"],
        "errored_trials": baseline["errored_trials"] + candidate["errored_trials"],
        "distinct_tasks": len(clusters),
        "difference_ci": newcombe_diff_ci(
            baseline["passes"],
            baseline["trials"],
            candidate["passes"],
            candidate["trials"],
        ),
        "cluster_bootstrap_ci": cluster_bootstrap_diff_ci(
            baseline_rows=baseline_rows,
            candidate_rows=candidate_rows,
            task_ids=task_ids,
        )
        if clusters
        else None,
        "per_task": _per_task(baseline_rows, candidate_rows, task_ids),
        "arms": {"baseline": baseline, "candidate": candidate},
        "metrics_source": met.MEASURED,
    }


# ── completeness, protocol equality, leakage ──────────────────────────

#: Fields every raw trial row must carry to be counted. A row missing one
#: is not a zero — it is a trial whose outcome is unknown, and averaging
#: over it would quietly bias the result.
REQUIRED_ROW_FIELDS = (
    "task_id",
    "trial",
    "passed",
    "score",
    "metrics_source",
    "total_tokens",
    "wall_time_s",
)


def trial_completeness(
    rows: list[dict[str, Any]],
    *,
    task_ids: list[str],
    trials_per_task: int,
    arm: str,
) -> dict[str, Any]:
    """Check an arm's rows against the protocol it claims to have run.

    Reports missing trials, duplicated (task, trial) pairs, trials that
    raised instead of finishing, and rows missing required fields. A
    result computed over an incomplete arm is not the result the protocol
    describes, so this is checked and published rather than assumed.

    A trial that raised is accounted for — it occupies its (task, trial)
    slot, so it is not also reported missing — but the arm that contains
    one is not complete: the protocol asked for an outcome there and got
    an exception.
    """
    expected = {(task_id, t) for task_id in task_ids for t in range(1, trials_per_task + 1)}
    seen: dict[tuple[str, int], int] = {}
    malformed: list[dict[str, Any]] = []
    errored: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if met.is_error_row(row):
            missing = [f for f in ("task_id", "trial") if row.get(f) is None]
        else:
            missing = [f for f in REQUIRED_ROW_FIELDS if row.get(f) is None]
        if missing:
            malformed.append({"row_index": index, "missing_fields": missing})
            continue
        key = (str(row["task_id"]), int(row["trial"]))
        seen[key] = seen.get(key, 0) + 1
        if met.is_error_row(row):
            errored.append(
                {
                    "trial": f"{key[0]}/trial-{key[1]}",
                    "error": str(row.get("error") or "")[:300],
                }
            )

    duplicates = sorted(f"{t}/trial-{n}" for (t, n), count in seen.items() if count > 1)
    missing_trials = sorted(f"{t}/trial-{n}" for (t, n) in expected - set(seen))
    unexpected = sorted(f"{t}/trial-{n}" for (t, n) in set(seen) - expected)

    return {
        "arm": arm,
        "expected_trials": len(expected),
        "observed_trials": len(rows),
        "distinct_trials": len(seen),
        "missing_trials": missing_trials,
        "duplicate_trials": duplicates,
        "unexpected_trials": unexpected,
        "errored_trials": errored,
        "malformed_rows": malformed,
        "complete": (
            not missing_trials
            and not duplicates
            and not unexpected
            and not errored
            and not malformed
            and len(rows) == len(expected)
        ),
    }


def _row_models(rows: list[dict[str, Any]]) -> list[str | None]:
    """The distinct models that served an arm's trials, as the rows say.

    ``None`` stands for a row that does not say, and is kept rather than
    dropped: an arm whose model is unknown cannot be shown to match.
    """
    models = {row.get("model") for row in rows}
    return sorted(models, key=lambda m: (m is None, str(m)))


def check_protocol_equality(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_ids: list[str],
    trials_per_task: int,
) -> dict[str, Any]:
    """Confirm both arms ran the same protocol under the same conditions.

    A pass-rate delta is only attributable to the harness if everything
    else was held constant. This checks the parts that are checkable from
    the artifacts: identical task set, identical per-task trial counts,
    one single ``metrics_source`` across both arms (so a mock trial can
    never be folded into a measured comparison), no trial that raised,
    and the same model.

    The model check reads the ``model`` each row records — the model that
    actually answered the trial's calls — and compares the two arms' sets.
    It takes no model argument: a caller-supplied pair was compared with
    itself, so the check could not fail. A row with no recorded model
    makes the check fail, because "unknown" cannot be shown to match.
    """
    baseline_complete = trial_completeness(
        baseline_rows, task_ids=task_ids, trials_per_task=trials_per_task, arm="baseline"
    )
    candidate_complete = trial_completeness(
        candidate_rows,
        task_ids=task_ids,
        trials_per_task=trials_per_task,
        arm="candidate",
    )

    def _per_task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row.get("task_id"))] = counts.get(str(row.get("task_id")), 0) + 1
        return counts

    baseline_counts = _per_task_counts(baseline_rows)
    candidate_counts = _per_task_counts(candidate_rows)
    sources = {
        str(r.get("metrics_source")) for r in (*baseline_rows, *candidate_rows)
    }
    baseline_models = _row_models(baseline_rows)
    candidate_models = _row_models(candidate_rows)
    errored = sum(
        1 for r in (*baseline_rows, *candidate_rows) if met.is_error_row(r)
    )

    checks = {
        "same_task_set": sorted(baseline_counts) == sorted(candidate_counts),
        "same_trials_per_task": baseline_counts == candidate_counts,
        "same_model": (
            bool(baseline_models)
            and baseline_models == candidate_models
            and None not in baseline_models
        ),
        "single_metrics_source": len(sources) == 1,
        "measured_only": sources == {met.MEASURED},
        "no_errored_trials": errored == 0,
        "baseline_complete": baseline_complete["complete"],
        "candidate_complete": candidate_complete["complete"],
    }
    return {
        "checks": checks,
        "identical_protocol": all(checks.values()),
        "metrics_sources": sorted(sources),
        "models": {"baseline": baseline_models, "candidate": candidate_models},
        "errored_trials": {
            "baseline": len(baseline_complete["errored_trials"]),
            "candidate": len(candidate_complete["errored_trials"]),
        },
        "baseline_completeness": baseline_complete,
        "candidate_completeness": candidate_complete,
        "baseline_trials_per_task": baseline_counts,
        "candidate_trials_per_task": candidate_counts,
    }


class LeakageError(RuntimeError):
    """Holdout material reached a process that may only see the search set.

    Raised when a holdout task appears in a search-only context, and by
    the proposer's isolation checks when its session touched holdout
    paths or the frozen task set changed while it ran.
    """


def check_task_set_isolation(
    *, search_dir: Path, holdout_dir: Path
) -> dict[str, Any]:
    """Confirm the search set and the holdout set share no task.

    A holdout number means "generalisation" only while the proposer never
    optimised against those tasks. Overlap makes the holdout arm a second
    search-set measurement wearing a different name.
    """
    search = {d.name for d in _task_dirs(search_dir)}
    holdout = {d.name for d in _task_dirs(holdout_dir)}
    overlap = sorted(search & holdout)
    return {
        "search_tasks": sorted(search),
        "holdout_tasks": sorted(holdout),
        "overlapping_tasks": overlap,
        "disjoint": not overlap,
    }


def _task_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(d for d in path.iterdir() if d.is_dir() and (d / "task.json").exists())


def reported_metric_sentence(summary: dict[str, Any]) -> str:
    """The one sentence a reader may quote, built only from the summary."""
    delta = summary.get("absolute_percentage_point_delta")
    if delta is None:
        return "No measured result: the experiment produced no trials."
    if delta > 0:
        change = f"Improved agent pass rate by {delta:+.1f} percentage points"
    elif delta < 0:
        change = f"Reduced agent pass rate by {abs(delta):.1f} percentage points"
    else:
        change = "Left agent pass rate unchanged (+0.0 percentage points)"
    sentence = (
        f"{change} "
        f"across {summary['total_trials']} task trials "
        f"({summary['baseline_passes']}/{summary['baseline_trials']} baseline vs "
        f"{summary['candidate_passes']}/{summary['candidate_trials']} evolved)."
    )
    # The point estimate never travels without the cluster-aware interval
    # and the cluster count: those are what say how much of this number is
    # the harness and how much is five tasks.
    ci = summary.get("cluster_bootstrap_ci") or {}
    if ci.get("lower") is not None:
        sentence += (
            f" 95% task-clustered bootstrap interval "
            f"[{ci['lower'] * 100:+.1f}, {ci['upper'] * 100:+.1f}] pp over "
            f"{ci['clusters']} evaluation tasks."
        )
    errored = summary.get("errored_trials") or 0
    if errored:
        sentence += f" {errored} trial(s) raised and are counted as failures."
    return sentence


def render_report(summary: dict[str, Any]) -> str:
    """Human-readable console report. Every number comes from the summary."""
    lines = [
        f"Baseline  ({summary['baseline_label']}): "
        f"{summary['baseline_passes']}/{summary['baseline_trials']} = "
        f"{_pct(summary['baseline_accuracy'])}",
        f"Candidate ({summary['candidate_label']}): "
        f"{summary['candidate_passes']}/{summary['candidate_trials']} = "
        f"{_pct(summary['candidate_accuracy'])}",
    ]
    delta = summary.get("absolute_percentage_point_delta")
    if delta is not None:
        lines.append(f"Absolute change: {delta:+.1f} percentage points")
    ci = summary.get("difference_ci") or {}
    if ci.get("lower") is not None:
        lines.append(
            f"95% CI on the difference: "
            f"[{ci['lower'] * 100:+.1f}, {ci['upper'] * 100:+.1f}] pp "
            f"({ci['method']}; assumes independent trials)"
        )
    cluster_ci = summary.get("cluster_bootstrap_ci") or {}
    if cluster_ci.get("lower") is not None:
        lines.append(
            f"95% CI, clustering by task: "
            f"[{cluster_ci['lower'] * 100:+.1f}, {cluster_ci['upper'] * 100:+.1f}] pp "
            f"({cluster_ci['method']}, {cluster_ci['resamples']} resamples, "
            f"seed {cluster_ci['seed']})"
        )
    lines.append(f"Total trials: {summary['total_trials']}")
    if summary.get("errored_trials"):
        lines.append(
            f"Errored trials (raised; counted as failures): {summary['errored_trials']}"
        )
    if summary.get("distinct_tasks") is not None:
        lines.append(f"Distinct evaluation tasks (clusters): {summary['distinct_tasks']}")
    if cluster_ci.get("limitation"):
        lines.append("")
        lines.append(f"LIMITATION: {cluster_ci['limitation']}")
    lines.append("")
    lines.append("Per task:")
    for task_id, row in summary.get("per_task", {}).items():
        lines.append(
            f"  {task_id}: {row['baseline_passes']}/{row['baseline_trials']} -> "
            f"{row['candidate_passes']}/{row['candidate_trials']} "
            f"({row['percentage_point_delta']:+.1f} pp)"
            if row["percentage_point_delta"] is not None
            else f"  {task_id}: incomplete"
        )
    lines.append("")
    lines.append("Reported metric:")
    lines.append(f'  "{reported_metric_sentence(summary)}"')
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


# ── raw result IO ─────────────────────────────────────────────────────


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write raw trial rows as JSONL, one object per trial.

    Per-call detail is dropped: it is large, and the summary depends
    only on the per-trial aggregates, which stay verifiable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            slim = {k: v for k, v in row.items() if k != "calls"}
            fh.write(json.dumps(slim, default=str) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read raw trial rows back from JSONL."""
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


# ── the experiment ────────────────────────────────────────────────────


ARMS = ("baseline", "candidate")


def interleaved_trial_specs(
    *,
    factories: dict[str, Callable[[], Any]],
    task_dirs: list[Path],
    trials: int,
    output_dir: Path,
    experiment_id: str,
    recording_dir_for: Callable[[str, str, int], Path | None] | None = None,
) -> list[bench.TrialSpec]:
    """Both arms' trials in one schedule, alternating which arm goes first.

    Order is trial-major: every task's trial 1 for both arms, then every
    task's trial 2, and so on. Within each (task, trial) pair the arm that
    starts first alternates, so neither arm systematically gets the
    earlier slot. Run sequentially, arm by arm, anything that drifts over
    the run's wall-clock time — provider latency, rate limiting, a model
    update — would land on one arm and read as a harness effect.
    """
    specs: list[bench.TrialSpec] = []
    task_specs = {
        td.name: json.loads((td / "task.json").read_text()) for td in task_dirs
    }
    for trial in range(1, trials + 1):
        for index, td in enumerate(task_dirs):
            order = ARMS if (trial + index) % 2 else tuple(reversed(ARMS))
            for arm in order:
                specs.append(
                    bench.TrialSpec(
                        task_dir=td,
                        spec=task_specs[td.name],
                        trial=trial,
                        harness_factory=factories[arm],
                        thread_id=met.inner_thread_id(
                            run_id=experiment_id,
                            thread_id=experiment_id,
                            candidate=arm,
                            task_id=td.name,
                            trial=trial,
                        ),
                        trace_dir=output_dir / "traces" / arm / f"{td.name}-trial-{trial}",
                        recording_dir=(
                            recording_dir_for(arm, td.name, trial)
                            if recording_dir_for is not None
                            else None
                        ),
                        arm=arm,
                    )
                )
    return specs


def declared_model(factory: Callable[[], Any]) -> str | None:
    """The model a harness factory says it will use, if it can say.

    Provenance only: what served each trial is read from the trial rows.
    A factory that cannot be built here (the real harnesses refuse to
    construct without an API key) reports ``None`` rather than a guess.
    """
    model = getattr(factory, "MODEL", None)
    if model is not None:
        return str(model)
    try:
        return getattr(factory(), "MODEL", None)
    except Exception:  # noqa: BLE001 — provenance must not fail the run
        return None


async def run_two_arm_experiment(
    *,
    repo_root: Path,
    config: dict[str, Any],
    experiment_id: str,
    tasks_dir: Path,
    baseline_label: str,
    candidate_label: str,
    baseline_factory: Callable[[], Any],
    candidate_factory: Callable[[], Any],
    arm_sources: dict[str, Path | None],
    model: str,
    trials: int,
    workers: int,
    output_dir: Path,
    checkpointer: Any = None,
    tracker: Any = None,
    on_trial: Callable[[str, dict[str, Any]], None] | None = None,
    record_trials_per_task: int = 0,
    arm_models: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Run both arms, derive the summary, check the methodology, persist.

    The single place a two-arm result is produced, so ``experiment`` and
    ``canonical-experiment`` cannot drift into measuring different things.
    Both arms share one worker pool and are interleaved trial by trial
    (see ``interleaved_trial_specs``).

    ``record_trials_per_task`` tapes the first N trials of each task in
    each arm for exact replay. Recording is bounded because a tape holds
    every model response verbatim; a handful is enough to demonstrate
    replay, and taping every trial is a large artifact for no extra
    evidence.

    ``model`` and ``arm_models`` are the declared models, recorded as
    provenance. The protocol check does not trust them: it compares the
    models the trial rows say actually served each arm.
    """
    from app.meta_harness import tracking as trk  # noqa: PLC0415 — optional sink

    task_dirs = bench.discover_tasks(tasks_dir)
    if not task_dirs:
        raise ValueError(f"no tasks found in {tasks_dir}")
    task_ids = [d.name for d in task_dirs]

    def _recording_dir_for(arm: str, task_id: str, trial: int) -> Path | None:
        if trial > record_trials_per_task:
            return None
        return output_dir / "recordings" / arm / f"{task_id}-trial-{trial}"

    specs = interleaved_trial_specs(
        factories={"baseline": baseline_factory, "candidate": candidate_factory},
        task_dirs=task_dirs,
        trials=trials,
        output_dir=output_dir,
        experiment_id=experiment_id,
        recording_dir_for=(
            _recording_dir_for
            if record_trials_per_task > 0 and checkpointer is not None
            else None
        ),
    )
    rows = await bench.run_trial_specs(
        specs,
        workers=workers,
        checkpointer=checkpointer,
        on_trial=(lambda row: on_trial(row["arm"], row)) if on_trial else None,
    )
    def _arm_rows(arm: str) -> list[dict[str, Any]]:
        return sorted(
            (r for r in rows if r.get("arm") == arm),
            key=lambda r: (str(r["task_id"]), int(r["trial"])),
        )

    baseline_rows = _arm_rows("baseline")
    candidate_rows = _arm_rows("candidate")

    resolved_config = {
        **config,
        "experiment_id": experiment_id,
        "arms": {"baseline": baseline_label, "candidate": candidate_label},
        "trials_per_task": trials,
        "workers": workers,
        "schedule": "interleaved",
        "task_set": str(tasks_dir.relative_to(repo_root)).replace("\\", "/")
        if tasks_dir.is_relative_to(repo_root)
        else str(tasks_dir),
        "tasks": task_ids,
        "total_trials": len(baseline_rows) + len(candidate_rows),
        "recorded_trials_per_task": record_trials_per_task,
    }
    environment = capture_environment(
        repo_root=repo_root,
        model=model,
        tasks=task_dirs,
        arm_sources=arm_sources,
        arm_models=arm_models,
    )
    summary = summarize(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        task_ids=task_ids,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    validation = check_protocol_equality(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        task_ids=task_ids,
        trials_per_task=trials,
    )
    paths = write_results(
        output_dir=output_dir,
        config=resolved_config,
        environment=environment,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        summary=summary,
        validation=validation,
    )
    if tracker is not None:
        trk.log_experiment(
            tracker,
            summary=summary,
            environment=environment,
            artifacts={
                f"{experiment_id}-summary": paths["summary"],
                f"{experiment_id}-baseline-rows": paths["baseline_results"],
                f"{experiment_id}-candidate-rows": paths["candidate_results"],
            },
        )

    return {
        "experiment_id": experiment_id,
        "config": resolved_config,
        "environment": environment,
        "summary": summary,
        "validation": validation,
        "paths": {k: str(v) for k, v in paths.items()},
        "output_dir": str(output_dir),
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
    }


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the committed canonical protocol."""
    config = json.loads(path.read_text())
    for key in ("experiment", "task_set", "trials_per_task", "workers"):
        if key not in config:
            raise ValueError(f"{path} is missing required key {key!r}")
    if int(config["trials_per_task"]) < 1:
        raise ValueError("trials_per_task must be >= 1")
    return config


def write_results(
    *,
    output_dir: Path,
    config: dict[str, Any],
    environment: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist config, provenance, raw rows, the derived summary, and the
    methodology checks that say whether the summary means what it claims."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": output_dir / "config.json",
        "environment": output_dir / "environment.json",
        "baseline_results": output_dir / "baseline-results.jsonl",
        "candidate_results": output_dir / "candidate-results.jsonl",
        "summary": output_dir / "summary.json",
        "report": output_dir / "REPORT.md",
    }
    runs_mod.write_json_atomic(paths["config"], config)
    runs_mod.write_json_atomic(paths["environment"], environment)
    write_rows(paths["baseline_results"], baseline_rows)
    write_rows(paths["candidate_results"], candidate_rows)
    runs_mod.write_json_atomic(paths["summary"], summary)
    if validation is not None:
        paths["validation"] = output_dir / "validation.json"
        runs_mod.write_json_atomic(paths["validation"], validation)
    paths["report"].write_text(
        _report_markdown(config, environment, summary, validation), encoding="utf-8"
    )
    return paths


def _validation_markdown(validation: dict[str, Any] | None) -> str:
    if not validation:
        return "_No methodology checks were recorded for this run._"
    lines = [
        "| Check | Result |",
        "|---|---|",
        *(
            f"| `{name}` | {'PASS' if ok else 'FAIL'} |"
            for name, ok in (validation.get("checks") or {}).items()
        ),
    ]
    if not validation.get("identical_protocol", False):
        lines.append("")
        lines.append(
            "**The two arms did not run an identical protocol.** The delta "
            "above is not attributable to the harness alone; see "
            "`validation.json`."
        )
    return "\n".join(lines)


def _statistics_limitation_markdown(summary: dict[str, Any]) -> str:
    """State the cluster-count limitation where a reader cannot miss it."""
    ci = summary.get("cluster_bootstrap_ci") or {}
    clusters = ci.get("clusters")
    if not clusters:
        return (
            "**No cluster-aware interval.** This experiment has no task with "
            "trials in both arms, so there is nothing to resample over."
        )
    if ci.get("informative"):
        return (
            f"Clustered over **{clusters} evaluation tasks**, "
            f"{ci['resamples']} resamples, seed `{ci['seed']}`."
        )
    return (
        f"> **Read the interval with this in mind: there are only "
        f"{clusters} task clusters.**\n"
        f">\n"
        f"> {ci.get('limitation', '')}\n"
        f">\n"
        f"> The trial count ({summary.get('total_trials')}) is large; the "
        f"number of *independent* units is not. Precision is bounded by the "
        f"number of tasks, not by the number of trials, and adding trials to "
        f"these same tasks cannot narrow it."
    )


def _report_markdown(
    config: dict[str, Any],
    environment: dict[str, Any],
    summary: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> str:
    git = environment.get("git", {})
    return f"""# {config.get('experiment', 'experiment')} — measured result

Generated by `meta-harness experiment`. Every number below is derived
from `baseline-results.jsonl` and `candidate-results.jsonl` by
`app.meta_harness.experiment.summarize`; nothing here is hand-entered.

- Commit: `{git.get('commit')}` (dirty: `{git.get('dirty')}`)
- Model: `{environment.get('model')}`
- Captured: `{environment.get('captured_at')}`
- Protocol: {len(environment.get('tasks', []))} tasks x \
{config.get('trials_per_task')} trials per arm, arms interleaved under one \
scheduler

```
{render_report(summary)}
```

## Methodology checks

Both arms must have run the identical protocol for the delta to be
attributable to the harness. These are derived from the raw rows:

{_validation_markdown(validation)}

## Reproducing

```bash
uv run meta-harness experiment --config benchmarks/pass-rate/config.json \\
    --candidate <candidate-name>
```

## Statistics

Two intervals are reported, and they answer different questions:

- **Newcombe hybrid score, on the difference in proportions** (built
  from each arm's Wilson score interval, which is also published per
  arm). Unlike a Wald interval it keeps a real width when an arm passes
  or fails every trial. It assumes every trial is an independent
  Bernoulli observation. It is not — repeated trials of one task are
  repeated looks at the same problem — so this interval mis-states
  precision and is kept for comparison with the naive reading. It is
  usually too narrow, but that direction is not guaranteed.
- **Task-cluster bootstrap (percentile).** Resamples **tasks** with
  replacement as the independent unit; every trial of a drawn task travels
  with it, and a drawn task contributes to both arms because both arms ran
  the identical task set. Deterministic under the recorded seed, so this
  interval can be recomputed from the published rows alone.

{_statistics_limitation_markdown(summary)}

No p-value and no significance verdict is reported. With this many task
clusters a hypothesis test would not be defensible, and stating an effect
with a cluster-aware interval is the honest form of the result.

## Limitations

- Results are tied to the task hashes in `environment.json`. Changing a
  task invalidates comparison with this experiment.
- This measures the frozen search tasks. Generalisation to unseen tasks is
  a separate holdout measurement.
"""
