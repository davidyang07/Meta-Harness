"""Pareto-frontier computation on (accuracy × tokens).

INTERFACES.md §2.2 frontier_val.json shape:
- ``candidates``: list of ``{name, accuracy, avg_tokens, metrics_source,
  dominated_by_names}``
- ``_pareto_names``: convenience subset where ``dominated_by_names == []``
- ``_best``: highest-accuracy candidate (ties broken by lower tokens)
- ``per_task``: per-task best candidate + pass_rate

Domination rule (maximize accuracy, minimize tokens): ``A`` dominates
``B`` iff ``A.accuracy >= B.accuracy`` AND ``A.avg_tokens <= B.avg_tokens``
AND at least one of those is strict.

**Unknown token counts do not participate in the cost axis.** A candidate
whose ``avg_tokens`` is ``None`` (never measured) is compared on accuracy
alone: it can be dominated by a strictly more accurate candidate, but it
can never dominate anything on a cost it never paid. Treating "unknown"
as ``0`` would make an unmeasured candidate dominate every real one.
"""

from __future__ import annotations

from typing import Any


def frontier_entry(name: str, scores: dict[str, Any]) -> dict[str, Any]:
    """Project a candidate's eval-result into one frontier row.

    Prefers the mean measured tokens-per-trial. Falls back to ``None``
    (unknown) rather than 0 so the domination rule can tell the two
    apart.
    """
    tokens = scores.get("mean_total_tokens_per_trial")
    if tokens is None:
        tokens = scores.get("median_total_tokens_per_trial")
    return {
        "name": name,
        "accuracy": float(scores.get("accuracy") or 0.0),
        "avg_tokens": float(tokens) if tokens is not None else None,
        "metrics_source": scores.get("metrics_source"),
    }


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True iff candidate ``a`` dominates ``b`` on (accuracy, tokens)."""
    a_acc, b_acc = a["accuracy"], b["accuracy"]
    if a_acc < b_acc:
        return False

    a_tok, b_tok = a.get("avg_tokens"), b.get("avg_tokens")
    if a_tok is None or b_tok is None:
        # One side has no measured cost: fall back to a strict accuracy
        # comparison so an unmeasured candidate never wins on cost.
        return a_acc > b_acc

    if a_tok > b_tok:
        return False
    return a_acc > b_acc or a_tok < b_tok


def compute_pareto(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate each candidate with ``dominated_by_names``. Returns the
    same list, with each entry mutated to include the field."""
    for c in candidates:
        c["dominated_by_names"] = [
            other["name"]
            for other in candidates
            if other["name"] != c["name"] and dominates(other, c)
        ]
    return candidates


def pareto_names(candidates: list[dict[str, Any]]) -> list[str]:
    """Names of candidates with ``dominated_by_names == []``."""
    return [c["name"] for c in candidates if not c.get("dominated_by_names")]


def best_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Highest-accuracy candidate; ties broken by lowest avg_tokens.

    A candidate with unknown tokens loses a tie to one with a measured
    count, since we can't claim it is cheaper.
    """
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (
            c["accuracy"],
            -(c["avg_tokens"] if c.get("avg_tokens") is not None else float("inf")),
        ),
    )


def build_frontier_val(
    iteration: int,
    candidates: list[dict[str, Any]],
    per_task_bests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the full frontier_val.json shape."""
    annotated = compute_pareto(list(candidates))
    sources = {c.get("metrics_source") for c in annotated if c.get("metrics_source")}
    return {
        "iteration": iteration,
        "candidates": annotated,
        "_pareto_names": pareto_names(annotated),
        "_best": best_candidate(annotated),
        # A frontier built from a single source can be labelled; a mixed
        # one is flagged so no UI presents it as measured.
        "metrics_source": sources.pop() if len(sources) == 1 else "mixed",
        "per_task": per_task_bests,
    }
