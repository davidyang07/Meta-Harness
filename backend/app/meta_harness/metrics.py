"""Measured token / cost / call metrics for inner-loop benchmarking.

Three levels, all derived mechanically from the level below:

- **per LLM call** — ``UsageRecorder`` wraps a harness' ``_call_llm`` and
  records the Anthropic SDK's ``usage`` block for every request.
- **per trial** — ``UsageRecorder.to_trial_row`` folds the calls of one
  (task, trial) into a row.
- **per candidate / run** — ``aggregate_trials`` folds trial rows into
  totals, means and medians.

Two rules the rest of the codebase relies on:

1. **Unknown cost is ``None``, never ``0.0``.** A model with no
   configured price yields ``cost_usd: None`` and sets
   ``cost_complete: False`` on the aggregate. Writing ``$0.00`` for
   "we didn't measure it" is how a Pareto frontier silently starts
   optimising against a fiction.
2. **Mock and measured numbers never mix.** Every payload carries
   ``metrics_source`` (``"measured"`` | ``"mock"``), and
   ``aggregate_trials`` refuses to fold rows whose source disagrees with
   the declared one.

Pricing is configuration, not a constant: ``META_HARNESS_PRICING`` may
point at a JSON file of ``{model: {input_usd_per_mtok, output_usd_per_mtok,
cache_write_usd_per_mtok, cache_read_usd_per_mtok}}`` to override or
extend the built-in table.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

MetricsSource = str  # "measured" | "mock"

MEASURED = "measured"
MOCK = "mock"


# ── pricing ───────────────────────────────────────────────────────────

# USD per million tokens. Extend or override via META_HARNESS_PRICING.
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input_usd_per_mtok": 1.0,
        "output_usd_per_mtok": 5.0,
        "cache_write_usd_per_mtok": 1.25,
        "cache_read_usd_per_mtok": 0.1,
    },
}

_pricing_cache: dict[str, dict[str, float]] | None = None


def load_pricing() -> dict[str, dict[str, float]]:
    """Return the model→price table, honouring ``META_HARNESS_PRICING``."""
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache
    table = {k: dict(v) for k, v in _DEFAULT_PRICING.items()}
    override = os.environ.get("META_HARNESS_PRICING")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(
                f"META_HARNESS_PRICING points at a missing file: {path}"
            )
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError(
                f"META_HARNESS_PRICING must contain a JSON object, got {type(loaded).__name__}"
            )
        for model, prices in loaded.items():
            table[str(model)] = {str(k): float(v) for k, v in dict(prices).items()}
    _pricing_cache = table
    return table


def reset_pricing_cache() -> None:
    """Drop the memoised pricing table. Used by tests that set the env var."""
    global _pricing_cache
    _pricing_cache = None


def call_cost_usd(call: dict[str, Any]) -> float | None:
    """Cost of one recorded LLM call, or ``None`` when the model has no price."""
    prices = load_pricing().get(str(call.get("model") or ""))
    if not prices:
        return None
    per_mtok = 1_000_000.0
    total = (
        int(call.get("input_tokens") or 0) * prices.get("input_usd_per_mtok", 0.0)
        + int(call.get("output_tokens") or 0) * prices.get("output_usd_per_mtok", 0.0)
        + int(call.get("cache_creation_input_tokens") or 0)
        * prices.get("cache_write_usd_per_mtok", 0.0)
        + int(call.get("cache_read_input_tokens") or 0)
        * prices.get("cache_read_usd_per_mtok", 0.0)
    ) / per_mtok
    return round(total, 8)


# ── per-call recording ────────────────────────────────────────────────


def _usage_field(usage: Any, name: str) -> int:
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class UsageRecorder:
    """Collects per-LLM-call usage for a single inner-loop trial."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, *, response: Any, model: str, latency_s: float) -> None:
        """Record one Anthropic ``Message`` response.

        Mocked responses in tests often carry no ``usage`` block at all;
        those record as a call with zero tokens rather than raising, so
        deterministic tests keep working.
        """
        usage = getattr(response, "usage", None)
        call = {
            "index": len(self.calls) + 1,
            "model": str(getattr(response, "model", None) or model),
            "input_tokens": _usage_field(usage, "input_tokens"),
            "output_tokens": _usage_field(usage, "output_tokens"),
            "cache_creation_input_tokens": _usage_field(
                usage, "cache_creation_input_tokens"
            ),
            "cache_read_input_tokens": _usage_field(usage, "cache_read_input_tokens"),
            "latency_s": round(latency_s, 4),
            "has_usage": usage is not None,
        }
        call["total_tokens"] = (
            call["input_tokens"]
            + call["output_tokens"]
            + call["cache_creation_input_tokens"]
            + call["cache_read_input_tokens"]
        )
        call["cost_usd"] = call_cost_usd(call)
        self.calls.append(call)

    def totals(self) -> dict[str, Any]:
        """Sum the recorded calls. ``cost_usd`` is None if any call is unpriced."""
        costs = [c["cost_usd"] for c in self.calls]
        cost_known = bool(costs) and all(c is not None for c in costs)
        return {
            "llm_calls": len(self.calls),
            "input_tokens": sum(c["input_tokens"] for c in self.calls),
            "output_tokens": sum(c["output_tokens"] for c in self.calls),
            "cache_creation_input_tokens": sum(
                c["cache_creation_input_tokens"] for c in self.calls
            ),
            "cache_read_input_tokens": sum(
                c["cache_read_input_tokens"] for c in self.calls
            ),
            "total_tokens": sum(c["total_tokens"] for c in self.calls),
            "cost_usd": round(sum(costs), 8) if cost_known else None,  # type: ignore[arg-type]
        }

    def to_trial_row(
        self,
        *,
        task_id: str,
        trial: int,
        passed: bool,
        score: float,
        wall_time_s: float,
    ) -> dict[str, Any]:
        """Fold this trial's calls into one raw benchmark row."""
        totals = self.totals()
        return {
            "task_id": task_id,
            "trial": trial,
            "passed": bool(passed),
            "score": float(score),
            "wall_time_s": round(wall_time_s, 3),
            "metrics_source": MEASURED,
            "calls": self.calls,
            **totals,
        }


def instrument_harness(harness: Any, recorder: UsageRecorder) -> None:
    """Wrap ``harness._call_llm`` so every request is recorded.

    Patches the instance, not the class, so concurrent trials of the
    same candidate each keep their own recorder.
    """
    original: Callable[..., Any] = harness._call_llm

    async def _recording_call_llm(*args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        response = await original(*args, **kwargs)
        recorder.record(
            response=response,
            model=getattr(harness, "MODEL", "unknown"),
            latency_s=time.monotonic() - started,
        )
        return response

    harness._call_llm = _recording_call_llm  # type: ignore[method-assign]


# ── mock rows ─────────────────────────────────────────────────────────


def mock_trial_metrics(
    *, task_id: str, trial: int, passed: bool, iteration: int
) -> dict[str, Any]:
    """A synthetic trial row, unmistakably tagged as mock.

    Token counts are deterministic placeholders. ``cost_usd`` is ``None``
    because no model was called — a mock trial has no real cost, and
    writing 0.0 would be indistinguishable from a measured free call.
    """
    total = 20_000 + iteration * 500 + trial * 10
    return {
        "task_id": task_id,
        "trial": trial,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "wall_time_s": 0.05,
        "metrics_source": MOCK,
        "llm_calls": 0,
        "input_tokens": int(total * 0.8),
        "output_tokens": int(total * 0.2),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_tokens": total,
        "cost_usd": None,
        "calls": [],
    }


# ── aggregation ───────────────────────────────────────────────────────


def aggregate_trials(
    rows: list[dict[str, Any]], *, metrics_source: MetricsSource
) -> dict[str, Any]:
    """Aggregate raw trial rows into candidate-level metrics.

    Raises ``ValueError`` if any row's ``metrics_source`` disagrees with
    the declared one — mixing a mock trial into a measured candidate is
    a correctness bug, not something to average over.
    """
    for row in rows:
        got = row.get("metrics_source")
        if got != metrics_source:
            raise ValueError(
                f"cannot aggregate {got!r} trial into {metrics_source!r} result"
            )

    totals = [int(r.get("total_tokens") or 0) for r in rows]
    costs = [r.get("cost_usd") for r in rows]
    cost_complete = bool(costs) and all(c is not None for c in costs)

    return {
        "total_trials": len(rows),
        "passed_trials": sum(1 for r in rows if r.get("passed")),
        "tokens": {
            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows),
            "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
            "cache_creation_input_tokens": sum(
                int(r.get("cache_creation_input_tokens") or 0) for r in rows
            ),
            "cache_read_input_tokens": sum(
                int(r.get("cache_read_input_tokens") or 0) for r in rows
            ),
            "total_tokens": sum(totals),
        },
        "total_llm_calls": sum(int(r.get("llm_calls") or 0) for r in rows),
        "total_wall_time_s": round(
            sum(float(r.get("wall_time_s") or 0.0) for r in rows), 3
        ),
        "mean_total_tokens_per_trial": (
            round(statistics.fmean(totals), 2) if totals else None
        ),
        "median_total_tokens_per_trial": (
            round(statistics.median(totals), 2) if totals else None
        ),
        "total_cost_usd": (
            round(sum(c for c in costs if c is not None), 6) if cost_complete else None
        ),
        "cost_complete": cost_complete,
        "metrics_source": metrics_source,
    }


def combine_run_metrics(
    *,
    benchmark: dict[str, Any],
    proposer: dict[str, Any],
) -> dict[str, Any]:
    """Combine benchmark (inner-loop) and proposer metrics for a whole run.

    Kept separate as well as combined: the proposer runs on a different
    model, on a different billing path (CLI subscription auth), and
    conflating the two makes a per-candidate cost look ~10× larger than
    it is.
    """
    b_cost = benchmark.get("total_cost_usd")
    p_cost = proposer.get("total_cost_usd")
    combined_known = b_cost is not None and p_cost is not None
    return {
        "benchmark": benchmark,
        "proposer": proposer,
        "combined": {
            "total_tokens": int(
                (benchmark.get("tokens") or {}).get("total_tokens") or 0
            )
            + int((proposer.get("tokens") or {}).get("total_tokens") or 0),
            "total_cost_usd": (
                round(float(b_cost) + float(p_cost), 6) if combined_known else None
            ),
            "cost_complete": combined_known,
        },
    }


# ── inner-loop thread naming ──────────────────────────────────────────


def inner_thread_id(
    *, run_id: str, thread_id: str, candidate: str, task_id: str, trial: int
) -> str:
    """Globally unique LangGraph thread id for one inner-loop trial.

    Namespacing by (run, branch, candidate, task, trial) is what keeps
    two branches benchmarking the same candidate label from writing into
    each other's checkpoint history, and it makes any inner checkpoint
    attributable back to the exact trial that produced it.
    """
    return f"inner::{run_id}::{thread_id}::{candidate}::{task_id}::trial-{trial}"


def parse_inner_thread_id(value: str) -> dict[str, Any] | None:
    """Inverse of :func:`inner_thread_id`; ``None`` if the shape doesn't match."""
    parts = value.split("::")
    if len(parts) != 6 or parts[0] != "inner" or not parts[5].startswith("trial-"):
        return None
    try:
        trial = int(parts[5].removeprefix("trial-"))
    except ValueError:
        return None
    return {
        "run_id": parts[1],
        "thread_id": parts[2],
        "candidate": parts[3],
        "task_id": parts[4],
        "trial": trial,
    }
