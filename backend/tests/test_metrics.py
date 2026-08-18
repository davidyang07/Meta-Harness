"""Per-call / per-trial / per-candidate metric collection and aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.meta_harness import metrics as met


class _Usage:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Response:
    def __init__(self, usage=None, model="claude-haiku-4-5-20251001"):
        self.usage = usage
        self.model = model


@pytest.fixture(autouse=True)
def _clean_pricing(monkeypatch):
    monkeypatch.delenv("META_HARNESS_PRICING", raising=False)
    met.reset_pricing_cache()
    yield
    met.reset_pricing_cache()


# ── per-call recording ────────────────────────────────────────────────


def test_recorder_captures_all_usage_fields():
    rec = met.UsageRecorder()
    rec.record(
        response=_Response(
            _Usage(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=10,
                cache_read_input_tokens=5,
            )
        ),
        model="claude-haiku-4-5-20251001",
        latency_s=1.25,
    )
    call = rec.calls[0]
    assert call["index"] == 1
    assert call["input_tokens"] == 100
    assert call["output_tokens"] == 50
    assert call["cache_creation_input_tokens"] == 10
    assert call["cache_read_input_tokens"] == 5
    assert call["total_tokens"] == 165
    assert call["latency_s"] == 1.25
    assert call["has_usage"] is True


def test_recorder_tolerates_mocked_responses_without_usage():
    """Deterministic tests stub _call_llm; that must not blow up."""
    rec = met.UsageRecorder()
    rec.record(response=_Response(None), model="m", latency_s=0.1)
    assert rec.calls[0]["total_tokens"] == 0
    assert rec.calls[0]["has_usage"] is False
    assert rec.totals()["llm_calls"] == 1


async def test_instrument_harness_records_every_call():
    class _Harness:
        MODEL = "claude-haiku-4-5-20251001"

        async def _call_llm(self, messages, tools, **kw):
            return _Response(_Usage(input_tokens=7, output_tokens=3))

    harness = _Harness()
    rec = met.UsageRecorder()
    met.instrument_harness(harness, rec)

    await harness._call_llm([], [])
    await harness._call_llm([], [])

    assert rec.totals()["llm_calls"] == 2
    assert rec.totals()["input_tokens"] == 14
    assert rec.totals()["output_tokens"] == 6


# ── cost ──────────────────────────────────────────────────────────────


def test_cost_is_computed_from_configured_pricing():
    call = {
        "model": "claude-haiku-4-5-20251001",
        "input_tokens": 1_000_000,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert met.call_cost_usd(call) == pytest.approx(1.0)


def test_unknown_model_yields_none_cost_not_zero():
    """A missing price must never be reported as $0.00."""
    call = {"model": "some-unpriced-model", "input_tokens": 5000, "output_tokens": 5000}
    assert met.call_cost_usd(call) is None


def test_pricing_is_configurable(tmp_path: Path, monkeypatch):
    table = tmp_path / "pricing.json"
    table.write_text(
        json.dumps({"my-model": {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 4.0}})
    )
    monkeypatch.setenv("META_HARNESS_PRICING", str(table))
    met.reset_pricing_cache()
    cost = met.call_cost_usd(
        {"model": "my-model", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    )
    assert cost == pytest.approx(6.0)


def test_missing_pricing_file_raises_rather_than_silently_zeroing(monkeypatch):
    monkeypatch.setenv("META_HARNESS_PRICING", "/definitely/not/here.json")
    met.reset_pricing_cache()
    with pytest.raises(FileNotFoundError):
        met.load_pricing()


def test_trial_cost_is_none_when_any_call_is_unpriced():
    rec = met.UsageRecorder()
    rec.record(
        response=_Response(_Usage(input_tokens=10, output_tokens=1)),
        model="claude-haiku-4-5-20251001",
        latency_s=0.1,
    )
    rec.record(
        response=_Response(_Usage(input_tokens=10, output_tokens=1), model="mystery"),
        model="mystery",
        latency_s=0.1,
    )
    assert rec.totals()["cost_usd"] is None


# ── aggregation ───────────────────────────────────────────────────────


def _row(task, trial, passed, tokens, cost=0.001, source=met.MEASURED):
    return {
        "task_id": task,
        "trial": trial,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "wall_time_s": 1.0,
        "metrics_source": source,
        "llm_calls": 3,
        "input_tokens": tokens // 2,
        "output_tokens": tokens - tokens // 2,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_tokens": tokens,
        "cost_usd": cost,
    }


def test_aggregate_computes_totals_mean_and_median():
    rows = [
        _row("t1", 1, True, 100),
        _row("t1", 2, False, 300),
        _row("t2", 1, True, 200),
    ]
    agg = met.aggregate_trials(rows, metrics_source=met.MEASURED)
    assert agg["total_trials"] == 3
    assert agg["passed_trials"] == 2
    assert agg["tokens"]["total_tokens"] == 600
    assert agg["mean_total_tokens_per_trial"] == 200.0
    assert agg["median_total_tokens_per_trial"] == 200.0
    assert agg["total_llm_calls"] == 9
    assert agg["total_cost_usd"] == pytest.approx(0.003)
    assert agg["cost_complete"] is True
    assert agg["metrics_source"] == met.MEASURED


def test_aggregate_marks_cost_incomplete_when_any_trial_is_unpriced():
    rows = [_row("t1", 1, True, 100), _row("t1", 2, True, 100, cost=None)]
    agg = met.aggregate_trials(rows, metrics_source=met.MEASURED)
    assert agg["total_cost_usd"] is None
    assert agg["cost_complete"] is False
    # Tokens are still fully reported.
    assert agg["tokens"]["total_tokens"] == 200


def test_aggregate_refuses_to_mix_mock_and_measured_rows():
    rows = [_row("t1", 1, True, 100), _row("t1", 2, True, 100, source=met.MOCK)]
    with pytest.raises(ValueError, match="cannot aggregate"):
        met.aggregate_trials(rows, metrics_source=met.MEASURED)


def test_mock_trial_metrics_are_tagged_and_have_no_fake_cost():
    row = met.mock_trial_metrics(task_id="t", trial=1, passed=True, iteration=2)
    assert row["metrics_source"] == met.MOCK
    assert row["cost_usd"] is None
    assert row["llm_calls"] == 0
    agg = met.aggregate_trials([row], metrics_source=met.MOCK)
    assert agg["metrics_source"] == met.MOCK
    assert agg["cost_complete"] is False


def test_empty_aggregate_reports_none_rather_than_zero():
    agg = met.aggregate_trials([], metrics_source=met.MEASURED)
    assert agg["total_trials"] == 0
    assert agg["mean_total_tokens_per_trial"] is None
    assert agg["median_total_tokens_per_trial"] is None
    assert agg["total_cost_usd"] is None


def test_combine_run_metrics_keeps_proposer_and_benchmark_separate():
    bench = met.aggregate_trials([_row("t", 1, True, 100)], metrics_source=met.MEASURED)
    proposer = {"tokens": {"total_tokens": 40}, "total_cost_usd": 0.5}
    combined = met.combine_run_metrics(benchmark=bench, proposer=proposer)
    assert combined["benchmark"]["tokens"]["total_tokens"] == 100
    assert combined["proposer"]["total_cost_usd"] == 0.5
    assert combined["combined"]["total_tokens"] == 140
    assert combined["combined"]["total_cost_usd"] == pytest.approx(0.501)


def test_combine_run_metrics_propagates_unknown_cost():
    bench = met.aggregate_trials(
        [_row("t", 1, True, 100, cost=None)], metrics_source=met.MEASURED
    )
    combined = met.combine_run_metrics(
        benchmark=bench, proposer={"tokens": {"total_tokens": 1}, "total_cost_usd": 0.5}
    )
    assert combined["combined"]["total_cost_usd"] is None
    assert combined["combined"]["cost_complete"] is False


# ── inner thread naming ───────────────────────────────────────────────


def test_inner_thread_ids_are_unique_across_branches():
    a = met.inner_thread_id(
        run_id="r", thread_id="r", candidate="c", task_id="t1", trial=1
    )
    b = met.inner_thread_id(
        run_id="r", thread_id="r.fork.ab", candidate="c", task_id="t1", trial=1
    )
    assert a != b
    assert met.parse_inner_thread_id(a) == {
        "run_id": "r",
        "thread_id": "r",
        "candidate": "c",
        "task_id": "t1",
        "trial": 1,
    }
    assert met.parse_inner_thread_id(b)["thread_id"] == "r.fork.ab"
    assert met.parse_inner_thread_id("not-an-inner-thread") is None
