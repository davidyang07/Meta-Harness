"""Cross-run memory holds measured results only, written idempotently.

Runs against LangGraph's in-memory store, so these properties are
checked on every machine, not only where Postgres is up:

- a mock-benchmarked acceptance never becomes a pattern (invariant 5);
- whatever feeds a proposer reads measured patterns only, including
  ignoring untagged patterns written before the tag existed;
- re-executing ``update_frontier`` (a resumed node) rewrites the same
  pattern instead of adding a second one (invariant 6).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import memory as mem  # noqa: E402
from app.meta_harness import metrics as met  # noqa: E402
from app.meta_harness import runs as runs_mod  # noqa: E402
from app.meta_harness.outer import OuterLoopRunner  # noqa: E402


async def test_add_pattern_refuses_anything_but_a_measured_result():
    store = InMemoryStore()
    for source in (met.MOCK, None, "mixed"):
        with pytest.raises(ValueError, match="only measured results"):
            await mem.add_pattern(
                store,
                pattern="p",
                mechanism_axis="a",
                score_delta=0.5,
                run_id="r",
                candidate="c",
                metrics_source=source,  # type: ignore[arg-type]
            )
    assert await mem.search_patterns(store, limit=50) == []


async def test_the_same_candidate_writes_the_same_key():
    store = InMemoryStore()
    kwargs: dict[str, Any] = dict(
        pattern="p",
        mechanism_axis="exploitation",
        score_delta=0.1,
        run_id="run-1",
        thread_id="run-1.fork.ab",
        candidate="cand",
        metrics_source=met.MEASURED,
    )
    first = await mem.add_pattern(store, **kwargs)
    second = await mem.add_pattern(store, **kwargs)
    assert first == second
    stored = await mem.search_patterns(store, limit=50)
    assert len(stored) == 1
    assert stored[0]["metrics_source"] == met.MEASURED
    assert stored[0]["candidate"] == "cand"

    other = await mem.add_pattern(store, **{**kwargs, "candidate": "cand-2"})
    assert other != first
    assert len(await mem.search_patterns(store, limit=50)) == 2


async def test_the_proposer_only_ever_sees_measured_patterns():
    store = InMemoryStore()
    ns = ("learned_patterns", mem.DEFAULT_DOMAIN)
    await mem.add_pattern(
        store,
        pattern="measured lesson",
        mechanism_axis="a",
        score_delta=0.05,
        run_id="r",
        candidate="c",
        metrics_source=met.MEASURED,
    )
    # Written directly, the way an older build or a mock run left them.
    await store.aput(
        ns,
        "legacy",
        {"pattern": "untagged lesson", "mechanism_axis": "a", "score_delta": 0.5,
         "evidence_run_ids": ["old"], "created_at": "2099-01-01T00:00:00+00:00"},
    )
    await store.aput(
        ns,
        "mock",
        {"pattern": "mock lesson", "mechanism_axis": "a", "score_delta": 0.5,
         "metrics_source": met.MOCK, "evidence_run_ids": ["demo"],
         "created_at": "2099-01-02T00:00:00+00:00"},
    )

    shown = await mem.patterns_for_proposer(store, limit=5)
    assert [p["pattern"] for p in shown] == ["measured lesson"]

    # The generic listing still shows everything, for inspection.
    assert len(await mem.search_patterns(store, limit=50)) == 3

    runner = _runner(REPO_ROOT / "runs" / "unused", store)
    prior = await runner._proposer_prior({"proposer_prior": "branch prior"})  # type: ignore[arg-type]
    assert "measured lesson" in prior
    assert "untagged lesson" not in prior and "mock lesson" not in prior
    assert prior.startswith("branch prior")


# ── update_frontier ───────────────────────────────────────────────────


def _runner(run_dir: Path, store: Any) -> OuterLoopRunner:
    return OuterLoopRunner(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        eval_tasks_dir=REPO_ROOT / "eval" / "tasks",
        mock_proposer=True,
        mock_bench=True,
        trials=1,
        bench_workers=1,
        memory_store=store,
    )


def _scored(name: str, accuracy: float, source: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "import_path": f"agents.{name}:C",
        "parent": None,
        "hypothesis": f"hypothesis for {name}",
        "axis": "exploitation",
        "iteration": extra.pop("iteration", 0),
        "status": "evaluated",
        "delta": None,
        "scores": {
            "accuracy": accuracy,
            "per_task": {"t": {"pass_rate": accuracy, "trials": []}},
            "metrics_source": source,
            "mean_total_tokens_per_trial": 100.0,
        },
        **extra,
    }


def _state(run_id: str, source: str) -> dict[str, Any]:
    baseline = _scored("baseline", 0.4, source)
    candidate = _scored("cand", 0.8, source, iteration=1, parent="baseline")
    return {
        "run_id": run_id,
        "iteration": 1,
        "budget_remaining": 1,
        "candidates": [baseline, candidate],
        "frontier": ["baseline"],
        "best_candidate": "baseline",
        "proposer_prior": "",
    }


async def test_a_measured_acceptance_is_stored_once_even_when_the_node_reruns(
    tmp_path: Path,
):
    store = InMemoryStore()
    run_dir = runs_mod.make_run_dir(tmp_path, "mem-measured", fresh=True)
    runner = _runner(run_dir, store)
    state = _state(run_dir.name, met.MEASURED)
    config = {"configurable": {"thread_id": run_dir.name}}

    first = await runner.update_frontier(copy.deepcopy(state), config)
    # A resumed run re-executes the node from the same checkpoint.
    await runner.update_frontier(copy.deepcopy(state), config)

    assert first["best_candidate"] == "cand"
    stored = await mem.search_patterns(store, limit=50)
    assert len(stored) == 1
    assert stored[0]["metrics_source"] == met.MEASURED
    assert stored[0]["score_delta"] == pytest.approx(0.4)
    assert stored[0]["evidence_run_ids"] == [run_dir.name]


async def test_a_mock_acceptance_is_never_stored(tmp_path: Path):
    store = InMemoryStore()
    run_dir = runs_mod.make_run_dir(tmp_path, "mem-mock", fresh=True)
    runner = _runner(run_dir, store)

    result = await runner.update_frontier(
        _state(run_dir.name, met.MOCK), {"configurable": {"thread_id": run_dir.name}}
    )

    assert result["best_candidate"] == "cand", "the mock acceptance itself still happens"
    assert await mem.search_patterns(store, limit=50) == []
