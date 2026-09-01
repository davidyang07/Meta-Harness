"""Outer-loop state machine: ``propose → validate → benchmark → update_frontier``.

Per Appendix B §B.6.1 / INTERFACES.md §1.1. **All nodes are async**
(step 7 refactor) so the outer graph integrates cleanly with
``AsyncPostgresSaver`` and concurrent branches (Appendix A).

The ``benchmark`` node uses ``asyncio.Semaphore`` for bounded
concurrency over (task × trial) tuples — explicitly **not**
``asyncio.gather`` over branches that may interrupt (per Appendix A
§A.4 Gotcha 2). Inner-loop trials don't use ``interrupt()``, so
gather over them is safe.

Two invariants this module enforces:

1. **Every filesystem write is thread-scoped.** Node bodies resolve
   ``thread_id`` from the LangGraph config and write through
   ``runs.py``'s thread-scoped helpers, so two branches forked from the
   same checkpoint never share a pending-eval handoff, frontier,
   evolution log, proposer session, or candidate directory.
2. **The baseline is a measured candidate, not an implicit zero.** The
   run's first benchmark evaluates ``agents/baseline.py`` under exactly
   the same task/trial protocol as evolved candidates, so every reported
   delta is relative to a real measurement.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.meta_harness import benchmark as bench
from app.meta_harness import candidates as cand_mod
from app.meta_harness import frontier as fr
from app.meta_harness import memory as mem
from app.meta_harness import metrics as met
from app.meta_harness import proposer as prp
from app.meta_harness import runs as runs_mod
from app.meta_harness import tracking as trk
from app.meta_harness.state import BASELINE_CANDIDATE_NAME, MetaHarnessState
from app.streaming import emit_run_event


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _thread_id(state: MetaHarnessState, config: RunnableConfig | None) -> str:
    configurable = (config or {}).get("configurable", {})
    return str(configurable.get("thread_id") or state["run_id"])


def _summary(state: MetaHarnessState, *, iteration: int | None = None) -> dict[str, Any]:
    return {
        "candidates_count": len(state.get("candidates") or []),
        "budget_remaining": state.get("budget_remaining"),
        "best_candidate": state.get("best_candidate"),
        "iteration": iteration if iteration is not None else state.get("iteration"),
    }


def _emit(
    state: MetaHarnessState,
    config: RunnableConfig | None,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Best-effort SSE emit — never crash graph nodes on streaming errors."""
    try:
        payload.setdefault("thread_id", _thread_id(state, config))
        emit_run_event(state["run_id"], event_type, payload)
    except Exception:  # noqa: BLE001 — SSE is best-effort; never crash a node
        pass


def baseline_candidate(run_id: str) -> dict[str, Any]:
    """The measured root of every run's search tree.

    ``agents/baseline.py`` is committed and immutable, so it needs no
    per-branch source snapshot: ``source_path`` stays ``None`` and the
    loader falls back to the ordinary module import.
    """
    return {
        "name": BASELINE_CANDIDATE_NAME,
        "import_path": "agents.baseline:BaselineHarness",
        "source_path": None,
        "parent": None,
        "hypothesis": "immutable starting harness (no overrides)",
        "axis": "baseline",
        "expected_score_delta": None,
        "iteration": 0,
        "status": "pending",
        "scores": None,
        "delta": None,
        "cost_usd": None,
    }


class OuterLoopRunner:
    """Builds the outer LangGraph for one run.

    Flags:
    - ``mock_proposer``: use ``proposer.mock_propose`` (step 5).
    - ``mock_bench``: skip the inner loop and synthesize scores per
      candidate. Synthesized results are always tagged
      ``metrics_source="mock"``.
    - ``checkpointer``: ``AsyncPostgresSaver`` (step 7) or ``None`` for
      in-memory. When set, it is threaded into every inner-loop trial so
      inner graph transitions are checkpointed too.
    - ``tracker``: experiment-tracking sink (``tracking.Tracker``).
      Defaults to the no-op tracker, so the loop behaves identically
      with or without a metrics backend configured.
    - ``recording_root``: when set, every inner-loop trial is taped for
      exact replay under ``<root>/<candidate>/<task>-trial-<n>/``.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        repo_root: Path,
        eval_tasks_dir: Path,
        mock_proposer: bool,
        mock_bench: bool,
        trials: int,
        bench_workers: int,
        skill_path: Path | None = None,
        checkpointer: Any = None,
        memory_store: Any = None,
        checkpoint_inner: bool = True,
        tracker: trk.Tracker | None = None,
        recording_root: Path | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.repo_root = repo_root
        self.eval_tasks_dir = eval_tasks_dir
        self.mock_proposer = mock_proposer
        self.mock_bench = mock_bench
        self.trials = trials
        self.bench_workers = bench_workers
        self.skill_path = skill_path
        self.checkpointer = checkpointer
        self.memory_store = memory_store
        self.checkpoint_inner = checkpoint_inner
        self.tracker = tracker or trk.NullTracker()
        self.recording_root = recording_root

    # ── propose ───────────────────────────────────────────────────────

    async def propose(
        self,
        state: MetaHarnessState,
        config: RunnableConfig = None,
    ) -> dict[str, Any]:
        thread_id = _thread_id(state, config)
        iteration = state["iteration"] + 1
        parent_name = state.get("best_candidate")
        if self.mock_proposer:
            payload = await asyncio.to_thread(
                prp.mock_propose,
                run_dir=self.run_dir,
                thread_id=thread_id,
                iteration=iteration,
                parent_name=parent_name,
                repo_root=self.repo_root,
            )
        else:
            if self.skill_path is None:
                raise ValueError("skill_path required for non-mock proposer")
            # Inject cross-run memory patterns into the proposer prior
            # (step 8). Patterns from prior runs are read from
            # PostgresStore and rendered as a Markdown section.
            proposer_prior = state.get("proposer_prior", "")
            if self.memory_store is not None:
                try:
                    patterns = await mem.search_patterns(
                        self.memory_store, limit=5,
                    )
                    memory_section = mem.format_patterns_for_prompt(patterns)
                    if memory_section:
                        proposer_prior = (
                            (proposer_prior + "\n\n" + memory_section)
                            if proposer_prior
                            else memory_section
                        )
                except Exception:  # noqa: BLE001 — memory is best-effort
                    pass
            # claude_propose spawns a subprocess. Wrap in to_thread to
            # avoid blocking the outer event loop while it runs.
            payload = await asyncio.to_thread(
                prp.claude_propose,
                run_dir=self.run_dir,
                thread_id=thread_id,
                iteration=iteration,
                parent_name=parent_name,
                repo_root=self.repo_root,
                skill_path=self.skill_path,
                proposer_prior=proposer_prior,
            )
        new_candidates = list(state.get("candidates") or [])
        for c in payload["candidates"]:
            label = str(c["name"])
            try:
                runs_mod.validate_artifact_name(label, kind="candidate")
                candidate_name = runs_mod.qualify_candidate_name(
                    label, thread_id, state["run_id"]
                )
            except ValueError as exc:
                raise ValueError(f"invalid proposer candidate: {exc}") from exc

            # Snapshot the authored source into this branch's private
            # directory before anything benchmarks it. A concurrently
            # running fork can rewrite agents/<label>.py at any moment;
            # the snapshot is what this branch actually evaluates.
            source_path = await asyncio.to_thread(
                cand_mod.snapshot_candidate_source,
                repo_root=self.repo_root,
                run_dir=self.run_dir,
                thread_id=thread_id,
                candidate_name=candidate_name,
                label=label,
            )

            new_candidates.append(
                {
                    "name": candidate_name,
                    "label": label,
                    "import_path": c["import_path"],
                    "source_path": str(source_path),
                    "source_sha256": cand_mod.source_sha256(source_path),
                    "parent": c.get("parent"),
                    "hypothesis": c.get("hypothesis", ""),
                    "axis": c.get("axis", "exploitation"),
                    "expected_score_delta": c.get("expected_score_delta"),
                    "iteration": iteration,
                    "status": "pending",
                    "scores": None,
                    "delta": None,
                    "cost_usd": None,
                }
            )
            _emit(
                state,
                config,
                "candidate-created",
                {
                    "candidate": candidate_name,
                    "label": label,
                    "parent_candidate_name": c.get("parent"),
                    "import_path": c["import_path"],
                    "parent": c.get("parent"),
                    "iteration": iteration,
                    "status": "seed",
                    "scores": {"accuracy": 0.0},
                    "delta": None,
                    "hypothesis": c.get("hypothesis", ""),
                    "axis": c.get("axis", "exploitation"),
                },
            )
        _emit(
            state,
            config,
            "state-update",
            {
                "node": "propose",
                "iteration": iteration,
                "ts": _now(),
                "summary": {
                    **_summary(state, iteration=iteration),
                    "candidates_count": len(new_candidates),
                },
            },
        )
        return {"iteration": iteration, "candidates": new_candidates}

    # ── validate ──────────────────────────────────────────────────────

    async def validate(
        self,
        state: MetaHarnessState,
        config: RunnableConfig = None,
    ) -> dict[str, Any]:
        candidate = state["candidates"][-1]
        error: str | None = None
        try:
            cls = await asyncio.to_thread(
                cand_mod.load_harness_class, candidate, repo_root=self.repo_root
            )
            cand_mod.assert_is_harness(cls, import_path=candidate["import_path"])
            candidate["status"] = "pending"
            valid = True
        except Exception as exc:  # noqa: BLE001 — record any error
            candidate["status"] = "smoke_failed"
            candidate["scores"] = {"error": str(exc)}
            error = str(exc)
            valid = False
        payload: dict[str, Any] = {
            "candidate": candidate["name"],
            "valid": valid,
        }
        if error:
            payload["error"] = error
        _emit(state, config, "validate-result", payload)
        _emit(
            state,
            config,
            "state-update",
            {
                "node": "validate",
                "iteration": state["iteration"],
                "ts": _now(),
                "summary": _summary(state),
            },
        )
        return {"candidates": state["candidates"], "_last_valid": valid}

    # ── benchmark ─────────────────────────────────────────────────────

    def _task_dirs(self) -> list[Path]:
        return sorted(
            d
            for d in self.eval_tasks_dir.iterdir()
            if d.is_dir() and (d / "task.json").exists()
        )

    async def evaluate_candidate(
        self,
        candidate: dict[str, Any],
        *,
        thread_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Benchmark one candidate over (tasks × trials) and persist the result.

        Returns the ``eval-result.json`` payload. Both the mock and the
        measured path produce the same schema; only ``metrics_source``
        differs, and callers must not mix the two.
        """
        task_dirs = self._task_dirs()

        if self.mock_bench:
            eval_result = self._mock_eval_result(
                candidate, task_dirs=task_dirs, thread_id=thread_id
            )
        else:
            eval_result = await self._measured_eval_result(
                candidate, task_dirs=task_dirs, thread_id=thread_id, run_id=run_id
            )

        eval_result["candidate"] = candidate["name"]
        eval_result["thread_id"] = thread_id
        eval_result["timestamp"] = _now()

        cand_dir = runs_mod.candidate_dir(self.run_dir, thread_id, candidate["name"])
        runs_mod.write_json_atomic(cand_dir / "eval-result.json", eval_result)
        return eval_result

    def _mock_eval_result(
        self,
        candidate: dict[str, Any],
        *,
        task_dirs: list[Path],
        thread_id: str,
    ) -> dict[str, Any]:
        """Deterministic synthetic scores, explicitly marked as mock.

        The baseline starts lower than evolved candidates so the mock
        demo shows a visible-but-honest progression; nothing here is
        ever presented as a measurement.
        """
        iteration = int(candidate.get("iteration") or 0)
        target_acc = min(0.95, 0.60 + iteration * 0.10)
        trial_rows: list[dict[str, Any]] = []
        for td in task_dirs:
            outcomes = [True] * int(round(self.trials * target_acc))
            outcomes += [False] * (self.trials - len(outcomes))
            for idx, passed in enumerate(outcomes, start=1):
                trial_rows.append(
                    met.mock_trial_metrics(
                        task_id=td.name,
                        trial=idx,
                        passed=passed,
                        iteration=iteration,
                    )
                )
        # Same summarizer as the measured path, so the two payloads have
        # identical shape and differ only in metrics_source.
        return bench.summarize(
            trial_rows,
            task_ids=[d.name for d in task_dirs],
            trials=self.trials,
            metrics_source=met.MOCK,
        )

    async def _measured_eval_result(
        self,
        candidate: dict[str, Any],
        *,
        task_dirs: list[Path],
        thread_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Run the real inner loop over (tasks × trials) and measure it.

        The Postgres checkpointer is threaded through to every inner
        trial, so inner graph transitions get their own checkpoint
        history under a thread id that identifies the exact
        (run, branch, candidate, task, trial) that produced it.
        """
        harness_class = await asyncio.to_thread(
            cand_mod.load_harness_class, candidate, repo_root=self.repo_root
        )
        traces_root = (
            runs_mod.candidate_dir(self.run_dir, thread_id, candidate["name"]) / "traces"
        )

        recordings_root = (
            self.recording_root / candidate["name"]
            if self.recording_root is not None
            else None
        )

        rows = await bench.run_trials(
            harness_factory=harness_class,
            task_dirs=task_dirs,
            trials=self.trials,
            workers=self.bench_workers,
            trace_dir_for=lambda task_id, trial: traces_root
            / f"{task_id}-trial-{trial}",
            inner_thread_id_for=lambda task_id, trial: met.inner_thread_id(
                run_id=run_id,
                thread_id=thread_id,
                candidate=candidate["name"],
                task_id=task_id,
                trial=trial,
            ),
            checkpointer=self.checkpointer if self.checkpoint_inner else None,
            on_trial=lambda row: trk.log_trial(self.tracker, row, arm=candidate["name"]),
            recording_dir_for=(
                (lambda task_id, trial: recordings_root / f"{task_id}-trial-{trial}")
                if recordings_root is not None
                else None
            ),
        )
        return bench.summarize(
            rows,
            task_ids=[d.name for d in task_dirs],
            trials=self.trials,
            metrics_source=met.MEASURED,
        )

    async def benchmark(
        self,
        state: MetaHarnessState,
        config: RunnableConfig = None,
    ) -> dict[str, Any]:
        thread_id = _thread_id(state, config)
        candidate = state["candidates"][-1]
        if candidate["status"] == "smoke_failed":
            _emit(
                state,
                config,
                "state-update",
                {
                    "node": "benchmark",
                    "iteration": state["iteration"],
                    "ts": _now(),
                    "summary": _summary(state),
                },
            )
            return {"candidates": state["candidates"]}

        eval_result = await self.evaluate_candidate(
            candidate, thread_id=thread_id, run_id=state["run_id"]
        )

        candidate["scores"] = eval_result
        candidate["status"] = "evaluated"
        candidate["cost_usd"] = eval_result.get("total_cost_usd")
        _emit(
            state,
            config,
            "eval-result",
            {
                "candidate": candidate["name"],
                "parent_candidate_name": candidate.get("parent"),
                "iteration": candidate.get("iteration"),
                "status": "evaluated",
                "accuracy": eval_result["accuracy"],
                "scores": {
                    "accuracy": eval_result["accuracy"],
                    "per_task": eval_result["per_task"],
                },
                "per_task": eval_result["per_task"],
                "tokens": eval_result["tokens"],
                "cost_usd": eval_result.get("total_cost_usd"),
                "metrics_source": eval_result["metrics_source"],
                "hypothesis": candidate.get("hypothesis", ""),
                "axis": candidate.get("axis"),
            },
        )
        _emit(
            state,
            config,
            "state-update",
            {
                "node": "benchmark",
                "iteration": state["iteration"],
                "ts": _now(),
                "summary": _summary(state),
            },
        )
        return {"candidates": state["candidates"]}

    # ── update_frontier ───────────────────────────────────────────────

    async def update_frontier(
        self,
        state: MetaHarnessState,
        config: RunnableConfig = None,
    ) -> dict[str, Any]:
        thread_id = _thread_id(state, config)
        candidate = state["candidates"][-1]
        scored_statuses = {"evaluated", "accepted", "rejected"}
        evaluated = [
            fr.frontier_entry(c["name"], c["scores"] or {})
            for c in state["candidates"]
            if c["status"] in scored_statuses and c.get("scores")
        ]
        per_task_bests: dict[str, dict[str, Any]] = {}
        for c in state["candidates"]:
            if c["status"] not in scored_statuses or not c.get("scores"):
                continue
            for task_id, info in (c["scores"] or {}).get("per_task", {}).items():
                cur = per_task_bests.get(task_id)
                if cur is None or info["pass_rate"] > cur["pass_rate"]:
                    per_task_bests[task_id] = {
                        "best_candidate": c["name"],
                        "pass_rate": info["pass_rate"],
                    }

        frontier = fr.build_frontier_val(state["iteration"], evaluated, per_task_bests)
        frontier["thread_id"] = thread_id
        runs_mod.write_frontier(self.run_dir, thread_id, frontier)

        # Deltas compare against the *measured* accuracy of the prior
        # best candidate — which, on iteration 1, is the benchmarked
        # baseline rather than an implicit zero.
        prev_best = state.get("best_candidate")
        prev_best_acc: float | None = None
        for c in state["candidates"]:
            if c["name"] == prev_best:
                prev_best_acc = (c["scores"] or {}).get("accuracy")
                break
        cand_acc = (candidate["scores"] or {}).get("accuracy", 0.0)
        baseline_acc = prev_best_acc if prev_best_acc is not None else 0.0
        delta = cand_acc - baseline_acc
        candidate["delta"] = round(delta, 4)
        accepted = candidate["status"] == "evaluated" and (
            prev_best_acc is None or cand_acc > prev_best_acc
        )
        candidate["status"] = "accepted" if accepted else "rejected"

        runs_mod.write_status(
            self.run_dir,
            thread_id,
            candidate["name"],
            {
                "candidate": candidate["name"],
                "thread_id": thread_id,
                "accepted": accepted,
                "parent": candidate.get("parent"),
                "compared_against": prev_best,
                "compared_against_accuracy": prev_best_acc,
                "delta": candidate["delta"],
                "reason": "accepted" if accepted else "regression",
            },
        )

        # Step 8: write cross-run memory pattern on accepted candidate.
        if accepted and self.memory_store is not None:
            try:
                await mem.add_pattern(
                    self.memory_store,
                    pattern=(
                        f"{candidate.get('hypothesis', 'unknown hypothesis')} "
                        f"— overrode {candidate.get('axis', 'unknown')} axis"
                    ),
                    mechanism_axis=candidate.get("axis", "unknown"),
                    score_delta=candidate["delta"],
                    run_id=state["run_id"],
                )
                _emit(
                    state,
                    config,
                    "memory-pattern-stored",
                    {
                        "namespace": ["learned_patterns", "coding-agent"],
                        "key": candidate["name"],
                        "score_delta": candidate["delta"],
                    },
                )
            except Exception:  # noqa: BLE001 — memory write is best-effort
                pass

        runs_mod.record_evolution_row(
            self.run_dir,
            thread_id,
            _evolution_row(candidate, iteration=state["iteration"]),
        )

        scores = candidate["scores"] or {}
        trk.log_iteration(
            self.tracker,
            iteration=state["iteration"],
            candidate=candidate["name"],
            accuracy=scores.get("accuracy"),
            delta=candidate["delta"],
            accepted=accepted,
            axis=candidate.get("axis"),
            metrics_source=scores.get("metrics_source"),
            mean_tokens=scores.get("mean_total_tokens_per_trial"),
            cost_usd=scores.get("total_cost_usd"),
            thread_id=thread_id,
            branch_id=thread_id.rpartition(".fork.")[2] or None,
            per_task=scores.get("per_task"),
        )
        trk.log_frontier(self.tracker, frontier, thread_id=thread_id)

        new_best = candidate["name"] if accepted else prev_best
        new_frontier_names = frontier.get("_pareto_names", [])
        _emit(
            state,
            config,
            "frontier-updated",
            {
                "candidate": candidate["name"],
                "parent_candidate_name": candidate.get("parent"),
                "iteration": state["iteration"],
                "frontier": new_frontier_names,
                "best_candidate": new_best,
                "best_score": (
                    cand_acc if new_best == candidate["name"] else prev_best_acc
                ),
                "status": (
                    "best"
                    if accepted and new_best == candidate["name"]
                    else "rejected"
                ),
                "accepted": accepted,
                "delta": candidate["delta"],
                "scores": {
                    "accuracy": cand_acc,
                    "per_task": (candidate["scores"] or {}).get("per_task", {}),
                },
                "hypothesis": candidate.get("hypothesis", ""),
                "axis": candidate.get("axis"),
            },
        )
        _emit(
            state,
            config,
            "iteration-complete",
            {
                "iteration": state["iteration"],
                "status": "improved" if accepted else "no_improvement",
            },
        )
        _emit(
            state,
            config,
            "state-update",
            {
                "node": "update_frontier",
                "iteration": state["iteration"],
                "ts": _now(),
                "summary": _summary(state),
            },
        )
        return {
            "candidates": state["candidates"],
            "frontier": new_frontier_names,
            "best_candidate": new_best,
            "budget_remaining": state["budget_remaining"] - 1,
        }

    # ── routing + compile ─────────────────────────────────────────────

    def _route_after_update(self, state: MetaHarnessState) -> str:
        return "propose" if state["budget_remaining"] > 0 else "end"

    def build(self) -> Any:
        g: StateGraph = StateGraph(MetaHarnessState)
        g.add_node("propose", self.propose)
        g.add_node("validate", self.validate)
        g.add_node("benchmark", self.benchmark)
        g.add_node("update_frontier", self.update_frontier)

        g.add_edge(START, "propose")
        g.add_edge("propose", "validate")
        g.add_edge("validate", "benchmark")
        g.add_edge("benchmark", "update_frontier")
        g.add_conditional_edges(
            "update_frontier",
            self._route_after_update,
            {"propose": "propose", "end": END},
        )
        return (
            g.compile(checkpointer=self.checkpointer)
            if self.checkpointer is not None
            else g.compile()
        )

    # ── baseline ──────────────────────────────────────────────────────

    async def benchmark_baseline(self, *, run_id: str) -> dict[str, Any]:
        """Measure ``agents/baseline.py`` before any candidate is proposed.

        Returns the baseline candidate dict with ``scores`` populated.
        This runs on the run's root thread: a fork inherits the measured
        baseline through checkpoint state rather than re-paying for it.
        """
        candidate = baseline_candidate(run_id)
        eval_result = await self.evaluate_candidate(
            candidate, thread_id=run_id, run_id=run_id
        )
        candidate["scores"] = eval_result
        candidate["status"] = "evaluated"
        candidate["delta"] = 0.0
        candidate["cost_usd"] = eval_result.get("total_cost_usd")

        runs_mod.write_status(
            self.run_dir,
            run_id,
            candidate["name"],
            {
                "candidate": candidate["name"],
                "thread_id": run_id,
                "accepted": True,
                "parent": None,
                "compared_against": None,
                "compared_against_accuracy": None,
                "delta": 0.0,
                "reason": "measured baseline (search root)",
            },
        )
        runs_mod.record_evolution_row(
            self.run_dir, run_id, _evolution_row(candidate, iteration=0)
        )
        return candidate


def _evolution_row(candidate: dict[str, Any], *, iteration: int) -> dict[str, Any]:
    """Project a candidate into one ``evolution_summary.jsonl`` row."""
    scores = candidate.get("scores") or {}
    acc = scores.get("accuracy", 0.0)
    delta = candidate.get("delta")
    return {
        "iteration": iteration,
        "candidate": candidate["name"],
        "label": candidate.get("label", candidate["name"]),
        "import_path": candidate["import_path"],
        "source_sha256": candidate.get("source_sha256"),
        "parent_candidate_name": candidate.get("parent"),
        "axis": candidate.get("axis"),
        "hypothesis": candidate.get("hypothesis", ""),
        "scores": {
            "accuracy": acc,
            "per_task": scores.get("per_task", {}),
        },
        "delta": delta,
        "outcome": (
            f"{acc:.1%} ({delta:+.1%})" if delta is not None else f"{acc:.1%}"
        ),
        "tokens": scores.get("mean_total_tokens_per_trial"),
        "cost_usd": scores.get("total_cost_usd"),
        "metrics_source": scores.get("metrics_source"),
    }


def initial_state(
    *, run_id: str, budget: int, seed_candidates: list[dict[str, Any]] | None = None
) -> MetaHarnessState:
    """Build the outer graph's initial state.

    ``seed_candidates`` normally holds exactly the measured baseline, so
    the very first proposed candidate is compared against a real number.
    """
    seeds = list(seed_candidates or [])
    best = seeds[-1]["name"] if seeds else None
    return {
        "run_id": run_id,
        "iteration": 0,
        "budget_remaining": budget,
        "candidates": seeds,
        "frontier": [c["name"] for c in seeds],
        "best_candidate": best,
        "proposer_prior": "",
    }


async def run_outer_loop(
    *,
    run_dir: Path,
    repo_root: Path,
    eval_tasks_dir: Path,
    mock_proposer: bool,
    mock_bench: bool,
    trials: int,
    bench_workers: int,
    budget: int,
    skill_path: Path | None = None,
    checkpointer: Any = None,
    memory_store: Any = None,
    evaluate_baseline: bool = True,
    tracker: Any = None,
    recording_root: Path | None = None,
) -> MetaHarnessState:
    """Run the outer loop end-to-end (async). Returns the final state."""
    runner = OuterLoopRunner(
        run_dir=run_dir,
        repo_root=repo_root,
        eval_tasks_dir=eval_tasks_dir,
        mock_proposer=mock_proposer,
        mock_bench=mock_bench,
        trials=trials,
        bench_workers=bench_workers,
        skill_path=skill_path,
        checkpointer=checkpointer,
        memory_store=memory_store,
        tracker=tracker,
        recording_root=recording_root,
    )
    runs_mod.write_manifest(
        run_dir,
        run_id=run_dir.name,
        thread_id=run_dir.name,
        budget=budget,
        trials=trials,
        mock_proposer=mock_proposer,
        mock_bench=mock_bench,
        metrics_source="mock" if mock_bench else "measured",
    )

    seeds: list[dict[str, Any]] = []
    if evaluate_baseline:
        seeds.append(await runner.benchmark_baseline(run_id=run_dir.name))

    graph = runner.build()
    final = await graph.ainvoke(
        initial_state(run_id=run_dir.name, budget=budget, seed_candidates=seeds),
        config={"configurable": {"thread_id": run_dir.name}, "recursion_limit": 200},
    )
    return final  # type: ignore[return-value]


async def resume_outer_loop(
    *,
    run_dir: Path,
    repo_root: Path,
    eval_tasks_dir: Path,
    checkpointer: Any,
    skill_path: Path | None = None,
) -> MetaHarnessState:
    """Resume an interrupted run from its last Postgres checkpoint.

    Reads the run's manifest.json to recover the original config
    (mock_proposer, mock_bench, trials, etc.) and resumes via
    ``graph.ainvoke(None, config={"thread_id": run_dir.name})``.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json missing in {run_dir}; cannot resume without run config"
        )
    manifest = json.loads(manifest_path.read_text())
    runner = OuterLoopRunner(
        run_dir=run_dir,
        repo_root=repo_root,
        eval_tasks_dir=eval_tasks_dir,
        mock_proposer=manifest.get("mock_proposer", False),
        mock_bench=manifest.get("mock_bench", False),
        trials=manifest.get("trials", 5),
        bench_workers=3,
        skill_path=skill_path,
        checkpointer=checkpointer,
    )
    graph = runner.build()
    config = {"configurable": {"thread_id": run_dir.name}, "recursion_limit": 200}

    # Guard: a thread that already ran to completion has no pending
    # nodes. Calling ``ainvoke(None, ...)`` on it re-enters the graph at
    # START and replays the whole loop, which double-appends every
    # evolution_summary row and re-spends the proposer budget. Return
    # the stored terminal state instead.
    snapshot = await graph.aget_state(config)
    values = getattr(snapshot, "values", None)
    if values and not getattr(snapshot, "next", ()):
        return values  # type: ignore[return-value]

    # ``None`` input + existing thread_id → resume from last checkpoint.
    final = await graph.ainvoke(None, config=config)
    return final  # type: ignore[return-value]


__all__ = [
    "OuterLoopRunner",
    "baseline_candidate",
    "initial_state",
    "resume_outer_loop",
    "run_outer_loop",
]
