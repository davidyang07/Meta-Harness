"""API flow assertions used by scripts/demo_acceptance.sh.

Drives the real backend through the demo path — create a run, read
checkpoints, fork one, confirm the branch is isolated — and asserts the
contracts the dashboard depends on. Uses the deterministic mock proposer
and mock benchmark, so it costs nothing.

    API_URL=http://127.0.0.1:8765 uv run python scripts/acceptance_api_flow.py
    API_URL=... uv run python scripts/acceptance_api_flow.py --sse-only
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


def request(
    method: str, path: str, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def wait_for(predicate, *, what: str, timeout_s: float = 120.0, interval: float = 0.5):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}; last={last!r}")


def check_sse(run_id: str) -> None:
    """Confirm the SSE stream replays this run's backlog."""
    req = urllib.request.Request(
        f"{API_URL}/runs/{run_id}/stream", headers={"Accept": "text/event-stream"}
    )
    seen: set[str] = set()
    with urllib.request.urlopen(req, timeout=30) as resp:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            line = resp.readline().decode("utf-8", "replace")
            if not line:
                break
            if line.startswith("event: "):
                seen.add(line[len("event: ") :].strip())
            if {"candidate-created", "eval-result", "frontier-updated"} <= seen:
                break
    missing = {"candidate-created", "eval-result", "frontier-updated"} - seen
    assert not missing, f"SSE stream missing event types: {sorted(missing)}; saw {sorted(seen)}"
    print(f"  SSE event types observed: {sorted(seen)}")


def main() -> int:
    sse_only = "--sse-only" in sys.argv
    run_id = f"acceptance-api-{int(time.time())}"

    status, _ = request(
        "POST",
        "/runs",
        {
            "run_name": run_id,
            "proposer": "mock",
            "mock_bench": True,
            "budget": 2,
            "trials": 2,
            "fresh": True,
        },
    )
    assert status == 201, f"POST /runs -> {status}"

    def _finished() -> dict[str, Any] | None:
        _, info = request("GET", f"/runs/{run_id}")
        return info if info.get("status") in {"completed", "failed"} else None

    info = wait_for(_finished, what="the run to finish")
    assert info["status"] == "completed", f"run failed: {info.get('error')}"

    try:
        if sse_only:
            check_sse(run_id)
            return 0

        # 1. Measured baseline is the root of the search tree.
        rows = info["summary_rows"]
        assert rows[0]["candidate"] == "baseline", rows[0]
        assert rows[0]["iteration"] == 0
        assert rows[0]["scores"]["accuracy"] > 0, "baseline must be measured"
        assert info["metrics_source"] == "mock", info.get("metrics_source")

        # 2. Checkpoint history exists and exposes a forkable point.
        status, ck = request("GET", f"/runs/{run_id}/checkpoints")
        assert status == 200 and len(ck["checkpoints"]) >= 4, ck
        forkable = next(
            c
            for c in ck["checkpoints"]
            if c["iteration"] == 1 and "propose" in (c["next"] or [])
        )

        status, detail = request(
            "GET", f"/runs/{run_id}/checkpoints/{forkable['checkpoint_id']}"
        )
        assert status == 200 and "iteration" in detail["state"], detail

        # 3. Two forks from the SAME checkpoint run concurrently.
        branches = []
        for label in ("left", "right"):
            status, branch = request(
                "POST",
                f"/runs/{run_id}/fork",
                {
                    "parent_checkpoint_id": forkable["checkpoint_id"],
                    "mods": {
                        "proposer_prior": f"acceptance {label}",
                        "budget_remaining": 1,
                    },
                    "name": label,
                },
            )
            assert status == 202, f"POST fork -> {status}: {branch}"
            branches.append(branch["thread_id"])
        assert branches[0] != branches[1]

        # 4. Both branches show up in the trajectory, on the right parent.
        def _both_in_trajectory():
            _, t = request("GET", f"/runs/{run_id}/trajectory")
            threads = {x["thread_id"] for x in t["trajectory"]["threads"]}
            return t if set(branches) <= threads else None

        trajectory = wait_for(_both_in_trajectory, what="both branches in trajectory")
        edges = trajectory["trajectory"]["edges"]
        assert len(edges) == 2, edges
        for edge in edges:
            assert edge["source"] == run_id
            assert edge["parent_checkpoint_id"] == forkable["checkpoint_id"]

        # 5. Branch artifacts are isolated: each branch contributed its own
        #    row, and no candidate name is claimed twice across the run.
        def _both_produced_rows():
            _, latest = request("GET", f"/runs/{run_id}")
            threads = {r["thread_id"] for r in latest["summary_rows"]}
            return latest if set(branches) <= threads else None

        latest = wait_for(_both_produced_rows, what="both branches to write rows")
        all_rows = latest["summary_rows"]
        names = [r["candidate"] for r in all_rows]
        assert len(names) == len(set(names)), f"duplicate candidate names: {names}"
        assert len(latest["branch_frontiers"]) == len(
            {r["thread_id"] for r in all_rows}
        ), "each branch must keep its own frontier"

        # 6. Diff and test output come from recorded artifacts.
        candidate = all_rows[-1]["candidate"]
        status, diff = request(
            "GET", f"/runs/{run_id}/candidates/{candidate}/diff"
        )
        assert status == 200 and diff["diff"], diff
        status, out = request(
            "GET", f"/runs/{run_id}/candidates/{candidate}/test-output"
        )
        assert status == 200 and "metrics_source" in out["output"], out

        status, _ = request("GET", f"/runs/{run_id}/candidates/nope-nope/diff")
        assert status == 404, "unknown candidate must 404"

        # 7. Memory endpoint answers with a usable shape.
        status, mem = request("GET", "/memory/coding-agent")
        assert status == 200 and isinstance(mem["entries"], list), mem

        print(f"  run {run_id}: {len(all_rows)} candidates across "
              f"{len({r['thread_id'] for r in all_rows})} branches")
        print(f"  branches: {[b.split('.fork.')[-1] for b in branches]}")
        return 0
    finally:
        request("DELETE", f"/runs/{run_id}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"ACCEPTANCE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
