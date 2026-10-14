"""The Claude proposer subprocess: isolation, handoff validation, lifetime.

A fake ``claude`` executable on ``PATH`` stands in for the CLI. It reads
what to do from a JSON file, records what it saw (its argv, its working
directory and every file in it), and emits stream-json the way the real
CLI does. The real CLI is never invoked.

What is checked:

- the proposer runs in a private staged copy that holds no holdout task,
  published result, evidence artifact, or other run's or branch's data;
- only the candidate source and ``pending_eval.json`` come back, the
  former straight into the branch snapshot (invariant 2);
- a change to the real ``eval/``/``benchmarks/`` fails the iteration;
- a tool call naming holdout material or the real repo fails it;
- the handoff must be one candidate, for this iteration, with a new
  name and a real file, and a stale handoff is never re-submitted;
- a timeout or a cancelled task kills the whole process tree.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from app.meta_harness import proposer as prp
from app.meta_harness import runs as runs_mod
from app.meta_harness.experiment import LeakageError

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the fake claude executable is a POSIX script"
)

HOLDOUT_ID = "task-secret-omega"

FAKE_CLAUDE = textwrap.dedent(
    '''\
    #!{python}
    """Fake ``claude`` CLI for the proposer tests."""
    import json, os, re, subprocess, sys, time
    from pathlib import Path

    cfg = json.loads(Path(os.environ["FAKE_CLAUDE_CONFIG"]).read_text())
    argv = sys.argv[1:]
    prompt = argv[argv.index("-p") + 1]
    pending_rel = re.search(r"Write `pending_eval.json` to: `([^`]+)`", prompt).group(1)
    thread_rel = str(Path(pending_rel).parent)
    iteration = int(re.search(r"Run iteration (\\d+)", prompt).group(1))
    parent = re.search(r"Parent candidate `[^`]+`: `([^`]+)`", prompt).group(1)
    cwd = Path.cwd()

    record = {{
        "argv": argv,
        "cwd": str(cwd),
        "api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "parent": parent,
        "parent_exists": (cwd / parent).is_file(),
        "listing": sorted(
            p.relative_to(cwd).as_posix() for p in cwd.rglob("*") if p.is_file()
        ),
    }}
    records = Path(cfg["record_dir"])
    records.mkdir(parents=True, exist_ok=True)
    (records / f"{{time.time_ns()}}-{{os.getpid()}}.json").write_text(json.dumps(record))

    def emit(event):
        print(json.dumps(event), flush=True)

    for event in cfg.get("events", []):
        emit(event)

    if cfg.get("spawn_child"):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        Path(cfg["pid_file"]).write_text(json.dumps([os.getpid(), child.pid]))

    if cfg.get("tamper_path"):
        target = Path(cfg["tamper_path"])
        target.write_text(target.read_text() + "\\n ")

    label = cfg.get("label", "cand")
    if label == "auto":
        label = f"cand_iter_{{iteration}}"
    if cfg.get("write_candidate", True):
        (cwd / "agents").mkdir(exist_ok=True)
        (cwd / "agents" / f"{{label}}.py").write_text(
            "from app.meta_harness.harness import CodingAgentHarness\\n\\n\\n"
            "class Cand(CodingAgentHarness):\\n"
            f"    AUTHORED_IN = {{thread_rel!r}}\\n"
        )
    for rel in cfg.get("extra_files", []):
        (cwd / rel).parent.mkdir(parents=True, exist_ok=True)
        (cwd / rel).write_text("not part of the handoff\\n")

    pending = cfg.get("pending", "auto")
    if pending == "auto":
        pending = {{
            "iteration": iteration,
            "candidates": [{{
                "name": label,
                "import_path": f"agents.{{label}}:Cand",
                "parent": "baseline",
                "hypothesis": "h",
                "axis": "exploration",
                "expected_score_delta": 0.01,
            }}],
        }}
    if pending is not None:
        (cwd / pending_rel).parent.mkdir(parents=True, exist_ok=True)
        (cwd / pending_rel).write_text(json.dumps(pending))

    time.sleep(cfg.get("sleep", 0))
    if cfg.get("result", True):
        emit({{"type": "result", "session_id": "s-1", "total_cost_usd": 0.01}})
    sys.exit(cfg.get("exit_code", 0))
    '''
)


class FakeClaude:
    """Configures the fake CLI and reads back what it recorded."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path = root / "fake-claude.json"
        self.record_dir = root / "records"
        self.configure()

    def configure(self, **cfg: Any) -> None:
        cfg.setdefault("record_dir", str(self.record_dir))
        self.config_path.write_text(json.dumps(cfg))

    def records(self) -> list[dict[str, Any]]:
        if not self.record_dir.is_dir():
            return []
        return [json.loads(p.read_text()) for p in sorted(self.record_dir.glob("*.json"))]

    def last(self) -> dict[str, Any]:
        records = self.records()
        assert records, "the fake claude never ran"
        return records[-1]


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "claude"
    script.write_text(FAKE_CLAUDE.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    fake = FakeClaude(tmp_path / "fake")
    fake.root.mkdir(exist_ok=True)
    fake.configure()
    monkeypatch.setenv("FAKE_CLAUDE_CONFIG", str(fake.config_path))
    return fake


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repository with everything the stage must leave out."""
    root = tmp_path / "repo"

    def put(rel: str, text: str = "x\n") -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    put("agents/__init__.py", "")
    put("agents/baseline.py", (REPO_ROOT / "agents" / "baseline.py").read_text())
    put("backend/app/meta_harness/harness.py", "# harness\n")
    put("skills/meta-harness-coding-agent/SKILL.md", "skill text\n")
    put("eval/tasks/task-alpha/task.json", '{"id": "task-alpha"}')
    put("eval/tasks/task-alpha/workspace/a.py")
    put(f"eval/holdout/{HOLDOUT_ID}/task.json", f'{{"id": "{HOLDOUT_ID}"}}')
    put("benchmarks/pass-rate/config.json", '{"experiment": "pass-rate"}')
    put(
        "benchmarks/holdout/config.json",
        json.dumps({"experiment": "holdout", "tasks": [HOLDOUT_ID]}),
    )
    put("benchmarks/results/old/summary.json", "{}")
    put("docs/evidence/replay-verification.json", "{}")
    put("docs/guide.md")
    put(".git/HEAD")
    put(".venv/bin/python")
    put("frontend/dashboard/node_modules/x.js")
    put("benchmark-results/x/summary.json", "{}")
    put(".env", "ANTHROPIC_API_KEY=sk-never-copied\n")
    put("runs/other-run/threads/other-run/thread.json", "{}")
    return root


def _run(repo: Path) -> Path:
    run_dir = runs_mod.make_run_dir(repo, "run-x")
    # A sibling branch's artifacts: never visible to another branch.
    runs_mod.write_frontier(run_dir, "run-x.fork.bbbb", {"sibling": True})
    return run_dir


def _skill(repo: Path) -> Path:
    return repo / "skills" / "meta-harness-coding-agent" / "SKILL.md"


def _propose(repo: Path, run_dir: Path, thread_id: str, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("iteration", 1)
    kwargs.setdefault("parent_name", "baseline")
    return prp.claude_propose(
        run_dir=run_dir,
        thread_id=thread_id,
        repo_root=repo,
        skill_path=_skill(repo),
        **kwargs,
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except OSError:
        return True
    return state != "Z"


def _wait_dead(pids: list[int], timeout: float = 10.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [p for p in pids if _alive(p)]
        if not alive:
            return []
        time.sleep(0.05)
    return [p for p in pids if _alive(p)]


# ── the staged copy ───────────────────────────────────────────────────


def test_the_proposer_runs_in_a_staged_copy_without_holdout_material(
    repo: Path, fake_claude: FakeClaude
) -> None:
    run_dir = _run(repo)
    # Something in the root thread a fork may read: its parent's snapshot.
    root_snapshot = runs_mod.candidate_source_path(run_dir, "run-x", "parent-cand")
    root_snapshot.write_text("# parent\n")
    thread_id = "run-x.fork.aaaa"
    # `loop --holdout` writes a post-hoc holdout evaluation into a thread.
    root_thread_dir = runs_mod.thread_dir(run_dir, "run-x")
    (root_thread_dir / "holdout-result.json").write_text("{}")
    holdout_trace = (
        runs_mod.candidate_dir(run_dir, "run-x", "parent-cand")
        / "holdout-traces"
        / f"{HOLDOUT_ID}-trial-1"
    )
    holdout_trace.mkdir(parents=True)
    (holdout_trace / "plan.json").write_text("{}")
    (holdout_trace.parent.parent / "traces").mkdir()
    (holdout_trace.parent.parent / "traces" / "plan.json").write_text("{}")

    _propose(repo, run_dir, thread_id, parent_name="parent-cand", iteration=2)

    seen = fake_claude.last()
    cwd = Path(seen["cwd"])
    assert repo.resolve() not in cwd.parents and cwd != repo.resolve()
    assert not cwd.exists(), "the stage is removed after the session"

    listing = seen["listing"]
    own = runs_mod.thread_dir(run_dir, thread_id).relative_to(repo).as_posix()
    root_thread = runs_mod.thread_dir(run_dir, "run-x").relative_to(repo).as_posix()
    sibling = runs_mod.thread_dir(run_dir, "run-x.fork.bbbb").relative_to(repo).as_posix()
    # What the proposer needs is there, at the paths the prompt uses.
    for present in (
        "agents/baseline.py",
        "backend/app/meta_harness/harness.py",
        "eval/tasks/task-alpha/task.json",
        "docs/guide.md",
        f"{own}/thread.json",
        f"{root_thread}/agents/parent-cand.py",
        f"{root_thread}/candidates/parent-cand/traces/plan.json",
    ):
        assert present in listing, present
    assert seen["parent"] == f"{root_thread}/agents/parent-cand.py"
    assert seen["parent_exists"] is True
    # What it must not see is not there at all.
    for absent in (
        "eval/holdout",
        "benchmarks/holdout",
        "benchmarks/results",
        "docs/evidence",
        ".git",
        ".venv",
        "frontend",
        "benchmark-results",
        "runs/other-run",
        sibling,
        ".env",
        f"{root_thread}/holdout-result.json",
        f"{root_thread}/candidates/parent-cand/holdout-traces",
    ):
        assert not any(p == absent or p.startswith(absent + "/") for p in listing), absent
    assert not any(HOLDOUT_ID in p for p in listing)


def test_only_the_candidate_and_the_handoff_come_back(
    repo: Path, fake_claude: FakeClaude
) -> None:
    run_dir = _run(repo)
    fake_claude.configure(extra_files=["agents/scratch_helper.py", "NOTES.md"])

    payload = _propose(repo, run_dir, "run-x")

    entry = payload["candidates"][0]
    snapshot = Path(entry["source_path"])
    assert snapshot == runs_mod.candidate_source_path(run_dir, "run-x", "cand")
    assert "AUTHORED_IN" in snapshot.read_text()
    assert entry["qualified_name"] == "cand"
    assert entry["source_sha256"]
    # The handoff landed in this branch's thread directory.
    pending = runs_mod.read_pending_eval(run_dir, "run-x")
    assert pending["candidates"][0]["name"] == "cand"
    # Nothing else crossed back, and the shared agents/ was never written.
    assert sorted(p.name for p in (repo / "agents").iterdir()) == [
        "__init__.py",
        "baseline.py",
    ]
    assert not (repo / "NOTES.md").exists()


def test_concurrent_forks_with_the_same_label_keep_their_own_source(
    repo: Path, fake_claude: FakeClaude
) -> None:
    """Invariant 2 for the real proposer: no shared file to race on."""
    run_dir = _run(repo)
    fake_claude.configure(sleep=0.5)
    forks = ["run-x.fork.left", "run-x.fork.right"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        payloads = list(pool.map(lambda t: _propose(repo, run_dir, t), forks))

    for thread_id, payload in zip(forks, payloads):
        snapshot = Path(payload["candidates"][0]["source_path"])
        own = runs_mod.thread_dir(run_dir, thread_id).relative_to(repo).as_posix()
        assert f"AUTHORED_IN = {own!r}" in snapshot.read_text(), thread_id
        assert snapshot.parent == runs_mod.thread_dir(run_dir, thread_id) / "agents"
    assert payloads[0]["candidates"][0]["source_sha256"] != (
        payloads[1]["candidates"][0]["source_sha256"]
    )


def test_deny_rules_for_holdout_and_results_are_passed_to_the_cli(
    repo: Path, fake_claude: FakeClaude
) -> None:
    run_dir = _run(repo)
    _propose(repo, run_dir, "run-x")
    argv = fake_claude.last()["argv"]
    assert "--disallowedTools" in argv
    rules = argv[argv.index("--disallowedTools") + 1 : argv.index("--disable-slash-commands")]
    for fragment in ("eval/holdout", "benchmarks/holdout", "benchmarks/results"):
        assert any(f"Read(./{fragment}/**)" == r for r in rules), fragment
        assert any(r.startswith("Bash(") and fragment in r for r in rules), fragment
    real = repo.resolve().as_posix().lstrip("/")
    assert f"Read(//{real}/**)" in rules


# ── isolation checks ──────────────────────────────────────────────────


def test_tampering_with_the_frozen_task_set_fails_the_iteration(
    repo: Path, fake_claude: FakeClaude
) -> None:
    run_dir = _run(repo)
    target = repo / "eval" / "tasks" / "task-alpha" / "task.json"
    # The path reaches the fake through its config, not a tool call, so
    # the audit cannot see it: only the before/after hash can.
    fake_claude.configure(tamper_path=str(target))

    with pytest.raises(LeakageError, match="eval/ or benchmarks/ changed"):
        _propose(repo, run_dir, "run-x")

    session = json.loads(
        (runs_mod.proposer_session_dir(run_dir, "run-x", 1) / "session.json").read_text()
    )
    assert session["isolation"]["frozen_tree_changes"] == ["eval/tasks/task-alpha/task.json"]
    # A void iteration hands nothing back.
    assert runs_mod.read_pending_eval(run_dir, "run-x") is None
    assert not runs_mod.candidate_source_path(run_dir, "run-x", "cand").exists()


def _tool_use(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "id": "t1", "name": name, "input": tool_input}
            ]
        },
    }


@pytest.mark.parametrize(
    "event_factory",
    [
        lambda repo: _tool_use("Read", {"file_path": "eval/holdout/x/task.json"}),
        lambda repo: _tool_use("Grep", {"pattern": "def", "path": f"EVAL\\{HOLDOUT_ID}"}),
        lambda repo: _tool_use("Bash", {"command": f"cat {repo.resolve()}/eval/tasks/a"}),
        lambda repo: _tool_use("Glob", {"pattern": "benchmarks/results/**/*.jsonl"}),
    ],
    ids=["holdout-path", "holdout-id", "real-repo-path", "results-path"],
)
def test_a_session_that_reaches_for_held_out_material_fails(
    repo: Path, fake_claude: FakeClaude, event_factory: Any
) -> None:
    run_dir = _run(repo)
    fake_claude.configure(events=[event_factory(repo)])

    with pytest.raises(LeakageError, match="referenced held-out material"):
        _propose(repo, run_dir, "run-x")

    session = json.loads(
        (runs_mod.proposer_session_dir(run_dir, "run-x", 1) / "session.json").read_text()
    )
    assert session["isolation"]["audit_hits"]
    assert runs_mod.read_pending_eval(run_dir, "run-x") is None


def test_ordinary_tool_calls_inside_the_stage_pass_the_audit(
    repo: Path, fake_claude: FakeClaude
) -> None:
    run_dir = _run(repo)
    fake_claude.configure(
        events=[
            _tool_use("Read", {"file_path": "eval/tasks/task-alpha/task.json"}),
            _tool_use("Bash", {"command": "python -c 'from agents.cand import *'"}),
        ]
    )
    payload = _propose(repo, run_dir, "run-x")
    assert payload["candidates"][0]["qualified_name"] == "cand"


# ── the handoff ───────────────────────────────────────────────────────


def test_a_stale_handoff_is_never_resubmitted(repo: Path, fake_claude: FakeClaude) -> None:
    """A proposer that writes nothing must fail, not reuse last iteration's file."""
    run_dir = _run(repo)
    runs_mod.write_pending_eval(
        run_dir,
        "run-x",
        {"iteration": 1, "candidates": [{"name": "old", "import_path": "agents.old:C"}]},
    )
    fake_claude.configure(pending=None, write_candidate=False)

    with pytest.raises(RuntimeError, match="did not write"):
        _propose(repo, run_dir, "run-x")
    assert runs_mod.read_pending_eval(run_dir, "run-x") is None


def test_a_sibling_branchs_handoff_does_not_satisfy_this_branch(
    repo: Path, fake_claude: FakeClaude
) -> None:
    run_dir = _run(repo)
    runs_mod.write_pending_eval(run_dir, "run-x.fork.bbbb", {"iteration": 1, "candidates": []})
    fake_claude.configure(pending=None)
    with pytest.raises(RuntimeError, match="did not write"):
        _propose(repo, run_dir, "run-x.fork.aaaa")


@pytest.mark.parametrize(
    ("pending", "message"),
    [
        ({"iteration": 1, "candidates": []}, "exactly one candidate"),
        (
            {
                "iteration": 1,
                "candidates": [
                    {"name": "cand", "import_path": "agents.cand:Cand"},
                    {"name": "cand2", "import_path": "agents.cand2:Cand"},
                ],
            },
            "exactly one candidate",
        ),
        (
            {"iteration": 7, "candidates": [{"name": "cand", "import_path": "agents.cand:Cand"}]},
            "iteration 7",
        ),
        (
            {"iteration": 1, "candidates": [{"name": "cand", "import_path": "agents.other:Cand"}]},
            "must be 'agents.cand:<ClassName>'",
        ),
        (
            {"iteration": 1, "candidates": [{"name": "ghost", "import_path": "agents.ghost:G"}]},
            "agents/ghost.py was not written",
        ),
        (
            {"iteration": 1, "candidates": [{"name": "../x", "import_path": "agents.x:X"}]},
            "invalid proposer candidate",
        ),
    ],
    ids=["none", "two", "wrong-iteration", "import-path", "no-file", "bad-name"],
)
def test_the_handoff_must_be_one_new_candidate_for_this_iteration(
    repo: Path, fake_claude: FakeClaude, pending: dict[str, Any], message: str
) -> None:
    run_dir = _run(repo)
    fake_claude.configure(pending=pending)
    with pytest.raises(prp.ProposalError, match=message):
        _propose(repo, run_dir, "run-x")


def test_a_name_already_used_in_the_run_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    (repo / "agents").mkdir(parents=True)
    (repo / "agents" / "cand.py").write_text("x = 1\n")
    run_dir = runs_mod.make_run_dir(repo, "run-y")
    payload = {"iteration": 2, "candidates": [{"name": "cand", "import_path": "agents.cand:C"}]}

    with pytest.raises(prp.ProposalError, match="already used"):
        prp.accept_proposal(
            payload,
            iteration=2,
            run_dir=run_dir,
            thread_id="run-y",
            existing_names={"baseline", "cand"},
            source_root=repo,
        )

    # Another thread of the same run already owns the name.
    (runs_mod.thread_dir(run_dir, "run-y.fork.zz") / "candidates" / "cand").mkdir(
        parents=True
    )
    with pytest.raises(prp.ProposalError, match="already used"):
        prp.accept_proposal(
            payload,
            iteration=2,
            run_dir=run_dir,
            thread_id="run-y",
            existing_names={"baseline"},
            source_root=repo,
        )


def test_re_accepting_in_the_same_thread_is_the_same_proposal(tmp_path: Path) -> None:
    """A resumed propose node re-proposes into its own thread; that is fine."""
    repo = tmp_path / "r"
    (repo / "agents").mkdir(parents=True)
    (repo / "agents" / "cand.py").write_text("x = 1\n")
    run_dir = runs_mod.make_run_dir(repo, "run-z")
    payload = {"iteration": 1, "candidates": [{"name": "cand", "import_path": "agents.cand:C"}]}
    for _ in range(2):
        accepted = prp.accept_proposal(
            payload,
            iteration=1,
            run_dir=run_dir,
            thread_id="run-z",
            existing_names={"baseline"},
            source_root=repo,
        )
    assert accepted["candidates"][0]["qualified_name"] == "cand"


# ── the parent the prompt points at ───────────────────────────────────


def test_the_prompt_points_at_the_source_the_parent_was_measured_with(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    run_dir = runs_mod.make_run_dir(repo, "run-p")
    assert (
        prp.resolve_parent_source(run_dir=run_dir, repo_root=repo, parent_name="baseline")
        == "agents/baseline.py"
    )
    assert (
        prp.resolve_parent_source(run_dir=run_dir, repo_root=repo, parent_name=None)
        == "agents/baseline.py"
    )
    # A fork's parent was benchmarked by the root thread: its snapshot is
    # there, not in the fork's own directory.
    snapshot = runs_mod.candidate_source_path(run_dir, "run-p", "cand")
    snapshot.write_text("x = 1\n")
    resolved = prp.resolve_parent_source(
        run_dir=run_dir,
        repo_root=repo,
        parent_name="cand",
        parent_candidate={"name": "cand", "source_path": str(snapshot)},
    )
    assert resolved == snapshot.relative_to(repo).as_posix()
    prompt = prp._render_proposer_prompt(
        2, run_dir, repo, "cand", "run-p.fork.abcd", resolved
    )
    assert f"`{resolved}`" in prompt
    assert "run-p.fork.abcd/agents/cand.py" not in prompt

    with pytest.raises(ValueError, match="no source snapshot"):
        prp.resolve_parent_source(run_dir=run_dir, repo_root=repo, parent_name="ghost")


# ── session log ───────────────────────────────────────────────────────


def test_the_cli_keeps_the_api_key_and_the_docstring_says_so(
    repo: Path, fake_claude: FakeClaude, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    run_dir = _run(repo)
    _propose(repo, run_dir, "run-x")
    assert fake_claude.last()["api_key"] == "test-key"
    assert "kept" in (prp.claude_propose.__doc__ or "")


def test_a_session_without_a_result_event_has_an_unknown_cost(
    repo: Path, fake_claude: FakeClaude
) -> None:
    run_dir = _run(repo)
    fake_claude.configure(result=False)
    _propose(repo, run_dir, "run-x")
    session = json.loads(
        (runs_mod.proposer_session_dir(run_dir, "run-x", 1) / "session.json").read_text()
    )
    assert session["cost_usd"] is None
    assert session["thread_id"] == "run-x"

    fake_claude.configure(result=True, label="cand2")
    _propose(repo, run_dir, "run-x", iteration=2)
    session = json.loads(
        (runs_mod.proposer_session_dir(run_dir, "run-x", 2) / "session.json").read_text()
    )
    assert session["cost_usd"] == 0.01


async def test_the_outer_loop_drives_the_real_proposer_path_end_to_end(
    repo: Path, fake_claude: FakeClaude
) -> None:
    """propose → validate → benchmark → update_frontier, real-proposer path.

    The second iteration's parent is the first iteration's candidate, so
    its prompt must point at that candidate's branch snapshot — and the
    file must exist in the proposer's stage.
    """
    from app.meta_harness.outer import run_outer_loop

    fake_claude.configure(label="auto")
    run_dir = runs_mod.make_run_dir(repo, "run-e2e")

    final = await run_outer_loop(
        run_dir=run_dir,
        repo_root=repo,
        eval_tasks_dir=repo / "eval" / "tasks",
        mock_proposer=False,
        mock_bench=True,
        trials=5,
        bench_workers=1,
        budget=2,
        skill_path=_skill(repo),
    )

    names = [c["name"] for c in final["candidates"]]
    assert names == ["baseline", "cand_iter_1", "cand_iter_2"]
    for c in final["candidates"][1:]:
        assert Path(c["source_path"]).parent == runs_mod.thread_dir(run_dir, "run-e2e") / "agents"
        assert c["status"] in {"accepted", "rejected"}
    first, second = fake_claude.records()[0], fake_claude.records()[1]
    assert first["parent"] == "agents/baseline.py"
    expected_parent = (
        runs_mod.candidate_source_path(run_dir, "run-e2e", "cand_iter_1")
        .relative_to(repo)
        .as_posix()
    )
    assert final["candidates"][1]["status"] == "accepted"
    assert second["parent"] == expected_parent
    assert second["parent_exists"] is True
    assert not (repo / "agents" / "cand_iter_1.py").exists()


def test_mock_proposer_labels_are_branch_unique() -> None:
    """Two forks on the same iteration must not author the same file."""
    root = prp.mock_label(3, "run-a", "run-a")
    fork_a = prp.mock_label(3, "run-a.fork.aaaa", "run-a")
    fork_b = prp.mock_label(3, "run-a.fork.bbbb", "run-a")
    assert root == "_mock_iter_3"
    assert len({root, fork_a, fork_b}) == 3


# ── lifetime: nothing outlives the node ───────────────────────────────


def test_a_timeout_kills_the_whole_process_tree(repo: Path, fake_claude: FakeClaude) -> None:
    run_dir = _run(repo)
    pid_file = fake_claude.root / "pids.json"
    fake_claude.configure(spawn_child=True, pid_file=str(pid_file), sleep=120)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exit_code=124"):
        _propose(repo, run_dir, "run-x", timeout_seconds=2)
    assert time.monotonic() - started < 30

    pids = json.loads(pid_file.read_text())
    assert _wait_dead(pids) == [], "the CLI or its child outlived the timeout"


async def test_cancelling_the_task_kills_the_whole_process_tree(
    repo: Path, fake_claude: FakeClaude
) -> None:
    run_dir = _run(repo)
    pid_file = fake_claude.root / "pids.json"
    fake_claude.configure(spawn_child=True, pid_file=str(pid_file), sleep=120)

    task = asyncio.create_task(
        prp.run_claude_proposer(
            run_dir=run_dir,
            thread_id="run-x",
            iteration=1,
            parent_name="baseline",
            repo_root=repo,
            skill_path=_skill(repo),
        )
    )
    deadline = time.monotonic() + 30
    while not pid_file.exists():
        assert time.monotonic() < deadline, "the fake claude never started"
        await asyncio.sleep(0.05)
    pids = json.loads(pid_file.read_text())

    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 30

    # Dead by the time the cancelled task returns, not eventually.
    assert [p for p in pids if _alive(p)] == [] or _wait_dead(pids, 2.0) == []
    session = json.loads(
        (runs_mod.proposer_session_dir(run_dir, "run-x", 1) / "session.json").read_text()
    )
    assert session["cancelled"] is True
    assert runs_mod.read_pending_eval(run_dir, "run-x") is None


def test_stage_isolation_check_catches_a_holdout_named_path(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "agents" / HOLDOUT_ID).mkdir(parents=True)
    with pytest.raises(LeakageError, match="holdout material"):
        prp.assert_stage_isolated(stage, holdout_ids=[HOLDOUT_ID])
    shutil.rmtree(stage / "agents" / HOLDOUT_ID)
    prp.assert_stage_isolated(stage, holdout_ids=[HOLDOUT_ID])
