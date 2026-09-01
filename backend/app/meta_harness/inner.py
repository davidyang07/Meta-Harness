"""Inner-loop state machine: ``orient → plan → act → verify → submit``.

Per Appendix C §C.8 / INTERFACES.md §1.2. **All nodes are async** so
the inner graph can be checkpointed via ``AsyncPostgresSaver`` (step 7)
and forks can run concurrently via ``asyncio.create_task`` (step 9).

Node bodies still issue some sync subprocess calls (``find``, ``pytest``)
because their wall time is short and bounded; we accept the brief
event-loop block. Use ``asyncio.to_thread`` if a future change makes
these long-running.

**Every crossing into the world goes through ``effects``.** Model calls,
tool dispatch, the workspace scan, the verify subprocess, the final-file
snapshot and every trace write are routed through
``app.meta_harness.effects``. With the default :class:`Effects` that is
exactly the behaviour it always was; with a recording or replaying
implementation the same graph re-executes against a tape instead of the
world (see ``recording.py``). Adding a new nondeterministic call to a
node body without routing it through ``effects`` silently breaks exact
replay, so don't.

**Message identity is deterministic.** ``act`` stamps a positional ``id``
on every message it produces. Without one, LangGraph's ``add_messages``
reducer assigns a fresh UUID per message, which (a) makes two executions
of the same recorded run produce different state and (b) makes a
re-executed ``act`` append duplicates instead of replacing its own
writes — the idempotency requirement for a node that can be re-entered
after an interrupt.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.meta_harness import effects as fx
from app.meta_harness import recording as rec
from app.meta_harness.harness import PLAN_TOOL_SCHEMA, CodingAgentHarness
from app.meta_harness.state import CodingAgentState
from app.meta_harness.tools import TOOL_SCHEMAS, execute_tool

ACT_TOOLS = TOOL_SCHEMAS  # all 6 fixed tools incl. task_complete


# ──────────────────────────────────────────────────────────────────────
# Phase 1 — orient
# ──────────────────────────────────────────────────────────────────────


def _depth_limited_tree(workspace: Path, max_depth: int = 3) -> str:
    """Build a depth-limited workspace tree (best-effort, sync)."""
    try:
        proc = subprocess.run(
            [
                "find",
                ".",
                "-maxdepth",
                str(max_depth),
                "-not",
                "-path",
                "*/__pycache__/*",
                "-not",
                "-path",
                "*/.*",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        return proc.stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def _scan_workspace(workspace: Path) -> dict[str, Any]:
    """Read everything the planner is shown. Sync; called via to_thread."""
    tree = _depth_limited_tree(workspace)

    has_python = (workspace / "pyproject.toml").exists() or any(
        workspace.rglob("*.py")
    )
    project_meta = {
        "lang": "python" if has_python else "unknown",
        "test_runner": "pytest" if has_python else "unknown",
    }

    tests: dict[str, str] = {}
    for test_file in list(workspace.rglob("test_*.py"))[:10]:
        if test_file.is_file():
            try:
                tests[str(test_file.relative_to(workspace))] = test_file.read_text()[
                    :4000
                ]
            except OSError:
                pass

    configs: dict[str, str] = {}
    for cfg_name in ["README.md", "pyproject.toml", "package.json", "Makefile"]:
        cfg_path = workspace / cfg_name
        if (
            cfg_path.exists()
            and cfg_path.is_file()
            and cfg_path.stat().st_size < 4000
        ):
            try:
                configs[cfg_name] = cfg_path.read_text()
            except OSError:
                pass

    return {
        "tree": tree[:2000],
        "project": project_meta,
        "configs": configs,
        "tests": tests,
    }


async def orient(
    state: CodingAgentState,
    harness: CodingAgentHarness,
    effects: fx.Effects | None = None,
) -> dict[str, Any]:
    """Phase 1: build initial context for the planner."""
    effects = effects or fx.Effects()
    effects.set_node("orient")
    workspace = Path(state["workspace_path"])

    summary = await effects.observe(
        rec.KIND_ORIENT,
        {"task_id": state["task"].get("id")},
        lambda: asyncio.to_thread(_scan_workspace, workspace),
    )

    trace_dir = _trace_dir_or_none(state)
    if trace_dir is not None:
        effects.write_trace(trace_dir / "orient.json", json.dumps(summary, indent=2))

    effects.end_step("orient")
    return {"orient_summary": summary}


# ──────────────────────────────────────────────────────────────────────
# Phase 2 — plan
# ──────────────────────────────────────────────────────────────────────


async def plan(
    state: CodingAgentState,
    harness: CodingAgentHarness,
    effects: fx.Effects | None = None,
) -> dict[str, Any]:
    """Phase 2: produce a structured plan via forced tool call (async)."""
    effects = effects or fx.Effects()
    effects.set_node("plan")
    orient_summary = state["orient_summary"] or {}
    summary = harness._build_initial_context(orient_summary)
    instruction = state["task"]["instruction"]

    prompt = harness.PLAN_PROMPT_TEMPLATE.format(
        instruction=instruction,
        tree=summary.get("tree", "")[:1500],
        lang=summary.get("project", {}).get("lang", "unknown"),
        test_runner=summary.get("project", {}).get("test_runner", "unknown"),
        tests=json.dumps(
            {k: v[:500] for k, v in summary.get("tests", {}).items()}, indent=2
        ),
    )

    messages = [{"role": "user", "content": prompt}]
    response = await harness._call_llm(
        messages=messages,
        tools=[PLAN_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_plan"},
    )

    plan_dict: dict[str, Any] = {}
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_plan":
            plan_dict = dict(block.input)
            break

    trace_dir = _trace_dir_or_none(state)
    if trace_dir is not None:
        effects.write_trace(trace_dir / "plan.json", json.dumps(plan_dict, indent=2))

    effects.end_step("plan")
    return {"plan": plan_dict}


# ──────────────────────────────────────────────────────────────────────
# Phase 3 — act (bounded ReAct, async)
# ──────────────────────────────────────────────────────────────────────


def _serialize_block(block: Any) -> dict[str, Any]:
    """Convert an Anthropic SDK content block into a plain dict."""
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": dict(block.input),
        }
    return {"type": str(btype), "raw": str(block)}


_ROLE_BY_MESSAGE_TYPE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "user",
}


def _as_api_message(message: Any) -> dict[str, Any]:
    """Normalise one state message back to the Anthropic wire shape.

    ``add_messages`` stores messages as LangChain ``BaseMessage`` objects,
    so on the second entry into ``act`` (the verify → act retry edge)
    ``state["messages"]`` is no longer the list of dicts ``act`` wrote.
    Sending those objects to the Anthropic SDK is not a valid request;
    normalising here is what makes the retry path work at all, and it is
    also what keeps a replayed request byte-identical to the recorded one.
    """
    if isinstance(message, dict):
        return dict(message)
    role = _ROLE_BY_MESSAGE_TYPE.get(getattr(message, "type", ""), "user")
    normalised: dict[str, Any] = {"role": role, "content": message.content}
    message_id = getattr(message, "id", None)
    if message_id:
        normalised["id"] = message_id
    return normalised


def _as_api_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [_as_api_message(m) for m in messages]


def _request_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip our bookkeeping ``id`` before the request goes to the model."""
    return [{k: v for k, v in m.items() if k != "id"} for m in messages]


def _messages_update(messages: list[dict[str, Any]]) -> list[Any]:
    """The value ``act`` writes to ``state["messages"]``.

    Stamps a deterministic positional id on every message, and prefixes a
    clear so the write *replaces* the trajectory rather than merging into
    it. Both halves matter:

    - Without stable ids, ``add_messages`` mints a random UUID per
      message and no two executions of the same recorded run produce the
      same state.
    - Without the clear, an ``act`` whose override-10 overflow strategy
      dropped messages would leave the dropped tail behind in state,
      because ``add_messages`` merges by id and never removes.

    Together they also make the node idempotent: re-executing ``act``
    after an interrupt rewrites the same list instead of appending a
    second copy.
    """
    return [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        *({**m, "id": f"m{index:04d}"} for index, m in enumerate(messages)),
    ]


async def act(
    state: CodingAgentState,
    harness: CodingAgentHarness,
    effects: fx.Effects | None = None,
) -> dict[str, Any]:
    """Phase 3: bounded ReAct over the 6 fixed tools (async)."""
    effects = effects or fx.Effects()
    effects.set_node("act")
    workspace = Path(state["workspace_path"])
    plan_dict = state["plan"] or {}
    trace_dir = _trace_dir_or_none(state)
    tool_log_path = (trace_dir / "act-tools.jsonl") if trace_dir else None

    messages = _as_api_messages(state.get("messages") or [])
    if not messages:
        messages.append(
            {"role": "user", "content": harness._compose_act_prompt(plan_dict)}
        )

    turn_count = state.get("turn_count", 0)
    act_complete = False

    while turn_count < harness.MAX_ACT_TURNS:
        if len(messages) > 40:
            messages = harness._summarize_for_overflow(messages)

        response = await harness._call_llm(_request_messages(messages), ACT_TOOLS)

        assistant_blocks: list[dict[str, Any]] = []
        tool_uses: list[Any] = []
        for block in response.content:
            assistant_blocks.append(_serialize_block(block))
            if getattr(block, "type", None) == "tool_use":
                tool_uses.append(block)

        messages.append({"role": "assistant", "content": assistant_blocks})

        if not tool_uses:
            break

        tool_results: list[dict[str, Any]] = []
        for tu in tool_uses:
            if tu.name == "task_complete":
                act_complete = True
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": "Task marked complete; running verify.",
                    }
                )
                if tool_log_path is not None:
                    _append_tool_log(
                        effects,
                        tool_log_path,
                        turn=turn_count + 1,
                        tool="task_complete",
                        tool_input={},
                        output_summary="complete",
                        is_error=False,
                    )
                continue

            # Tool dispatch is sync (subprocess-based). Wrap in
            # to_thread so we don't block the event loop on long
            # bash commands.
            tool_input = dict(tu.input)
            result = await effects.observe(
                rec.KIND_TOOL,
                {"tool": tu.name, "input": tool_input},
                lambda name=tu.name, args=tool_input: asyncio.to_thread(
                    execute_tool, name, workspace, **args
                ),
            )
            formatted = harness._format_tool_result(tu.name, result)
            is_error = result.get("status") == "error"
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": formatted,
                    "is_error": is_error,
                }
            )
            if tool_log_path is not None:
                _append_tool_log(
                    effects,
                    tool_log_path,
                    turn=turn_count + 1,
                    tool=tu.name,
                    tool_input=tool_input,
                    output_summary=formatted[:400],
                    is_error=is_error,
                )

        messages.append({"role": "user", "content": tool_results})
        turn_count += 1

        if act_complete:
            break

    if trace_dir is not None:
        effects.write_trace(
            trace_dir / "act-messages.jsonl",
            "".join(json.dumps(m, default=str) + "\n" for m in messages),
        )

    effects.end_step("act")
    return {"messages": _messages_update(messages), "turn_count": turn_count}


def _append_tool_log(
    effects: fx.Effects,
    path: Path,
    *,
    turn: int,
    tool: str,
    tool_input: dict[str, Any],
    output_summary: str,
    is_error: bool,
) -> None:
    entry = {
        "turn": turn,
        "tool": tool,
        "input": tool_input,
        "output_summary": output_summary,
        "is_error": is_error,
    }
    effects.append_trace(path, json.dumps(entry, default=str) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Phase 4 — verify
# ──────────────────────────────────────────────────────────────────────


def _run_verify_subprocess(workspace: Path, test_command: str) -> tuple[bool, str]:
    """Sync helper for verify (called via asyncio.to_thread).

    **Trust boundary.** ``test_command`` runs through a shell. It comes
    from ``eval/tasks/<id>/task.json``, which is committed repository
    content — not user input, not model output, and not anything the
    proposer can write (the proposer authors ``agents/*.py`` only). A
    task definition is as trusted as the rest of the source tree.

    Do not widen this: if task specs ever become user-supplied, this call
    needs an argv list and an allowlist first. The workspace itself is a
    disposable sandbox copy (``sandbox.sandbox_for``), so a command can
    destroy its own workspace but not the task's pristine source.
    """
    try:
        proc = subprocess.run(  # noqa: S602 — see the trust boundary above
            test_command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        return proc.returncode == 0, (proc.stdout + "\n" + proc.stderr)[-2000:]
    except subprocess.TimeoutExpired as exc:
        # With ``text=True`` the partial output on a timeout is already
        # ``str``. Calling ``.decode()`` on it turned every verify
        # timeout into an AttributeError, which surfaced as a crashed
        # node rather than the failed trial it actually is.
        out = _as_text(exc.stdout) + _as_text(exc.stderr)
        return False, (out + "\n[timeout]")[-2000:]


def _as_text(stream: str | bytes | None) -> str:
    """Decode captured subprocess output regardless of ``text`` mode."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return str(stream)


async def verify(
    state: CodingAgentState,
    harness: CodingAgentHarness,
    effects: fx.Effects | None = None,
) -> dict[str, Any]:
    """Phase 4: run the task's test_command + persist verify.json."""
    effects = effects or fx.Effects()
    effects.set_node("verify")
    workspace = Path(state["workspace_path"])
    test_command = state["task"].get("test_command", "pytest -q")
    attempt = state.get("verify_attempts", 0) + 1

    tests_pass, output = await effects.observe(
        rec.KIND_VERIFY,
        {"command": test_command, "attempt": attempt},
        lambda: asyncio.to_thread(_run_verify_subprocess, workspace, test_command),
    )

    verify_result = {
        "tests_pass": tests_pass,
        "tests_failed": [],
        "test_output": output,
        "lint_pass": True,
        "lint_errors": [],
        "out_of_plan_changes": [],
    }

    trace_dir = _trace_dir_or_none(state)
    if trace_dir is not None:
        effects.write_trace(
            trace_dir / "verify.json", json.dumps(verify_result, indent=2)
        )

    effects.end_step("verify")
    return {"verify_result": verify_result, "verify_attempts": attempt}


# ──────────────────────────────────────────────────────────────────────
# Phase 5 — submit
# ──────────────────────────────────────────────────────────────────────


def _snapshot_workspace(workspace: Path) -> dict[str, str]:
    """Read back every small file in the workspace. Sync; via to_thread."""
    final_files: dict[str, str] = {}
    for f in sorted(workspace.rglob("*")):
        if f.is_file() and f.stat().st_size < 50_000:
            try:
                final_files[str(f.relative_to(workspace))] = f.read_text()
            except (OSError, UnicodeDecodeError):
                pass
    return final_files


async def submit(
    state: CodingAgentState,
    harness: CodingAgentHarness,
    effects: fx.Effects | None = None,
) -> dict[str, Any]:
    """Phase 5: snapshot workspace, write score.json + summary.md +
    final-files.json."""
    effects = effects or fx.Effects()
    effects.set_node("submit")
    workspace = Path(state["workspace_path"])
    verify_result = state.get("verify_result") or {}
    score = 1.0 if verify_result.get("tests_pass") else 0.0

    final_files = await effects.observe(
        rec.KIND_FILES,
        {"task_id": state["task"].get("id")},
        lambda: asyncio.to_thread(_snapshot_workspace, workspace),
    )

    trace_dir = _trace_dir_or_none(state)
    if trace_dir is not None:
        effects.write_trace(
            trace_dir / "final-files.json", json.dumps(final_files, indent=2)
        )
        effects.write_trace(
            trace_dir / "score.json",
            json.dumps(
                {
                    "passed": bool(verify_result.get("tests_pass")),
                    "score": score,
                    "why": (
                        "all tests green"
                        if score == 1.0
                        else "tests failed after retries exhausted"
                    ),
                },
                indent=2,
            ),
        )
        effects.write_trace(
            trace_dir / "summary.md",
            f"""# Trial summary

- Task: {state["task"].get("id", "unknown")}
- Score: {score}
- Turns: {state.get("turn_count", 0)}
- Verify attempts: {state.get("verify_attempts", 0)}
- Tests pass: {verify_result.get("tests_pass", False)}
""",
        )

    effects.end_step("submit")
    return {"score": score, "final_files": final_files}


# ──────────────────────────────────────────────────────────────────────
# Routing + graph build + run
# ──────────────────────────────────────────────────────────────────────


def _route_after_verify(state: CodingAgentState, max_verify_retries: int = 3) -> str:
    """Conditional edge: loop back to act on test failure if budget remains."""
    verify_result = state.get("verify_result") or {}
    if verify_result.get("tests_pass", False):
        return "submit"
    if state.get("verify_attempts", 0) >= max_verify_retries:
        return "submit"
    return "act"


def _route_after_verify_for_harness(
    state: CodingAgentState, harness: CodingAgentHarness
) -> str:
    """Conditional edge using the harness' retry policy."""
    verify_result = state.get("verify_result") or {}
    if verify_result.get("tests_pass", False):
        return "submit"
    if state.get("verify_attempts", 0) >= harness.MAX_VERIFY_RETRIES:
        return "submit"
    return "act" if harness.should_loop_back_to_act(verify_result) else "submit"


def build_inner_graph(
    harness: CodingAgentHarness,
    *,
    checkpointer: Any = None,
    effects: fx.Effects | None = None,
) -> Any:
    """Compile the inner-loop ``StateGraph``. ``checkpointer`` is passed
    through to ``compile()``; ``None`` means no checkpointer (in-memory
    only, used by tests and by mock-bench).

    ``effects`` is the boundary to the world (default: live execution).
    The graph's *shape* is identical in every mode — that is the point:
    a replay re-runs this exact state machine and recomputes its own
    routing decisions.

    Wraps each phase function in an async closure that captures
    ``harness`` — sync lambdas would return coroutines without awaiting,
    which LangGraph rejects as ``InvalidUpdateError``.
    """
    fx_impl = effects or fx.Effects()

    async def _orient(s: CodingAgentState) -> dict[str, Any]:
        return await orient(s, harness, fx_impl)

    async def _plan(s: CodingAgentState) -> dict[str, Any]:
        return await plan(s, harness, fx_impl)

    async def _act(s: CodingAgentState) -> dict[str, Any]:
        return await act(s, harness, fx_impl)

    async def _verify(s: CodingAgentState) -> dict[str, Any]:
        return await verify(s, harness, fx_impl)

    async def _submit(s: CodingAgentState) -> dict[str, Any]:
        return await submit(s, harness, fx_impl)

    g: StateGraph = StateGraph(CodingAgentState)
    g.add_node("orient", _orient)
    g.add_node("plan", _plan)
    g.add_node("act", _act)
    g.add_node("verify", _verify)
    g.add_node("submit", _submit)

    g.add_edge(START, "orient")
    g.add_edge("orient", "plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "verify")
    g.add_conditional_edges(
        "verify",
        lambda s: _route_after_verify_for_harness(s, harness),
        {"act": "act", "submit": "submit"},
    )
    g.add_edge("submit", END)
    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()


def _trace_dir_or_none(state: CodingAgentState) -> Path | None:
    raw = state["task"].get("_trace_dir")
    return Path(raw) if raw else None


def initial_inner_state(
    *, task_dict: dict[str, Any], workspace: Path | str
) -> CodingAgentState:
    """The inner loop's entry state. Shared by live runs and replays."""
    return {
        "task": task_dict,
        "workspace_path": str(workspace),
        "orient_summary": None,
        "plan": None,
        "messages": [],
        "turn_count": 0,
        "verify_attempts": 0,
        "verify_result": None,
        "final_files": None,
        "score": None,
    }


async def run_inner_loop(
    harness: CodingAgentHarness,
    *,
    task_dict: dict[str, Any],
    workspace: Path,
    trace_dir: Path | None = None,
    thread_id: str = "inner-trial-1",
    checkpointer: Any = None,
    effects: fx.Effects | None = None,
) -> CodingAgentState:
    """Run one inner-loop trial. Async."""
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        task_dict = dict(task_dict)
        task_dict["_trace_dir"] = str(trace_dir)

    initial_state = initial_inner_state(task_dict=task_dict, workspace=workspace)

    graph = build_inner_graph(harness, checkpointer=checkpointer, effects=effects)
    final_state = await graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100},
    )
    return final_state  # type: ignore[return-value]
