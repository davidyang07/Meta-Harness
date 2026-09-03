"""Nothing in a node body may reach the world except through ``effects``.

Exact replay rests on one structural property: every nondeterministic
thing an inner-loop run does is routed through
``app.meta_harness.effects``, so a replaying implementation can serve it
from a tape instead. ``test_exact_replay.py`` proves that *today's*
graph replays exactly. It cannot prove that tomorrow's does — a node
that grows a direct ``subprocess.run`` or a fresh API client still
replays "successfully" while quietly doing the real thing, because the
tape is simply never consulted for that call.

That failure is silent, which is what makes it worth a static test. The
checks below read `inner.py` as a syntax tree and assert the shape:
every crossing sits inside a producer handed to ``effects.observe``, and
no graph node reaches the world on its own.

They are deliberately about `inner.py`. `outer.py` is not replayed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import recording as rec  # noqa: E402

INNER = REPO_ROOT / "backend" / "app" / "meta_harness" / "inner.py"
EFFECTS = REPO_ROOT / "backend" / "app" / "meta_harness" / "effects.py"

#: The graph's node coroutines. `build_inner_graph` registers exactly
#: these five, and a sixth would have to be added here consciously.
NODE_FUNCTIONS = ("orient", "plan", "act", "verify", "submit")

#: Calls that reach outside the process. A node body may not make one
#: directly; a producer passed to ``effects.observe`` may.
CROSSINGS = (
    "asyncio.to_thread",
    "subprocess.run",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.call",
    "os.system",
    "os.popen",
)


def _tree() -> ast.Module:
    return ast.parse(INNER.read_text(encoding="utf-8"))


def _dotted(node: ast.AST) -> str:
    """`asyncio.to_thread` for an Attribute chain, else ''."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    links: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            links[child] = parent
    return links


def _observe_producers(tree: ast.Module) -> list[ast.AST]:
    """Every callable handed to an ``effects.observe(...)`` call."""
    producers: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _dotted(node.func).endswith("observe"):
            continue
        for argument in [*node.args, *(kw.value for kw in node.keywords)]:
            if isinstance(argument, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
                producers.append(argument)
    return producers


def _inside(node: ast.AST, container: ast.AST) -> bool:
    return any(candidate is node for candidate in ast.walk(container))


def _enclosing_function(
    node: ast.AST, links: dict[ast.AST, ast.AST]
) -> ast.AST | None:
    current = links.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = links.get(current)
    return None


# ── the boundary itself ───────────────────────────────────────────────


def _module_functions(tree: ast.Module) -> dict[str, ast.AST]:
    """Top-level `def`/`async def` in the module, by name."""
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _names_called(scope: ast.AST) -> set[str]:
    """Bare names invoked as functions anywhere inside `scope`."""
    called: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
        # `asyncio.to_thread(_helper, ...)` passes the callable as an
        # argument rather than calling it, so catch that shape too.
        if isinstance(node, ast.Call):
            for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                if isinstance(argument, ast.Name):
                    called.add(argument.id)
    return called


def _behind_the_boundary(tree: ast.Module) -> tuple[list[ast.AST], set[str]]:
    """Regions that may legitimately reach the world.

    A crossing is allowed inside a producer handed to
    ``effects.observe``, and inside any module-level helper reachable
    from such a producer — ``observe(..., lambda:
    asyncio.to_thread(_run_verify_subprocess, ...))`` puts the helper
    behind the boundary just as surely as inlining its body would.

    Returns the producer nodes and the names of the reachable helpers.
    """
    producers = _observe_producers(tree)
    functions = _module_functions(tree)

    reachable: set[str] = set()
    frontier = set()
    for producer in producers:
        frontier |= _names_called(producer) & set(functions)
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        reachable.add(name)
        frontier |= _names_called(functions[name]) & set(functions)
    return producers, reachable


def test_every_world_crossing_in_inner_sits_behind_the_effects_boundary():
    """A direct crossing replays "fine" while doing the real thing."""
    tree = _tree()
    producers, reachable = _behind_the_boundary(tree)
    assert producers, "no effects.observe producers found; the scan is broken"

    functions = _module_functions(tree)
    allowed = [*producers, *(functions[name] for name in reachable)]

    escapes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name not in CROSSINGS:
            continue
        if not any(_inside(node, region) for region in allowed):
            escapes.append(f"{name} at inner.py:{node.lineno}")

    assert escapes == [], (
        "these calls reach the world without passing through "
        f"effects.observe, which silently breaks exact replay: {escapes}"
    )


def test_a_helper_behind_the_boundary_is_not_also_called_in_front_of_it():
    """The one-hop allowance holds only while it is the *only* path.

    ``_run_verify_subprocess`` is exempt from the scan above because
    every route to it runs through ``effects.observe``. The moment a
    node body calls it directly, that exemption is laundering a real
    crossing, and the tape is bypassed for that call.
    """
    tree = _tree()
    producers, reachable = _behind_the_boundary(tree)
    functions = _module_functions(tree)
    links = _parents(tree)

    leaks: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func.id if isinstance(node.func, ast.Name) else None
        if target not in reachable:
            continue
        inside_producer = any(_inside(node, producer) for producer in producers)
        owner = _enclosing_function(node, links)
        inside_helper = owner is not None and owner.name in reachable
        if not (inside_producer or inside_helper):
            where = owner.name if owner is not None else "module scope"
            leaks.append(f"{where} calls {target} at inner.py:{node.lineno}")

    assert leaks == [], (
        "a helper that owns a world-crossing is reachable without going "
        f"through effects.observe: {leaks}"
    )


def test_no_graph_node_reaches_the_world_on_its_own():
    """The five node coroutines delegate; they do not execute."""
    tree = _tree()
    links = _parents(tree)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if not name.startswith("subprocess.") and name not in ("os.system", "os.popen"):
            continue
        owner = _enclosing_function(node, links)
        if owner is not None and owner.name in NODE_FUNCTIONS:
            offenders.append(f"{owner.name} calls {name} at inner.py:{node.lineno}")

    assert offenders == [], (
        "a node body runs a subprocess directly rather than through a "
        f"helper behind effects.observe: {offenders}"
    )


def test_no_node_constructs_its_own_model_client():
    """The LLM boundary is instrumented on the harness instance.

    ``effects.instrument_harness_for_effects`` wraps whatever
    ``_call_llm`` the candidate supplied. A node that built its own
    client would sidestep that wrapper entirely.
    """
    source = INNER.read_text(encoding="utf-8")
    for banned in ("Anthropic(", "AsyncAnthropic(", "ChatAnthropic(", "OpenAI("):
        assert banned not in source, f"inner.py constructs a client: {banned}"


def test_the_effect_kinds_are_a_closed_set_and_all_are_used():
    """A new kind is a deliberate change to the tape format."""
    assert rec.EFFECT_KINDS == {
        rec.KIND_LLM,
        rec.KIND_TOOL,
        rec.KIND_ORIENT,
        rec.KIND_VERIFY,
        rec.KIND_FILES,
    }

    combined = INNER.read_text(encoding="utf-8") + EFFECTS.read_text(encoding="utf-8")
    for kind in ("KIND_LLM", "KIND_TOOL", "KIND_ORIENT", "KIND_VERIFY", "KIND_FILES"):
        assert kind in combined, f"{kind} is declared but never routed"


# ── the replaying implementation cannot cheat ─────────────────────────


def test_replay_effects_hold_no_reference_to_a_producer():
    """The proof that replay issues no call is structural, not a promise.

    ``ReplayEffects.observe`` must not name its ``produce`` argument in
    its body at all — not in a fallback, not in an ``except``, not
    behind a flag.
    """
    effects_tree = ast.parse(EFFECTS.read_text(encoding="utf-8"))
    replay_observe = None
    for node in ast.walk(effects_tree):
        if isinstance(node, ast.ClassDef) and node.name == "ReplayEffects":
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "observe"
                ):
                    replay_observe = item
    assert replay_observe is not None, "ReplayEffects.observe not found"

    used = {
        child.id
        for child in ast.walk(replay_observe)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    assert "produce" not in used, (
        "ReplayEffects.observe references its producer; replay can no "
        "longer be shown to issue zero calls by inspection"
    )
