"""The boundary between the inner loop's state machine and the world.

Every nondeterministic thing an inner-loop run does — call a model, run a
tool, scan a workspace, shell out to pytest, snapshot final files — goes
through one of three implementations of the same interface:

- :class:`LiveEffects` does the real thing. This is the default and the
  behaviour of every run that is not being recorded.
- :class:`RecordingEffects` does the real thing *and* writes it to a
  :class:`~app.meta_harness.recording.TapeWriter`.
- :class:`ReplayEffects` does none of it. It serves each request from a
  recorded tape and raises :class:`ReplayDivergence` the moment the graph
  asks for something the tape does not have at that position.

Why an interface rather than monkey-patching: replay has to be *proved*
not to touch the world, and the proof is that ``ReplayEffects`` has no
code path that calls the producer at all. ``produce`` is a callable the
node hands in; ``ReplayEffects.observe`` never invokes it.

Trace files are also routed here. In replay they are suppressed, so a
replay of a run cannot overwrite that run's own recorded artifacts.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.meta_harness import recording as rec


class Effects:
    """Live execution: every observation actually happens."""

    #: True when this instance is serving from a tape rather than the world.
    replaying: bool = False
    #: True when this instance is appending to a tape.
    recording: bool = False

    async def observe(
        self,
        kind: str,
        key_input: Any,
        produce: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Perform one boundary crossing and return its result."""
        return await produce()

    def write_trace(self, path: Path, text: str) -> None:
        """Write a per-trial trace artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def append_trace(self, path: Path, text: str) -> None:
        """Append a line to a per-trial JSONL trace artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)

    def set_node(self, node: str | None) -> None:
        """Tell the tape which node is currently executing."""

    def end_step(self, node: str) -> None:
        """Record that a graph node finished."""


LiveEffects = Effects


class RecordingEffects(Effects):
    """Live execution that also writes an execution tape."""

    replaying = False
    recording = True

    def __init__(self, writer: rec.TapeWriter) -> None:
        self.writer = writer

    async def observe(
        self,
        kind: str,
        key_input: Any,
        produce: Callable[[], Awaitable[Any]],
    ) -> Any:
        # The key is computed *before* the world is touched. A producer
        # that mutates its own inputs (a ``_call_llm`` override adding
        # ``cache_control`` to the last message is the common case) would
        # otherwise be keyed on what it did to the request rather than on
        # the request, and replay — which never runs the producer — would
        # ask for a different key.
        key = rec.effect_key(kind, key_input)
        value = await produce()
        encode, _ = rec.CODECS[kind]
        self.writer.append(kind, key, encode(value))
        return value

    def set_node(self, node: str | None) -> None:
        self.writer.set_node(node)

    def end_step(self, node: str) -> None:
        self.writer.mark_step(node)


class ReplayEffects(Effects):
    """Serves every observation from a recorded tape. Touches nothing.

    ``produce`` is accepted for interface compatibility and deliberately
    never called: that is what makes "no external model calls" a property
    of the code rather than a claim in a docstring.
    """

    replaying = True
    recording = False

    def __init__(self, reader: rec.TapeReader) -> None:
        self.reader = reader
        self.steps: list[str] = []

    async def observe(
        self,
        kind: str,
        key_input: Any,
        produce: Callable[[], Awaitable[Any]],
    ) -> Any:
        payload = self.reader.next(kind, rec.effect_key(kind, key_input))
        _, decode = rec.CODECS[kind]
        return decode(payload)

    def write_trace(self, path: Path, text: str) -> None:
        """No-op: a replay must not rewrite the recorded run's artifacts."""

    def append_trace(self, path: Path, text: str) -> None:
        """No-op: a replay must not rewrite the recorded run's artifacts."""

    def end_step(self, node: str) -> None:
        self.steps.append(node)


def llm_key(
    messages: list[Any],
    tools: list[Any],
    tool_choice: dict[str, Any] | None,
    *,
    system: Any = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """The key input for one model call: the whole request configuration.

    The messages are built entirely from earlier recorded results, so if
    anything upstream differed, this key differs too. The rest is what
    the harness sends alongside them — the system prompt, the model, the
    token budget and the full tool schemas. Hashing only tool *names*
    let a harness with a different ``SYSTEM_PROMPT``, ``MODEL`` or
    ``MAX_TOKENS`` replay another configuration's tape as verified.
    """
    return {
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "system": system,
        "model": model,
        "max_tokens": max_tokens,
    }


def llm_config(harness: Any) -> dict[str, Any]:
    """The request configuration a harness sends with every model call."""
    return {
        "system": getattr(harness, "SYSTEM_PROMPT", None),
        "model": getattr(harness, "MODEL", None),
        "max_tokens": getattr(harness, "MAX_TOKENS", None),
    }


def instrument_harness_for_effects(harness: Any, effects: Effects) -> None:
    """Route a harness' ``_call_llm`` through the effects boundary.

    Wraps the *instance*, not the class, and wraps whatever the candidate
    overrode — so a candidate that reimplements ``_call_llm`` with
    caching or reordering is recorded and replayed at the same boundary
    as the baseline.

    Compose with ``metrics.instrument_harness``: usage recording wraps
    the response this returns, so a replayed run reports the recorded
    token counts.

    The key is a deep copy of the request taken before the call, and the
    wrapped override receives its own deep copy: whatever it does to its
    arguments can neither change the key nor reach the graph's state.
    """
    original: Callable[..., Awaitable[Any]] = harness._call_llm

    async def _effectful_call_llm(
        messages: list[Any],
        tools: list[Any],
        *,
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        key_input = copy.deepcopy(
            llm_key(messages, tools, tool_choice, **llm_config(harness))
        )
        return await effects.observe(
            rec.KIND_LLM,
            key_input,
            lambda: original(
                copy.deepcopy(messages),
                copy.deepcopy(tools),
                tool_choice=copy.deepcopy(tool_choice),
            ),
        )

    harness._call_llm = _effectful_call_llm  # type: ignore[method-assign]


__all__ = [
    "Effects",
    "LiveEffects",
    "RecordingEffects",
    "ReplayEffects",
    "instrument_harness_for_effects",
    "llm_config",
    "llm_key",
]
