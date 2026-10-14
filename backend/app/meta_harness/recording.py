"""The execution tape: every nondeterministic input an inner-loop run consumed.

An inner-loop run is a deterministic state machine driving a
nondeterministic world. The state machine is ``inner.build_inner_graph``;
the world is model responses, subprocess output and the filesystem. If
every crossing of that boundary is recorded in order, the run can be
re-executed later against the tape instead of the world — and then
"replay" means the graph really runs, the routing decisions are really
recomputed, and the only thing supplied from storage is what the world
said.

That is the difference between this module and ``replay.replay_events``,
which reads persisted checkpoints back without running anything.

**What a tape holds**

- ``entries`` — one row per boundary crossing, in the order the run made
  them: model responses, tool inputs/outputs, workspace scans, verify
  subprocess results, the grade, final-file snapshots.
- ``steps`` — one row per completed graph node, carrying the tape offset
  at that moment. This is what makes replay *from a checkpoint* possible:
  the checkpoint identifies a step, the step identifies where in the tape
  the continuation begins.
- ``manifest`` — the configuration and provenance the run happened under:
  model id, the harness class and the path and SHA-256 of the source file
  that defines it, task hash, thread id, commit.

**Keys, and why divergence is detectable**

Every entry is keyed by a SHA-256 over its *inputs* — for an LLM call
the messages, the full tool schemas, the tool choice, the system prompt,
the model and ``max_tokens``; for a tool call its name and arguments. During
replay the reader demands that the key the graph asks for equals the key
recorded at that position. Since each request's inputs are built from the
results of earlier requests, a single divergence anywhere upstream
changes every key downstream and the replay fails loudly instead of
quietly producing a different run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Bump when the on-disk tape schema changes incompatibly. 2.0.0: the
#: grade became its own effect, LLM keys gained the request configuration,
#: and non-text content blocks are stored as their wire shape.
RECORDING_SCHEMA_VERSION = "2.0.0"

MANIFEST_FILENAME = "manifest.json"
TAPE_FILENAME = "tape.jsonl"
STEPS_FILENAME = "steps.jsonl"

#: Effect kinds a tape may contain. A kind not listed here is a bug, not
#: a forward-compatible extension — replay must never silently pass
#: through an effect it does not understand.
KIND_LLM = "llm"
KIND_TOOL = "tool"
KIND_ORIENT = "orient"
KIND_VERIFY = "verify"
KIND_FILES = "files"
KIND_GRADE = "grade"

EFFECT_KINDS = frozenset(
    {KIND_LLM, KIND_TOOL, KIND_ORIENT, KIND_VERIFY, KIND_FILES, KIND_GRADE}
)


class RecordingError(RuntimeError):
    """Base class for tape read/write failures."""


class ReplayDivergence(RecordingError):
    """The graph asked for an effect the tape does not have at this point.

    Raised — never swallowed, never downgraded to a warning — because a
    replay that silently continues past a divergence is not a replay.
    """


def canonical_json(value: Any) -> str:
    """Deterministic JSON encoding used for every hash in this module."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_of(value: Any) -> str:
    """SHA-256 over the canonical JSON encoding of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def effect_key(kind: str, key_input: Any) -> str:
    """The content key for one boundary crossing.

    Derived from the *inputs* only. Two runs that reach the same point
    with the same request produce the same key; a run that diverges
    produces a different one, and it stays different for every
    subsequent request because later inputs embed earlier results.
    """
    if kind not in EFFECT_KINDS:
        raise RecordingError(f"unknown effect kind: {kind!r}")
    return sha256_of({"kind": kind, "input": key_input})


# ── model-response codec ──────────────────────────────────────────────


@dataclass(frozen=True)
class ReplayedUsage:
    """The subset of an Anthropic ``usage`` block the metrics layer reads."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(frozen=True)
class ReplayedBlock:
    """One content block, shaped like the SDK block the inner loop reads.

    ``fields`` is the block's full wire shape as recorded. Attributes the
    dataclass does not declare (``thinking``, ``signature``, ``data`` …)
    resolve from it, and ``model_dump`` returns it, so a replayed block
    serialises to exactly the dict its live original did.
    """

    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        fields = self.__dict__.get("fields") or {}
        if name in fields:
            return fields[name]
        raise AttributeError(name)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(self.fields)


@dataclass(frozen=True)
class ReplayedMessage:
    """A recorded model response, replayed in place of a live call.

    Duck-types the fields ``inner.py`` and ``metrics.py`` actually touch:
    ``content`` blocks with ``type``/``text``/``id``/``name``/``input``,
    plus ``model``, ``usage`` and ``stop_reason``.
    """

    content: list[ReplayedBlock]
    model: str | None = None
    usage: ReplayedUsage | None = None
    stop_reason: str | None = None


def _json_safe(value: Any) -> Any:
    """``value`` round-tripped through JSON: plain types, key order kept.

    Not ``canonical_json``: sorting keys would reorder a tool input such
    as the plan, which the act prompt then renders in a different order
    on replay than it did live.
    """
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def serialize_content_block(block: Any) -> dict[str, Any]:
    """One response content block in the shape the Messages API accepts back.

    Used both to write the assistant turn into the conversation and to
    record the block on the tape, so the live and the replayed run hold
    the same dict. ``str(block)`` — the old fallback for anything that
    was not text or a tool call — was neither: a ``thinking`` block came
    back as ``{"type": "thinking", "raw": "ThinkingBlock(...)"}``, which
    the API rejects, and its text differed between an SDK object and a
    replayed one.
    """
    btype = str(getattr(block, "type", None))
    if btype == "text":
        return {"type": "text", "text": getattr(block, "text", "") or ""}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", None),
            "name": getattr(block, "name", None),
            "input": _json_safe(dict(getattr(block, "input", None) or {})),
        }
    if btype == "thinking":
        return {
            "type": "thinking",
            "thinking": str(getattr(block, "thinking", "") or ""),
            "signature": str(getattr(block, "signature", "") or ""),
        }
    if btype == "redacted_thinking":
        return {
            "type": "redacted_thinking",
            "data": str(getattr(block, "data", "") or ""),
        }
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump(mode="json", exclude_none=True)
        except TypeError:
            dumped = dump()
        if isinstance(dumped, dict):
            return _json_safe({**dumped, "type": btype})
    public = {
        key: value
        for key, value in (getattr(block, "__dict__", None) or {}).items()
        if not key.startswith("_") and value is not None
    }
    return _json_safe({**public, "type": btype})


def encode_llm_response(response: Any) -> dict[str, Any]:
    """Serialise a model response into the tape's plain-JSON shape."""
    blocks = [
        serialize_content_block(block)
        for block in getattr(response, "content", None) or []
    ]

    usage = getattr(response, "usage", None)
    usage_payload: dict[str, Any] | None = None
    if usage is not None:
        usage_payload = {
            name: _int_field(usage, name)
            for name in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        }
    return {
        "content": blocks,
        "model": getattr(response, "model", None),
        "usage": usage_payload,
        "stop_reason": getattr(response, "stop_reason", None),
    }


def _int_field(usage: Any, name: str) -> int:
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def decode_llm_response(payload: dict[str, Any]) -> ReplayedMessage:
    """Rebuild a replayable model response from its tape payload."""
    blocks = [
        ReplayedBlock(
            type=str(b.get("type")),
            text=b.get("text"),
            id=b.get("id"),
            name=b.get("name"),
            input=dict(b.get("input") or {}),
            fields=dict(b),
        )
        for b in payload.get("content") or []
    ]
    usage_payload = payload.get("usage")
    usage = (
        ReplayedUsage(
            input_tokens=int(usage_payload.get("input_tokens") or 0),
            output_tokens=int(usage_payload.get("output_tokens") or 0),
            cache_creation_input_tokens=int(
                usage_payload.get("cache_creation_input_tokens") or 0
            ),
            cache_read_input_tokens=int(
                usage_payload.get("cache_read_input_tokens") or 0
            ),
        )
        if usage_payload is not None
        else None
    )
    return ReplayedMessage(
        content=blocks,
        model=payload.get("model"),
        usage=usage,
        stop_reason=payload.get("stop_reason"),
    )


def _identity(value: Any) -> Any:
    return value


def encode_verify(value: tuple[bool, str]) -> list[Any]:
    """``(tests_pass, output)`` → a JSON pair."""
    passed, output = value
    return [bool(passed), str(output)]


def decode_verify(payload: Any) -> tuple[bool, str]:
    passed, output = payload
    return bool(passed), str(output)


#: kind → (encode, decode). Every kind in ``EFFECT_KINDS`` must appear.
CODECS: dict[str, tuple[Any, Any]] = {
    KIND_LLM: (encode_llm_response, decode_llm_response),
    KIND_TOOL: (_identity, _identity),
    KIND_ORIENT: (_identity, _identity),
    KIND_VERIFY: (encode_verify, decode_verify),
    KIND_FILES: (_identity, _identity),
    KIND_GRADE: (_identity, _identity),
}

assert set(CODECS) == set(EFFECT_KINDS), "every effect kind needs a codec"


# ── tape ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TapeEntry:
    """One recorded boundary crossing."""

    seq: int
    kind: str
    key: str
    node: str | None
    payload: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "key": self.key,
            "node": self.node,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TapeEntry":
        return cls(
            seq=int(data["seq"]),
            kind=str(data["kind"]),
            key=str(data["key"]),
            node=data.get("node"),
            payload=data.get("payload"),
        )


@dataclass(frozen=True)
class StepMark:
    """A completed graph node, and where the tape stood when it finished.

    ``index`` counts node completions from 0 in execution order, which is
    also LangGraph's super-step order for the same thread. That is the
    join key between a tape and a checkpoint history.

    ``checkpoint_id`` and ``state_hash`` are filled in after the run, when
    the checkpoint history is read back — they are what turns "replay this
    tape" into "replay from *this checkpoint*".
    """

    index: int
    node: str
    tape_length: int
    checkpoint_id: str | None = None
    state_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "node": self.node,
            "tape_length": self.tape_length,
            "checkpoint_id": self.checkpoint_id,
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepMark":
        return cls(
            index=int(data["index"]),
            node=str(data["node"]),
            tape_length=int(data["tape_length"]),
            checkpoint_id=data.get("checkpoint_id"),
            state_hash=data.get("state_hash"),
        )


class TapeWriter:
    """Accumulates a run's effects in execution order."""

    def __init__(self) -> None:
        self.entries: list[TapeEntry] = []
        self.steps: list[StepMark] = []
        self._current_node: str | None = None

    def set_node(self, node: str | None) -> None:
        """Tag subsequent entries with the node that produced them."""
        self._current_node = node

    def append(self, kind: str, key: str, payload: Any) -> TapeEntry:
        entry = TapeEntry(
            seq=len(self.entries),
            kind=kind,
            key=key,
            node=self._current_node,
            payload=payload,
        )
        self.entries.append(entry)
        return entry

    def mark_step(self, node: str) -> StepMark:
        mark = StepMark(
            index=len(self.steps), node=node, tape_length=len(self.entries)
        )
        self.steps.append(mark)
        return mark

    def tape_hash(self) -> str:
        """A digest over the whole tape, for provenance."""
        return sha256_of([e.to_dict() for e in self.entries])


class TapeReader:
    """Serves a recorded tape back in order, refusing any divergence."""

    def __init__(self, entries: list[TapeEntry], *, start: int = 0) -> None:
        self._entries = list(entries)
        self._position = start
        self._start = start
        self._consumed = 0

    @property
    def position(self) -> int:
        return self._position

    @property
    def consumed(self) -> int:
        return self._consumed

    def remaining(self) -> int:
        return len(self._entries) - self._position

    def next(self, kind: str, key: str) -> Any:
        """Return the payload of the next entry, or raise ``ReplayDivergence``."""
        if self._position >= len(self._entries):
            raise ReplayDivergence(
                f"the replayed graph requested a {kind!r} effect at tape position "
                f"{self._position}, but the recording ended at "
                f"{len(self._entries)} entries. The replayed execution ran "
                f"longer than the recorded one."
            )
        entry = self._entries[self._position]
        if entry.kind != kind or entry.key != key:
            raise ReplayDivergence(
                f"tape divergence at position {self._position}: recorded "
                f"{entry.kind}/{entry.key[:12]} but the replayed graph asked for "
                f"{kind}/{key[:12]}. The replayed execution took a different "
                f"path than the recorded one."
            )
        self._position += 1
        self._consumed += 1
        return entry.payload


@dataclass
class Recording:
    """A tape plus the provenance of the execution that produced it."""

    manifest: dict[str, Any]
    entries: list[TapeEntry]
    steps: list[StepMark]

    @property
    def recording_id(self) -> str:
        return str(self.manifest.get("recording_id"))

    @property
    def thread_id(self) -> str:
        return str(self.manifest.get("thread_id"))

    def tape_hash(self) -> str:
        return sha256_of([e.to_dict() for e in self.entries])

    def step_for_index(self, index: int) -> StepMark | None:
        for mark in self.steps:
            if mark.index == index:
                return mark
        return None

    def reader_from_step(self, index: int | None) -> TapeReader:
        """A reader positioned just after step ``index`` completed.

        ``None`` means "from the beginning" — replay of the whole run.
        """
        if index is None:
            return TapeReader(self.entries, start=0)
        mark = self.step_for_index(index)
        if mark is None:
            raise RecordingError(
                f"recording {self.recording_id} has no step with index {index}"
            )
        return TapeReader(self.entries, start=mark.tape_length)

    def continuation_nodes(self, index: int | None) -> list[str]:
        """Node names the recorded run executed after step ``index``."""
        if index is None:
            return [m.node for m in self.steps]
        return [m.node for m in self.steps if m.index > index]


def build_manifest(
    *,
    recording_id: str,
    thread_id: str,
    task: dict[str, Any],
    workspace_path: str,
    model: str | None,
    harness_class: str,
    candidate_source_sha256: str | None = None,
    task_sha256: str | None = None,
    git_commit: str | None = None,
    harness_source_path: str | None = None,
    harness_source_sha256: str | None = None,
    candidate_source_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance block stored beside a tape.

    ``harness_source_path`` is repo-relative POSIX when the file is inside
    the repository. ``candidate_source_path`` is set only for a harness
    loaded from a source snapshot, whose synthetic module name cannot be
    imported again in another process.

    No environment variables are captured, for the same reason
    ``experiment.capture_environment`` captures none: that is how an API
    key ends up in a committed artifact.
    """
    manifest = {
        "schema_version": RECORDING_SCHEMA_VERSION,
        "recording_id": recording_id,
        "thread_id": thread_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task.get("id"),
        "task_sha256": task_sha256,
        "workspace_path": workspace_path,
        "model": model,
        "harness_class": harness_class,
        "harness_source_path": harness_source_path,
        "harness_source_sha256": harness_source_sha256,
        "candidate_source_path": candidate_source_path,
        "candidate_source_sha256": candidate_source_sha256,
        "git_commit": git_commit,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_recording(
    directory: Path, *, manifest: dict[str, Any], writer: TapeWriter
) -> Path:
    """Persist a tape to ``directory``. Returns the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    full_manifest = {
        **manifest,
        "entry_count": len(writer.entries),
        "step_count": len(writer.steps),
        "tape_sha256": writer.tape_hash(),
    }
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(full_manifest, indent=2, default=str), encoding="utf-8"
    )
    with (directory / TAPE_FILENAME).open("w", encoding="utf-8") as fh:
        for entry in writer.entries:
            fh.write(json.dumps(entry.to_dict(), default=str) + "\n")
    with (directory / STEPS_FILENAME).open("w", encoding="utf-8") as fh:
        for mark in writer.steps:
            fh.write(json.dumps(mark.to_dict()) + "\n")
    return directory


def read_recording(directory: Path) -> Recording:
    """Load a tape written by :func:`write_recording`."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RecordingError(f"no recording manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [
        TapeEntry.from_dict(json.loads(line))
        for line in _lines(directory / TAPE_FILENAME)
    ]
    steps = [
        StepMark.from_dict(json.loads(line))
        for line in _lines(directory / STEPS_FILENAME)
    ]

    recorded_hash = manifest.get("tape_sha256")
    actual_hash = sha256_of([e.to_dict() for e in entries])
    if recorded_hash and recorded_hash != actual_hash:
        raise RecordingError(
            f"recording at {directory} is corrupt: manifest declares tape "
            f"{recorded_hash[:12]} but the entries hash to {actual_hash[:12]}"
        )
    return Recording(manifest=manifest, entries=entries, steps=steps)


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_recordings(root: Path) -> list[Path]:
    """Recording directories directly under ``root``, sorted by id."""
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if (d / MANIFEST_FILENAME).is_file())


def discover_recordings(root: Path) -> list[Path]:
    """Every recording under ``root``, at any depth, sorted by path.

    Recordings live at different depths depending on who wrote them — a
    branch's own ``threads/<thread>/recordings/<id>/``, or an
    experiment's ``recordings/<arm>/<id>/`` — so a caller pointed at a
    run or a result directory needs the recursive search, not just the
    immediate children.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    direct = list_recordings(root)
    if direct:
        return direct
    return sorted(
        manifest.parent
        for manifest in root.rglob(MANIFEST_FILENAME)
        if (manifest.parent / TAPE_FILENAME).is_file()
    )


__all__ = [
    "CODECS",
    "EFFECT_KINDS",
    "KIND_FILES",
    "KIND_GRADE",
    "KIND_LLM",
    "KIND_ORIENT",
    "KIND_TOOL",
    "KIND_VERIFY",
    "RECORDING_SCHEMA_VERSION",
    "Recording",
    "RecordingError",
    "ReplayDivergence",
    "ReplayedBlock",
    "ReplayedMessage",
    "ReplayedUsage",
    "StepMark",
    "TapeEntry",
    "TapeReader",
    "TapeWriter",
    "build_manifest",
    "canonical_json",
    "discover_recordings",
    "effect_key",
    "list_recordings",
    "read_recording",
    "serialize_content_block",
    "sha256_of",
    "write_recording",
]
