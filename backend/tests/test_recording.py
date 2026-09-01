"""The execution tape: keys, ordering, divergence, codecs and integrity.

These tests pin the properties exact replay rests on. If a tape can be
read out of order, or a mismatched request can be served, or a corrupt
tape can be loaded silently, then "exact replay" means nothing — so each
of those is asserted to fail loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.meta_harness import recording as rec  # noqa: E402


# ── keys ──────────────────────────────────────────────────────────────


def test_effect_key_is_stable_and_order_independent():
    a = rec.effect_key(rec.KIND_TOOL, {"tool": "read_file", "input": {"path": "a.py"}})
    b = rec.effect_key(rec.KIND_TOOL, {"input": {"path": "a.py"}, "tool": "read_file"})
    assert a == b


def test_effect_key_changes_with_any_input_change():
    base = rec.effect_key(rec.KIND_TOOL, {"tool": "read_file", "path": "a.py"})
    assert base != rec.effect_key(rec.KIND_TOOL, {"tool": "read_file", "path": "b.py"})
    assert base != rec.effect_key(rec.KIND_LLM, {"tool": "read_file", "path": "a.py"})


def test_effect_key_rejects_an_unknown_kind():
    with pytest.raises(rec.RecordingError, match="unknown effect kind"):
        rec.effect_key("telepathy", {})


def test_every_effect_kind_has_a_codec():
    assert set(rec.CODECS) == set(rec.EFFECT_KINDS)


# ── writer / reader ───────────────────────────────────────────────────


def _writer_with(entries: list[tuple[str, str, object]]) -> rec.TapeWriter:
    writer = rec.TapeWriter()
    for kind, key_input, payload in entries:
        writer.append(kind, rec.effect_key(kind, key_input), payload)
    return writer


def test_reader_serves_entries_in_recorded_order():
    writer = _writer_with(
        [
            (rec.KIND_TOOL, {"n": 1}, {"status": "ok", "content": "one"}),
            (rec.KIND_TOOL, {"n": 2}, {"status": "ok", "content": "two"}),
        ]
    )
    reader = rec.TapeReader(writer.entries)

    assert reader.next(rec.KIND_TOOL, rec.effect_key(rec.KIND_TOOL, {"n": 1}))[
        "content"
    ] == "one"
    assert reader.next(rec.KIND_TOOL, rec.effect_key(rec.KIND_TOOL, {"n": 2}))[
        "content"
    ] == "two"
    assert reader.remaining() == 0
    assert reader.consumed == 2


def test_reader_refuses_a_request_the_tape_does_not_have_here():
    writer = _writer_with([(rec.KIND_TOOL, {"n": 1}, {"content": "one"})])
    reader = rec.TapeReader(writer.entries)

    with pytest.raises(rec.ReplayDivergence, match="tape divergence at position 0"):
        reader.next(rec.KIND_TOOL, rec.effect_key(rec.KIND_TOOL, {"n": 99}))


def test_reader_refuses_a_request_of_the_right_key_but_wrong_kind():
    writer = _writer_with([(rec.KIND_TOOL, {"n": 1}, {"content": "one"})])
    reader = rec.TapeReader(writer.entries)

    with pytest.raises(rec.ReplayDivergence):
        reader.next(rec.KIND_LLM, rec.effect_key(rec.KIND_TOOL, {"n": 1}))


def test_reader_refuses_to_run_past_the_end_of_the_tape():
    reader = rec.TapeReader([])
    with pytest.raises(rec.ReplayDivergence, match="the recording ended"):
        reader.next(rec.KIND_TOOL, rec.effect_key(rec.KIND_TOOL, {"n": 1}))


def test_reader_can_start_partway_through():
    writer = _writer_with(
        [
            (rec.KIND_TOOL, {"n": 1}, {"content": "one"}),
            (rec.KIND_TOOL, {"n": 2}, {"content": "two"}),
        ]
    )
    reader = rec.TapeReader(writer.entries, start=1)
    assert reader.remaining() == 1
    assert reader.next(rec.KIND_TOOL, rec.effect_key(rec.KIND_TOOL, {"n": 2}))[
        "content"
    ] == "two"


# ── step marks position the continuation ──────────────────────────────


def test_step_marks_record_the_tape_offset_at_each_node_boundary():
    writer = rec.TapeWriter()
    writer.append(rec.KIND_ORIENT, "k0", {})
    writer.mark_step("orient")
    writer.append(rec.KIND_LLM, "k1", {})
    writer.mark_step("plan")
    writer.append(rec.KIND_TOOL, "k2", {})
    writer.append(rec.KIND_TOOL, "k3", {})
    writer.mark_step("act")

    assert [(m.index, m.node, m.tape_length) for m in writer.steps] == [
        (0, "orient", 1),
        (1, "plan", 2),
        (2, "act", 4),
    ]


def test_recording_positions_a_reader_after_a_given_step(tmp_path: Path):
    writer = rec.TapeWriter()
    for index in range(4):
        writer.append(rec.KIND_TOOL, rec.effect_key(rec.KIND_TOOL, {"n": index}), {})
        writer.mark_step(f"node-{index}")
    recording = rec.Recording(
        manifest={"recording_id": "r", "thread_id": "t"},
        entries=writer.entries,
        steps=writer.steps,
    )

    assert recording.reader_from_step(None).remaining() == 4
    assert recording.reader_from_step(1).remaining() == 2
    assert recording.continuation_nodes(1) == ["node-2", "node-3"]
    assert recording.continuation_nodes(None) == [
        "node-0",
        "node-1",
        "node-2",
        "node-3",
    ]


def test_reader_from_an_unknown_step_raises():
    recording = rec.Recording(manifest={}, entries=[], steps=[])
    with pytest.raises(rec.RecordingError, match="no step with index"):
        recording.reader_from_step(7)


# ── model-response codec ──────────────────────────────────────────────


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Usage:
    input_tokens = 100
    output_tokens = 25
    cache_creation_input_tokens = 5
    cache_read_input_tokens = 7


class _Message:
    content = [
        _Block(type="text", text="thinking"),
        _Block(type="tool_use", id="tu1", name="read_file", input={"path": "a.py"}),
    ]
    model = "claude-haiku-4-5-20251001"
    usage = _Usage()
    stop_reason = "tool_use"


def test_llm_response_round_trips_through_the_tape():
    payload = rec.encode_llm_response(_Message())
    replayed = rec.decode_llm_response(payload)

    assert replayed.model == "claude-haiku-4-5-20251001"
    assert replayed.stop_reason == "tool_use"
    assert [b.type for b in replayed.content] == ["text", "tool_use"]
    assert replayed.content[0].text == "thinking"
    assert replayed.content[1].name == "read_file"
    assert replayed.content[1].input == {"path": "a.py"}


def test_replayed_usage_carries_the_recorded_token_counts():
    """A replayed trial must report what the recorded trial actually spent."""
    replayed = rec.decode_llm_response(rec.encode_llm_response(_Message()))
    assert replayed.usage.input_tokens == 100
    assert replayed.usage.output_tokens == 25
    assert replayed.usage.cache_creation_input_tokens == 5
    assert replayed.usage.cache_read_input_tokens == 7


def test_a_response_without_usage_records_as_absent_not_as_zero():
    class _NoUsage:
        content = []
        model = "m"
        usage = None
        stop_reason = None

    payload = rec.encode_llm_response(_NoUsage())
    assert payload["usage"] is None
    assert rec.decode_llm_response(payload).usage is None


def test_verify_codec_round_trips():
    encode, decode = rec.CODECS[rec.KIND_VERIFY]
    assert decode(encode((True, "2 passed"))) == (True, "2 passed")
    assert decode(encode((False, "1 failed"))) == (False, "1 failed")


# ── persistence ───────────────────────────────────────────────────────


def _sample_recording(tmp_path: Path) -> Path:
    writer = rec.TapeWriter()
    writer.set_node("orient")
    writer.append(rec.KIND_ORIENT, rec.effect_key(rec.KIND_ORIENT, {}), {"tree": "."})
    writer.mark_step("orient")
    directory = tmp_path / "rec-1"
    manifest = rec.build_manifest(
        recording_id="rec-1",
        thread_id="thread-1",
        task={"id": "task-001-fix-typo"},
        workspace_path="/tmp/ws",
        model="scripted-model",
        harness_class="tests.harness_doubles:FixTypoHarness",
    )
    rec.write_recording(directory, manifest=manifest, writer=writer)
    return directory


def test_recording_round_trips_through_disk(tmp_path: Path):
    directory = _sample_recording(tmp_path)
    loaded = rec.read_recording(directory)

    assert loaded.recording_id == "rec-1"
    assert loaded.thread_id == "thread-1"
    assert loaded.manifest["task_id"] == "task-001-fix-typo"
    assert loaded.manifest["model"] == "scripted-model"
    assert len(loaded.entries) == 1
    assert loaded.entries[0].node == "orient"
    assert [m.node for m in loaded.steps] == ["orient"]


def test_manifest_captures_no_environment_variables(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-appear")
    directory = _sample_recording(tmp_path)
    blob = (directory / rec.MANIFEST_FILENAME).read_text()
    assert "sk-should-never-appear" not in blob
    assert "ANTHROPIC_API_KEY" not in blob


def test_a_tampered_tape_fails_to_load(tmp_path: Path):
    """A tape whose entries no longer hash to the manifest is not evidence."""
    directory = _sample_recording(tmp_path)
    tape = directory / rec.TAPE_FILENAME
    entry = json.loads(tape.read_text().strip())
    entry["payload"] = {"tree": "tampered"}
    tape.write_text(json.dumps(entry) + "\n")

    with pytest.raises(rec.RecordingError, match="corrupt"):
        rec.read_recording(directory)


def test_reading_a_missing_recording_names_the_path(tmp_path: Path):
    with pytest.raises(rec.RecordingError, match="no recording manifest"):
        rec.read_recording(tmp_path / "nope")


def test_list_recordings_finds_only_real_recordings(tmp_path: Path):
    _sample_recording(tmp_path)
    (tmp_path / "not-a-recording").mkdir()
    assert [d.name for d in rec.list_recordings(tmp_path)] == ["rec-1"]


# ── discovery ─────────────────────────────────────────────────────────


def test_discover_recordings_finds_them_at_any_depth(tmp_path: Path):
    """Recordings sit at different depths depending on who wrote them.

    A branch writes ``threads/<thread>/recordings/<id>/``; an experiment
    writes ``recordings/<arm>/<id>/``. A caller pointed at a run, or at a
    result directory, must find both.
    """
    nested = tmp_path / "threads" / "t1" / "recordings" / "task-a-trial-1"
    _sample_recording(nested.parent)
    (nested.parent / "rec-1").rename(nested)

    found = rec.discover_recordings(tmp_path)

    assert [d.name for d in found] == ["task-a-trial-1"]


def test_discover_recordings_prefers_direct_children(tmp_path: Path):
    _sample_recording(tmp_path)
    deep = tmp_path / "deeper" / "nested"
    _sample_recording(deep)

    assert [d.name for d in rec.discover_recordings(tmp_path)] == ["rec-1"]


def test_discover_recordings_on_an_empty_tree_returns_nothing(tmp_path: Path):
    (tmp_path / "not-a-recording").mkdir()
    assert rec.discover_recordings(tmp_path) == []
    assert rec.discover_recordings(tmp_path / "missing") == []
