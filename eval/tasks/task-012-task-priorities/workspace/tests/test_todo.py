import json

import pytest

import storage
from models import Task
from service import TodoService

V1_EXPORT = json.dumps(
    {
        "version": 1,
        "tasks": [
            {"id": 2, "title": "write report", "done": False, "tags": ["work"]},
            {"id": 5, "title": "water plants", "done": True, "tags": []},
        ],
    }
)


def _summary(service):
    return [(t.id, t.title, t.done, t.tags, t.priority) for t in service.all()]


# ── existing behaviour ────────────────────────────────────────────────


def test_add_assigns_sequential_ids():
    service = TodoService()
    first = service.add("first")
    second = service.add("second", tags=["x"])
    assert (first.id, second.id) == (1, 2)
    assert second.tags == ["x"]
    assert service.next_id == 3


def test_blank_titles_are_rejected():
    service = TodoService()
    with pytest.raises(ValueError):
        service.add("   ")
    assert service.all() == []
    assert service.next_id == 1


def test_complete_and_pending():
    service = TodoService()
    for title in ("a", "b", "c"):
        service.add(title)
    service.complete(2)
    assert [t.id for t in service.pending()] == [1, 3]
    assert service.get(2).done is True


def test_unknown_ids_raise_key_error():
    service = TodoService()
    with pytest.raises(KeyError):
        service.get(99)
    with pytest.raises(KeyError):
        service.complete(99)


def test_tags_are_copied():
    service = TodoService()
    tags = ["a"]
    task = service.add("x", tags=tags)
    tags.append("b")
    assert task.tags == ["a"]


# ── priorities on the model ───────────────────────────────────────────


def test_task_priority_defaults_to_normal():
    assert Task(id=1, title="x").priority == "normal"


@pytest.mark.parametrize("priority", ["low", "normal", "high"])
def test_task_accepts_the_three_priorities(priority):
    assert Task(id=1, title="x", priority=priority).priority == priority


@pytest.mark.parametrize("priority", ["urgent", "High", "NORMAL", "", None, 1])
def test_task_rejects_anything_else(priority):
    with pytest.raises(ValueError):
        Task(id=1, title="x", priority=priority)


# ── priorities in the service ─────────────────────────────────────────


def test_add_with_priority():
    service = TodoService()
    assert service.add("x", priority="high").priority == "high"
    assert service.add("y").priority == "normal"
    assert service.add("z", tags=["t"], priority="low").priority == "low"


def test_a_rejected_add_changes_nothing():
    service = TodoService()
    service.add("a")
    with pytest.raises(ValueError):
        service.add("b", priority="urgent")
    assert [t.id for t in service.all()] == [1]
    assert service.next_id == 2
    assert service.add("c").id == 2


def test_set_priority():
    service = TodoService()
    task = service.add("a")
    service.set_priority(task.id, "low")
    assert service.get(task.id).priority == "low"
    with pytest.raises(ValueError):
        service.set_priority(task.id, "urgent")
    assert service.get(task.id).priority == "low"
    with pytest.raises(KeyError):
        service.set_priority(42, "high")


def _mixed_service():
    service = TodoService()
    service.add("l1", priority="low")
    service.add("n1")
    service.add("h1", priority="high")
    service.add("l2", priority="low")
    service.add("h2", priority="high")
    service.add("n2", priority="normal")
    service.complete(5)
    return service


def test_pending_by_priority():
    service = _mixed_service()
    assert [t.id for t in service.pending(order="priority")] == [3, 2, 6, 1, 4]


def test_pending_defaults_to_id_order():
    service = _mixed_service()
    assert [t.id for t in service.pending()] == [1, 2, 3, 4, 6]
    assert [t.id for t in service.pending(order="id")] == [1, 2, 3, 4, 6]


def test_changing_a_priority_reorders_pending():
    service = _mixed_service()
    service.set_priority(4, "high")
    service.set_priority(3, "low")
    assert [t.id for t in service.pending(order="priority")] == [4, 2, 6, 1, 3]


@pytest.mark.parametrize("order", ["title", "", "PRIORITY", None])
def test_pending_rejects_unknown_orders(order):
    with pytest.raises(ValueError):
        _mixed_service().pending(order=order)


# ── the export format ─────────────────────────────────────────────────


def test_export_is_version_2_and_records_priorities():
    service = TodoService()
    service.add("a", priority="high")
    service.add("b")
    payload = json.loads(service.export())
    assert payload["version"] == 2
    assert [row["priority"] for row in payload["tasks"]] == ["high", "normal"]
    assert {"id", "title", "done", "tags", "priority"} <= set(payload["tasks"][0])


def test_round_trip_through_the_service():
    service = TodoService()
    service.add("a", tags=["x", "y"], priority="low")
    service.add("b", priority="high")
    service.add("c")
    service.complete(2)
    restored = TodoService.from_export(service.export())
    assert _summary(restored) == [
        (1, "a", False, ["x", "y"], "low"),
        (2, "b", True, [], "high"),
        (3, "c", False, [], "normal"),
    ]
    assert restored.next_id == service.next_id == 4
    assert [t.id for t in restored.pending(order="priority")] == [3, 1]


def test_round_trip_through_storage():
    tasks = [
        Task(id=1, title="a", priority="high"),
        Task(id=2, title="b", done=True, tags=["t"], priority="low"),
        Task(id=7, title="c"),
    ]
    assert storage.loads(storage.dumps(tasks)) == tasks


def test_version_1_exports_still_load_as_normal_priority():
    service = TodoService.from_export(V1_EXPORT)
    assert _summary(service) == [
        (2, "write report", False, ["work"], "normal"),
        (5, "water plants", True, [], "normal"),
    ]
    assert service.next_id == 6
    assert service.add("new", priority="high").id == 6
    assert [t.id for t in service.pending(order="priority")] == [6, 2]


def test_re_exporting_a_version_1_file_writes_version_2():
    payload = json.loads(TodoService.from_export(V1_EXPORT).export())
    assert payload["version"] == 2
    assert [row["priority"] for row in payload["tasks"]] == ["normal", "normal"]
    assert storage.loads(json.dumps(payload)) == storage.loads(V1_EXPORT)


@pytest.mark.parametrize("version", [0, 3, "2", None])
def test_unknown_versions_are_rejected(version):
    with pytest.raises(ValueError):
        storage.loads(json.dumps({"version": version, "tasks": []}))


def test_a_missing_version_is_rejected():
    with pytest.raises(ValueError):
        storage.loads(json.dumps({"tasks": []}))


def test_version_2_tasks_need_a_valid_priority():
    row = {"id": 1, "title": "a", "done": False, "tags": []}
    with pytest.raises(ValueError):
        storage.loads(json.dumps({"version": 2, "tasks": [row]}))
    with pytest.raises(ValueError):
        storage.loads(json.dumps({"version": 2, "tasks": [{**row, "priority": "urgent"}]}))
    assert storage.loads(json.dumps({"version": 2, "tasks": [{**row, "priority": "low"}]})) == [
        Task(id=1, title="a", priority="low")
    ]
