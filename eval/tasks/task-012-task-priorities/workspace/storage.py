"""The JSON export format.

An export is a JSON object ``{"version": N, "tasks": [...]}`` with one
object per task.
"""

from __future__ import annotations

import json
from typing import Iterable

from models import Task

FORMAT_VERSION = 1


def _task_to_dict(task: Task) -> dict:
    return {
        "id": task.id,
        "title": task.title,
        "done": task.done,
        "tags": list(task.tags),
    }


def _task_from_dict(row: dict) -> Task:
    return Task(
        id=row["id"],
        title=row["title"],
        done=bool(row.get("done", False)),
        tags=list(row.get("tags", [])),
    )


def dumps(tasks: Iterable[Task]) -> str:
    """Serialise ``tasks`` in the current export format."""
    payload = {
        "version": FORMAT_VERSION,
        "tasks": [_task_to_dict(task) for task in tasks],
    }
    return json.dumps(payload, sort_keys=True)


def loads(text: str) -> list[Task]:
    """Parse an export produced by ``dumps``."""
    payload = json.loads(text)
    version = payload.get("version")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported export version: {version!r}")
    return [_task_from_dict(row) for row in payload["tasks"]]
