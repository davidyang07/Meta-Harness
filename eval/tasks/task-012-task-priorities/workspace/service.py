"""The application-facing API for managing tasks."""

from __future__ import annotations

from typing import Iterable

import storage
from models import Task


class TodoService:
    def __init__(self, tasks: Iterable[Task] = ()) -> None:
        self._tasks: dict[int, Task] = {}
        for task in tasks:
            if task.id in self._tasks:
                raise ValueError(f"duplicate task id {task.id}")
            self._tasks[task.id] = task
        self._next_id = max(self._tasks, default=0) + 1

    @property
    def next_id(self) -> int:
        """The id the next added task will get."""
        return self._next_id

    def add(self, title: str, tags: Iterable[str] = ()) -> Task:
        task = Task(id=self._next_id, title=title, tags=list(tags))
        self._tasks[task.id] = task
        self._next_id += 1
        return task

    def get(self, task_id: int) -> Task:
        return self._tasks[task_id]

    def complete(self, task_id: int) -> None:
        self._tasks[task_id].done = True

    def all(self) -> list[Task]:
        """Every task, by id."""
        return [self._tasks[task_id] for task_id in sorted(self._tasks)]

    def pending(self) -> list[Task]:
        """Tasks not yet done, by id."""
        return [task for task in self.all() if not task.done]

    def export(self) -> str:
        return storage.dumps(self.all())

    @classmethod
    def from_export(cls, text: str) -> "TodoService":
        return cls(storage.loads(text))
