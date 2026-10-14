"""The task record."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    id: int
    title: str
    done: bool = False
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or self.id < 1:
            raise ValueError(f"task id must be a positive int, got {self.id!r}")
        if not self.title.strip():
            raise ValueError("task title must not be blank")
