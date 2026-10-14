# todo

A small task tracker.

- `models.py` defines the `Task` record.
- `storage.py` converts tasks to and from the JSON export format.
- `service.py` holds `TodoService`, the API the application calls.

## Feature request: task priorities

Every task gets a priority: one of `"low"`, `"normal"` or `"high"`.

**Model.** `Task` takes `priority` as a keyword argument and stores it in a
`priority` attribute. It defaults to `"normal"`. Any other value,
including a differently-capitalised one such as `"High"`, raises
`ValueError` when the task is created.

**Service.**

- `TodoService.add(title, tags=(), priority="normal")` creates a task with
  the given priority. An add that is rejected leaves the service exactly as
  it was: no task is stored and no id is used up.
- `TodoService.set_priority(task_id, priority)` changes a task's priority,
  with the same validation. An unknown id raises `KeyError`.
- `TodoService.pending(order="id")` keeps its current behaviour by default.
  `pending(order="priority")` returns the pending tasks highest priority
  first (`high`, then `normal`, then `low`) and by id within the same
  priority. Any other `order` raises `ValueError`.

**Export format.** Exports move to format version 2, which records every
task's priority under the key `"priority"`.

- `storage.loads` still accepts version 1 exports, which were written before
  priorities existed; every task in them gets priority `"normal"`.
- A version 2 task whose priority is missing or invalid raises `ValueError`.
- Any version other than 1 or 2 raises `ValueError`, as today.

A service restored with `TodoService.from_export` must be indistinguishable
from the one that was exported: same tasks, priorities, completion state,
tags, and the same next id.
