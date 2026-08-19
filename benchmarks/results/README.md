# Published benchmark results

Immutable evidence for this project's quantitative results. One directory
per experiment id, committed deliberately (unlike `benchmark-results/`,
which is runner scratch and is gitignored).

Each published directory contains everything needed to independently
recompute the headline number:

```
<experiment-id>/
├── config.json             # the resolved protocol that was run
├── environment.json        # commit SHA, model, task hashes, harness hashes
├── baseline-results.jsonl  # raw per-trial rows, control arm
├── candidate-results.jsonl # raw per-trial rows, evolved arm
├── summary.json            # derived from the two JSONL files
└── REPORT.md               # generated, human-readable
```

To verify a published summary yourself:

```python
from app.meta_harness import experiment as exp
d = "benchmarks/results/<experiment-id>"
summary = exp.summarize(
    baseline_rows=exp.read_rows(f"{d}/baseline-results.jsonl"),
    candidate_rows=exp.read_rows(f"{d}/candidate-results.jsonl"),
    task_ids=[t["task_id"] for t in json.load(open(f"{d}/environment.json"))["tasks"]],
    baseline_label="baseline",
    candidate_label="candidate",
)
```

The recomputed `absolute_percentage_point_delta` must equal the one in
the committed `summary.json`.

Per-trial `traces/` are deliberately not published: they are large and
are not needed to verify the summary.

## Status

No canonical 200-trial experiment has been published yet. See
[`../pass-rate/README.md`](../pass-rate/README.md) for the exact
command that produces one, and `docs/CAPABILITIES.md` for which
capabilities are measured and which are still pending.
