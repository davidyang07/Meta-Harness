# Evidence artifacts

Machine-written inputs to [`../RESUME_EVIDENCE.md`](../RESUME_EVIDENCE.md).
Every file here is produced by a command, not by hand, and the evidence
document reads them rather than describing them.

| File | Written by | Settles |
|---|---|---|
| `replay-verification.json` | `uv run meta-harness verify-replay <recordings-dir>` | exact recorded-execution replay |
| `version-graph.json` | `uv run meta-harness report version-graph <run-name>` | the checkpoint DAG, branch refs, immutability |
| `wandb-offline.json` | `uv run meta-harness report wandb-check` | the W&B adapter works offline, and without W&B |

A missing file is not a soft pass. `RESUME_EVIDENCE.md` reports the
corresponding claim as `UNSUPPORTED` and names the command that would
produce it.

## What each one has to say to count

**`replay-verification.json`** — `all_verified: true`,
`model_calls_issued: 0`, and at least one replay from a stored
checkpoint. Each recording is replayed twice: once from the start of the
run and once from a mid-run checkpoint, because those are two different
claims ("this tape reproduces the run" and "any stored checkpoint is a
valid entry point"). A report containing only whole-run replays supports
the first and not the second, and the evidence document says so rather
than passing on the easier half. Every replay reports its five checks:
`no_divergence`, `node_sequence_identical`,
`per_step_state_hashes_identical`, `final_state_byte_identical`,
`tape_fully_consumed`.

`recorded_models` says what produced the model turns on the tape, and it
is reproduced in the evidence document's row. The committed artifact
currently reads `scripted-offline`: it was produced by
`scripts/record_replay_evidence.py`, which runs the real inner graph, the
real six tools, the real pytest verify and real Postgres checkpoints, and
scripts only the model's side of the conversation — because this
environment has no provider credentials. That demonstrates the replay
machinery, which is what the claim is about. It is not a stand-in for the
pass-rate measurement, which stays `UNSUPPORTED` until a provider run
happens. `meta-harness resume-experiment --record-trials N` overwrites
this artifact with recordings of real provider calls.

**`version-graph.json`** — `immutable: true`, plus the checkpoint DAG,
the branch registry with each branch's fork point, and each branch's
private working tree. Immutability is re-checked by re-reading every
stored checkpoint and confirming it still hashes to what it hashed to; a
mismatch would invalidate every replay and fork claim, so it is reported
as a failure rather than rounded off.

**`wandb-offline.json`** — `ok: true`. `WANDB_MODE=offline` is forced, so
the probe never touches the network and never needs credentials. A
missing `wandb` package is `ok: true` with a reason: the property being
checked is that the repository runs without it. The adapter's actual
translation into W&B's API is covered by `tests/test_tracking.py` with an
injected fake module.

## Credentials

None of these commands read or write credentials, and none of these files
may contain one. The generators capture no environment variables — the
same rule `experiment.capture_environment` follows, and for the same
reason.

## Regenerating everything

```bash
docker compose -f infra/docker-compose.yml up -d postgres

# replay evidence — with credentials:
uv run meta-harness inner --task task-001-fix-typo --candidate baseline --record
uv run meta-harness verify-replay runs/inner-test
# replay evidence — without credentials (scripted model turns, real everything else):
uv run python scripts/record_replay_evidence.py
uv run meta-harness verify-replay runs/replay-evidence

# version graph — needs no credentials, a mock run is enough
uv run meta-harness loop --proposer mock --mock-bench --budget 3 --fresh --run-name vg
uv run meta-harness fork vg --checkpoint <id>          # twice, from the same checkpoint
uv run meta-harness report version-graph vg

uv run meta-harness report wandb-check
uv run meta-harness report resume-evidence
```

`uv run meta-harness resume-experiment` does all of this as part of the
full measurement pipeline.
