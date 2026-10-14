"""Time-travel fork REST endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.runs import get_run_dir, get_run_graph, resolve_run_thread
from app.meta_harness.branches import worktree_add
from app.meta_harness.state import MetaHarnessState
from app.streaming import emit_run_event


router = APIRouter(tags=["forks"])

#: State keys a fork may not change. ``run_id`` is the run's identity:
#: overriding it would send the branch's events to another run's stream.
_IMMUTABLE_KEYS = frozenset({"run_id"})


class ForkRequest(BaseModel):
    parent_checkpoint_id: str
    mods: dict[str, Any] = Field(default_factory=dict)
    parent_thread_id: str | None = None
    name: str | None = None


def _mods_summary(mods: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": sorted(mods),
        "count": len(mods),
    }


def _validate_mods(mods: dict[str, Any]) -> None:
    """Reject mods that are not outer-loop state keys, or that are identity."""
    allowed = set(MetaHarnessState.__annotations__) - _IMMUTABLE_KEYS
    bad = sorted(set(mods) - allowed)
    if bad:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"mods may only set outer-loop state keys {sorted(allowed)}; got {bad}",
        )


@router.post("/runs/{run_id}/fork", status_code=status.HTTP_202_ACCEPTED)
async def fork_run(
    run_id: str,
    payload: ForkRequest,
    request: Request,
) -> dict[str, Any]:
    """Fork a checkpoint of ``run_id`` into a new concurrent branch.

    ``parent_thread_id`` must be the run's root thread (the default) or
    one of its forks; naming another run's thread is a 404, rather than a
    branch whose parent later cannot be found.
    """
    get_run_dir(request, run_id)
    parent_thread_id = resolve_run_thread(run_id, payload.parent_thread_id)
    _validate_mods(payload.mods)
    graph = await get_run_graph(request, run_id)
    try:
        metadata, _task = await worktree_add(
            graph,
            run_id=run_id,
            parent_thread_id=parent_thread_id,
            parent_checkpoint_id=payload.parent_checkpoint_id,
            mods=payload.mods,
            name=payload.name,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from None

    emit_run_event(
        run_id,
        "fork-created",
        {
            "thread_id": metadata.thread_id,
            "parent_thread_id": metadata.parent_thread_id,
            "parent_checkpoint_id": metadata.parent_checkpoint_id,
            "mods_summary": _mods_summary(metadata.mods),
        },
    )
    return {
        "thread_id": metadata.thread_id,
        "status": metadata.status,
        "parent_checkpoint_id": metadata.parent_checkpoint_id,
        "branch_id": metadata.branch_id,
    }
