"""Checkpoint history REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.api.runs import get_run_dir, get_run_graph, resolve_run_thread
from app.meta_harness.branches import get_checkpoint_state, get_state_history


router = APIRouter(tags=["checkpoints"])

_THREAD_QUERY = Query(
    default=None,
    description=(
        "Thread to read: the run's root thread (the default) or one of its "
        "forks. Any other thread is a 404."
    ),
)


@router.get("/runs/{run_id}/checkpoints")
async def list_checkpoints(
    run_id: str,
    request: Request,
    thread_id: str | None = _THREAD_QUERY,
) -> dict[str, Any]:
    get_run_dir(request, run_id)
    thread = resolve_run_thread(run_id, thread_id)
    graph = await get_run_graph(request, run_id)
    history = await get_state_history(graph, thread_id=thread)
    return {"checkpoints": [record.to_dict() for record in history]}


@router.get("/runs/{run_id}/checkpoints/{checkpoint_id}")
async def get_checkpoint(
    run_id: str,
    checkpoint_id: str,
    request: Request,
    thread_id: str | None = _THREAD_QUERY,
) -> dict[str, Any]:
    get_run_dir(request, run_id)
    thread = resolve_run_thread(run_id, thread_id)
    graph = await get_run_graph(request, run_id)
    history = await get_state_history(graph, thread_id=thread)
    record = next((item for item in history if item.checkpoint_id == checkpoint_id), None)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="checkpoint not found",
        )
    try:
        state = await get_checkpoint_state(
            graph,
            thread_id=thread,
            checkpoint_id=checkpoint_id,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from None
    return {
        "checkpoint_id": checkpoint_id,
        "thread_id": thread,
        "state": state,
        "ts": record.ts,
        "node": record.node,
    }
