"""Branch metadata and trajectory REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.api.runs import get_run_dir
from app.meta_harness.branches import (
    BranchOwnedElsewhere,
    cancel_branch_outcome,
    list_branches,
    reconstruct_trajectory,
)
from app.streaming import emit_run_event


router = APIRouter(tags=["branches"])


@router.get("/runs/{run_id}/branches")
async def list_run_branches(run_id: str, request: Request) -> dict[str, Any]:
    get_run_dir(request, run_id)
    return {
        "branches": [branch.to_dict() for branch in list_branches(run_id=run_id)]
    }


@router.get("/runs/{run_id}/trajectory")
async def get_run_trajectory(run_id: str, request: Request) -> dict[str, Any]:
    get_run_dir(request, run_id)
    return {"trajectory": reconstruct_trajectory(run_id)}


@router.post("/runs/{run_id}/branches/{thread_id}/cancel")
async def cancel_run_branch(
    run_id: str,
    thread_id: str,
    request: Request,
) -> dict[str, str]:
    """Stop one branch of ``run_id``.

    The branch must belong to the run in the URL: a fork of another run
    is a 404 here, and nothing is emitted on this run's stream for it.
    ``branch-cancelled`` is emitted only when the branch's status actually
    changed — cancelling a branch that already finished is a no-op, not a
    cancellation. A branch another live process drives is a 409.
    """
    get_run_dir(request, run_id)
    try:
        metadata, changed = await cancel_branch_outcome(thread_id, run_id=run_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="branch not found",
        ) from None
    except BranchOwnedElsewhere as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from None

    if changed:
        emit_run_event(
            run_id,
            "branch-cancelled",
            {
                "thread_id": metadata.thread_id,
                "reason": "requested",
            },
        )
    return {"status": metadata.status}
