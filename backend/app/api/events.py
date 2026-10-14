"""SSE endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from app.api.runs import get_run_dir
from app.streaming import channel_for_run, event_registry


router = APIRouter(tags=["events"])


@router.get("/runs/{run_id}/stream")
async def stream_run_events(
    run_id: str,
    request: Request,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    """Multiplexed SSE stream for one run (all its branches).

    ``run_id`` must name an existing run: an arbitrary string would
    otherwise open a channel nobody can ever publish to. When an API
    token is configured this route also accepts ``?token=``, because
    ``EventSource`` cannot send an ``Authorization`` header.
    """
    get_run_dir(request, run_id)

    async def _events():
        async for chunk in event_registry.subscribe(
            channel_for_run(run_id),
            last_event_id=last_event_id,
        ):
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
