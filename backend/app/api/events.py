"""SSE endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse

from app.api.runs import get_run_dir
from app.auth import STREAM_TICKET_TTL_S, configured_api_token, mint_stream_ticket
from app.streaming import channel_for_run, event_registry


router = APIRouter(tags=["events"])


@router.post("/runs/{run_id}/stream-ticket")
async def create_stream_ticket(run_id: str, request: Request) -> dict[str, Any]:
    """Trade the bearer token for a ticket that opens this run's stream.

    ``EventSource`` cannot send an ``Authorization`` header, so the
    stream is authorised by its URL, and a URL is logged. The ticket is
    what goes in it: one run, ``expires_in`` seconds. With no token
    configured there is nothing to trade, and ``ticket`` is ``None``.
    """
    get_run_dir(request, run_id)
    token = configured_api_token()
    if token is None:
        return {"ticket": None, "expires_in": None}
    return {
        "ticket": mint_stream_ticket(token, run_id),
        "expires_in": STREAM_TICKET_TTL_S,
    }


@router.get("/runs/{run_id}/stream")
async def stream_run_events(
    run_id: str,
    request: Request,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    resume_after: Annotated[str | None, Query(alias="last_event_id")] = None,
) -> StreamingResponse:
    """Multiplexed SSE stream for one run (all its branches).

    ``run_id`` must name an existing run: an arbitrary string would
    otherwise open a channel nobody can ever publish to. When an API
    token is configured this route also accepts ``?ticket=`` (see
    ``create_stream_ticket``). ``?last_event_id=`` stands in for the
    ``Last-Event-ID`` header when a client reconnects with a new
    ``EventSource``, which cannot set it.
    """
    get_run_dir(request, run_id)

    async def _events():
        async for chunk in event_registry.subscribe(
            channel_for_run(run_id),
            last_event_id=last_event_id or resume_after,
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
