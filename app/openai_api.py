"""OpenAI-compatible adapter: exposes the DRAG pipeline as model "drag".

Lets Open WebUI (or any OpenAI-API client) chat against our retrieval pipeline.
Open WebUI becomes a pure chat UI — it does NOT run its own RAG; docling stays
the source of truth. Reuses query.answer() unchanged.
"""
import json
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .query import answer

router = APIRouter(prefix="/v1")
MODEL_ID = "drag"


def _format_sources(sources) -> str:
    if not sources:
        return ""
    lines = [f"- {s['book']} p.{s['page']}" + (f" — {s['section']}" if s.get("section") else "") for s in sources]
    return "\n\n**Sources:**\n" + "\n".join(lines)


def _chunk(delta: dict, finish: str | None = None) -> str:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "drag"}],
    }


@router.post("/chat/completions")
async def chat_completions(body: dict):
    messages = body.get("messages", [])
    # ponytail: RAG is per-question; use the last user turn as the query, ignore history.
    question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

    async def gen():
        sources = None
        async for kind, payload in answer(question):
            if kind == "token":
                yield f"data: {_chunk({'content': payload})}\n\n"
            elif kind == "sources":
                sources = payload
        yield f"data: {_chunk({'content': _format_sources(sources)})}\n\n"
        yield f"data: {_chunk({}, finish='stop')}\n\n"
        yield "data: [DONE]\n\n"

    if body.get("stream", True):
        return StreamingResponse(gen(), media_type="text/event-stream")

    # non-streaming fallback for clients that probe without stream=true
    parts, sources = [], None
    async for kind, payload in answer(question):
        if kind == "token":
            parts.append(payload)
        elif kind == "sources":
            sources = payload
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "".join(parts) + _format_sources(sources)}, "finish_reason": "stop"}
        ],
    }
