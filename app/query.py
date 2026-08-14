"""Hybrid search (dense + BM25/RRF) -> bge rerank -> Qwen3.6-27B streamed answer."""
import json

import httpx
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from . import config, ingest
from .reranker import rerank


async def _client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=config.QDRANT_URL)


def missing_collection(e: UnexpectedResponse, fallback):
    """Qdrant 404 = the collection doesn't exist yet (fresh stack, nothing ingested):
    return fallback. Any other status re-raises. Single home for this semantic —
    used by search() and the /books endpoints."""
    if e.status_code == 404:
        return fallback
    raise e


async def search(question: str, book: str | None = None, limit: int = config.SEARCH_LIMIT):
    """Hybrid search, RRF fusion. Returns Qdrant ScoredPoint list."""
    qvec = (await ingest.embed_texts([question], is_query=True))[0]
    flt = (
        models.Filter(must=[models.FieldCondition(key="book", match=models.MatchValue(value=book))])
        if book
        else None
    )
    client = await _client()
    try:
        res = await client.query_points(
            collection_name=config.COLLECTION,
            prefetch=[
                models.Prefetch(query=qvec, using="dense", limit=config.PREFETCH_LIMIT),
                models.Prefetch(
                    query=models.Document(text=question, model="Qdrant/bm25"),
                    using="bm25",
                    limit=config.PREFETCH_LIMIT,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return res.points
    except UnexpectedResponse as e:
        return missing_collection(e, [])
    finally:
        await client.close()


def _context_block(points) -> str:
    return "\n\n".join(
        f"[{p.payload['book']} p.{p.payload.get('page')} - {p.payload.get('section','')}]\n{p.payload['content']}"
        for p in points
    )


async def generate(question: str, points):
    """Stream tokens from llama-server /v1/chat/completions (OpenAI SSE).

    Uses the chat endpoint (not raw /completion) so the model applies its chat
    template and enable_thinking=False keeps chain-of-thought out of the answer —
    Qwen3.6-27B is a reasoning model that otherwise dumps its thinking visibly.
    """
    body = {
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{_context_block(points)}\n\nQuestion: {question}"},
        ],
        "stream": True,
        "max_tokens": config.GEN_MAX_TOKENS,
        "temperature": config.GEN_TEMPERATURE,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(base_url=config.LLAMA_GEN_URL, timeout=None) as client:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta.get("content"):
                    yield delta["content"]


async def answer(question: str, book: str | None = None):
    """Full pipeline. Yields ('token', str) then ('sources', list) at the end."""
    points = await search(question, book)
    if not points:
        yield ("token", "No relevant context found in the indexed books.")
        return

    ranked = rerank(
        question,
        [p.payload["content"] for p in points],
        top_k=config.FINAL_TOPK,
    )
    top_contents = {c for c, _ in ranked}
    top_points = [p for p in points if p.payload["content"] in top_contents][: config.FINAL_TOPK]

    async for tok in generate(question, top_points):
        yield ("token", tok)

    yield (
        "sources",
        [
            {"book": p.payload["book"], "page": p.payload.get("page"), "section": p.payload.get("section")}
            for p in top_points
        ],
    )
