"""FastAPI: /query (SSE), /ingest, /books, DELETE /books/{book}."""
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sse_starlette.sse import EventSourceResponse

from . import config
from .ingest import ingest_pdf
from .query import answer, missing_collection
from .openai_api import router as openai_router

app = FastAPI(title="DRAG")
app.include_router(openai_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query_endpoint(body: dict):
    question = body.get("question")
    if not question:
        raise HTTPException(400, "question required")

    async def gen():
        async for kind, payload in answer(question, body.get("book")):
            yield {"event": kind, "data": json.dumps(payload)}

    return EventSourceResponse(gen())


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    pdf_dir = Path(config.PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / file.filename
    dest.write_bytes(await file.read())
    count = await ingest_pdf(dest)
    return {"book": dest.stem, "chunks": count}


@app.get("/books")
async def books_endpoint():
    client = AsyncQdrantClient(url=config.QDRANT_URL)
    # ponytail: scroll whole collection for distinct books — fine for a few thousand points.
    seen = {}
    offset = None
    try:
        while True:
            res, offset = await client.scroll(
                collection_name=config.COLLECTION, limit=512, offset=offset, with_payload=True
            )
            for p in res:
                b = p.payload.get("book")
                if b and b not in seen:
                    seen[b] = p.payload
            if offset is None:
                break
        return [
            {"book": b, "page": p.get("page"), "section": p.get("section")} for b, p in seen.items()
        ]
    except UnexpectedResponse as e:
        return missing_collection(e, [])
    finally:
        await client.close()


@app.delete("/books/{book}")
async def delete_book(book: str):
    client = AsyncQdrantClient(url=config.QDRANT_URL)
    try:
        await client.delete(
            collection_name=config.COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="book", match=models.MatchValue(value=book))]
                )
            ),
        )
        return {"deleted": book}
    except UnexpectedResponse as e:
        return missing_collection(e, {"deleted": book})
    finally:
        await client.close()
