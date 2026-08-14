"""PDF -> docling -> HybridChunker -> llama-server embed -> Qdrant upsert (dense + BM25)."""
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

import httpx
from qdrant_client import AsyncQdrantClient, models

from . import config


def _chunk_type(text: str) -> str:
    # ponytail: table heuristic — markdown pipe rows. docling's table label is deeper to dig out.
    pipe_lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    return "table" if len(pipe_lines) >= 2 else "text"


def parse_pdf(pdf_path: Path) -> list:
    """Parse PDF via docling, chunk with HybridChunker. Returns docling ChunkItems.

    ponytail: docling parse is ~minutes and deterministic per file — pickle cache keyed on
    path+mtime so ingest/query iteration doesn't re-parse. Delete .parse_cache to force.
    """
    import pickle

    pdf_path = Path(pdf_path)
    cache_dir = Path(".parse_cache")
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"{pdf_path.stem}_{int(pdf_path.stat().st_mtime)}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.chunking import HybridChunker

    # Pin OCR backend to torch: matches the .pth weights baked into the image
    # (Containerfile). docling's default is OcrAutoOptions, which only lands on torch
    # via an onnxruntime/easyocr ImportError cascade — pinning makes build<->runtime
    # alignment explicit, not coincidental. Cache below is keyed on path+mtime, so a
    # backend change also needs .parse_cache cleared.
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    ocr_options=RapidOcrOptions(backend="torch")
                )
            )
        }
    )
    doc = converter.convert(str(pdf_path)).document
    chunker = HybridChunker(
        tokenizer=config.EMBED_TOKENIZER,
        chunk_size=config.CHUNK_TOKENS,
        max_tokens=config.MAX_TOKENS,
    )
    chunks = list(chunker.chunk(dl_doc=doc))
    with open(cache, "wb") as f:
        pickle.dump(chunks, f)
    return chunks


def _payload(chunk, book: str) -> dict:
    meta = getattr(chunk, "meta", None)
    # docling keeps page in meta.doc_items[].prov[].page_no (meta.page_numbers is empty here).
    page = None
    for di in getattr(meta, "doc_items", None) or []:
        for prov in getattr(di, "prov", None) or []:
            if getattr(prov, "page_no", None):
                page = prov.page_no
                break
        if page:
            break
    headings = getattr(meta, "headings", None) or []
    return {
        "content": chunk.text,
        "book": book,
        "page": page,
        "section": " / ".join(headings),
        "chunk_type": _chunk_type(chunk.text),
    }


async def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed via llama-server /v1/embeddings, batched. Disk-cached by content hash —
    ponytail: CPU embed of 500+ chunks is ~10min; caching survives crashes and re-ingests."""
    pfx = config.QUERY_PREFIX if is_query else ""
    cache_dir = Path(".embed_cache")
    cache_dir.mkdir(exist_ok=True)

    def _key(s: str) -> str:
        return hashlib.sha1(s.encode()).hexdigest()

    results: list[list[float] | None] = [None] * len(texts)
    todo: list[str] = []
    todo_idx: list[int] = []
    for i, t in enumerate(texts):
        s = pfx + t
        cf = cache_dir / f"{_key(s)}.json"
        if cf.exists():
            results[i] = json.loads(cf.read_text())
        else:
            todo.append(s)
            todo_idx.append(i)

    if todo:
        # 64-chunk batch on CPU-limited llama-embed takes >5min; 300s ReadTimeout killed the upload mid-ingest
        async with httpx.AsyncClient(base_url=config.LLAMA_EMBED_URL, timeout=1800) as client:
            for j in range(0, len(todo), 64):
                batch = todo[j : j + 64]
                r = await client.post("/v1/embeddings", json={"input": batch})
                r.raise_for_status()
                for k, d in enumerate(r.json()["data"]):
                    vec = d["embedding"]
                    results[todo_idx[j + k]] = vec
                    (cache_dir / f"{_key(batch[k])}.json").write_text(json.dumps(vec))
    return results  # type: ignore[return-value]


async def ensure_collection(client) -> None:
    cols = await client.get_collections()
    if config.COLLECTION in [c.name for c in cols.collections]:
        return
    await client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            # Qdrant computes BM25 server-side from the Document text on upsert.
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


async def ingest_pdf(pdf_path: Path, book: str | None = None) -> int:
    """Parse, chunk, embed, upsert. Returns chunk count."""
    pdf_path = Path(pdf_path)
    book = book or pdf_path.stem
    chunks = parse_pdf(pdf_path)
    if not chunks:
        return 0

    client = AsyncQdrantClient(url=config.QDRANT_URL)
    # ensure_collection first — surface schema bugs before the ~10min embed cost.
    await ensure_collection(client)

    payloads = [_payload(c, book) for c in chunks]
    vectors = await embed_texts([p["content"] for p in payloads])

    points = [
        models.PointStruct(
            id=str(uuid4()),
            vector={
                "dense": vec,
                "bm25": models.Document(text=p["content"], model="Qdrant/bm25"),
            },
            payload=p,
        )
        for p, vec in zip(payloads, vectors)
    ]
    # ponytail: small batches — Qdrant upsert limit per request.
    for i in range(0, len(points), 256):
        await client.upsert(config.COLLECTION, points=points[i : i + 256])
    await client.close()
    return len(points)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m app.ingest <pdf_path> [book_name]")
    n = asyncio.run(ingest_pdf(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None))
    print(f"ingested {n} chunks")
