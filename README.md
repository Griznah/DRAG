# DRAG

Local RAG for RPG PDFs. docling parses → Qwen3-Embedding → Qdrant (dense + BM25/RRF) → bge-reranker-v2-m3 → external LLM. Chat UI via Open WebUI.

## Stack

```
Browser → open-webui (:3000) ──OpenAI API──► drag-app (:8000)
                                               │  /v1/chat/completions  /query  /ingest  /books
                                               ├─► qdrant (:6333)        dense + server-side BM25
                                               ├─► llama-embed (:8081)   Qwen3-Embedding-0.6B (gguf baked in)
                                               └─► <LLM_GEN_URL>         Qwen3.6-27B (external box)
```

| Service | Image | Port |
|---|---|---|
| open-webui | `ghcr.io/open-webui/open-webui:main` | 3000 |
| drag-app | `ghcr.io/griznah/drag-app` (non-root, uid 1001) | 8000 |
| llama-embed | `ghcr.io/griznah/drag-llama-embed` (gguf baked) | 8081 |
| qdrant | `docker.io/qdrant/qdrant:v1.19.0` | 6333 |

## Run

```bash
uv sync
podman compose up -d --build      # builds drag-app + drag-llama-embed
```

First ingest/query is slow — `drag-app` downloads docling + bge-reranker models (~4GB) into the `hf-cache` volume; persists after. The embed gguf is already in its image.

Ingest a book, then ask:

```bash
curl -F file=@book.pdf http://localhost:8000/ingest
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"drag","stream":true,"messages":[{"role":"user","content":"How do grapples work?"}]}'
```

UI: http://localhost:3000 (first signup = admin), pick model **drag**.

## Config (`app/config.py`, all env-overridable)

| Var | Default | What |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant |
| `LLAMA_EMBED_URL` | `http://localhost:8081` | embedding server (in-cluster llama-embed, or the GPU box — below) |
| `LLAMA_GEN_URL` | `http://localhost:8080` | **external 27B box** |
| `QDRANT_COLLECTION` | `drag` | collection name |
| `CHUNK_TOKENS` / `MAX_TOKENS` | 512 / 1024 | HybridChunker size / cap |
| `SEARCH_LIMIT` / `FINAL_TOPK` | 15 / 8 | hybrid candidates → reranked top-k |
| `PDF_DIR` | `/app/pdf` | where uploads land |

## API

| Method | Path | |
|---|---|---|
| GET | `/health` | liveness |
| POST | `/v1/chat/completions` | OpenAI-compatible (for Open WebUI) |
| POST | `/query` | raw SSE (`token`/`sources` events) |
| POST | `/ingest` | multipart PDF upload → docling → embed → store |
| GET | `/books` / DELETE `/books/{book}` | list / remove |

## Images / release

Tag-triggered CI (`.github/workflows/docker-publish.yaml`): truffleHog → image-test (build + non-root check) → build & push both images to GHCR with semver + `latest` tags.

```bash
git tag v0.1.3 && git push --tags
```

## Kubernetes

Manifests live in **[k8s-apps](https://github.com/Griznah/k8s-apps)** under `div/drag/` (Argo CD ApplicationSet). Not in this repo.

## Embeddings on the GPU box

Cluster CPU too slow for the embed model? Run it on the external 27B box instead — no app change, it's pure `LLAMA_EMBED_URL`. llama.cpp serves one model per process, so it's a second `llama-server` next to the 27B (0.6B Q8 ≈ 0.7GB VRAM):

```bash
# on the gen box (it already has the llama.cpp binary)
llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --embedding -ngl 99 --port 8081 --host 0.0.0.0
# or containerized: build the GPU variant and run it
podman build -f Dockerfile.embed --build-arg CUDA=cu124 -t drag-llama-embed:gpu
podman run --gpus all -e N_GPU_LAYERS=99 -p 8081:8080 drag-llama-embed:gpu
```

Same `/v1/embeddings` API either way. Then in k8s-apps (`div/drag/`): set `LLAMA_EMBED_URL=http://<gen-box>:8081` and delete the in-cluster llama-embed deployment. Ingest-time CPU embed (~10min/book) drops to seconds; `.embed_cache` in drag-app still saves re-embeds across restarts.

## License

MIT
