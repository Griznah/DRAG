# AGENTS.md

Guidance for AI agents in this repo. Local RAG for RPG PDFs: docling → Qwen3-Embedding → Qdrant (dense + BM25/RRF) → bge-reranker → external LLM. Chat UI via Open WebUI.

## Workflow (mandatory)

**All work on branches. No direct `main`. PR-based.**

- Branch off `main`: `feat/...`, `fix/...`, `chore/...`, `docs/...`.
- Commit, push branch, open PR. Self-merge OK for solo, still PR.
- CI gates every PR? No — CI triggers on **tags** only (`v*.*.*`). So PR safety = run checks locally, review diff yourself. Don't push tags from a branch that didn't build clean.
- Releases = git tag → CI builds/pushes images (see Images).

## What lives here vs not

- **Here:** app code (`app/`), two image recipes (`Containerfile` = drag-app, `Dockerfile.embed` = embed server), local `compose.yaml`, CI.
- **NOT here:** Kubernetes manifests. Those live in [`k8s-apps`](https://github.com/Griznah/k8s-apps) under `div/drag/` (Argo CD). k8s change → that repo, separate PR.

## Stack / data flow

```
open-webui --OpenAI API--> drag-app (FastAPI)
                            ├─ qdrant        dense + server-side BM25, RRF fusion
                            ├─ llama-embed   Qwen3-Embedding-0.6B (gguf baked in image)
                            └─ LLAMA_GEN_URL external 27B box (Qwen3.6, chat endpoint)
```

- `/ingest` (multipart PDF) → docling parse → embed → Qdrant upsert. **Creates the `drag` collection.**
- `/v1/chat/completions`, `/query` → hybrid search → bge-rerank → streamed generation.
- **Query/chat paths are guarded against a missing collection.** `/query` and `/v1/chat/completions` route through `search()`, which catches `UnexpectedResponse(404)` on the not-yet-created `drag` collection and returns `[]` → the chat emits "No relevant context found in the indexed books." (No 500, no broken SSE.) **`/books` and `DELETE /books/{book}` are NOT guarded** — they hit Qdrant (`scroll`/`delete`) directly, so on a fresh stack they raise `UnexpectedResponse` → HTTP 500. Ingest a book first so `ensure_collection()` runs.

## Invariants (don't change without knowing why)

- **drag-app runs non-root, uid 1001.** CI image-test asserts it (`docker run ... id -u` must be 1001). Anything that needs to write into its install dir at runtime **will fail** — see next bullet.
- **Models that download into their package dir must be baked at build, not fetched at runtime.** rapidocr (docling's OCR backend) drops PP-OCRv6 weights into `site-packages/rapidocr/models/` on first init; uid 1001 can't write there → `PermissionError`. Bake via `RUN python -c "from rapidocr import RapidOCR; RapidOCR()"` in `Containerfile` (as root, before `USER 1001`). rapidocr has no cache-relocating env var. Same pattern as the gguf baked into `Dockerfile.embed`.
- **`HF_HOME=/data/hf-cache` (PVC) is the writable model dir at runtime.** docling + bge-reranker weights (~4GB) land there on first ingest, persist after. The PVC mounts over `/data` in k8s; the image chowns `/data` to 1001 at build.
- **`TORCHDYNAMO_DISABLE=1`.** `python:3.12-slim` has no g++; torch eager avoids `build-essential` in the image. Don't remove without re-adding the compiler.
- **All config env-overridable, defaults localhost.** No settings framework, no extra dep. See `app/config.py`. External LLM endpoint (`LLAMA_GEN_URL`) is the one value that must be set per-env, not localhost.
- **`uv sync --frozen --no-dev`.** Lock is law. Bump deps deliberately, commit `uv.lock`.

## Local dev

`compose.yaml` runs qdrant + llama-embed + drag-webui; **app runs from your venv**, not the container (faster iteration). Per the header comment in `compose.yaml`.

```shell
uv sync
podman compose up -d --build    # qdrant + llama-embed + open-webui
uv run uvicorn app.api:app --reload --port 8000
```

Ingest + query:
```shell
curl -F file=@book.pdf http://localhost:8000/ingest
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"drag","stream":true,"messages":[{"role":"user","content":"How do grapples work?"}]}'
```

First ingest slow (~10min) — model downloads + parse. `.parse_cache/` + `.embed_cache/` (gitignored) speed reruns.

## Tests

`tests/test_ingest.py`. No framework ceremony — run with `uv run pytest`. Non-trivial logic (parser, reranker, money/security paths) leaves a check behind; trivial one-liners don't.

## Images / release

Tag-triggered CI (`.github/workflows/docker-publish.yaml`): truffleHog full-history scan → image-test (build + non-root check + smoke import) → build & push `drag-app` + `drag-llama-embed` to GHCR with semver + `latest`.

```shell
git tag v0.1.x && git push --tags
```

Both images are **public** on GHCR (no imagePullSecret, by convention). Keep them secret-free — CI secretscan runs on full history; a leak blocks the build.

## Commands

```shell
uv sync                            # install
uv run pytest                      # tests
uv run uvicorn app.api:app --reload
podman compose up -d --build       # backing containers (qdrant/embed/webui)
_build.sh                          # rebuild drag-app image locally

# images after a code change:
git tag vX.Y.Z && git push --tags  # CI builds + pushes :latest + :vX.Y.Z
```

## Files

| File | What |
|---|---|
| `app/api.py` | FastAPI routes (`/health`, `/ingest`, `/query`, `/v1/chat/completions`, `/books`) |
| `app/config.py` | env-driven config, all defaults |
| `app/ingest.py` | docling parse → chunk → embed → Qdrant; `ensure_collection()` |
| `app/query.py` | hybrid search (RRF) → rerank → streamed gen |
| `app/openai_api.py` | OpenAI-compatible adapter for Open WebUI |
| `app/reranker.py` | bge-reranker-v2-m3 wrapper |
| `Containerfile` | drag-app image (non-root 1001) |
| `Dockerfile.embed` | embed image (gguf baked) |
| `compose.yaml` | local dev stack |
