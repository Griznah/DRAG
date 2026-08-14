# ponytail: single stage — all deps are runtime, multi-stage buys nothing here
FROM python:3.12-slim
# opencv (docling tableformer) runtime libs absent from -slim; install at base so
# uv sync layer above still caches on pyproject/lock, not on apt.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# ponytail: pin uv by digest — :latest = non-reproducible build. Bump deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.12.3@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/usr/local
# ponytail: python:3.12-slim has no g++; torch dynamo/inductor JIT-compile needs one.
# Disable dynamo (run eager) — avoids apt/build-essential in the image, docling parses fine.
ENV TORCHDYNAMO_DISABLE=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
# rapidocr (docling OCR backend) downloads model weights into its package dir on
# first init — bake at build as root so non-root uid 1001 runtime reads them
# instead of PermissionError on site-packages. No env var relocates its cache.
# ENGINE MUST BE TORCH: rapidocr's own default is onnxruntime (not installed here);
# docling's auto-OCR tries onnxruntime->ImportError, easyocr->ImportError, then torch
# (installed) -> selects it. So runtime uses the torch engine and downloads .pth
# files. A bare RapidOCR() here would pick onnxruntime and fail the build; baking onnx
# models would be ignored at runtime and leave the PermissionError unfixed. Force
# torch to match runtime exactly.
RUN python -c "from rapidocr import RapidOCR; from rapidocr.utils.typings import EngineType; RapidOCR(params={'Det.engine_type':EngineType.TORCH,'Cls.engine_type':EngineType.TORCH,'Rec.engine_type':EngineType.TORCH}); print('rapidocr torch models cached')"
COPY app/ /app/app/
# Non-root: chown the writable cwd (parse/embed caches land under /app) and /data
# (HF_HOME + PDF dir; PVCs mount over these in K8s) and drop privileges.
RUN useradd -u 1001 -r -M drag \
    && mkdir -p /data/hf-cache /data/pdf \
    && chown -R 1001:1001 /app /data
USER 1001
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
