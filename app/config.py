"""Env-driven config. Module constants — no settings framework, no extra dep."""
import os

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "drag")

# --- llama-server: embeddings ---
LLAMA_EMBED_URL = os.getenv("LLAMA_EMBED_URL", "http://localhost:8081")

# --- llama-server: generation (external box) ---
LLAMA_GEN_URL = os.getenv("LLAMA_GEN_URL", "http://localhost:8080")

# --- embeddings model ---
# Qwen3-Embedding-0.6B -> 1024 dims. GGUF lives in llama-embed container.
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
# HF id used only for the HybridChunker tokenizer (token counting), aligned to embed model.
EMBED_TOKENIZER = os.getenv("EMBED_TOKENIZER", "Qwen/Qwen3-Embedding-0.6B")
# Qwen3-Embedding wants a "query: " prefix on queries, plain passages.
QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "query: ")

# --- chunking ---
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "512"))
# Qwen tokenizer doesn't expose model_max_length — HybridChunker needs it explicitly.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# --- hybrid search / rerank / generation ---
PREFETCH_LIMIT = int(os.getenv("PREFETCH_LIMIT", "20"))
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "15"))
FINAL_TOPK = int(os.getenv("FINAL_TOPK", "8"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "1024"))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.3"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are an RPG rules assistant. Answer using ONLY the provided context from the "
    "rule books. If the answer is not in the context, say you don't know. "
    "Cite book and page numbers when relevant.",
)

PDF_DIR = os.getenv("PDF_DIR", "/app/pdf")
