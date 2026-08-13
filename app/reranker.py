"""bge-reranker-v2-m3 via FlagEmbedding, lazy singleton."""
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def get_reranker():
    """Load once, reuse. ~2GB RAM, fp32 on CPU."""
    from FlagEmbedding import FlagReranker

    return FlagReranker(config.RERANKER_MODEL)


def rerank(question: str, passages: list[str], top_k: int) -> list[tuple[str, float]]:
    """Return [(passage, score)] sorted desc, truncated to top_k."""
    if not passages:
        return []
    reranker = get_reranker()
    scores = reranker.compute_score([[question, p] for p in passages], normalize=True)
    # single-pair input returns a bare float, normalize to list
    if isinstance(scores, float):
        scores = [scores]
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
