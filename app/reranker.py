"""bge-reranker-v2-m3 cross-encoder via transformers (transformers 5.x compatible).

FlagEmbedding's wrapper calls tokenizer.prepare_for_model, removed in transformers 5.0,
and FlagEmbedding 1.4.0 never fixed it — so score the model directly. FlagReranker with
normalize=True == sigmoid of the cross-encoder logit, which we replicate here.
"""
from functools import lru_cache

import torch

from . import config


@lru_cache(maxsize=1)
def _model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.RERANKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.RERANKER_MODEL)
    model.eval()
    return tok, model


def rerank(question: str, passages: list[str], top_k: int) -> list[tuple[str, float]]:
    """Return [(passage, relevance)] sorted desc, truncated to top_k. Scores are
    sigmoid(logit) in [0,1], matching FlagReranker(normalize=True)."""
    if not passages:
        return []
    tok, model = _model()
    with torch.no_grad():
        inputs = tok(
            [[question, p] for p in passages],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        logits = model(**inputs).logits.squeeze(-1)
        scores = torch.sigmoid(logits).tolist()
    if isinstance(scores, float):  # single-passage edge case
        scores = [scores]
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
