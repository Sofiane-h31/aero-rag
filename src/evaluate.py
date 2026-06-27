"""A lightweight reliability check for generated answers.

Generation is hard to evaluate automatically. One useful, cheap signal is
*groundedness*: how well the answer is supported by the retrieved context.
We embed the answer and each retrieved chunk and take the maximum cosine
similarity. A low score flags an answer that may not be grounded in the
sources -> a candidate for human review.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_embedder: SentenceTransformer | None = None


def _get_embedder(device: str = "cpu") -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL, device=device)
    return _embedder


def groundedness(answer: str, contexts: list[str],
                 embedder=None, device: str = "cpu") -> float:
    """Return the max cosine similarity between the answer and any context chunk.

    Pass an already-loaded `embedder` (e.g. the pipeline's) to avoid reloading
    the model. If none is given, a module-level one is lazily created.
    """
    if not contexts or not answer.strip():
        return 0.0
    emb = embedder if embedder is not None else _get_embedder(device)
    a = emb.encode([answer], normalize_embeddings=True)[0]
    c = emb.encode(contexts, normalize_embeddings=True)
    return float(np.max(c @ a))