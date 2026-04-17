"""
Sentence-Transformer Embedder
===============================

Encodes text into dense vector embeddings using the `all-MiniLM-L6-v2` model
from the sentence-transformers library.

Model details:
    - Output dimension: 384
    - Max sequence length: 256 tokens
    - Fast, lightweight, and suitable for semantic search

Usage:
    >>> from agent.rag.embedder import embed_texts, get_embedding_model
    >>> embeddings = embed_texts(["example text 1", "example text 2"])
    >>> print(embeddings.shape)  # (2, 384)
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Module-level cache ───────────────────────────────────────────────────────

_model_cache = None


def get_embedding_model():
    """
    Load and cache the sentence-transformer model.

    Returns the model singleton — loaded only on first call.

    Returns
    -------
    SentenceTransformer
        The loaded embedding model.
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embeddings. "
            "Install with: pip install sentence-transformers"
        )

    logger.info("Loading embedding model: %s", MODEL_NAME)
    _model_cache = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model loaded successfully (dim=%d)", EMBEDDING_DIM)
    return _model_cache


def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a list of text strings into dense vector embeddings.

    Parameters
    ----------
    texts : List[str]
        List of text strings to embed.
    batch_size : int
        Batch size for encoding (default 64).
    show_progress : bool
        Whether to show a progress bar during encoding.
    normalize : bool
        If True, L2-normalize embeddings (required for cosine similarity
        with FAISS IndexFlatIP).

    Returns
    -------
    np.ndarray
        Float32 array of shape (len(texts), 384).

    Raises
    ------
    ValueError
        If texts is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts")

    model = get_embedding_model()

    logger.info("Embedding %d texts (batch_size=%d)", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    # Ensure float32 for FAISS compatibility
    embeddings = embeddings.astype(np.float32)

    # L2 normalize for cosine similarity via inner product
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms

    logger.info("Embedding complete: shape=%s", embeddings.shape)
    return embeddings
