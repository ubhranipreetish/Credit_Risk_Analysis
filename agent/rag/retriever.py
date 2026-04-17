"""
RAG Retriever
==============

Retrieves the most relevant document chunks for a given query using
FAISS similarity search. This is the primary interface for the RAG pipeline.

Usage:
    >>> from agent.rag.retriever import retrieve_docs
    >>> results = retrieve_docs("high risk borrower loan approval rules", top_k=5)
    >>> for r in results:
    ...     print(f"[{r['score']:.3f}] {r['source']} p.{r['page']}: {r['text'][:80]}...")
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Module-level cache for loaded index ──────────────────────────────────────

_cached_index = None
_cached_metadata = None
_cached_index_dir = None


def _ensure_index_loaded(index_dir: str = "rag/faiss_index"):
    """Load and cache the FAISS index + metadata (only on first call or dir change)."""
    global _cached_index, _cached_metadata, _cached_index_dir

    if _cached_index is not None and _cached_index_dir == index_dir:
        return _cached_index, _cached_metadata

    from agent.rag.vector_store import load_faiss_index

    _cached_index, _cached_metadata = load_faiss_index(index_dir)
    _cached_index_dir = index_dir
    return _cached_index, _cached_metadata


def retrieve_docs(
    query: str,
    top_k: int = 5,
    index_dir: str = "rag/faiss_index",
    score_threshold: Optional[float] = None,
) -> List[dict]:
    """
    Retrieve the most relevant document chunks for a query.

    Embeds the query using the same sentence-transformer model, then
    performs FAISS similarity search to find the closest chunks.

    Parameters
    ----------
    query : str
        Natural language query (e.g., "high risk borrower loan approval rules").
    top_k : int
        Maximum number of results to return (default 5).
    index_dir : str
        Path to the directory containing the persisted FAISS index.
    score_threshold : float, optional
        Minimum similarity score (0-1 range for normalized vectors).
        Results below this threshold are excluded.

    Returns
    -------
    List[dict]
        Ordered list of matching chunks (best match first), each containing:
        - text: str       — chunk text content
        - source: str     — originating document filename
        - page: int       — originating page number
        - score: float    — similarity score (higher = more relevant)
        - chunk_id: int   — global chunk identifier

    Raises
    ------
    FileNotFoundError
        If the FAISS index has not been built yet.
    ValueError
        If query is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    # Load index (cached after first call)
    index, metadata = _ensure_index_loaded(index_dir)

    # Embed the query using the same model
    from agent.rag.embedder import embed_texts

    query_embedding = embed_texts([query.strip()])

    # Clamp top_k to available vectors
    effective_k = min(top_k, index.ntotal)
    if effective_k == 0:
        logger.warning("FAISS index is empty — no results to return")
        return []

    # Search
    scores, indices = index.search(query_embedding, effective_k)

    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 for missing results

        if score_threshold is not None and score < score_threshold:
            continue

        chunk_meta = metadata[idx]
        results.append({
            "text": chunk_meta["text"],
            "source": chunk_meta.get("source", "unknown"),
            "page": chunk_meta.get("page", 0),
            "score": round(float(score), 4),
            "chunk_id": chunk_meta.get("chunk_id", idx),
        })

    logger.info(
        "Retrieved %d chunks for query: '%s' (top score: %.4f)",
        len(results),
        query[:50],
        results[0]["score"] if results else 0.0,
    )
    return results
