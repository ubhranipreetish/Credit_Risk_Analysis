"""
FAISS Vector Store
===================

Builds and manages a FAISS index for fast similarity search over
document chunk embeddings. Uses IndexFlatIP (inner product) with
L2-normalized vectors to achieve cosine similarity.

Persistence:
    - FAISS index saved as `index.faiss`
    - Chunk metadata saved as `chunks_metadata.json`

Usage:
    >>> from agent.rag.vector_store import build_faiss_index, load_faiss_index
    >>> index = build_faiss_index(chunks, save_dir="rag/faiss_index")
    >>> index, metadata = load_faiss_index("rag/faiss_index")
"""

import os
import json
import logging
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── File names for persistence ───────────────────────────────────────────────

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "chunks_metadata.json"


def build_faiss_index(
    chunks: List[dict],
    save_dir: str = "rag/faiss_index",
) -> "faiss.Index":
    """
    Build a FAISS index from document chunks and persist to disk.

    Steps:
        1. Extract text from chunks
        2. Generate embeddings using sentence-transformers
        3. Build FAISS IndexFlatIP index
        4. Save index + metadata to disk

    Parameters
    ----------
    chunks : List[dict]
        List of chunk dicts from `chunker.chunk_documents()`.
        Each must have key 'text'; metadata keys are preserved.
    save_dir : str
        Directory to save the FAISS index and metadata.

    Returns
    -------
    faiss.Index
        The built FAISS index.

    Raises
    ------
    ValueError
        If chunks is empty.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is required for vector storage. "
            "Install with: pip install faiss-cpu"
        )

    from agent.rag.embedder import embed_texts, EMBEDDING_DIM

    if not chunks:
        raise ValueError("Cannot build index from empty chunk list")

    # ── Step 1: Extract texts ────────────────────────────────────────────
    texts = [chunk["text"] for chunk in chunks]
    logger.info("Building FAISS index for %d chunks", len(texts))

    # ── Step 2: Generate embeddings ──────────────────────────────────────
    embeddings = embed_texts(texts, show_progress=True)

    # ── Step 3: Build FAISS index ────────────────────────────────────────
    # IndexFlatIP = exact inner-product search
    # With L2-normalized vectors, inner product = cosine similarity
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, EMBEDDING_DIM)

    # ── Step 4: Persist to disk ──────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)

    index_path = os.path.join(save_dir, INDEX_FILENAME)
    faiss.write_index(index, index_path)
    logger.info("FAISS index saved to: %s", index_path)

    # Save chunk metadata (everything except 'text' goes into a compact format,
    # but we keep 'text' for retrieval display)
    metadata_path = os.path.join(save_dir, METADATA_FILENAME)
    metadata = []
    for chunk in chunks:
        metadata.append({
            "text": chunk["text"],
            "source": chunk.get("source", "unknown"),
            "page": chunk.get("page", 0),
            "chunk_id": chunk.get("chunk_id", 0),
        })

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Chunk metadata saved to: %s (%d entries)", metadata_path, len(metadata))

    return index


def load_faiss_index(
    save_dir: str = "rag/faiss_index",
) -> Tuple["faiss.Index", List[dict]]:
    """
    Load a persisted FAISS index and its chunk metadata from disk.

    Parameters
    ----------
    save_dir : str
        Directory containing `index.faiss` and `chunks_metadata.json`.

    Returns
    -------
    Tuple[faiss.Index, List[dict]]
        The FAISS index and corresponding chunk metadata list.

    Raises
    ------
    FileNotFoundError
        If the index or metadata files are missing.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is required. Install with: pip install faiss-cpu"
        )

    index_path = os.path.join(save_dir, INDEX_FILENAME)
    metadata_path = os.path.join(save_dir, METADATA_FILENAME)

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            f"Run build_rag_index() first to create the index."
        )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Chunk metadata not found at '{metadata_path}'. "
            f"Run build_rag_index() first."
        )

    index = faiss.read_index(index_path)
    logger.info("Loaded FAISS index: %d vectors", index.ntotal)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info("Loaded %d chunk metadata entries", len(metadata))

    return index, metadata
