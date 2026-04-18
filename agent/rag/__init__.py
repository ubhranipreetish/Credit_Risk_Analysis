"""
RAG Pipeline Package
=====================

Full Retrieval-Augmented Generation pipeline:
    Load PDFs/TXT → Chunk → Embed (sentence-transformers) → Store (FAISS) → Retrieve

Usage:
    >>> from agent.rag import build_rag_index, retrieve_docs
    >>> build_rag_index("rag/rag_docs", "rag/faiss_index")
    >>> results = retrieve_docs("high risk borrower loan approval rules")
"""

from agent.rag.retriever import retrieve_docs
from agent.rag.vector_store import (
    build_faiss_index,
    load_faiss_index,
    ensure_faiss_index_ready,
    has_faiss_index,
)
from agent.rag.retriever import preload_retriever_index, initialize_retriever_safe_mode


def build_rag_index(
    docs_dir: str = "rag/rag_docs",
    index_dir: str = "rag/faiss_index",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> int:
    """
    End-to-end RAG index construction pipeline.

    Orchestrates: Load documents → Chunk → Embed → Build FAISS index → Save.

    Parameters
    ----------
    docs_dir : str
        Path to directory containing .pdf and/or .txt regulation documents.
    index_dir : str
        Path where FAISS index and chunk metadata will be persisted.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    int
        Total number of chunks indexed.
    """
    from agent.rag.document_loader import load_documents
    from agent.rag.chunker import chunk_documents

    # Step 1: Load raw documents
    documents = load_documents(docs_dir)

    # Step 2: Chunk into retrieval units
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=chunk_overlap)

    # Step 3 + 4: Embed and store in FAISS
    index = build_faiss_index(chunks, save_dir=index_dir)

    return len(chunks)


__all__ = [
    "build_rag_index",
    "retrieve_docs",
    "load_faiss_index",
    "ensure_faiss_index_ready",
    "has_faiss_index",
    "preload_retriever_index",
    "initialize_retriever_safe_mode",
]
