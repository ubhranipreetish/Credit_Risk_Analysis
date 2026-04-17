"""
Text Chunker
==============

Splits documents into overlapping chunks for embedding and retrieval.
Uses character-based splitting with configurable chunk size and overlap.

Overlap ensures context continuity across chunk boundaries, which improves
retrieval quality for queries that span logical sections.

Usage:
    >>> from agent.rag.chunker import chunk_documents
    >>> chunks = chunk_documents(documents, chunk_size=500, overlap=100)
    >>> print(f"Created {len(chunks)} chunks")
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split a text string into overlapping chunks.

    Parameters
    ----------
    text : str
        The input text to split.
    chunk_size : int
        Maximum number of characters per chunk.
    overlap : int
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    List[str]
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # If not the last chunk, try to break at a sentence/paragraph boundary
        if end < text_length:
            # Look for a good break point (newline, period, or space)
            # within the last 20% of the chunk
            search_start = end - int(chunk_size * 0.2)
            best_break = -1

            # Prefer paragraph breaks
            newline_pos = text.rfind("\n\n", search_start, end)
            if newline_pos != -1:
                best_break = newline_pos + 2

            # Then sentence breaks
            if best_break == -1:
                for sep in [". ", ".\n", "? ", "! "]:
                    pos = text.rfind(sep, search_start, end)
                    if pos != -1:
                        best_break = pos + len(sep)
                        break

            # Then any whitespace
            if best_break == -1:
                space_pos = text.rfind(" ", search_start, end)
                if space_pos != -1:
                    best_break = space_pos + 1

            if best_break != -1:
                end = best_break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Move forward by (end - overlap), but ensure we make progress
        start = max(start + 1, end - overlap)

    return chunks


def chunk_documents(
    documents: List[dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[dict]:
    """
    Split a list of documents into overlapping chunks with metadata.

    Each chunk preserves its source document information (filename, page number)
    for citation in the final report.

    Parameters
    ----------
    documents : List[dict]
        List of document entries from `document_loader.load_documents()`.
        Each dict must have keys: 'source', 'page', 'text'.
    chunk_size : int
        Maximum characters per chunk (default 500).
    overlap : int
        Overlapping characters between consecutive chunks (default 100).

    Returns
    -------
    List[dict]
        List of chunk dicts, each containing:
        - text: str         — chunk text content
        - source: str       — originating filename
        - page: int         — originating page number
        - chunk_id: int     — global chunk index (0-based)

    Raises
    ------
    ValueError
        If chunk_size <= 0 or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        text = doc.get("text", "")
        if not text or not text.strip():
            continue

        text_chunks = _split_text(text, chunk_size, overlap)

        for chunk_text in text_chunks:
            all_chunks.append({
                "text": chunk_text,
                "source": doc.get("source", "unknown"),
                "page": doc.get("page", 0),
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents), len(all_chunks), chunk_size, overlap,
    )
    return all_chunks
