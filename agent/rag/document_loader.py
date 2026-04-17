"""
Document Loader
================

Loads text content from .pdf and .txt files in a directory.
Returns structured documents with source metadata for downstream chunking.

Supported formats:
    - PDF: Extracted page-by-page using PyPDF2
    - TXT: Read as plain text (UTF-8)

Usage:
    >>> from agent.rag.document_loader import load_documents
    >>> docs = load_documents("rag/rag_docs")
    >>> print(f"Loaded {len(docs)} document pages")
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)


def _load_pdf(file_path: str) -> List[dict]:
    """
    Extract text from a PDF file, one entry per page.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    List[dict]
        List of dicts with keys: 'source', 'page', 'text'.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError(
            "PyPDF2 is required for PDF loading. "
            "Install it with: pip install PyPDF2"
        )

    documents = []
    filename = os.path.basename(file_path)

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info("Loading PDF: %s (%d pages)", filename, total_pages)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                documents.append({
                    "source": filename,
                    "page": page_num,
                    "text": text.strip(),
                })
            else:
                logger.debug(
                    "Skipping empty page %d in %s", page_num, filename
                )

    except Exception as e:
        logger.error("Failed to read PDF '%s': %s", filename, e)

    return documents


def _load_txt(file_path: str) -> List[dict]:
    """
    Load a plain text file as a single document entry.

    Parameters
    ----------
    file_path : str
        Path to the .txt file.

    Returns
    -------
    List[dict]
        Single-element list with keys: 'source', 'page', 'text'.
    """
    filename = os.path.basename(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            logger.warning("Empty text file: %s", filename)
            return []

        logger.info("Loading TXT: %s (%d chars)", filename, len(text))
        return [{"source": filename, "page": 1, "text": text}]

    except Exception as e:
        logger.error("Failed to read text file '%s': %s", filename, e)
        return []


# ── Supported file extensions → loader functions ─────────────────────────────

_LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
}


def load_documents(docs_dir: str) -> List[dict]:
    """
    Load all supported documents from a directory.

    Scans the directory for .pdf and .txt files and extracts their text content.
    Unsupported file types are silently skipped.

    Parameters
    ----------
    docs_dir : str
        Path to the directory containing regulation/guideline documents.

    Returns
    -------
    List[dict]
        List of document entries, each containing:
        - source: str   — filename
        - page: int     — page number (1 for txt files)
        - text: str     — extracted text content

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no documents could be loaded from the directory.
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir}. "
            f"Please create it and add .pdf or .txt files."
        )

    all_documents = []
    supported_files = 0

    for filename in sorted(os.listdir(docs_dir)):
        file_path = os.path.join(docs_dir, filename)

        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        loader = _LOADERS.get(ext)

        if loader is None:
            logger.debug("Skipping unsupported file: %s", filename)
            continue

        supported_files += 1
        docs = loader(file_path)
        all_documents.extend(docs)

    if not all_documents:
        raise ValueError(
            f"No text could be extracted from {supported_files} file(s) in '{docs_dir}'. "
            f"Ensure files contain readable text content."
        )

    logger.info(
        "Loaded %d document entries from %d files in '%s'",
        len(all_documents), supported_files, docs_dir,
    )
    return all_documents
