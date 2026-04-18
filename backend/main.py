from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from typing import List

from backend.api.routes.analyze import router as analyze_router
from backend.api.routes.health import router as health_router
from backend.core.handlers import register_exception_handlers
from agent.rag import (
    ensure_faiss_index_ready,
    preload_retriever_index,
    initialize_retriever_safe_mode,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure structured logging once for the process."""
    if logging.getLogger().handlers:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format='{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    )


def _parse_allowed_origins() -> List[str]:
    """Parse ALLOWED_ORIGINS env var into a normalized list."""
    raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["http://localhost:8501", "http://localhost:3000"]


_configure_logging()
_allowed_origins = _parse_allowed_origins()
_allow_credentials = "*" not in _allowed_origins

app = FastAPI(
    title="Credit Risk Agent API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(
    "Configured CORS",
    extra={
        "allowed_origins": _allowed_origins,
        "allow_credentials": _allow_credentials,
    },
)

app.include_router(health_router)
app.include_router(analyze_router)
register_exception_handlers(app)


@app.on_event("startup")
def initialize_rag_system() -> None:
    """Initialize RAG at startup without hard-failing the application."""
    index_dir = os.environ.get("RAG_INDEX_DIR", "rag/faiss_index")
    docs_dir = os.environ.get("RAG_DOCS_DIR", "rag/rag_docs")
    auto_build = os.environ.get("RAG_AUTO_BUILD", "true").strip().lower() in {"1", "true", "yes"}

    logger.info(
        "Starting RAG initialization (index_dir='%s', docs_dir='%s', auto_build=%s)",
        index_dir,
        docs_dir,
        auto_build,
    )

    try:
        status = ensure_faiss_index_ready(
            index_dir=index_dir,
            docs_dir=docs_dir,
            auto_build_if_missing=auto_build,
        )
        if status.get("ready"):
            preload_retriever_index(index_dir=index_dir)
            logger.info("RAG startup initialization complete: %s", status)
        else:
            reason = str(status.get("reason", "index unavailable"))
            initialize_retriever_safe_mode(index_dir=index_dir, reason=reason)
            logger.warning("RAG startup entered SAFE MODE: %s", status)
    except Exception as exc:
        initialize_retriever_safe_mode(index_dir=index_dir, reason=str(exc))
        logger.exception("RAG startup initialization failed; continuing in SAFE MODE: %s", exc)
