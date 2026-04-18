"""Compatibility entrypoint for ASGI servers.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from backend.main import app
