"""ASGI entry point for the durable RAG Studio API."""

from __future__ import annotations

import logging

from rag_system.api import create_app
from rag_system.bootstrap import build_production_runtime


logging.basicConfig(level=logging.INFO, format="%(message)s")

runtime = build_production_runtime()
app = create_app(
    platform=runtime.platform,
    authenticator=runtime.authenticator,
    rate_limiter=runtime.rate_limiter,
    logger=runtime.event_logger,
    readiness=runtime.ready,
    shutdown=runtime.close,
)
