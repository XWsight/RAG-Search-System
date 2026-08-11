"""Run the RAG Studio web application."""

from __future__ import annotations

import os

from rag_system.bootstrap import build_service
from rag_system.ui import create_demo


os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

service, settings = build_service()
demo = create_demo(service, settings)


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        share=False,
        show_error=False,
        max_file_size=settings.max_file_bytes,
        strict_cors=True,
        enable_monitoring=False,
    )
