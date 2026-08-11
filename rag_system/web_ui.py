"""Same-origin product interface for the production HTTP API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles


_WEB_ROOT = Path(__file__).resolve().parent / "web_ui"
_APP_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "base-uri 'none'; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "img-src 'self' data:; "
        "object-src 'none'; "
        "script-src 'self'; "
        "style-src 'self'"
    ),
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


def mount_web_ui(app: FastAPI) -> None:
    """Mount the packaged browser client without changing the API boundary."""

    index_file = _WEB_ROOT / "index.html"
    assets = _WEB_ROOT / "assets"
    if not index_file.is_file() or not assets.is_dir():
        raise RuntimeError("packaged web interface assets are unavailable")

    @app.get("/", include_in_schema=False)
    def product_root() -> RedirectResponse:
        return RedirectResponse(url="/app", status_code=307)

    @app.get("/app", include_in_schema=False)
    @app.get("/app/", include_in_schema=False)
    def product_app() -> FileResponse:
        return FileResponse(
            index_file,
            media_type="text/html",
            headers=_APP_SECURITY_HEADERS,
        )

    app.mount("/app/assets", StaticFiles(directory=assets), name="web-assets")


__all__ = ["mount_web_ui"]
