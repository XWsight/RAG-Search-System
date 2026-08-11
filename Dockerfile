# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS wheels

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements.txt ./

# Resolve dependencies once, then install the resulting wheel set without
# network access in the runtime stage.
RUN python -m pip install --no-cache-dir --upgrade pip==25.1.1 && \
    python -m pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/data \
    XDG_CACHE_HOME=/data/cache \
    HF_HOME=/data/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

RUN groupadd --gid "${APP_GID}" app && \
    useradd --uid "${APP_UID}" --gid "${APP_GID}" \
      --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements.txt ./
COPY --from=wheels /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels \
      -r requirements.txt && \
    rm -rf /wheels

COPY --chown=app:app rag_system ./rag_system
COPY --chown=app:app api_app.py ./api_app.py

# An empty named volume mounted at /data inherits this ownership on first use.
RUN mkdir -p /data/cache/huggingface && chown -R app:app /app /data

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).read()"]

STOPSIGNAL SIGTERM

ENTRYPOINT ["uvicorn", "api_app:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "30", "--no-access-log"]
