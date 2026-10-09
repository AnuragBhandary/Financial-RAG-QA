# API image. CPU-only torch: inside Docker on a Mac there is no MPS, so this is for
# deployment-shaped runs; local development uses `uv run finrag serve` on the host GPU.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra models --no-install-project

COPY src ./src
COPY web ./web
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra models

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uv", "run", "--no-sync", "finrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
