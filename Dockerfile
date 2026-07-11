# CPU repro image (environment + router + OPE; NO training — that is GPU-only, Phase 3).
# Phase 0 ships a minimal, buildable stub; the CPU demo entrypoint lands with Phase 2.
FROM python:3.11-slim

# Install uv (single static binary) from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (better layer caching), then the project.
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Default: show the CLI is alive. Real CPU demo (docker run) is wired in Phase 2/5.
CMD ["uv", "run", "python", "-c", "import specialist_router; print(specialist_router.__version__)"]
