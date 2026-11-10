# syntax=docker/dockerfile:1

FROM python:3.11-slim

# libgomp1: LightGBM's OpenMP runtime dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependency layer first, cached independently of source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Source and the minimal runtime data the API needs.
COPY src/ src/
COPY data/processed/feature_spec.json data/processed/feature_spec.json
COPY reports/model_registry.json reports/model_registry.json
COPY reports/production_approvals.json reports/production_approvals.json
COPY params.yaml params.yaml
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

RUN groupadd --system app && useradd --system --gid app --no-create-home app \
    && chown -R app:app /app
USER app

ENV MODEL_ALIAS=production \
    PORT=8000 \
    PATH="/app/.venv/bin:$PATH" \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health').status == 200 else 1)"

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
