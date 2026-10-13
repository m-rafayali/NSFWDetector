# Runtime image: onnxruntime only (no PyTorch). Model weights are mounted read-only at /models.
# Build/run with Podman: `podman-compose up -d --build` (see README).
FROM docker.io/library/python:3.12.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv==0.12.5
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project && pip uninstall -y uv
COPY ctc_mod ctc_mod
COPY config.yaml .
RUN useradd --system --uid 10001 ctc && mkdir -p /data && chown ctc /data
USER ctc
ENV PATH=/app/.venv/bin:$PATH CTC_DB=/data/ctc.db CTC_MODELS_DIR=/models
EXPOSE 8000
# ONE worker process on purpose: in-memory rate limits/cache; concurrency via CTC_CONCURRENCY threads.
# (Healthcheck lives in compose.yaml: OCI images built by Podman drop HEALTHCHECK.)
CMD ["uvicorn", "ctc_mod.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--timeout-keep-alive", "5"]
