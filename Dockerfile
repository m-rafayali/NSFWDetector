# Runtime image: onnxruntime only (no PyTorch). Model weights are mounted read-only at /models.
FROM python:3.12.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv==0.12.5
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project && pip uninstall -y uv
COPY ctc_mod ctc_mod
COPY config.yaml .
RUN useradd --system --uid 10001 ctc && mkdir -p /data && chown ctc /data
USER ctc
ENV PATH=/app/.venv/bin:$PATH CTC_DB=/data/ctc.db CTC_KEYS_FILE=/run/keys.yaml CTC_MODELS_DIR=/models
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')"
# ONE worker process on purpose: in-memory rate limits/cache; concurrency via CTC_CONCURRENCY threads.
CMD ["uvicorn", "ctc_mod.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--timeout-keep-alive", "5"]
