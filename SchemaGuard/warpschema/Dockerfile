# ============================================================
# WarpSchema — Multi-stage Production Build
# Copyright (c) 2026 Tarek Clarke. All Rights Reserved.
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps in an isolated layer
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the model so it's baked into the image (no runtime download)
RUN python -c "\
from transformers import AutoModel, AutoTokenizer; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
AutoModel.from_pretrained('distilbert-base-uncased')"

# --- Stage 2: Production ---
FROM python:3.12-slim AS production

LABEL maintainer="Tarek Clarke"
LABEL description="WarpSchema — Self-Healing Data Proxy"

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local
# Copy cached model from builder
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application source
COPY app.py engine.py config.py ./

# Non-root user for security
RUN useradd --create-home schemaguard
USER schemaguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
