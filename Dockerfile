# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch CPU-only (separate layer → Docker cache hit on rebuilds) ──────────
# CPU tier is what HF free Spaces runs on.
# Full float32 precision — no resolution or quality compromise.
RUN pip install --no-cache-dir \
        torch==2.2.2 \
        torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

# ── App deps ──────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        fastapi==0.111.0 \
        "uvicorn[standard]==0.29.0" \
        pillow==10.3.0 \
        numpy==1.26.4 \
        python-multipart==0.0.9 \
        "huggingface_hub>=0.23.0"

# ── App code ──────────────────────────────────────────────────────────────────
COPY api_server.py .

# ── HF Spaces requires port 7860 ──────────────────────────────────────────────
EXPOSE 7860

# Long timeout for first-request inference on CPU
CMD ["uvicorn", "api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--timeout-keep-alive", "300", \
     "--workers", "1"]
