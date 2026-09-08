# Hugging Face Spaces (Docker SDK) image for Veritas.
# One container serves BOTH the REST API and the web UI (FastAPI static mount).
# HF Spaces free tier: 2 vCPU / 16 GB RAM — comfortably fits PyTorch + the
# sentence-transformers embedding model. HF Spaces expect the app on port 7860.

FROM python:3.12-slim

# Non-root user (HF Spaces convention: uid 1000, writable home).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    VERITAS_DATA_DIR=/home/user/app/data

WORKDIR /home/user/app

# Install CPU-only PyTorch first (much smaller/faster than the default CUDA wheel),
# then the project. Doing torch first lets the project resolve against it.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user . .

RUN pip install --no-cache-dir -e .

# Pre-download the embedding model into the image so first request is fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860

CMD ["uvicorn", "veritas.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
