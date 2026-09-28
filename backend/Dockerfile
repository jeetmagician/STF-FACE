# CPU image. For CUDA see backend/Dockerfile.gpu.
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libGL and libglib are required by OpenCV even in headless builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Optional InsightFace backend. Off by default: its pretrained weights are
# licensed for non-commercial research use only.
ARG INSTALL_INSIGHTFACE=false
COPY requirements-insightface.txt .
RUN if [ "$INSTALL_INSIGHTFACE" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends build-essential && \
        pip install -r requirements-insightface.txt && \
        apt-get purge -y build-essential && apt-get autoremove -y && \
        rm -rf /var/lib/apt/lists/*; \
    fi

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests

# Download model weights at build time so the image is self-contained and the
# container does not need network access at runtime.
RUN mkdir -p assets/models assets/calibration && \
    python scripts/download_models.py || \
    echo "WARNING: model download failed at build time; run it in the container before first use."

# Never run as root.
RUN useradd --create-home --shell /usr/sbin/nologin facet && \
    chown -R facet:facet /app
USER facet

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
