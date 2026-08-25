# syntax=docker/dockerfile:1.7
#
# Multi-stage build. The builder installs the (large) Python toolchain and
# fetches model weights; the runtime keeps only the venv, the weight cache, and
# ffmpeg. Nothing else follows it across.
#
# Notable choices:
#   * CPU-only torch from the PyTorch CPU index. The default PyPI wheel bundles
#     CUDA and is ~6 GB; the CPU wheel is a fraction of that, and this service
#     is deliberately CPU-bound (see README "Scaling").
#   * Weights are baked in, so the runtime sets HF_HUB_OFFLINE=1 and needs no
#     network. See scripts/download_models.py.
#   * espeak-ng (~5 MB) is installed so `make sample` and the smoke test can
#     generate speech fixtures inside the container, with no dataset download.

ARG PYTHON_VERSION=3.11-slim

# ---------------------------------------------------------------- builder ---
FROM python:${PYTHON_VERSION} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# torch/torchaudio come from the CPU index; everything else from PyPI. Split
# into two steps so a PyPI-only resolver cannot silently pull a CUDA wheel.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.11.0 torchaudio==2.11.0 \
 && pip install -r requirements.txt

# Fetch weights into a cache directory we then copy into the runtime stage.
ENV HF_HOME=/opt/models
COPY app/ /build/app/
COPY scripts/download_models.py scripts/export_onnx.py /build/scripts/
ARG VA_AGE_GENDER_MODEL=audeering/wav2vec2-large-robust-6-ft-age-gender
ARG VA_LANGUAGE_MODEL=openai/whisper-tiny
ARG VA_ENABLE_LANGUAGE_ID=true
ENV VA_AGE_GENDER_MODEL=${VA_AGE_GENDER_MODEL} \
    VA_LANGUAGE_MODEL=${VA_LANGUAGE_MODEL} \
    VA_ENABLE_LANGUAGE_ID=${VA_ENABLE_LANGUAGE_ID}
RUN cd /build && python scripts/download_models.py

# Export to ONNX at build time. On linux/aarch64 this runs 2.1x faster than the
# PyTorch path (213 ms vs 453 ms on a 5 s chunk, 4 threads) because the
# manylinux torch wheel has no Accelerate-class BLAS. The export is verified
# against PyTorch at several input lengths before the build is allowed to
# succeed, so a silently-wrong graph cannot ship.
RUN cd /build && python scripts/export_onnx.py --out /opt/models/onnx/age_gender.onnx

# ---------------------------------------------------------------- runtime ---
FROM python:${PYTHON_VERSION} AS runtime

# ffmpeg: decodes every container/codec a telephony vendor might send.
# espeak-ng: offline speech fixtures for the smoke test.
# curl: container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/models /opt/models

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/models \
    HF_HUB_OFFLINE=1 \
    VA_ONNX_PATH=/opt/models/onnx/age_gender.onnx \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    VA_HOST=0.0.0.0 \
    VA_PORT=8000

WORKDIR /app
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser eval/ /app/eval/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser tests/ /app/tests/
COPY --chown=appuser:appuser pytest.ini /app/

USER appuser
EXPOSE 8000

# Liveness only. Readiness (weights loaded AND warmed) is /ready, which is what
# a load balancer should gate traffic on -- see app/routers/health.py.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# One uvicorn worker per container by design. Scale with replicas, not workers:
# each worker would hold its own full copy of the weights in RSS, and the
# threading is already tuned per-process. See README "Scaling".
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--no-access-log", "--timeout-keep-alive", "65"]
