FROM python:3.13.7-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    H3_HOST=0.0.0.0 \
    H3_PORT=8000 \
    H3_COMFY_CORE_PATH=/opt/mediaforge/ComfyUI \
    H3_UPSCALE_RUNTIME_PATH=/opt/mediaforge/SeedVR2 \
    H3_MODEL_DIR=/models \
    H3_UPSCALE_MODEL_DIR=/models/SEEDVR2 \
    H3_DATA_DIR=/data \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.26
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY mediaforge_service ./mediaforge_service
RUN uv sync --locked --extra local --extra ui --no-dev --no-editable --no-cache
COPY scripts/package_runtime.py ./scripts/package_runtime.py
RUN python scripts/package_runtime.py --output /opt/mediaforge

FROM base AS final
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/mediaforge /opt/mediaforge
VOLUME ["/models", "/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('H3_PORT','8000')+'/healthz', timeout=4)"
CMD ["python", "-m", "mediaforge_service"]
