FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    H3_COMFY_CORE_PATH=/opt/minimaxh3/ComfyUI \
    H3_UPSCALE_RUNTIME_PATH=/opt/minimaxh3/SeedVR2 \
    H3_MODEL_DIR=/models \
    H3_UPSCALE_MODEL_DIR=/models/SEEDVR2 \
    H3_DATA_DIR=/data

RUN apt-get update \
    && apt-get install --no-install-recommends -y git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY minimaxh3_service ./minimaxh3_service
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision==0.23.0+cu126 \
    && pip install --no-cache-dir ".[local]" \
    && minimaxh3-setup --runtime-only \
        --runtime-path "$H3_COMFY_CORE_PATH" \
        --upscale-runtime-path "$H3_UPSCALE_RUNTIME_PATH"

VOLUME ["/models", "/data"]
EXPOSE 8000
CMD ["python", "-m", "minimaxh3_service"]
