from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniMax H3 portable service")
    parser.add_argument("--check", action="store_true", help="check CUDA, runtimes and model paths")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    load_dotenv(root / ".env", override=False)
    defaults = {
        "H3_HOST": "127.0.0.1",
        "H3_COMFY_CORE_PATH": str(root / "runtime" / "ComfyUI"),
        "H3_UPSCALE_RUNTIME_PATH": str(root / "runtime" / "SeedVR2"),
        "H3_MODEL_DIR": str(root / "models"),
        "H3_DATA_DIR": str(root / "data"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("H3_UPSCALE_MODEL_DIR", str(Path(os.environ["H3_MODEL_DIR"]) / "SEEDVR2"))
    from minimaxh3_service.config import Settings

    settings = Settings.from_env()
    if args.check:
        from minimaxh3_service.native import NativeH3Runner
        from minimaxh3_service.upscale import SeedVR2UpscaleRunner

        video = NativeH3Runner(settings, lambda _: Path()).readiness()
        upscale = SeedVR2UpscaleRunner(settings, lambda _: Path()).readiness()
        print(json.dumps({"video": video, "upscale": upscale}, ensure_ascii=False, indent=2))
        return 0 if video["status"] == "ready" and upscale["status"] == "ready" else 1

    settings.model_root.mkdir(parents=True, exist_ok=True)
    settings.upscale_model_root.mkdir(parents=True, exist_ok=True)
    print(f"MiniMax H3: http://127.0.0.1:{settings.port}/docs", flush=True)
    print(f"Models: {settings.model_root.resolve()}", flush=True)
    print(f"Data: {settings.data_root.resolve()}", flush=True)
    from minimaxh3_service.__main__ import main as serve

    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
