"""Install pinned official Video Depth Anything Small sources and weights."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import httpx

from .config import Settings

RUNTIME_COMMIT = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
MODEL_REVISION = "256875362cff76724b920335dfb4b29dd611f66e"
REPOSITORY = "https://github.com/DepthAnything/Video-Depth-Anything.git"
MODEL_URL = f"https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/{MODEL_REVISION}/video_depth_anything_vits.pth"


def main(argv=None):
    defaults = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=defaults.depth_runtime_path)
    parser.add_argument("--model", type=Path, default=defaults.depth_model_path)
    args = parser.parse_args(argv)
    runtime, model = args.runtime.resolve(), args.model.resolve()
    if not runtime.exists():
        runtime.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(runtime)], check=True)
        subprocess.run(["git", "-C", str(runtime), "checkout", "--detach", RUNTIME_COMMIT], check=True)
    else:
        revision = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
        if revision != RUNTIME_COMMIT:
            raise RuntimeError(f"Existing runtime is not the pinned revision; choose a new --runtime directory: {runtime}")
    model.parent.mkdir(parents=True, exist_ok=True)
    if not model.is_file():
        partial = model.with_suffix(".download")
        try:
            with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                count = 0
                with partial.open("wb") as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        stream.write(chunk)
                        count += len(chunk)
                expected = response.headers.get("content-length")
                if not count or (expected and count != int(expected)):
                    raise RuntimeError("Incomplete model download")
            partial.replace(model)
        finally:
            partial.unlink(missing_ok=True)
    print(f"Depth runtime: {runtime}\nSmall model: {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
