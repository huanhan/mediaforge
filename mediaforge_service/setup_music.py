"""Install YuE2 in an isolated Python 3.12 environment and download pinned weights."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

RUNTIME_REVISION = "1647252d68b70cbead046ffd3f7caf036dde47ee"
MODEL_ID = "m-a-p/YuE2-3B"
VAE_ID = "m-a-p/YuE2-Vae"
MODELS = (
    (MODEL_ID, "8f24312f187763b9854adeec87e437e2b03bbff1", 7261441640,
     "1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9"),
    (VAE_ID, "152733a19ad43aa67e367f9b5503ef8075bb5126", 530512720,
     "807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346"),
)


def runtime_python(root=Path(".runtime/yue2-env")):
    return Path(root) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def missing_model_files(root):
    missing = []
    for model_id, _, size, _ in MODELS:
        folder = Path(root) / model_id.split("/")[-1]
        for name in ("config.json", "weights_manifest.json", "model.safetensors"):
            path = folder / name
            if not path.is_file() or not path.stat().st_size or (name == "model.safetensors" and path.stat().st_size != size):
                missing.append(str(path))
        if model_id == MODEL_ID and not (folder / "qwen.tiktoken").is_file():
            missing.append(str(folder / "qwen.tiktoken"))
    return missing


def download_weights(url, target, size, expected_hash, *, chunk_size=32 * 1024 * 1024):
    """Bounded range requests also work behind proxies that buffer whole responses."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size:
        with target.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() == expected_hash:
                return
    chunks = [(offset, min(chunk_size, size - offset),
               target.with_name(f"{target.name}.chunk-{index:06d}"))
              for index, offset in enumerate(range(0, size, chunk_size))]

    def fetch(chunk):
        offset, length, path = chunk
        if path.is_file() and path.stat().st_size == length:
            return length
        for attempt in range(3):
            try:
                end = offset + length - 1
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{size}":
                        raise ValueError("Invalid YuE2 download Content-Range")
                    content = response.read(length + 1)
                    if len(content) != length:
                        raise ValueError("Incomplete YuE2 download range")
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
                return length
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    complete, reported = 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, chunk) for chunk in chunks]):
            complete += future.result()
            if time.monotonic() - reported > 10 or complete == size:
                print(f"{target.parent.name}: {complete / size:.1%} ({complete // 1048576} MiB)", flush=True)
                reported = time.monotonic()
    partial = target.with_suffix(".safetensors.part")
    digest = hashlib.sha256()
    with partial.open("wb") as destination:
        for _, _, path in chunks:
            with path.open("rb") as source:
                while data := source.read(8 * 1024 * 1024):
                    destination.write(data)
                    digest.update(data)
    if partial.stat().st_size != size or digest.hexdigest() != expected_hash:
        partial.unlink(missing_ok=True)
        for _, _, path in chunks:
            path.unlink(missing_ok=True)
        raise RuntimeError(f"YuE2 weight integrity failed: {target}; rerun setup to download again")
    partial.replace(target)
    for _, _, path in chunks:
        path.unlink(missing_ok=True)


def download_models(root):
    from huggingface_hub import snapshot_download

    for model_id, revision, size, expected_hash in MODELS:
        folder = Path(root) / model_id.split("/")[-1]
        snapshot_download(model_id, revision=revision, local_dir=str(folder), max_workers=2,
                          allow_patterns=["config.json", "generation_config.json", "yue2_generation_config.json",
                                          "weights_manifest.json", "qwen.tiktoken", "LICENSE",
                                          "MODEL_LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/*.txt", "README.md"])
        weights = folder / "model.safetensors"
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        download_weights(f"{endpoint}/{model_id}/resolve/{revision}/model.safetensors", weights, size, expected_hash)
        (folder / "REVISION").write_text(revision + "\n", encoding="utf-8")
        print(f"Verified {model_id}: {folder}", flush=True)
    missing = missing_model_files(root)
    if missing:
        raise RuntimeError("Incomplete YuE2 installation: " + ", ".join(missing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path(".runtime/yue2-env"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/music/yue2"))
    parser.add_argument("--python", default="3.12", help="Python 3.12 interpreter or version for uv venv")
    parser.add_argument("--runtime-only", action="store_true", help="Install dependencies without model downloads")
    parser.add_argument("--models-only", action="store_true", help="Download using this interpreter's huggingface-hub")
    args = parser.parse_args(argv)
    if args.runtime_only and args.models_only:
        parser.error("--runtime-only and --models-only cannot be combined")
    model_root = args.model_dir.expanduser().resolve()
    if args.models_only:
        download_models(model_root)
        return 0
    uv = shutil.which("uv")
    if uv is None:
        parser.error("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    env_root = args.runtime_dir.expanduser().resolve()
    interpreter = runtime_python(env_root)
    if not interpreter.is_file():
        subprocess.run([uv, "venv", "--python", args.python, str(env_root)], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(interpreter), "torch==2.10.0",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(interpreter),
                    f"git+https://github.com/multimodal-art-projection/YuE.git@{RUNTIME_REVISION}"], check=True)
    if not args.runtime_only:
        subprocess.run([str(interpreter), str(Path(__file__).resolve()), "--models-only",
                        "--model-dir", str(model_root)], check=True)
    print(f"YuE2 interpreter: {interpreter}\nModel directory: {model_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
