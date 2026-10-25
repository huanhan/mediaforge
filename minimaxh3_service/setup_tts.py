"""Install an isolated Qwen3-TTS runtime and pinned 1.7B model checkpoints."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess

MODELS = {
    "CustomVoice": ("0c0e3051f131929182e2c023b9537f8b1c68adfe", 3833402552,
                    "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137"),
    "Base": ("fd4b254389122332181a7c3db7f27e918eec64e3", 3857413744,
             "38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c"),
}
TOKENIZER_SIZE = 682293092
TOKENIZER_SHA = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"

def model_name(variant):
    return f"Qwen3-TTS-12Hz-1.7B-{variant}"

def missing_model_files(root, variant):
    folder = Path(root) / model_name(variant)
    required = {"config.json": None, "generation_config.json": None, "tokenizer_config.json": None,
                "preprocessor_config.json": None, "vocab.json": None, "merges.txt": None,
                "model.safetensors": MODELS[variant][1], "speech_tokenizer/config.json": None,
                "speech_tokenizer/preprocessor_config.json": None,
                "speech_tokenizer/model.safetensors": TOKENIZER_SIZE}
    return [str(folder / name) for name, size in required.items()
            if not (folder / name).is_file() or not (folder / name).stat().st_size
            or (size is not None and (folder / name).stat().st_size != size)]

def download_models(root):
    from filelock import FileLock
    Path(root).mkdir(parents=True, exist_ok=True)
    with FileLock(str(Path(root) / ".setup.lock")):
        _download_models(root)

def _download_models(root):
    from huggingface_hub import snapshot_download
    from minimaxh3_service.setup_music import download_weights
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    for variant, (revision, size, checksum) in MODELS.items():
        repo = "Qwen/" + model_name(variant)
        folder = Path(root) / model_name(variant)
        snapshot_download(repo, revision=revision, local_dir=str(folder), max_workers=2,
                          allow_patterns=["*.json", "*.txt", "*.md", "LICENSE", "speech_tokenizer/*.json"])
        for filename, length, digest in [("model.safetensors", size, checksum),
                                         ("speech_tokenizer/model.safetensors", TOKENIZER_SIZE, TOKENIZER_SHA)]:
            download_weights(f"{endpoint}/{repo}/resolve/{revision}/{filename}", folder / filename, length, digest)
        (folder / "REVISION").write_text(revision + "\n", encoding="utf-8")
        if missing_model_files(root, variant):
            raise RuntimeError(f"Incomplete Qwen3-TTS checkpoint: {folder}")
        print(f"Verified {repo}", flush=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path(".runtime/qwen3-tts-env"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tts"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models-only", action="store_true")
    group.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args(argv)
    if args.models_only:
        download_models(args.model_dir.resolve())
        return
    uv = shutil.which("uv")
    if not uv:
        parser.error("Install uv first")
    interpreter = args.runtime_dir.resolve() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.is_file():
        subprocess.run([uv, "venv", "--python", "3.12", str(args.runtime_dir.resolve())], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(interpreter), "torch==2.8.0", "torchaudio==2.8.0",
                    "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(interpreter), "qwen-tts==0.1.1"], check=True)
    if not args.runtime_only:
        download_models(args.model_dir.resolve())
    print(f"TTS runtime: {interpreter}", flush=True)

if __name__ == "__main__":
    main()
