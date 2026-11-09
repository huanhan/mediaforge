"""Install pinned Breeze TTS 2 for explicitly accepted research/non-commercial use."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

MANIFEST = json.loads(Path(__file__).with_name("breeze_tts_models.json").read_text(encoding="utf-8"))
SOURCE_REVISION = MANIFEST["source_revision"]

def model_name():
    return "Breeze-TTS-2"

def model_folder(root):
    return Path(root) / model_name()

def missing_model_files(root):
    folder = model_folder(root)
    return [str(folder / model["destination"] / entry["name"])
            for model in MANIFEST["models"] for entry in model["files"]
            if not (folder / model["destination"] / entry["name"]).is_file()
            or (folder / model["destination"] / entry["name"]).stat().st_size != entry["size"]]

def install_source(folder):
    folder = Path(folder).resolve()
    revision_file = folder / "H3_REVISION"
    if revision_file.is_file() and revision_file.read_text().strip() == SOURCE_REVISION:
        return
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f"Source folder is not the pinned revision; choose an empty --source-dir: {folder}")
    url = f"https://codeload.github.com/breezeblue-ai/breeze-tts/tar.gz/{SOURCE_REVISION}"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive_bytes = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        prefix = archive.getmembers()[0].name.split("/")[0]
        for member in archive.getmembers():
            target = (folder / Path(member.name).relative_to(prefix)).resolve()
            if not target.is_relative_to(folder):
                raise ValueError("Unsafe source archive path")
            if member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    revision_file.write_text(SOURCE_REVISION + "\n", encoding="utf-8")

def download_models(root):
    from filelock import FileLock
    from mediaforge_service.setup_music import download_weights
    folder = model_folder(root).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    def download(item):
        model, entry = item
        target = folder / model["destination"] / entry["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint}/{model['repo']}/resolve/{model['revision']}/{entry['name']}"
        if entry["sha256"]:
            download_weights(url, target, entry["size"], entry["sha256"])
        elif not target.is_file() or target.stat().st_size != entry["size"]:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if len(data) != entry["size"]:
                raise RuntimeError(f"Incomplete model file: {entry['name']}")
            partial = target.with_suffix(target.suffix + ".partial")
            partial.write_bytes(data)
            partial.replace(target)
        print(f"Verified {model['repo']}/{entry['name']}", flush=True)
    with FileLock(str(folder / ".setup.lock")):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(download, [(model, entry) for model in MANIFEST["models"] for entry in model["files"]]))
        (folder / "NOTICE").write_text(
            "Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial License Agreement. "
            "Copyright (c) 2026 RESONIA, INC. All Rights Reserved.\n"
            "Inference code: Apache-2.0. Audio tokenizer based on Qwen3-TTS: Apache-2.0.\n"
            "This integration does not grant commercial rights to model weights or self-hosted outputs.\n", encoding="utf-8")
        (folder / "H3_LICENSE_ACCEPTANCE.json").write_text(json.dumps({
            "license": MANIFEST["license"], "revision": MANIFEST["models"][0]["revision"],
            "use": "research-and-non-commercial", "accepted": True}, indent=2), encoding="utf-8")
        (folder / "H3_MODELS.json").write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path(".runtime/breeze-tts-env"))
    parser.add_argument("--source-dir", type=Path, default=Path(".runtime/breeze-tts-src"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tts"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models-only", action="store_true")
    group.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--accept-non-commercial-license", action="store_true",
                        help="Accept the pinned BreezeBlue Research and Non-Commercial License; no commercial rights")
    args = parser.parse_args(argv)
    if not args.accept_non_commercial_license:
        parser.error("Read licenses/BREEZE_MODEL_LICENSE.txt, then pass --accept-non-commercial-license for non-commercial use")
    if args.models_only:
        download_models(args.model_dir)
        return
    uv = shutil.which("uv")
    if not uv:
        parser.error("Install uv first")
    install_source(args.source_dir)
    interpreter = args.runtime_dir.resolve() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Avoid accidentally inheriting a system Conda interpreter and its DLL search path.
    subprocess.run([uv, "venv", "--managed-python", "--python", "3.11", "--allow-existing",
                    str(args.runtime_dir.resolve())], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(interpreter), "torch==2.9.1", "torchaudio==2.9.1",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(interpreter), "-r", str(args.source_dir.resolve() / "requirements.txt")], check=True)
    # Upstream is a source checkout, not a Python distribution; isolate it from the API.
    site = subprocess.check_output([str(interpreter), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True).strip()
    (Path(site) / "breeze_source.pth").write_text(str(args.source_dir.resolve()) + "\n", encoding="utf-8")
    if not args.runtime_only:
        download_models(args.model_dir)
    print(f"Breeze TTS 2 runtime: {interpreter}", flush=True)

if __name__ == "__main__":
    main()
