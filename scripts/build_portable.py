"""Build a Windows x64 ZIP with Python, locked CUDA dependencies and pinned runtimes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from package_runtime import ROOT, package_runtimes


PYTHON_VERSION = "3.13.7"
VC_RUNTIME_FILES = (
    "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll",
    "msvcp140_2.dll", "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
    "concrt140.dll", "vcomp140.dll", "libomp140.x86_64.dll",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(output: Path, runtime_source: Path | None, vc_runtime: Path) -> Path:
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Build the Windows portable package on Windows x64")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Use Python 3.13 to build this release")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("The build machine needs uv; the released package does not")
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    name = f"minimaxh3-{metadata['version']}-windows-x64"
    output.mkdir(parents=True, exist_ok=True)
    package = output / name
    archive_path = output / f"{name}.zip"
    if package.exists() or archive_path.exists():
        raise FileExistsError(f"Release already exists: {package}; choose a new --output directory")
    missing_dlls = [name for name in VC_RUNTIME_FILES if not (vc_runtime / name).is_file()]
    if missing_dlls:
        raise RuntimeError(f"Missing x64 redistributable DLLs in {vc_runtime}: {missing_dlls}")
    cache = ROOT / "build" / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    python_zip = cache / f"python-{PYTHON_VERSION}-embed-amd64.zip"
    url = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{python_zip.name}"
    if not python_zip.is_file():
        print(f"Downloading {url}", flush=True)
        partial = python_zip.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        partial.replace(python_zip)
    python_root = package / "python"
    python_root.mkdir(parents=True)
    with zipfile.ZipFile(python_zip) as archive:
        archive.extractall(python_root)
    (python_root / "python313._pth").write_text(
        "python313.zip\n.\nLib/site-packages\n../app\nimport site\n", encoding="ascii"
    )
    for filename in VC_RUNTIME_FILES:
        shutil.copy2(vc_runtime / filename, python_root / filename)
    requirements = package / "requirements-lock.txt"
    subprocess.run([
        uv, "export", "--locked", "--extra", "local", "--no-dev", "--no-emit-project",
        "--no-header", "--no-annotate", "--emit-index-url", "--output-file", str(requirements),
    ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([
        uv, "pip", "install", "--python", sys.executable,
        "--target", str(python_root / "Lib" / "site-packages"),
        "--requirements", str(requirements), "--require-hashes", "--no-deps",
        "--link-mode", "copy", "--index-strategy", "unsafe-best-match",
    ], cwd=ROOT, check=True)
    app = package / "app" / "minimaxh3_service"
    app.mkdir(parents=True)
    for source in sorted((ROOT / "minimaxh3_service").glob("*.py")):
        shutil.copy2(source, app / source.name)
    package_runtimes(package / "runtime", runtime_source)
    for filename in ("start.cmd", "check.cmd", "launch.py", ".env.example"):
        shutil.copy2(ROOT / "packaging" / "windows" / filename, package / filename)
    for filename in ("DISTRIBUTION.md", "MODELS.md", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(ROOT / "docs" / filename, package / filename)
    shutil.copy2(ROOT / "docs" / "openapi.json", package / "openapi.json")
    for relative in ("models/diffusion_models", "models/text_encoders", "models/vae", "models/loras", "models/SEEDVR2", "data/input", "data/output"):
        (package / relative).mkdir(parents=True)
    manifest = {
        "name": name,
        "python": {"version": PYTHON_VERSION, "url": url, "sha256": sha256(python_zip)},
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "model_weights_included": False,
        "user_data_included": False,
        "files": {},
    }
    print("Hashing release files...", flush=True)
    for path in sorted(package.rglob("*")):
        if path.is_file():
            manifest["files"][path.relative_to(package).as_posix()] = sha256(path)
    (package / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Creating ZIP64 archive...", flush=True)
    partial_archive = archive_path.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in sorted(package.rglob("*")):
            archive.write(path, path.relative_to(output).as_posix())
    partial_archive.replace(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(f"{sha256(archive_path)}  {archive_path.name}\n", encoding="ascii")
    print(f"Built {archive_path} ({archive_path.stat().st_size:,} bytes)", flush=True)
    return archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--runtime-source", type=Path, help="existing pinned runtime repositories; omit to fetch")
    parser.add_argument("--vc-runtime", type=Path, default=Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32")
    args = parser.parse_args()
    build(args.output.resolve(), args.runtime_source.resolve() if args.runtime_source else None, args.vc_runtime.resolve())
