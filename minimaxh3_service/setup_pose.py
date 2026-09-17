"""Install pinned DWPose sources and hash-verified local model files."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

from .config import Settings

RUNTIME_REVISION = "6907bdcc259a6a048d41a365e840d22274f9256c"


def install_runtime(root):
    root = root.resolve()
    marker = root / "REVISION"
    if marker.is_file() and marker.read_text().strip() == RUNTIME_REVISION:
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Existing DWPose directory has an unknown revision; choose a new --runtime")
    url = f"https://api.github.com/repos/Tencent/MimicMotion/zipball/{RUNTIME_REVISION}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts[1:]
            if not parts or parts[0] not in {"mimicmotion", "LICENSE", "environment.yaml"}:
                continue
            destination = root.joinpath(*parts).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Unsafe runtime archive path")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(item))
    marker.write_text(RUNTIME_REVISION)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(entry, root):
    target = root / entry["path"]
    if target.is_file() and target.stat().st_size == entry["size"] and sha256(target) == entry["sha256"]:
        print(f"Verified {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".download")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == entry["size"]:
            break
        if offset > entry["size"]:
            raise RuntimeError(f"Invalid partial model size: {partial}")
        try:
            req = urllib.request.Request(entry["url"], headers={"Range": f"bytes={offset}-", "User-Agent": "minimaxh3-setup-pose"})
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Unexpected download range")
                elif response.status == 200:
                    offset = 0
                else:
                    raise RuntimeError(f"Unexpected HTTP status {response.status}")
                with partial.open("ab" if offset else "wb") as stream:
                    last = time.monotonic()
                    while chunk := response.read(4 * 1024 * 1024):
                        stream.write(chunk)
                        offset += len(chunk)
                        if offset > entry["size"]:
                            raise RuntimeError("Download exceeds expected model size")
                        if time.monotonic() - last >= 10:
                            print(f"{entry['path']}: {offset / entry['size']:.0%}", flush=True)
                            last = time.monotonic()
        except OSError as exc:
            print(f"Download attempt {attempt + 1}: {exc}", flush=True)
    if not partial.is_file() or partial.stat().st_size != entry["size"]:
        raise RuntimeError(f"Incomplete model download, rerun to resume: {partial}")
    if sha256(partial) != entry["sha256"]:
        partial.unlink()
        raise RuntimeError(f"Model checksum mismatch: {target}")
    partial.replace(target)
    print(f"Installed {target}", flush=True)


def main(argv=None):
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=settings.pose_runtime_path)
    parser.add_argument("--models", type=Path, default=settings.pose_model_path)
    args = parser.parse_args(argv)
    install_runtime(args.runtime.expanduser())
    manifest = json.loads(Path(__file__).with_name("pose_models.json").read_text(encoding="utf-8"))
    for entry in manifest:
        download(entry, args.models.expanduser())
    print("Installed local DWPose. See docs/SKELETON.md for usage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
