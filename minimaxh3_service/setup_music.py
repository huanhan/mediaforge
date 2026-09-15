"""Download the pinned ACE-Step checkpoint for native, offline music generation."""
import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

FILENAME = "ace_step_v1_3.5b.safetensors"
SIZE = 7699743341
SHA256 = "f07cad74c4adce52ca14ca1bdf74cf3c14cbafb0823b95eca4459467fa369f40"
URL = "https://modelscope.cn/models/Comfy-Org/ACE-Step_ComfyUI_repackaged/resolve/32f755ece177429125250f24a6b02f1b3a13839f/all_in_one/" + FILENAME


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(folder):
    target = folder / FILENAME
    folder.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == SIZE and digest(target) == SHA256:
        print(f"Verified {target}", flush=True)
        return target
    part = target.with_suffix(".safetensors.part")
    for attempt in range(5):
        offset = part.stat().st_size if part.exists() else 0
        if offset == SIZE:
            break
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Invalid download range")
                elif response.status == 200:
                    offset = 0
                else:
                    raise ValueError(f"Download returned HTTP {response.status}")
                with part.open("ab" if offset else "wb") as destination:
                    last = time.monotonic()
                    while chunk := response.read(4 * 1024 * 1024):
                        offset += len(chunk)
                        if offset > SIZE:
                            raise ValueError("Checkpoint exceeds expected size")
                        destination.write(chunk)
                        if time.monotonic() - last > 15:
                            print(f"ACE-Step download: {offset / SIZE:.1%} ({offset // 1048576} MiB)", flush=True)
                            last = time.monotonic()
            if offset == SIZE:
                break
        except (OSError, ValueError) as exc:
            print(f"Download attempt {attempt + 1}: {exc}", flush=True)
    if not part.exists() or part.stat().st_size != SIZE or digest(part) != SHA256:
        raise RuntimeError("Checkpoint incomplete or checksum mismatch; no model installed")
    part.replace(target)
    print(f"Verified and installed {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/music"))
    args = parser.parse_args()
    download(args.model_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
