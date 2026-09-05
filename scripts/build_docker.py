"""Build and export the Linux amd64 GPU image with its Compose file and instructions."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build(output: Path, skip_build: bool) -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    tag = f"minimaxh3:{version}"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"minimaxh3-{version}-linux-amd64.docker.tar"
    if archive.exists():
        raise FileExistsError(f"Artifact exists: {archive}; choose another --output directory")
    if not skip_build:
        subprocess.run(["docker", "build", "--platform", "linux/amd64", "--progress", "plain", "-t", tag, "."], cwd=ROOT, check=True)
    metadata = json.loads(subprocess.check_output(["docker", "image", "inspect", tag], text=True))[0]
    if (metadata["Os"], metadata["Architecture"]) != ("linux", "amd64"):
        raise RuntimeError(f"Expected linux/amd64 image: {tag}")
    partial = archive.with_suffix(".tar.partial")
    print(f"Exporting {tag} ({metadata['Id']})...", flush=True)
    subprocess.run(["docker", "save", "--output", str(partial), tag], check=True)
    partial.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".tar.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    archive.with_suffix(".image.json").write_text(json.dumps({
        "tag": tag, "id": metadata["Id"], "os": metadata["Os"],
        "architecture": metadata["Architecture"], "size": metadata["Size"],
        "archive_sha256": digest,
    }, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(ROOT / "compose.yaml", output / "compose.yaml")
    for filename in ("DISTRIBUTION.md", "MODELS.md", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(ROOT / "docs" / filename, output / filename)
    print(f"Built {archive} ({archive.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--skip-build", action="store_true", help="export an already-built release image")
    args = parser.parse_args()
    build(args.output.resolve(), args.skip_build)
