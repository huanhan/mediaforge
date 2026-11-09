"""Build and export the Linux amd64 GPU image with its Compose file and instructions."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
from datetime import datetime
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
    shutil.copy2(ROOT / "README.md", output / "README.md")
    shutil.copy2(ROOT / "packaging" / "THIRD_PARTY_NOTICES.md", output / "THIRD_PARTY_NOTICES.md")
    instructions = (ROOT / "packaging" / "docker" / "使用说明-Docker.txt").read_text(encoding="utf-8-sig")
    (output / "使用说明-Docker.txt").write_text(
        instructions.replace("{version}", version), encoding="utf-8-sig",
    )
    print(f"Built {archive} ({archive.stat().st_size:,} bytes)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "dist" / datetime.now().strftime("docker-%Y%m%d-%H%M%S-%f"))
    parser.add_argument("--skip-build", action="store_true", help="export an already-built release image")
    args = parser.parse_args()
    output = args.output.resolve()
    print(f"Building Docker release. Output: {output}", flush=True)
    try:
        build(output, args.skip_build)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Build failed: {exc}", flush=True)
        print(f"Existing artifacts are preserved in: {output}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
