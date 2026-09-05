"""Export pinned inference source and static resources, never local model/data trees."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = {
    "ComfyUI": ("https://github.com/comfy-org/ComfyUI.git", "native.py", "COMFY_CORE_COMMIT"),
    "SeedVR2": (
        "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git",
        "upscale.py",
        "SEEDVR2_COMMIT",
    ),
}
# Fixed prompt embeddings shipped with the backend, not model checkpoints.
STATIC_EMBEDDINGS = {"pos_emb.pt", "neg_emb.pt"}
WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}


def pinned_commit(filename: str, constant: str) -> str:
    tree = ast.parse((ROOT / "minimaxh3_service" / filename).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == constant for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing runtime pin: {constant}")


def include_runtime_file(runtime: str, filename: str) -> bool:
    path = PurePosixPath(filename)
    if any(part in {".git", "__pycache__"} for part in path.parts) or path.parts[0] in {
        "models", "input", "output",
    }:
        return False
    if runtime == "SeedVR2" and filename in STATIC_EMBEDDINGS:
        return True
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return False
    if runtime == "ComfyUI":
        return path.parts[0] == "comfy" or filename == "LICENSE" or (
            len(path.parts) == 1 and path.suffix == ".py"
        )
    return path.parts[0] in {"src", "configs_3b", "configs_7b"} or filename in {
        "inference_cli.py", "LICENSE", "__init__.py",
    }


def git(repository: Path, *arguments: str) -> list[str]:
    return ["git", "-c", f"safe.directory={repository.resolve().as_posix()}", "-C", str(repository), *arguments]


def package_runtimes(output: Path, source_root: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    with tempfile.TemporaryDirectory(prefix="h3-runtime-") as temporary:
        scratch = Path(temporary)
        for name, (url, filename, constant) in RUNTIMES.items():
            commit = pinned_commit(filename, constant)
            repository = source_root / name if source_root else scratch / name
            if source_root is None:
                subprocess.run(["git", "init", str(repository)], check=True)
                subprocess.run(git(repository, "fetch", "--depth=1", url, commit), check=True)
            else:
                head = subprocess.check_output(git(repository, "rev-parse", "HEAD"), text=True).strip()
                if head != commit:
                    raise RuntimeError(f"{repository}: expected {commit}, got {head}")
            archive_path = scratch / f"{name}.tar"
            subprocess.run(git(repository, "archive", "--format=tar", f"--output={archive_path}", commit), check=True)
            files = {}
            with tarfile.open(archive_path) as archive:
                for member in archive:
                    if not member.isfile() or not include_runtime_file(name, member.name):
                        continue
                    relative = PurePosixPath(member.name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"Invalid archive path: {member.name}")
                    target = output / name / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("wb") as destination:
                        content = source.read()
                        destination.write(content)
                    files[member.name] = hashlib.sha256(content).hexdigest()
            manifest[name] = {"repository": url, "commit": commit, "files": files}
            print(f"Packaged {name}: {commit}, {len(files)} files", flush=True)
    (output / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    package_runtimes(args.output.resolve(), args.source_root.resolve() if args.source_root else None)
