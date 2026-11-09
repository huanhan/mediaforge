"""Extract a portable ZIP elsewhere and verify its manifest, isolation and HTTP startup."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def verify(archive: Path, destination: Path, models: Path | None, skip_extract: bool = False) -> None:
    if not skip_extract:
        destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise AssertionError(f"Unsafe archive member: {path}")
            if len(path.parts) > 1 and path.parts[1] in {"models", "data"}:
                assert item.is_dir(), f"User data or model included: {path}"
            assert ".git" not in path.parts, f"Git metadata included: {path}"
            assert path.suffix.lower() != ".safetensors", f"Model included: {path}"
        if not skip_extract:
            print("Extracting and checking ZIP CRCs...", flush=True)
            zipped.extractall(destination)
    package, = (path for path in destination.iterdir() if path.is_dir())
    manifest = json.loads((package / "release-manifest.json").read_text(encoding="utf-8"))
    print(f"Verifying {len(manifest['files'])} file hashes...", flush=True)
    for relative, expected in manifest["files"].items():
        with (package / relative).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, relative
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith(("H3_", "PYTHON", "VIRTUAL_ENV", "CONDA", "UV_"))}
    environment["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    environment["HF_HUB_OFFLINE"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    python = package / "python" / "python.exe"
    with tempfile.TemporaryDirectory(prefix="h3-portable-smoke-") as temporary:
        cwd = Path(temporary)
        result = subprocess.run([str(python), "-I", str(package / "launch.py"), "--check"], cwd=cwd, env=environment, capture_output=True, text=True, check=False)
        assert result.returncode == 1, result.stdout + result.stderr
        check = json.loads(result.stdout)
        assert check["video"]["missing_models"] and check["upscale"]["missing_models"], check
        assert not check["video"]["missing_runtime"] and check["video"]["reason"] is None, check
        assert not check["upscale"]["missing_runtime"] and not check["upscale"]["missing_dependencies"], check
        assert check["video"]["devices"], check
        print("Relocated Python, CUDA and empty model directory checks passed", flush=True)
        if models:
            environment["H3_MODEL_DIR"] = str(models)
            environment["H3_UPSCALE_MODEL_DIR"] = str(models / "SEEDVR2")
            result = subprocess.run([str(python), "-I", str(package / "launch.py"), "--check"], cwd=cwd, env=environment, capture_output=True, text=True, check=False)
            assert result.returncode in {0, 1}, result.stdout + result.stderr
            check = json.loads(result.stdout)
            assert check["video"]["status"] == "ready", check
            print(f"External H3 models ready; optional upscale: {check['upscale']['status']}", flush=True)
            environment["H3_COMFY_CORE_PATH"] = str(package / "runtime" / "ComfyUI")
            environment["H3_UPSCALE_RUNTIME_PATH"] = str(package / "runtime" / "SeedVR2")
            smoke = (
                "from pathlib import Path; import torch; "
                "from mediaforge_service.config import Settings; "
                "from mediaforge_service.native import NativeH3Runner; "
                "from mediaforge_service.upscale import SeedVR2UpscaleRunner; "
                "s=Settings.from_env(); r=NativeH3Runner(s,lambda _:Path()); "
                "r._import_core(); u=SeedVR2UpscaleRunner(s,lambda _:Path()); "
                "u._build_args(u._import_cli()); "
                "a=torch.ones((16,16),device='cuda'); assert (a@a).sum().item()==4096; "
                "print('Pinned H3 core, DynamicVRAM, SeedVR2 CLI and CUDA compute passed')"
            )
            subprocess.run([str(python), "-I", "-c", smoke], cwd=cwd, env=environment, check=True)
        environment["H3_DATA_DIR"] = str(cwd / "data")
        environment["H3_HOST"] = "127.0.0.1"
        with socket.socket() as bound:
            bound.bind(("127.0.0.1", 0))
            port = bound.getsockname()[1]
        environment["H3_PORT"] = str(port)
        with (destination / "smoke-service.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(python), "-I", str(package / "launch.py")], cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Service exited with {process.returncode}; see {log.name}")
                    try:
                        with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                            assert response.status == 200
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.5)
                else:
                    raise TimeoutError("Portable service did not start")
                for endpoint in ("openapi.json", "docs"):
                    with opener.open(f"http://127.0.0.1:{port}/{endpoint}", timeout=10) as response:
                        assert response.status == 200
                if models:
                    with opener.open(f"http://127.0.0.1:{port}/readyz", timeout=30) as response:
                        readiness = json.load(response)
                        assert readiness["status"] == "ready", readiness
                print("Portable HTTP health, API docs and external model readiness passed", flush=True)
            finally:
                process.terminate()
                process.wait(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--skip-extract", action="store_true", help="recheck an existing extraction")
    args = parser.parse_args()
    verify(args.archive.resolve(), args.destination.resolve(), args.models.resolve() if args.models else None, args.skip_extract)
