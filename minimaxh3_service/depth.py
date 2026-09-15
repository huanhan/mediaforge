"""Video Depth Anything adapter; isolated from ComfyUI's global imports/CUDA state."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from .backend import OutputArtifact
from .config import Settings
from .models import DepthRequest
from .tasks import GenerationCanceled


class VideoDepthRunner:
    def __init__(self, settings: Settings, resolve_input):
        self.settings = settings
        self.resolve_input = resolve_input

    def readiness(self) -> dict:
        runtime = self.settings.depth_runtime_path.expanduser().resolve()
        model = self.settings.depth_model_path.expanduser().resolve()
        missing_runtime = [name for name in (
            "video_depth_anything/video_depth.py", "utils/util.py"
        ) if not (runtime / name).is_file()]
        missing_dependencies = [name for name in (
            "torch", "torchvision", "numpy", "cv2", "av", "einops", "easydict", "tqdm"
        ) if importlib.util.find_spec(name) is None]
        device = self.settings.device.lower()
        reason = None if device == "cpu" or device == "cuda" or (
            device.startswith("cuda:") and device[5:].isdigit()
        ) else "Depth supports cpu or cuda[:index] devices"
        return {
            "status": "ready" if not missing_runtime and not missing_dependencies and model.is_file() and not reason else "not_ready",
            "backend": "video_depth_anything", "model": "vits", "depth_type": "relative_inverse_depth",
            "runtime_path": str(runtime), "model_path": str(model),
            "missing_runtime": missing_runtime, "missing_models": [] if model.is_file() else [str(model)],
            "missing_dependencies": missing_dependencies, "reason": reason,
            "max_frames": self.settings.depth_max_frames, "models_loaded": False,
        }

    def process(self, request: DepthRequest, output: OutputArtifact, cancel, progress):
        if cancel.is_set():
            raise GenerationCanceled("depth canceled")
        source = self.resolve_input(request.file_id)
        output.path.parent.mkdir(parents=True, exist_ok=True)
        raw = output.path.with_suffix(".npz")
        log = output.path.with_suffix(".log")
        arguments = [sys.executable, "-m", "minimaxh3_service.depth_worker",
                     "--source", str(source), "--output", str(output.path),
                     "--runtime", str(self.settings.depth_runtime_path.expanduser().resolve()),
                     "--model", str(self.settings.depth_model_path.expanduser().resolve()),
                     "--device", self.settings.device, "--input-size", str(request.input_size),
                     "--max-resolution", str(request.max_resolution),
                     "--max-frames", str(self.settings.depth_max_frames)]
        if request.save_raw:
            arguments.append("--save-raw")
        process = None
        success = False
        try:
            progress(0.05)
            with log.open("wb") as stream:
                process = subprocess.Popen(
                    arguments, stdout=stream, stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).resolve().parent.parent),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                while process.poll() is None:
                    if cancel.wait(0.25):
                        raise GenerationCanceled("depth canceled")
            if cancel.is_set():
                raise GenerationCanceled("depth canceled")
            if process.returncode:
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 4000))
                    detail = stream.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Video depth inference failed: {detail}")
            artifacts = [output]
            if request.save_raw:
                artifacts.append(OutputArtifact(raw.name, output.subfolder, raw))
            if any(not item.path.is_file() or item.path.stat().st_size == 0 for item in artifacts):
                raise RuntimeError("Depth worker did not produce all requested outputs")
            progress(0.99)
            success = True
            return artifacts
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not success:
                output.path.unlink(missing_ok=True)
                raw.unlink(missing_ok=True)
            output.path.with_suffix(".partial.mp4").unlink(missing_ok=True)
            raw.with_suffix(".partial.npz").unlink(missing_ok=True)
            log.unlink(missing_ok=True)

    def release_models(self) -> None:
        # The worker exits before process() returns, releasing all model memory.
        pass

    def close(self) -> None:
        self.release_models()
