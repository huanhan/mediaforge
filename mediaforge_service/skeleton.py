from __future__ import annotations
import json, os, subprocess, sys, importlib.util
from pathlib import Path
from .backend import OutputArtifact
from .tasks import GenerationCanceled

class SkeletonRunner:
    def __init__(self, settings, resolve_input):
        self.settings, self.resolve_input = settings, resolve_input
    def readiness(self):
        root = self.settings.pose_model_path.expanduser().resolve()
        runtime = self.settings.pose_runtime_path.expanduser().resolve()
        missing = [n for n in ("DWPose/yolox_l.onnx", "DWPose/dw-ll_ucoco_384.onnx") if not (root/n).is_file()]
        cuda_ready = importlib.util.find_spec("onnxruntime") is not None
        return {"status": "ready" if not missing and (runtime/"mimicmotion/dwpose/dwpose_detector.py").is_file() and cuda_ready else "not_ready",
                "backend": "dwpose", "local_only": True, "missing_models": missing,
                "device": self.settings.device, "cuda_provider": cuda_ready,
                "runtime_path": str(runtime), "model_path": str(root)}
    def process(self, request, output, cancel, progress):
        source = self.resolve_input(request.file_id); output.path.parent.mkdir(parents=True, exist_ok=True)
        config = output.path.with_suffix(".request.json"); state = output.path.with_suffix(".progress"); log = output.path.with_suffix(".log")
        config.write_text(json.dumps({"source": str(source), "output": str(output.path), "progress": str(state),
            "runtime": str(self.settings.pose_runtime_path.expanduser().resolve()), "models": str(self.settings.pose_model_path.expanduser().resolve()),
            "request": request.model_dump(), "device": self.settings.device}), encoding="utf-8")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        process = None
        success = False
        try:
            with log.open("wb") as stream:
                process = subprocess.Popen([sys.executable, "-m", "mediaforge_service.skeleton_worker", str(config)], cwd=str(Path(__file__).parent.parent), env=env, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                while process.poll() is None:
                    if cancel.wait(.25): raise GenerationCanceled("skeleton canceled")
                    try: progress(float(state.read_text()))
                    except (OSError, ValueError): pass
            if process.returncode:
                detail = log.read_text(errors="replace")[-4000:]
                raise RuntimeError(f"Skeleton extraction failed: {detail}")
            if not output.path.is_file() or output.path.stat().st_size == 0: raise RuntimeError("Skeleton worker produced no video")
            progress(.99); success = True; return [output]
        finally:
            if process is not None and process.poll() is None: process.terminate()
            if not success: output.path.unlink(missing_ok=True)
            for p in (config, state, log, output.path.with_suffix('.partial.mp4')): p.unlink(missing_ok=True)
    def release_models(self): pass
    def close(self): pass
