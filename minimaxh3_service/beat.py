"""Cancellable Beat This! worker, using the same task lifecycle as video generation."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .config import Settings
from .tasks import GenerationCanceled


SAMPLE_RATE = 22050


class BeatAnalysisRunner:
    def __init__(self, settings: Settings, resolve_input):
        self.settings = settings
        self.resolve_input = resolve_input

    def readiness(self) -> dict:
        model = self.settings.beat_model_path.expanduser().resolve()
        missing = [name for name in ("beat_this", "torch", "torchaudio", "numpy", "scipy")
                   if importlib.util.find_spec(name) is None]
        if shutil.which("ffmpeg") is None:
            missing.append("ffmpeg")
        return {
            "status": "ready" if not missing and model.is_file() else "not_ready",
            "backend": "beat_this", "model_path": str(model),
            "device": self.settings.beat_device, "models_loaded": False,
            "missing_dependencies": missing,
            "missing_models": [] if model.is_file() else [str(model)],
            "max_duration_seconds": self.settings.beat_max_seconds or None,
            "lyrics_alignment": False,
        }

    def _run(self, arguments, log: Path, cancel, deadline: float):
        if cancel.is_set():
            raise GenerationCanceled("beat analysis canceled")
        process = None
        try:
            with log.open("wb") as stream:
                process = subprocess.Popen(
                    arguments, stdout=stream, stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).resolve().parent.parent),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                while process.poll() is None:
                    if cancel.wait(0.2):
                        raise GenerationCanceled("beat analysis canceled")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("beat analysis exceeded H3_BEAT_TIMEOUT_SECONDS")
            if cancel.is_set():
                raise GenerationCanceled("beat analysis canceled")
            if process.returncode:
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 4000))
                    detail = stream.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Beat analysis failed: {detail}")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def process(self, request, output, cancel, progress):
        source = self.resolve_input(request.file_id)
        output.path.parent.mkdir(parents=True, exist_ok=True)
        pcm = output.path.with_suffix(".f32")
        partial = output.path.with_suffix(".partial.json")
        log = output.path.with_suffix(".log")
        deadline = time.monotonic() + self.settings.beat_timeout_seconds
        success = False
        try:
            progress(0.05)
            # Decode from the start without silence removal or independently timed chunks.
            # Only an explicitly configured deployment limit bounds decoding.
            limit = ["-t", str(self.settings.beat_max_seconds + 1)] if self.settings.beat_max_seconds else []
            self._run([
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
                "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                *limit, "-f", "f32le", str(pcm),
            ], log, cancel, deadline)
            duration = pcm.stat().st_size / (4 * SAMPLE_RATE)
            if self.settings.beat_max_seconds and duration > self.settings.beat_max_seconds:
                raise ValueError(f"Audio exceeds the {self.settings.beat_max_seconds:g} second limit")
            if duration < 1:
                raise ValueError("Audio must contain at least one second of decodable samples")
            progress(0.2)
            self._run([
                sys.executable, "-m", "minimaxh3_service.beat_worker", "--pcm", str(pcm),
                "--output", str(partial), "--model", str(self.settings.beat_model_path.expanduser().resolve()),
                "--device", self.settings.beat_device, "--threads", str(self.settings.beat_cpu_threads),
                "--file-id", request.file_id, "--title", request.title,
            ], log, cancel, deadline)
            document = json.loads(partial.read_text(encoding="utf-8"))
            if document.get("schema_version") != 1 or document.get("source_file_id") != request.file_id:
                raise RuntimeError("Beat worker returned an invalid analysis")
            if cancel.is_set():
                raise GenerationCanceled("beat analysis canceled")
            partial.replace(output.path)
            progress(0.99)
            success = True
        finally:
            for temporary in (pcm, partial, log):
                temporary.unlink(missing_ok=True)
            if not success:
                output.path.unlink(missing_ok=True)

    def release_models(self):
        # Child process exit releases CPU/GPU resources before the next queued job.
        pass

    def close(self):
        self.release_models()
