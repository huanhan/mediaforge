"""Local audio transcription, isolated from H3's imports and GPU model lifetime."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .backend import OutputArtifact
from .config import Settings
from .setup_asr import MODEL_ID, REQUIRED_FILES
from .tasks import GenerationCanceled


SAMPLE_RATE = 16000


class TranscriptionRunner:
    def __init__(self, settings: Settings, resolve_input):
        self.settings = settings
        self.resolve_input = resolve_input

    @property
    def max_seconds(self):
        return self.settings.asr_max_seconds

    def worker_arguments(self, request, pcm, partial, extra):
        return [sys.executable, "-m", "minimaxh3_service.transcription_worker", "--pcm", str(pcm),
                "--output", str(partial), "--model", str(self.settings.asr_model_path.expanduser().resolve()),
                "--device", self.settings.asr_device, "--threads", str(self.settings.asr_cpu_threads),
                "--file-id", request.file_id, "--title", request.title, "--language", request.language,
                "--chunk-seconds", str(request.chunk_seconds)]

    def validate_document(self, document, request):
        return (document.get("schema_version") == 1 and document.get("source_file_id") == request.file_id
                and isinstance(document.get("text"), str) and document.get("timing") == "chunk_boundaries_only")

    def readiness(self):
        root = self.settings.asr_model_path.expanduser().resolve()
        missing = [name for name in ("torch", "transformers", "numpy", "accelerate")
                   if importlib.util.find_spec(name) is None]
        if shutil.which("ffmpeg") is None:
            missing.append("ffmpeg")
        if "transformers" not in missing:
            version = importlib.metadata.version("transformers")
            if tuple(int(v) for v in version.split(".")[:2]) < (5, 13):
                missing.append("transformers>=5.13")
        missing_models = [name for name in REQUIRED_FILES if not (root / name).is_file() or not (root / name).stat().st_size]
        return {"status": "ready" if not missing and not missing_models else "not_ready",
                "backend": "qwen3_asr", "model": MODEL_ID, "model_path": str(root),
                "device": self.settings.asr_device, "models_loaded": False,
                "missing_dependencies": missing, "missing_models": missing_models,
                "max_duration_seconds": self.settings.asr_max_seconds,
                "word_timestamps": False, "lyrics_alignment": False}

    def _run(self, arguments, log, cancel, deadline):
        if cancel.is_set():
            raise GenerationCanceled("transcription canceled")
        process = None
        try:
            with log.open("wb") as stream:
                environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                               "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"}
                process = subprocess.Popen(arguments, stdout=stream, stderr=subprocess.STDOUT, env=environment,
                                           cwd=str(Path(__file__).resolve().parent.parent),
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                while process.poll() is None:
                    if cancel.wait(.2):
                        raise GenerationCanceled("transcription canceled")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("transcription exceeded H3_ASR_TIMEOUT_SECONDS")
            if cancel.is_set():
                raise GenerationCanceled("transcription canceled")
            if process.returncode:
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 6000))
                    detail = stream.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Qwen3-ASR transcription failed: {detail}")
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
        pcm, partial, log = (output.path.with_suffix(ext) for ext in (".f32", ".partial.json", ".log"))
        text_path = output.path.with_suffix(".txt")
        extra = output.path.with_suffix(".request.json")
        deadline = time.monotonic() + self.settings.asr_timeout_seconds
        success = False
        try:
            progress(.05)
            self._run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
                       "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                       "-t", str(self.max_seconds+1), "-f", "f32le", str(pcm)], log, cancel, deadline)
            duration = pcm.stat().st_size / (4*SAMPLE_RATE)
            if duration > self.max_seconds:
                raise ValueError(f"Audio exceeds the {self.max_seconds:g} second limit")
            if duration < 1:
                raise ValueError("Audio must contain at least one second of decodable samples")
            progress(.15)
            self._run(self.worker_arguments(request, pcm, partial, extra), log, cancel, deadline)
            document = json.loads(partial.read_text(encoding="utf-8"))
            if not self.validate_document(document, request):
                raise RuntimeError("Transcription worker returned an invalid result")
            if cancel.is_set():
                raise GenerationCanceled("transcription canceled")
            text_path.write_text(document["text"] + "\n", encoding="utf-8")
            partial.replace(output.path)
            progress(.99)
            success = True
            return [output, OutputArtifact(text_path.name, output.subfolder, text_path)]
        finally:
            for temporary in (pcm, partial, log, extra):
                temporary.unlink(missing_ok=True)
            if not success:
                output.path.unlink(missing_ok=True)
                text_path.unlink(missing_ok=True)

    def release_models(self):
        # The subprocess exits before returning any output, freeing all GPU resources.
        pass

    def close(self):
        self.release_models()
