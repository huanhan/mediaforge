"""YuE2 music runner; its separate interpreter owns all model and CUDA state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import wave

from .backend import OutputArtifact
from .setup_music import MODEL_ID, VAE_ID, missing_model_files
from .tasks import GenerationCanceled


class MusicRunner:
    supports_interrupt = True

    def __init__(self, settings):
        self.settings = settings
        self._probe_cache = None
        self._probe_lock = threading.Lock()

    def _command(self):
        return [str(self.settings.music_python.expanduser().resolve()),
                str(Path(__file__).with_name("music_worker.py").resolve())]

    def _environment(self):
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1", "TOKENIZERS_PARALLELISM": "false"}
        for name in ("PYTHONPATH", "PYTHONHOME"):
            environment.pop(name, None)
        return environment

    def _process_options(self):
        return dict(env=self._environment(), cwd=str(Path(__file__).resolve().parent.parent),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def readiness(self):
        missing = missing_model_files(self.settings.music_model_path.expanduser().resolve())
        if not self.settings.music_python.expanduser().is_file():
            missing.append("YuE2 Python environment: run python -m minimaxh3_service.setup_music")
        result = dict(status="not_ready", backend="yue2", model=MODEL_ID, vae=VAE_ID,
                      models_loaded=False, instrumental=True, lyrics=True, sample_rate=48000,
                      max_duration_seconds=300, supports_interrupt=True,
                      device=self.settings.music_device, missing=missing)
        if missing:
            return result
        with self._probe_lock:
            if self._probe_cache is not None and time.monotonic() - self._probe_cache[0] < 30:
                return {**result, **self._probe_cache[1]}
            try:
                probe = subprocess.run(self._command() + ["--check", "--device", self.settings.music_device],
                                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=60, **self._process_options())
                if probe.returncode:
                    raise RuntimeError(probe.stderr[-2000:] or "YuE2 runtime check failed")
                detail = None
                # Some Windows console integrations append ANSI reset bytes at exit.
                for line in probe.stdout.splitlines():
                    try:
                        candidate = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(candidate, dict) and "status" in candidate:
                        detail = candidate
                if not isinstance(detail, dict) or detail.get("status") not in {"ready", "not_ready"}:
                    raise ValueError("Invalid YuE2 runtime check")
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                detail = dict(status="not_ready", reason=str(exc))
            self._probe_cache = (time.monotonic(), detail)
        return {**result, **detail}

    def _run(self, arguments, log, status_file, cancel, progress):
        process = None
        deadline = time.monotonic() + self.settings.music_timeout_seconds
        last_progress = 0.0
        try:
            if cancel.is_set():
                raise GenerationCanceled("music task canceled")
            with log.open("wb") as stream:
                process = subprocess.Popen(arguments, stdout=stream, stderr=subprocess.STDOUT,
                                           **self._process_options())
                while process.poll() is None:
                    if cancel.wait(.2):
                        raise GenerationCanceled("music task canceled")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("YuE2 exceeded H3_MUSIC_TIMEOUT_SECONDS")
                    try:
                        value = float(json.loads(status_file.read_text(encoding="utf-8"))["progress"])
                        if last_progress < value < 1:
                            progress(value)
                            last_progress = value
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
            if cancel.is_set():
                raise GenerationCanceled("music task canceled")
            if process.returncode:
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 5000))
                    detail = stream.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"YuE2 generation failed: {detail}")
        finally:
            if process is not None and process.poll() is None:
                if os.name == "nt":
                    # Windows venv python.exe is a launcher with a real Python child.
                    # Killing only the launcher leaves inference and CUDA allocations alive.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    @staticmethod
    def _validate_audio(path, maximum_seconds):
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) != (2, 48000, 2):
                raise ValueError("YuE2 must return stereo 48 kHz 16-bit PCM WAV")
            if not 0 < frames <= maximum_seconds * 48000:
                raise ValueError("YuE2 returned empty or overlong audio")
            remaining = frames
            while remaining:
                count = min(remaining, 65536)
                if len(audio.readframes(count)) != count * 4:
                    raise ValueError("YuE2 returned truncated WAV data")
                remaining -= count
        return frames / 48000

    def generate(self, request, output, cancel, progress):
        output.path.parent.mkdir(parents=True, exist_ok=True)
        published = [output.path, output.path.with_suffix(".json"), output.path.with_suffix(".abc")]
        success = False
        try:
            with tempfile.TemporaryDirectory(prefix=".yue2-", dir=output.path.parent) as temporary:
                root = Path(temporary)
                request_path, status_file = root / "request.json", root / "progress.json"
                request_path.write_text(json.dumps(request.model_dump(), ensure_ascii=False), encoding="utf-8")
                args = self._command() + ["--request", str(request_path), "--output", str(root / "audio.wav"),
                    "--model-dir", str(self.settings.music_model_path.expanduser().resolve()),
                    "--device", self.settings.music_device,
                    "--memory-budget-gib", str(self.settings.music_memory_budget_gib),
                    "--progress", str(status_file)]
                if self.settings.music_offload_ar:
                    args.append("--offload-ar")
                progress(.02)
                self._run(args, root / "worker.log", status_file, cancel, progress)
                duration = self._validate_audio(root / "audio.wav", request.duration_seconds)
                metadata = json.loads((root / "audio.json").read_text(encoding="utf-8"))
                if (metadata.get("schema_version") != 1 or metadata.get("model") != MODEL_ID
                        or metadata.get("sample_rate") != 48000
                        or metadata.get("duration_seconds") != duration
                        or not isinstance(metadata.get("truncated"), dict)):
                    raise ValueError("Invalid YuE2 result metadata")
                if cancel.is_set():
                    raise GenerationCanceled("music task canceled")
                artifacts = []
                for suffix in (".wav", ".json", ".abc"):
                    source = (root / "audio.wav").with_suffix(suffix)
                    if not source.is_file():
                        continue
                    target = output.path.with_suffix(suffix)
                    source.replace(target)
                    artifacts.append(OutputArtifact(target.name, output.subfolder, target))
                progress(.99)
                success = True
                return artifacts
        finally:
            if not success:
                for path in published:
                    path.unlink(missing_ok=True)

    def release_models(self):
        # The worker is reaped before generate returns; no model lives in the API process.
        pass

    def close(self):
        self.release_models()
