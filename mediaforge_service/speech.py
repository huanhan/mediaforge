"""Breeze TTS 2 runner, sharing the serialized H3 GPU task queue."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import wave
from .music import MusicRunner
from .setup_tts import model_name, missing_model_files, MANIFEST
from .backend import OutputArtifact
from .tasks import GenerationCanceled

class SpeechRunner(MusicRunner):
    def __init__(self, settings, resolve_input):
        super().__init__(settings)
        self.resolve_input = resolve_input

    def _command(self):
        return [str(self.settings.tts_python.expanduser().resolve()), str(Path(__file__).with_name("speech_worker.py"))]

    def readiness(self):
        missing = []
        if not self.settings.tts_python.is_file():
            missing.append("Run python -m mediaforge_service.setup_tts")
        if not shutil.which("ffmpeg"):
            missing.append("ffmpeg")
        absent = missing_model_files(self.settings.tts_model_path)
        try:
            accepted = json.loads((self.settings.tts_model_path / model_name() / "H3_LICENSE_ACCEPTANCE.json").read_text())
            if accepted.get("accepted") is not True or accepted.get("revision") != MANIFEST["models"][0]["revision"]:
                raise ValueError("license revision mismatch")
        except (OSError, ValueError):
            missing.append("Accept the Breeze non-commercial license via setup_tts --accept-non-commercial-license")
        from .speech_models import SPEAKERS
        for speaker in SPEAKERS:
            if not Path(__file__).with_name("voice_presets").joinpath(speaker + ".wav").is_file():
                missing.append(f"Preset reference: {speaker}")
        result = {"status": "ready" if not missing and not absent else "not_ready",
                "backend": "breeze_tts", "model": "Breeze-TTS-2", "models_loaded": False,
                "device": self.settings.tts_device, "presets": not absent, "voice_clone": not absent,
                "modes": ["clone", "direction", "design"], "speed_range": [0.5, 2.0], "speed_control": "postprocess-atempo",
                "license": MANIFEST["license"], "license_url": MANIFEST["license_url"], "commercial_use": False,
                "reference_text_required": True, "execution": "eager",
                "languages": ["Chinese", "English", "Auto"],
                "missing_dependencies": missing, "missing_models": absent,
                "sample_rate": 24000, "max_text_length": 1000, "reference_seconds": [3, 30]}
        if result["status"] != "ready":
            return result
        with self._probe_lock:
            if self._probe_cache is not None and time.monotonic() - self._probe_cache[0] < 60:
                return {**result, **self._probe_cache[1]}
            try:
                probe = subprocess.run(self._command() + ["--check", "--device", self.settings.tts_device,
                    "--model-dir", str((self.settings.tts_model_path / model_name()).resolve())],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, **self._process_options())
                detail = None
                for line in probe.stdout.splitlines():
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict) and value.get("status") in {"ready", "not_ready"}:
                            detail = value
                    except ValueError:
                        pass
                if probe.returncode or detail is None:
                    raise RuntimeError(probe.stderr[-1500:] or "TTS runtime check failed")
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                detail = {"status": "not_ready", "reason": str(error)}
            self._probe_cache = (time.monotonic(), detail)
        return {**result, **detail}

    @property
    def timeout_seconds(self):
        return self.settings.tts_timeout_seconds

    def generate(self, request, output, cancel, progress):
        output.path.parent.mkdir(parents=True, exist_ok=True)
        success = False
        try:
            with tempfile.TemporaryDirectory(prefix=".tts-", dir=output.path.parent) as temporary:
                root = Path(temporary)
                payload = request.model_dump()
                status = root / "progress.json"
                for field, file_id in (("reference_audio", request.reference_file_id),):
                    if not file_id:
                        continue
                    source = self.resolve_input(file_id)
                    reference = root / (field + ".wav")
                    self._run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
                               "-i", str(source), "-map", "0:a:0", "-t", "31", "-ac", "1", "-ar", "24000",
                               "-c:a", "pcm_s16le", str(reference)], root / "reference.log", status, cancel, progress)
                    with wave.open(str(reference), "rb") as wav:
                        duration = wav.getnframes() / wav.getframerate()
                    if not 3 <= duration <= 30:
                        raise ValueError("Reference recording must be between 3 and 30 seconds")
                    payload[field] = str(reference)
                if request.mode != "design" and not request.reference_file_id:
                    reference = Path(__file__).with_name("voice_presets") / (request.speaker + ".wav")
                    if not reference.is_file():
                        raise RuntimeError("Default voice reference is missing; restore voice_presets")
                    payload["reference_audio"] = str(reference.resolve())
                    presets = json.loads(reference.with_name("manifest.json").read_text(encoding="utf-8"))
                    payload["reference_text"] = presets[request.speaker]["text"]
                request_path = root / "request.json"
                request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                self._run(self._command() + ["--request", str(request_path), "--output", str(root / "raw.wav"),
                    "--model-dir", str((self.settings.tts_model_path / model_name()).resolve()),
                    "--device", self.settings.tts_device, "--progress", str(status)],
                    root / "worker.log", status, cancel, progress)
                self._run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(root / "raw.wav"),
                    "-af", f"atempo={request.speed},volume={request.volume_db}dB,alimiter=limit=0.95:level=false:latency=true", "-ar", "24000",
                    "-ac", "1", "-c:a", "pcm_s16le", str(root / "speech.wav")], root / "encode.log", status, cancel, progress)
                with wave.open(str(root / "speech.wav"), "rb") as wav:
                    frames = wav.getnframes()
                    if wav.getframerate() != 24000 or frames <= 0 or frames > 600 * 24000 or len(wav.readframes(frames)) != frames * 2:
                        raise RuntimeError("Invalid speech output")
                if cancel.is_set():
                    raise GenerationCanceled("speech canceled")
                (root / "speech.wav").replace(output.path)
                metadata = {"schema_version": 2, "model": model_name(), "text": request.text,
                            "license": MANIFEST["license"], "commercial_use": False, "speed_control": "postprocess-atempo",
                            "duration_seconds": frames / 24000, "sample_rate": 24000, "speed": request.speed,
                            "parameters": request.model_dump(exclude={"reference_file_id", "reference_text"}),
                            "speaker": "voice_design" if request.mode == "design" else "recorded_voice" if request.reference_file_id else request.speaker}
                meta = output.path.with_suffix(".json")
                meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                success = True
                return [output, OutputArtifact(meta.name, output.subfolder, meta)]
        finally:
            if not success:
                output.path.unlink(missing_ok=True)
                output.path.with_suffix(".json").unlink(missing_ok=True)
