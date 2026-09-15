"""Adapter for the official self-hosted Music 3 SGLang-Omni Speech API."""
import wave

import httpx

from .backend import BackendError
from .tasks import GenerationCanceled


class MusicRunner:
    supports_interrupt = False

    def validate_request(self, request):
        if not request.lyrics:
            raise BackendError("Remote Music 3 requires lyrics", status_code=422, code="music_lyrics_required")
    def __init__(self, settings, *, transport=None):
        self.settings = settings
        self.transport = transport

    def _client(self, timeout):
        return httpx.Client(
            base_url=self.settings.music_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {self.settings.music_api_key}"} if self.settings.music_api_key else {},
            timeout=timeout, transport=self.transport, trust_env=False, follow_redirects=False,
        )

    def readiness(self):
        result = {"status": "not_ready", "backend": "sglang-omni", "model": self.settings.music_model}
        if not self.settings.music_base_url:
            return {**result, "reason": "Configure H3_MUSIC_BASE_URL; see docs/MUSIC.md"}
        try:
            with self._client(3) as client:
                response = client.get("v1/models")
                response.raise_for_status()
                models = response.json().get("data", [])
                if any(item.get("id") == self.settings.music_model for item in models):
                    return {**result, "status": "ready"}
                return {**result, "reason": "Configured Music 3 model is not served"}
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return {**result, "reason": "Music service unavailable or invalid model catalog"}

    def generate(self, request, output, cancel, progress):
        self.validate_request(request)
        if cancel.is_set():
            raise GenerationCanceled("music task canceled")
        payload = {
            "model": self.settings.music_model, "input": request.lyrics,
            "instructions": request.prompt, "seed": request.seed,
            "max_new_tokens": request.duration_seconds * 25,
            "response_format": "wav", "stream": False,
        }
        # Bound downloaded PCM data; five minutes of stereo 32 kHz 16-bit audio is ~39 MB.
        limit = (request.duration_seconds + 2) * 32000 * 4 + 1024 * 1024
        try:
            with self._client(httpx.Timeout(self.settings.music_timeout_seconds, connect=10)) as client:
                with client.stream("POST", "v1/audio/speech", json=payload) as response:
                    if response.status_code != 200:
                        raise BackendError(f"Music service returned HTTP {response.status_code}", code="music_upstream_error")
                    output.path.parent.mkdir(parents=True, exist_ok=True)
                    size = 0
                    with output.path.open("wb") as target:
                        for chunk in response.iter_bytes():
                            if cancel.is_set():
                                raise GenerationCanceled("music task canceled")
                            size += len(chunk)
                            if size > limit:
                                raise ValueError("Music response exceeds the requested audio size limit")
                            target.write(chunk)
            with wave.open(str(output.path), "rb") as audio:
                frames = audio.getnframes()
                if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) != (2, 32000, 2) or frames <= 0:
                    raise ValueError("Music service must return non-empty stereo 32 kHz 16-bit PCM WAV")
                if frames > (request.duration_seconds + 2) * 32000 or len(audio.readframes(frames)) != frames * 4:
                    raise ValueError("Music service returned truncated or overlong WAV audio")
            if cancel.is_set():
                raise GenerationCanceled("music task canceled")
            progress(0.99)
        except Exception:
            output.path.unlink(missing_ok=True)
            raise

    def release_models(self):
        # External service owns its model lifecycle; this adapter cannot unload it.
        pass

    def close(self):
        pass
