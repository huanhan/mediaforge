"""Offline ACE-Step inference through the existing Comfy core (no Comfy server)."""
import gc
import importlib.util
import wave

from .backend import BackendError
from .tasks import GenerationCanceled


class NativeMusicRunner:
    supports_interrupt = True

    def __init__(self, settings, core_provider):
        self.settings = settings
        self.core_provider = core_provider
        self._core = None
        self._model = self._clip = self._vae = None

    def readiness(self):
        from .setup_music import SIZE
        path = self.settings.music_model_path.expanduser().resolve()
        missing = []
        if not path.is_file() or path.stat().st_size != SIZE:
            missing.append("ACE-Step checkpoint: run python -m minimaxh3_service.setup_music")
        if not (self.settings.runtime_path / "comfy" / "sd.py").is_file():
            missing.append("Comfy inference core")
        for name in ("torch", "torchaudio", "sentencepiece", "safetensors"):
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        return {"status": "not_ready" if missing else "ready", "backend": "native-ace-step",
                "model": "ACE-Step-v1-3.5B", "models_loaded": self._model is not None,
                "instrumental": True, "lyrics": True, "sample_rate": 44100,
                "max_duration_seconds": 120, "supports_interrupt": True, "missing": missing}

    def validate_request(self, request):
        if request.duration_seconds > 120:
            raise BackendError("本地 ACE-Step 当前支持 10–120 秒，请缩短时长", status_code=422, code="music_duration_limit")

    def generate(self, request, output, cancel, progress):
        self.validate_request(request)
        def check_cancel():
            if cancel.is_set():
                raise GenerationCanceled("music task canceled")
        check_cancel()
        core = self.core_provider._import_core(check_readiness=False)
        self._core = core
        torch = core.torch
        try:
            if self._model is None:
                progress(0.03)
                self._model, self._clip, self._vae, *_ = core.sd.load_checkpoint_guess_config(
                    str(self.settings.music_model_path.expanduser().resolve()),
                    output_vae=True, output_clip=True, embedding_directory=[],
                )
                self._vae.throw_exception_if_invalid()
            check_cancel()
            with torch.inference_mode():
                # Empty lyrics select the model's instrumental path.
                lyrics = request.lyrics or "[instrumental]"
                tokens = self._clip.tokenize(request.prompt, lyrics=lyrics)
                positive = self._clip.encode_from_tokens_scheduled(tokens)
                positive = [[tensor, {**meta, "lyrics_strength": 1.0}] for tensor, meta in positive]
                negative = [[torch.zeros_like(tensor), {**meta, "lyrics_strength": 0.0}] for tensor, meta in positive]
                length = int(request.duration_seconds * 44100 / 4096)
                latent = torch.zeros((1, 8, 16, length), device="cpu")
                noise = core.sample.prepare_noise(latent, request.seed)
                check_cancel()
                progress(0.12)
                def callback(step, x0, x, total):
                    check_cancel()
                    progress(0.12 + 0.70 * (step + 1) / total)
                samples = core.sample.sample(self._model, noise, 50, 5.0, "euler", "simple",
                                             positive, negative, latent, callback=callback,
                                             disable_pbar=True, seed=request.seed)
                check_cancel()
                progress(0.85)
                decoded = self._vae.decode(samples)
                # Comfy VAE returns batch/time/channels for audio as for images.
                audio = decoded[0].float().cpu()
                if audio.ndim != 2:
                    raise ValueError(f"Invalid ACE-Step audio shape: {tuple(audio.shape)}")
                if audio.shape[-1] != 2 and audio.shape[0] == 2:
                    audio = audio.transpose(0, 1)
                if audio.shape[-1] != 2 or not torch.isfinite(audio).all() or audio.numel() == 0:
                    raise ValueError("ACE-Step returned invalid stereo audio")
                pcm = (audio.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().astype("<i2").tobytes()
                check_cancel()
                output.path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output.path), "wb") as wav:
                    wav.setnchannels(2)
                    wav.setsampwidth(2)
                    wav.setframerate(44100)
                    wav.writeframes(pcm)
                progress(0.99)
        except Exception:
            output.path.unlink(missing_ok=True)
            self.release_models()
            raise

    def release_models(self):
        if self._core is None:
            return
        self._core.model_management.unload_all_models()
        self._model = self._clip = self._vae = None
        gc.collect()
        self._core.model_management.soft_empty_cache()

    def close(self):
        self.release_models()
