from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    allow_interrupt_running: bool = False
    max_queue_size: int = 8
    idle_release_seconds: float = 5.0

    runtime_path: Path = Path(".runtime/ComfyUI")
    model_root: Path = Path("models")
    data_root: Path = Path("data")
    device: str = "cuda"
    reserve_vram_gb: float = 1.0
    vram_mode: str = "dynamic"

    fl2va_model: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    ref2va_model: str = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    text_encoder: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    video_vae: str = "minimax_h3_video_vae_fp16.safetensors"
    audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors"

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        max_upload_mb = int(os.getenv("H3_MAX_UPLOAD_MB", "2048"))
        return cls(
            host=os.getenv("H3_HOST", defaults.host),
            port=int(os.getenv("H3_PORT", str(defaults.port))),
            api_key=os.getenv("H3_API_KEY") or None,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            allow_interrupt_running=_env_bool("H3_ALLOW_INTERRUPT_RUNNING"),
            max_queue_size=int(os.getenv("H3_MAX_QUEUE_SIZE", str(defaults.max_queue_size))),
            idle_release_seconds=float(
                os.getenv("H3_IDLE_RELEASE_SECONDS", str(defaults.idle_release_seconds))
            ),
            runtime_path=Path(os.getenv("H3_COMFY_CORE_PATH", str(defaults.runtime_path))),
            model_root=Path(os.getenv("H3_MODEL_DIR", str(defaults.model_root))),
            data_root=Path(os.getenv("H3_DATA_DIR", str(defaults.data_root))),
            device=os.getenv("H3_DEVICE", defaults.device),
            reserve_vram_gb=float(os.getenv("H3_RESERVE_VRAM_GB", str(defaults.reserve_vram_gb))),
            vram_mode=os.getenv("H3_VRAM_MODE", defaults.vram_mode).lower(),
            fl2va_model=os.getenv("H3_FL2VA_MODEL", defaults.fl2va_model),
            ref2va_model=os.getenv("H3_REF2VA_MODEL", defaults.ref2va_model),
            text_encoder=os.getenv("H3_TEXT_ENCODER", defaults.text_encoder),
            video_vae=os.getenv("H3_VIDEO_VAE", defaults.video_vae),
            audio_vae=os.getenv("H3_AUDIO_VAE", defaults.audio_vae),
        )

    @property
    def model_names(self) -> dict[str, str]:
        return {
            "fl2va": self.fl2va_model,
            "ref2va": self.ref2va_model,
            "text_encoder": self.text_encoder,
            "video_vae": self.video_vae,
            "audio_vae": self.audio_vae,
        }

    @property
    def model_paths(self) -> dict[str, Path]:
        root = self.model_root.expanduser().resolve()
        return {
            "fl2va": root / "diffusion_models" / self.fl2va_model,
            "ref2va": root / "diffusion_models" / self.ref2va_model,
            "text_encoder": root / "text_encoders" / self.text_encoder,
            "video_vae": root / "vae" / self.video_vae,
            "audio_vae": root / "vae" / self.audio_vae,
        }

    @property
    def input_root(self) -> Path:
        return self.data_root.expanduser().resolve() / "input"

    @property
    def output_root(self) -> Path:
        return self.data_root.expanduser().resolve() / "output"
