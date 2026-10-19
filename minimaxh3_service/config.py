from __future__ import annotations

import os
import math
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
    upscale_idle_release_seconds: float = 5.0
    music_base_url: str = ""
    music_backend: str = "native"
    music_model_path: Path = Path("models/music/ace_step_v1_3.5b.safetensors")
    music_api_key: str | None = None
    music_model: str = "MiniMaxAI/MiniMax-Music3"
    music_timeout_seconds: float = 1800.0

    runtime_path: Path = Path(".runtime/ComfyUI")
    upscale_runtime_path: Path = Path(".runtime/SeedVR2")
    model_root: Path = Path("models")
    upscale_model_root: Path = Path("models/SEEDVR2")
    data_root: Path = Path("data")
    device: str = "cuda"
    depth_runtime_path: Path = Path(".runtime/Video-Depth-Anything")
    depth_model_path: Path = Path("models/depth/video_depth_anything_vits.pth")
    depth_max_frames: int = 720
    pose_runtime_path: Path = Path(".runtime/DWPose")
    pose_model_path: Path = Path("models/pose")
    pose_max_frames: int = 240
    pose_timeout_seconds: float = 3600.0
    beat_model_path: Path = Path("models/beat/final0.ckpt")
    beat_device: str = "cpu"
    beat_max_seconds: float = 900.0
    beat_timeout_seconds: float = 600.0
    beat_cpu_threads: int = 4
    asr_model_path: Path = Path("models/asr/Qwen3-ASR-1.7B")
    asr_device: str = "cuda"
    asr_max_seconds: float = 900.0
    asr_timeout_seconds: float = 1800.0
    asr_cpu_threads: int = 4
    reserve_vram_gb: float = 1.0
    vram_mode: str = "dynamic"
    enable_acceleration_lora: bool = True
    eager_chunk_rows: int = 2048
    mlp_chunk_rows: int = 0
    eager_memory_fraction: float = 0.7

    fl2va_model: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    ref2va_model: str = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    fl2va_acceleration_lora: str = (
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    )
    ref2va_acceleration_lora: str = (
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    )
    text_encoder: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    video_vae: str = "minimax_h3_video_vae_fp16.safetensors"
    audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors"

    upscale_dit_model: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    upscale_vae_model: str = "ema_vae_fp16.safetensors"
    upscale_blocks_to_swap: int = 16
    upscale_vae_tile_size: int = 512
    upscale_swap_io: bool = True
    upscale_seed: int = 42

    def __post_init__(self) -> None:
        if self.asr_device != "cpu" and self.asr_device != "cuda" and not (
            self.asr_device.startswith("cuda:") and self.asr_device[5:].isdigit()
        ):
            raise ValueError("H3_ASR_DEVICE must be cpu or cuda[:index]")
        if not 1 <= self.asr_cpu_threads <= 64:
            raise ValueError("H3_ASR_CPU_THREADS must be between 1 and 64")
        for name, value in (("MAX_SECONDS", self.asr_max_seconds), ("TIMEOUT_SECONDS", self.asr_timeout_seconds)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"H3_ASR_{name} must be finite and positive")
        if self.beat_device != "cpu" and self.beat_device != "cuda" and not (
            self.beat_device.startswith("cuda:") and self.beat_device[5:].isdigit()
        ):
            raise ValueError("H3_BEAT_DEVICE must be cpu or cuda[:index]")
        if not 1 <= self.beat_cpu_threads <= 64:
            raise ValueError("H3_BEAT_CPU_THREADS must be between 1 and 64")
        for name, value in (("MAX_SECONDS", self.beat_max_seconds), ("TIMEOUT_SECONDS", self.beat_timeout_seconds)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"H3_BEAT_{name} must be finite and positive")
        if self.pose_max_frames < 2:
            raise ValueError("H3_POSE_MAX_FRAMES must be at least 2")
        if not math.isfinite(self.pose_timeout_seconds) or self.pose_timeout_seconds <= 0:
            raise ValueError("H3_POSE_TIMEOUT_SECONDS must be finite and positive")
        if self.depth_max_frames < 1:
            raise ValueError("H3_DEPTH_MAX_FRAMES must be positive")
        if self.music_backend not in {"native", "remote"}:
            raise ValueError("H3_MUSIC_BACKEND must be native or remote")
        if self.music_timeout_seconds <= 0 or not math.isfinite(self.music_timeout_seconds):
            raise ValueError("H3_MUSIC_TIMEOUT_SECONDS must be finite and positive")
        if self.music_base_url:
            from urllib.parse import urlsplit
            url = urlsplit(self.music_base_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("H3_MUSIC_BASE_URL must be an HTTP(S) service URL without credentials, query or fragment")
        if self.eager_chunk_rows < 0:
            raise ValueError("H3_EAGER_CHUNK_ROWS must be >= 0 (0 disables the policy)")
        if self.mlp_chunk_rows < 0:
            raise ValueError("H3_MLP_CHUNK_ROWS must be >= 0 (0 disables MLP streaming)")
        if not 0 < self.eager_memory_fraction <= 1:
            raise ValueError("H3_EAGER_MEMORY_FRACTION must be in (0, 1]")

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        max_upload_mb = int(os.getenv("H3_MAX_UPLOAD_MB", "2048"))
        return cls(
            asr_model_path=Path(os.getenv("H3_ASR_MODEL_PATH", str(defaults.asr_model_path))),
            asr_device=os.getenv("H3_ASR_DEVICE", defaults.asr_device),
            asr_max_seconds=float(os.getenv("H3_ASR_MAX_SECONDS", str(defaults.asr_max_seconds))),
            asr_timeout_seconds=float(os.getenv("H3_ASR_TIMEOUT_SECONDS", str(defaults.asr_timeout_seconds))),
            asr_cpu_threads=int(os.getenv("H3_ASR_CPU_THREADS", str(defaults.asr_cpu_threads))),
            beat_model_path=Path(os.getenv("H3_BEAT_MODEL_PATH", str(defaults.beat_model_path))),
            beat_device=os.getenv("H3_BEAT_DEVICE", defaults.beat_device),
            beat_max_seconds=float(os.getenv("H3_BEAT_MAX_SECONDS", str(defaults.beat_max_seconds))),
            beat_timeout_seconds=float(os.getenv("H3_BEAT_TIMEOUT_SECONDS", str(defaults.beat_timeout_seconds))),
            beat_cpu_threads=int(os.getenv("H3_BEAT_CPU_THREADS", str(defaults.beat_cpu_threads))),
            pose_runtime_path=Path(os.getenv("H3_POSE_RUNTIME_PATH", str(defaults.pose_runtime_path))),
            pose_model_path=Path(os.getenv("H3_POSE_MODEL_PATH", str(defaults.pose_model_path))),
            pose_max_frames=int(os.getenv("H3_POSE_MAX_FRAMES", str(defaults.pose_max_frames))),
            pose_timeout_seconds=float(os.getenv("H3_POSE_TIMEOUT_SECONDS", str(defaults.pose_timeout_seconds))),
            music_backend=os.getenv("H3_MUSIC_BACKEND", defaults.music_backend),
            music_model_path=Path(os.getenv("H3_MUSIC_MODEL_PATH", str(defaults.music_model_path))),
            music_base_url=os.getenv("H3_MUSIC_BASE_URL", "").rstrip("/"),
            music_api_key=os.getenv("H3_MUSIC_API_KEY") or None,
            music_model=os.getenv("H3_MUSIC_MODEL", defaults.music_model),
            music_timeout_seconds=float(os.getenv("H3_MUSIC_TIMEOUT_SECONDS", str(defaults.music_timeout_seconds))),
            host=os.getenv("H3_HOST", defaults.host),
            port=int(os.getenv("H3_PORT", str(defaults.port))),
            api_key=os.getenv("H3_API_KEY") or None,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            allow_interrupt_running=_env_bool("H3_ALLOW_INTERRUPT_RUNNING"),
            max_queue_size=int(os.getenv("H3_MAX_QUEUE_SIZE", str(defaults.max_queue_size))),
            idle_release_seconds=float(
                os.getenv("H3_IDLE_RELEASE_SECONDS", str(defaults.idle_release_seconds))
            ),
            upscale_idle_release_seconds=float(
                os.getenv(
                    "H3_UPSCALE_IDLE_RELEASE_SECONDS",
                    str(defaults.upscale_idle_release_seconds),
                )
            ),
            runtime_path=Path(os.getenv("H3_COMFY_CORE_PATH", str(defaults.runtime_path))),
            upscale_runtime_path=Path(
                os.getenv("H3_UPSCALE_RUNTIME_PATH", str(defaults.upscale_runtime_path))
            ),
            model_root=Path(os.getenv("H3_MODEL_DIR", str(defaults.model_root))),
            upscale_model_root=Path(
                os.getenv("H3_UPSCALE_MODEL_DIR", str(defaults.upscale_model_root))
            ),
            data_root=Path(os.getenv("H3_DATA_DIR", str(defaults.data_root))),
            device=os.getenv("H3_DEVICE", defaults.device),
            depth_runtime_path=Path(os.getenv("H3_DEPTH_RUNTIME_PATH", str(defaults.depth_runtime_path))),
            depth_model_path=Path(os.getenv("H3_DEPTH_MODEL_PATH", str(defaults.depth_model_path))),
            depth_max_frames=int(os.getenv("H3_DEPTH_MAX_FRAMES", str(defaults.depth_max_frames))),
            reserve_vram_gb=float(os.getenv("H3_RESERVE_VRAM_GB", str(defaults.reserve_vram_gb))),
            vram_mode=os.getenv("H3_VRAM_MODE", defaults.vram_mode).lower(),
            eager_chunk_rows=int(os.getenv("H3_EAGER_CHUNK_ROWS", str(defaults.eager_chunk_rows))),
            mlp_chunk_rows=int(os.getenv("H3_MLP_CHUNK_ROWS", str(defaults.mlp_chunk_rows))),
            eager_memory_fraction=float(os.getenv("H3_EAGER_MEMORY_FRACTION", str(defaults.eager_memory_fraction))),
            enable_acceleration_lora=_env_bool(
                "H3_ENABLE_ACCELERATION_LORA", defaults.enable_acceleration_lora
            ),
            fl2va_model=os.getenv("H3_FL2VA_MODEL", defaults.fl2va_model),
            ref2va_model=os.getenv("H3_REF2VA_MODEL", defaults.ref2va_model),
            fl2va_acceleration_lora=os.getenv(
                "H3_FL2VA_ACCELERATION_LORA", defaults.fl2va_acceleration_lora
            ),
            ref2va_acceleration_lora=os.getenv(
                "H3_REF2VA_ACCELERATION_LORA", defaults.ref2va_acceleration_lora
            ),
            text_encoder=os.getenv("H3_TEXT_ENCODER", defaults.text_encoder),
            video_vae=os.getenv("H3_VIDEO_VAE", defaults.video_vae),
            audio_vae=os.getenv("H3_AUDIO_VAE", defaults.audio_vae),
            upscale_dit_model=os.getenv(
                "H3_UPSCALE_DIT_MODEL", defaults.upscale_dit_model
            ),
            upscale_vae_model=os.getenv(
                "H3_UPSCALE_VAE_MODEL", defaults.upscale_vae_model
            ),
            upscale_blocks_to_swap=int(
                os.getenv(
                    "H3_UPSCALE_BLOCKS_TO_SWAP",
                    str(defaults.upscale_blocks_to_swap),
                )
            ),
            upscale_vae_tile_size=int(
                os.getenv(
                    "H3_UPSCALE_VAE_TILE_SIZE",
                    str(defaults.upscale_vae_tile_size),
                )
            ),
            upscale_swap_io=_env_bool(
                "H3_UPSCALE_SWAP_IO", defaults.upscale_swap_io
            ),
            upscale_seed=int(os.getenv("H3_UPSCALE_SEED", str(defaults.upscale_seed))),
        )

    @property
    def model_names(self) -> dict[str, str]:
        return {
            "fl2va": self.fl2va_model,
            "ref2va": self.ref2va_model,
            "fl2va_acceleration_lora": self.fl2va_acceleration_lora,
            "ref2va_acceleration_lora": self.ref2va_acceleration_lora,
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
            "fl2va_acceleration_lora": root / "loras" / self.fl2va_acceleration_lora,
            "ref2va_acceleration_lora": root / "loras" / self.ref2va_acceleration_lora,
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

    @property
    def upscale_model_paths(self) -> dict[str, Path]:
        root = self.upscale_model_root.expanduser().resolve()
        return {
            "dit": root / self.upscale_dit_model,
            "vae": root / self.upscale_vae_model,
        }
