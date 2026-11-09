from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .native import COMFY_CORE_COMMIT
from .upscale import SEEDVR2_COMMIT


COMFY_REPOSITORY = "https://github.com/comfy-org/ComfyUI.git"
SEEDVR2_REPOSITORY = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
MODEL_REPOSITORY = "Comfy-Org/MiniMax-H3"
UPSCALE_MODEL_REPOSITORY = "numz/SeedVR2_comfyUI"
MODEL_LICENSE = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
BASE_MODEL_FILES = (
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
)
REF2V_MODEL_FILE = "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FL2VA_ACCELERATION_LORA_FILE = (
    "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
)
REF2VA_ACCELERATION_LORA_FILE = (
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
)
UPSCALE_MODEL_FILES = (
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "ema_vae_fp16.safetensors",
)


def _run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def install_runtime(runtime_path: Path) -> None:
    expected = runtime_path / "comfy" / "ldm" / "minimax" / "model.py"
    if expected.is_file():
        head_file = runtime_path / ".git" / "HEAD"
        installed_commit = head_file.read_text(encoding="ascii").strip() if head_file.is_file() else ""
        if not re.fullmatch(r"[0-9a-f]{40}", installed_commit):
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=runtime_path,
                check=True,
                capture_output=True,
                text=True,
            )
            installed_commit = result.stdout.strip()
        if installed_commit != COMFY_CORE_COMMIT:
            raise RuntimeError(
                f"runtime at {runtime_path} is not the pinned commit {COMFY_CORE_COMMIT}; "
                "move it aside and rerun setup"
            )
        print(f"Runtime already installed: {runtime_path}")
        return
    if runtime_path.exists():
        raise RuntimeError(
            f"runtime path exists but is incomplete: {runtime_path}; move it aside and rerun setup"
        )

    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = runtime_path.with_name(f".{runtime_path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        _run("git", "clone", "--filter=blob:none", "--no-checkout", COMFY_REPOSITORY, str(temporary))
        _run("git", "sparse-checkout", "set", "comfy", cwd=temporary)
        _run("git", "checkout", COMFY_CORE_COMMIT, cwd=temporary)
        if not (temporary / "node_helpers.py").is_file() or not (
            temporary / "comfy" / "sd.py"
        ).is_file():
            raise RuntimeError("the sparse Comfy inference runtime is incomplete")
        temporary.replace(runtime_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Installed pinned inference runtime: {runtime_path}")


def install_upscale_runtime(runtime_path: Path) -> None:
    expected = runtime_path / "inference_cli.py"
    if expected.is_file():
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=runtime_path,
            check=True,
            capture_output=True,
            text=True,
        )
        installed_commit = result.stdout.strip()
        if installed_commit != SEEDVR2_COMMIT:
            raise RuntimeError(
                f"upscale runtime at {runtime_path} is not the pinned commit "
                f"{SEEDVR2_COMMIT}; move it aside and rerun setup"
            )
        print(f"Upscale runtime already installed: {runtime_path}")
        return
    if runtime_path.exists():
        raise RuntimeError(
            f"upscale runtime path exists but is incomplete: {runtime_path}; "
            "move it aside and rerun setup"
        )

    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = runtime_path.with_name(f".{runtime_path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        _run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            SEEDVR2_REPOSITORY,
            str(temporary),
        )
        _run("git", "checkout", SEEDVR2_COMMIT, cwd=temporary)
        if not (temporary / "inference_cli.py").is_file() or not (
            temporary / "src" / "core" / "generation_utils.py"
        ).is_file():
            raise RuntimeError("the SeedVR2 inference runtime is incomplete")
        temporary.replace(runtime_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Installed pinned SeedVR2 runtime: {runtime_path}")


def download_models(
    model_root: Path,
    *,
    with_ref2v: bool,
    with_acceleration_lora: bool,
    token: str | None,
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is missing; run `uv sync --extra local` first") from exc

    files = list(BASE_MODEL_FILES)
    if with_ref2v:
        files.append(REF2V_MODEL_FILE)
    if with_acceleration_lora:
        files.append(FL2VA_ACCELERATION_LORA_FILE)
        if with_ref2v:
            files.append(REF2VA_ACCELERATION_LORA_FILE)
    model_root.mkdir(parents=True, exist_ok=True)
    for filename in files:
        destination = model_root / filename
        if destination.is_file():
            print(f"Model already present: {destination}")
            continue
        print(f"Downloading {filename} ...")
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=filename,
            local_dir=model_root,
            token=token,
        )


def download_upscale_models(model_root: Path, *, token: str | None) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is missing; run `uv sync --extra local` first") from exc

    model_root.mkdir(parents=True, exist_ok=True)
    for filename in UPSCALE_MODEL_FILES:
        destination = model_root / filename
        if destination.is_file():
            print(f"Upscale model already present: {destination}")
            continue
        print(f"Downloading SeedVR2 {filename} ...")
        hf_hub_download(
            repo_id=UPSCALE_MODEL_REPOSITORY,
            filename=filename,
            local_dir=model_root,
            token=token,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install the standalone local MiniMax H3 runtime")
    parser.add_argument("--runtime-path", type=Path, default=Path(".runtime/ComfyUI"))
    parser.add_argument(
        "--upscale-runtime-path",
        type=Path,
        default=Path(".runtime/SeedVR2"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--upscale-model-dir",
        type=Path,
        default=Path("models/SEEDVR2"),
    )
    parser.add_argument("--with-ref2v", action="store_true", help="also download the 21 GB Ref2V model")
    parser.add_argument(
        "--without-acceleration-lora",
        action="store_false",
        dest="with_acceleration_lora",
        default=True,
        help="skip the Turbo LoRAs downloaded by default",
    )
    parser.add_argument(
        "--without-upscale",
        action="store_false",
        dest="with_upscale",
        default=True,
        help="skip the SeedVR2 runtime and image upscale weights",
    )
    parser.add_argument("--runtime-only", action="store_true", help="install code without model weights")
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="confirm acceptance of the MiniMax H3 Community License",
    )
    parser.add_argument("--hf-token", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime_path = args.runtime_path.expanduser().resolve()
    upscale_runtime_path = args.upscale_runtime_path.expanduser().resolve()
    model_root = args.model_dir.expanduser().resolve()
    upscale_model_root = args.upscale_model_dir.expanduser().resolve()
    install_runtime(runtime_path)
    if args.with_upscale:
        install_upscale_runtime(upscale_runtime_path)
    if args.runtime_only:
        return
    if not args.accept_license:
        print(f"Read the model license first: {MODEL_LICENSE}", file=sys.stderr)
        print(
            "Rerun with --accept-license to download approximately 45 GB of weights.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    download_models(
        model_root,
        with_ref2v=args.with_ref2v,
        with_acceleration_lora=args.with_acceleration_lora,
        token=args.hf_token,
    )
    if args.with_upscale:
        download_upscale_models(upscale_model_root, token=args.hf_token)
    print("Local MiniMax H3 and image upscale runtimes are installed.")


if __name__ == "__main__":
    main()
