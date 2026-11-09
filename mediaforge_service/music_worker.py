"""Standalone entry point executed by the isolated YuE2 interpreter, not H3 Python."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time


def check_runtime(device):
    import torch
    from yue2 import YuE2Pipeline  # noqa: F401

    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "yue2-infer")}
    expected = {"torch": "2.10.0", "transformers": "4.57.6", "yue2-infer": "0.1.6"}
    result = dict(status="ready", versions=versions)
    if any(versions[name].split("+")[0] != value for name, value in expected.items()):
        return {**result, "status": "not_ready", "reason": "Run setup_music to install the pinned YuE2 runtime"}
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return {**result, "status": "not_ready", "reason": "YuE2 requires a CUDA-enabled PyTorch and NVIDIA GPU"}
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                return {**result, "status": "not_ready", "reason": "YuE2 requires CUDA BF16 support"}
            props = torch.cuda.get_device_properties(device)
            result.update(gpu=props.name, vram_gib=round(props.total_memory / 2**30, 2))
            if props.total_memory < 24 * 2**30:
                result["warning"] = "Below the upstream 24 GB recommendation; long requests may run out of memory"
    return result


def generate(args):
    import numpy as np
    import soundfile as sf
    from yue2 import YuE2Pipeline
    # This file is also launched directly, outside the service package.
    if __package__:
        from .music_memory import yue2_memory_scope
    else:
        from music_memory import yue2_memory_scope

    request = json.loads(args.request.read_text(encoding="utf-8"))
    max_tokens = request["duration_seconds"] * 25  # VAE: 1 semantic frame = 1920 / 48000 s.
    sampling = {"max_tokens": max_tokens, "min_tokens": min(200, max_tokens)}

    def report(value):
        if args.progress is not None:
            temporary = args.progress.with_suffix(".tmp")
            temporary.write_text(json.dumps({"progress": value}), encoding="utf-8")
            try:
                temporary.replace(args.progress)
            except PermissionError:
                # Windows readers can briefly hold a handle without delete sharing.
                pass

    started = time.monotonic()
    report(.03)
    with yue2_memory_scope(on_progress=lambda done, total: report(.72 + .16 * done / total)) as memory_policy, \
         YuE2Pipeline.from_pretrained(str(args.model_dir / "YuE2-3B"),
            vae=str(args.model_dir / "YuE2-Vae"), local_files_only=True, device=args.device,
            memory_budget_gib=args.memory_budget_gib, offload_ar=args.offload_ar,
            backend="torch-eager", progress=False) as pipe:
        report(.08)
        counts = {"abc": 0, "semantic": 0}

        def on_token(phase, token):
            counts[phase] += 1
            if counts[phase] % 25 == 0:
                base, span, limit = (.08, .22, 4096) if phase == "abc" else (.30, .40, max_tokens)
                report(base + span * min(counts[phase] / limit, 1))

        style = request["prompt"]
        if not request["lyrics"]:
            style += ", instrumental, no vocals"
        plan = pipe.plan(style=style, lyrics=request["lyrics"], seed=request["seed"],
                         cot=request["cot"], abc=request["abc"], cfg_scale=request["cfg_scale"], on_token=on_token)
        report(.30)
        semantic = pipe.generate_semantic(plan, sampling=sampling, on_token=on_token)
        report(.72)
        latents = pipe.synthesize(semantic)
        report(.88)
        audio = pipe.decode(latents)
        if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError("YuE2 returned invalid audio samples")
        maximum_frames = request["duration_seconds"] * 48000
        cropped = len(audio) > maximum_frames
        audio = np.clip(audio[:maximum_frames], -1, 1)
        sf.write(str(args.output), audio, 48000, subtype="PCM_16")
        metadata = dict(schema_version=1, model="m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae",
                        sample_rate=48000, duration_seconds=len(audio) / 48000,
                        truncated=dict(abc=bool(plan.truncated), semantic=bool(semantic.truncated), audio_cropped=cropped),
                        request=request, elapsed_seconds=round(time.monotonic() - started, 3),
                        config=pipe.effective_config(plan.request, semantic_sampling=sampling), weights=pipe.weights)
        metadata["config"]["memory_policy"] = memory_policy
        args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if plan.abc:
            args.output.with_suffix(".abc").write_text(plan.abc, encoding="utf-8")
        report(.98)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--memory-budget-gib", type=float, default=12)
    parser.add_argument("--offload-ar", action="store_true")
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(check_runtime(args.device)))
    else:
        if args.request is None or args.output is None or args.model_dir is None:
            parser.error("--request, --output and --model-dir are required")
        generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
