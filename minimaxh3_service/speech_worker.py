"""Isolated, offline Breeze worker. Eager execution fits the 12 GB deployment target."""
import argparse
import json
import re
from pathlib import Path

MAX_AUDIO_FRAMES = 1500

def inference_options(request):
    options = {"id": "shot", "speaker": "S0", "text": request["text"],
               "ref_audio_path": request["reference_audio"], "ref_text": request["reference_text"]}
    if request.get("mode", "clone") == "direction":
        options["instruction"] = request["instruction"]
    return options

def split_text(text, limit=200):
    # Preserve every non-whitespace character and keep inline events intact.
    segments = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        while len(paragraph) > limit:
            boundaries = [m.end() for m in re.finditer(r"[。！？.!?；;，,\s]", paragraph[:limit])]
            end = boundaries[-1] if boundaries else limit
            # Never cut through an inline [event] / (event).
            for opening, closing in (("[", "]"), ("(", ")")):
                start = paragraph.rfind(opening, 0, end)
                if start >= 0 and paragraph.find(closing, start) >= end:
                    end = start or paragraph.find(closing, start) + 1
            segments.append(paragraph[:end].strip())
            paragraph = paragraph[end:].strip()
        if paragraph:
            segments.append(paragraph)
    return [s for s in segments if s]

def checked_audio_chunks(runtime, inputs, *, request_id, seed, max_steps=MAX_AUDIO_FRAMES):
    generated_steps = 0
    frames = 0
    def observe_token(_):
        nonlocal generated_steps
        generated_steps += 1
    for chunk in runtime.iter_audio_chunks(inputs, request_id=request_id, seed=seed,
                                           token_observer=observe_token):
        frames += chunk.codec_frames
        yield chunk
    # Reserved/invalid codec frames are skipped by the decoder, but still count
    # toward the model's generation limit. The runner discards partial output.
    if generated_steps >= max_steps:
        raise RuntimeError("Breeze reached the audio limit; split the shot instead of using truncated speech")
    if frames == 0:
        raise RuntimeError("Breeze returned no speech for a text segment")

def main():
    import os
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--request")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--progress")
    args = parser.parse_args()
    # Upstream calls torch.cuda.set_device(), which requires an explicit index.
    if args.device == "cuda":
        args.device = "cuda:0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import numpy as np
    import soundfile as sf
    # H3 also has models.py. Upstream's absolute `models.*` imports must resolve
    # inside its isolated source checkout, not beside this worker script.
    from importlib.util import find_spec
    spec = find_spec("breeze_infer")
    if spec is None or spec.origin is None:
        raise RuntimeError("Breeze source is not installed; run setup_tts")
    sys.path.insert(0, str(Path(spec.origin).resolve().parent.parent))
    from breeze_infer.runtime import load_runtime, set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
    if args.check:
        available = args.device == "cpu" or (torch.cuda.is_available() and
            int(args.device.split(":")[1]) < torch.cuda.device_count())
        print(json.dumps({"status": "ready" if available else "not_ready", "reason": "" if available else "CUDA unavailable"}))
        return
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    progress = lambda value: Path(args.progress).write_text(json.dumps({"progress": value}), encoding="utf-8")
    progress(.05)
    tokenizer, model, audio_tokenizer = load_runtime(Path(args.model_dir), device=args.device, attn_implementation="eager")
    update_generation_config_for_breeze(model)
    config = FastStreamingConfig(max_new_tokens=MAX_AUDIO_FRAMES, max_seq_len=4096, fast_all=False,
        temperature=float(request["temperature"]), top_p=float(request["top_p"]),
        top_k=request["top_k"], repetition_penalty=float(request["repetition_penalty"]))
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)
    segments = split_text(request["text"])
    cfg = float(request["cfg_scale"]) if request.get("mode") == "direction" else 1.0
    progress(.15)
    with torch.inference_mode(), sf.SoundFile(args.output, mode="w", samplerate=runtime.sample_rate,
                                             channels=1, subtype="PCM_16") as output:
        for index, text in enumerate(segments):
            seed = (request["seed"] + index) % 2147483648
            set_all_seeds(seed)
            item = inference_options({**request, "text": text})
            inputs = prepare_inputs(tokenizer, audio_tokenizer, model, [item], get_template(select_template_name(item)),
                                    guidance_scale=cfg, guidance_scale_ref=None, guidance_scale_ins=None)
            if inputs["input_ids"].shape[1] + MAX_AUDIO_FRAMES >= config.max_seq_len:
                raise ValueError("Reference/text context too long; shorten the reference or split this shot")
            frames = 0
            for chunk in checked_audio_chunks(runtime, inputs, request_id=f"shot-{index}", seed=seed):
                if not np.isfinite(chunk.audio).all():
                    raise RuntimeError("Breeze returned invalid audio")
                frames += chunk.codec_frames
                output.write(chunk.audio)
                progress(.15 + .75 * (index + min(.95, frames / MAX_AUDIO_FRAMES)) / len(segments))
            if index + 1 < len(segments):
                output.write(np.zeros(round(runtime.sample_rate * request["pause_ms"] / 1000), dtype=np.float32))
    wave, sr = sf.read(args.output, dtype="float32")
    if not np.isfinite(wave).all() or wave.size == 0 or len(wave) / sr > 600 or np.max(np.abs(wave)) < .0001:
        raise RuntimeError("Breeze returned empty, silent or invalid audio")
    progress(.95)

if __name__ == "__main__":
    main()
