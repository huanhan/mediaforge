"""Runs in the isolated TTS interpreter, which releases all CUDA memory on exit."""
import argparse
import json
from pathlib import Path
import re

def chunks(text, limit=100):
    parts = []
    current = ""
    for sentence in re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]*|[。！？!?；;\n]+", text):
        while len(sentence) > limit:
            if current:
                parts.append(current); current = ""
            split = max(sentence.rfind("，", 0, limit), sentence.rfind(",", 0, limit), sentence.rfind(" ", 0, limit))
            cut = split + 1 if split > limit // 2 else limit
            parts.append(sentence[:cut]); sentence = sentence[cut:]
        if len(current) + len(sentence) > limit and current:
            parts.append(current); current = ""
        current += sentence
    if current.strip():
        parts.append(current)
    return parts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--request")
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    parser.add_argument("--progress")
    args = parser.parse_args()
    import torch
    import numpy as np
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    if args.check:
        available = args.device == "cpu" or torch.cuda.is_available()
        print(json.dumps({"status": "ready" if available else "not_ready", "reason": "" if available else "CUDA unavailable"}))
        return
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    torch.manual_seed(request["seed"])
    progress = lambda value: Path(args.progress).write_text(json.dumps({"progress": value}), encoding="utf-8")
    progress(.08)
    model = Qwen3TTSModel.from_pretrained(args.model_dir, device_map=args.device, local_files_only=True,
        dtype=torch.float32 if args.device == "cpu" else torch.bfloat16, attn_implementation="sdpa")
    progress(.20)
    clone = None
    if request.get("reference_audio"):
        clone = model.create_voice_clone_prompt(ref_audio=request["reference_audio"],
                    ref_text=request["reference_text"], x_vector_only_mode=False)
    pieces = chunks(request["text"])
    waves = []
    with torch.inference_mode():
        for index, text in enumerate(pieces):
            common = dict(text=text, language=request["language"], max_new_tokens=2048)
            if clone is not None:
                audio, sr = model.generate_voice_clone(**common, voice_clone_prompt=clone)
            else:
                audio, sr = model.generate_custom_voice(**common, speaker=request["speaker"], instruct=request["instruct"])
            wave = np.asarray(audio[0], dtype=np.float32).reshape(-1)
            # Hitting the generation budget can silently cut a sentence. Do not publish it.
            if not np.isfinite(wave).all() or len(wave) == 0 or len(wave) / sr >= 160:
                raise RuntimeError("Speech was empty, invalid, or hit the generation limit; shorten this shot's text")
            waves.append(wave)
            if index < len(pieces) - 1:
                waves.append(np.zeros(round(sr * .12), dtype=np.float32))
            progress(.20 + .72 * (index + 1) / len(pieces))
    sf.write(args.output, np.concatenate(waves), sr, subtype="PCM_16")

if __name__ == "__main__":
    main()
