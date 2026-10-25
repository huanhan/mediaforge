"""Runs in the isolated TTS interpreter, which releases all CUDA memory on exit."""
import argparse
import json
from pathlib import Path

def inference_options(request):
    mode = request.get("emotion_mode", "text")
    return dict(
        spk_audio_prompt=request["reference_audio"], text=request["text"],
        emo_audio_prompt=request.get("emotion_reference_audio") if mode == "reference" else None,
        emo_alpha=float(request.get("emotion_strength", 0.65)),
        emo_vector=request.get("emotion_vector") if mode == "vector" else None,
        use_emo_text=mode == "text", emo_text=request.get("emotion_text"),
        use_random=False, duration_factor=1 / request.get("speed", 1),
        interval_silence=request.get("pause_ms", 200),
        max_text_tokens_per_segment=request.get("max_segment_tokens", 120),
        temperature=float(request.get("temperature", 0.8)), top_p=float(request.get("top_p", 0.8)),
        top_k=request.get("top_k", 30), repetition_penalty=float(request.get("repetition_penalty", 10)),
        text_normalization=request.get("text_normalization", True),
        max_mel_tokens=1500, verbose=False,
    )

def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--request")
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    parser.add_argument("--progress")
    args = parser.parse_args()
    import os
    import random
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    import torch
    import numpy as np
    import soundfile as sf
    from indextts.infer_v2_5 import IndexTTS2
    if args.check:
        available = args.device == "cpu" or torch.cuda.is_available()
        print(json.dumps({"status": "ready" if available else "not_ready", "reason": "" if available else "CUDA unavailable"}))
        return
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    torch.manual_seed(request["seed"])
    np.random.seed(request["seed"])
    random.seed(request["seed"])
    progress = lambda value: Path(args.progress).write_text(json.dumps({"progress": value}), encoding="utf-8")
    progress(.08)
    model = IndexTTS2(cfg_path=str(Path(args.model_dir) / "config.yaml"), model_dir=args.model_dir,
                     device=args.device, use_bf16=args.device != "cpu", use_cuda_kernel=False,
                     use_deepspeed=False, use_accel=False, use_torch_compile=False,
                     use_qwen_emo=request.get("emotion_mode", "text") == "text")
    progress(.20)
    languages = {"Chinese": "ZH", "English": "EN", "Japanese": "JA", "Spanish": "ES", "Arabic": "AR"}
    language = request["language"]
    if language == "Auto":
        import langid
        langid.set_languages(["zh", "en", "ja", "es", "ar"])
        lang = langid.classify(request["text"])[0].upper()
    else:
        lang = languages[language]
    model.gr_progress = lambda value, desc=None: progress(.20 + .72 * max(0, min(1, float(value))))
    with torch.inference_mode():
        model.infer(**inference_options(request), output_path=args.output, lang=lang)
    wave, sr = sf.read(args.output, dtype="float32")
    if not np.isfinite(wave).all() or wave.size == 0 or len(wave) / sr > 600 or np.max(np.abs(wave)) < 0.0001:
        raise RuntimeError("IndexTTS returned empty, silent or invalid audio")
    progress(.95)

if __name__ == "__main__":
    main()
