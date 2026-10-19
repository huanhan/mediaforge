"""Qwen3-ASR transcription. No fabricated word or sentence timestamps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .setup_asr import MODEL_ID

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 2048


def chunk_ranges(sample_count, chunk_seconds, sample_rate=SAMPLE_RATE):
    size = int(chunk_seconds * sample_rate)
    if size <= 0:
        raise ValueError("chunk_seconds must be positive")
    return [(start, min(start+size, sample_count)) for start in range(0, sample_count, size)]


def assemble_result(request, signal, transcribe, model_revision=None):
    if len(signal) < SAMPLE_RATE or not np.all(np.isfinite(signal)):
        raise ValueError("Audio is too short or contains non-finite samples")
    chunks = []
    warnings = ["Automatic transcription may contain lyric errors. Chunk start/end values describe input windows, not lyric timestamps. No forced alignment was performed."]
    for start, end in chunk_ranges(len(signal), request.chunk_seconds):
        window = signal[start:end]
        if float(np.max(np.abs(window))) < 1e-5:
            text, language, truncated = "", None, False
        else:
            text, language, truncated = transcribe(window)
        chunks.append({"start": start/SAMPLE_RATE, "end": end/SAMPLE_RATE,
                       "text": text, "language": language, "token_limit_reached": truncated})
        if truncated:
            warnings.append(f"Input window {start/SAMPLE_RATE:g}-{end/SAMPLE_RATE:g}s reached the output token limit; transcription may be incomplete.")
    if len(chunks) > 1:
        warnings.append("Long audio was split into non-overlapping input windows. Lyrics crossing a boundary may need correction.")
    text = "\n".join(chunk["text"].strip() for chunk in chunks if chunk["text"].strip())
    if not text:
        warnings.append("No lyrics or speech were recognized.")
    return {"schema_version": 1, "backend": "qwen3_asr", "model": MODEL_ID,
            "model_revision": model_revision, "source_file_id": request.file_id, "title": request.title,
            "duration_seconds": len(signal)/SAMPLE_RATE, "sample_rate": SAMPLE_RATE,
            "language_hint": request.language, "timing": "chunk_boundaries_only", "text": text,
            "chunks": chunks, "warnings": warnings}


def create_transcriber(model_path, device, threads, language):
    """Lazily load ASR; discarding the callback releases its model references."""
    model = processor = None

    def transcribe(window):
        nonlocal model, processor
        import torch
        if model is None:
            from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
            torch.set_num_threads(threads)
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; install CUDA PyTorch or set H3_ASR_DEVICE=cpu")
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
            model = Qwen3ASRForConditionalGeneration.from_pretrained(
                str(model_path), dtype=dtype, device_map=device, local_files_only=True,
                attn_implementation="sdpa").eval()
        with torch.inference_mode():
            inputs = processor.apply_transcription_request(
                audio=window, language=None if language == "auto" else language,
            ).to(model.device, model.dtype)
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            # Decode one sequence explicitly; return format differs for batch/single inputs.
            parsed = processor.decode(generated[0], return_format="parsed")
            eos = model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            truncated = generated.shape[1] >= MAX_NEW_TOKENS and int(generated[0, -1]) not in eos_ids
            return parsed["transcription"], parsed.get("language"), truncated
    return transcribe


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--language", choices=["auto", "zh", "en", "yue", "ja", "ko"], default="zh")
    parser.add_argument("--chunk-seconds", type=int, choices=range(30, 181), default=180)
    args = parser.parse_args(argv)
    started = time.monotonic()
    signal = np.fromfile(args.pcm, dtype="<f4")
    transcribe = create_transcriber(args.model, args.device, args.threads, args.language)

    revision = args.model / "REVISION"
    result = assemble_result(args, signal, transcribe, revision.read_text().strip() if revision.is_file() else None)
    result["elapsed_seconds"] = round(time.monotonic()-started, 3)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
