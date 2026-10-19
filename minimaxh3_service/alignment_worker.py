import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .models import AlignmentRequest
from .setup_asr import ALIGNER_ID


def make_cues(lines, tokens_by_line, aligned, duration):
    expected = [token for tokens in tokens_by_line for token in tokens]
    if expected != [word["text"] for word in aligned]:
        raise ValueError("Alignment tokens do not match the confirmed lyrics")
    words = []
    previous = 0
    for item in aligned:
        start, end = float(item["start_time"]), float(item["end_time"])
        if not math.isfinite(start) or not math.isfinite(end) or start < previous-.081 or end < start or end > duration+.16:
            raise ValueError("Invalid alignment timestamps; check the supplied lyrics")
        start, end = max(previous, min(start, duration)), min(end, duration)
        if end < start:
            raise ValueError("Overlapping alignment timestamps")
        words.append({"text": item["text"], "start": round(start, 3), "end": round(end, 3)})
        previous = end
    cues, offset = [], 0
    for line, tokens in zip(lines, tokens_by_line):
        if not tokens:
            raise ValueError("Each lyric line must contain words")
        selected = words[offset:offset+len(tokens)]
        start, end = selected[0]["start"], selected[-1]["end"]
        if end <= start:
            raise ValueError("A lyric line could not be aligned; check the supplied lyrics")
        cues.append({"id": f"line_{len(cues)+1}", "start": start, "end": end, "text": line})
        offset += len(tokens)
    return cues, words


def main():
    parser = argparse.ArgumentParser()
    for name in ("pcm", "output", "request", "model"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    request = AlignmentRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    audio = np.fromfile(args.pcm, dtype="<f4")
    duration = len(audio)/16000
    if not 1 <= duration <= 180 or not np.all(np.isfinite(audio)) or np.max(np.abs(audio)) < 1e-5:
        raise ValueError("Alignment requires 1–180 seconds of non-silent audio")
    import torch
    from transformers import AutoProcessor, Qwen3ASRForTokenClassification
    torch.set_num_threads(args.threads)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = Qwen3ASRForTokenClassification.from_pretrained(str(args.model), dtype=dtype,
                device_map=args.device, local_files_only=True, attn_implementation="sdpa").eval()
    lines = [line.strip() for line in request.lyrics.splitlines() if line.strip()]
    tokens_by_line = [processor.split_words_for_alignment(line, request.language) for line in lines]
    with torch.inference_mode():
        inputs, word_lists = processor.prepare_forced_aligner_inputs(audio=audio, transcript="\n".join(lines), language=request.language)
        inputs = inputs.to(model.device, model.dtype)
        logits = model(**inputs).logits
        aligned = processor.decode_forced_alignment(logits, inputs["input_ids"], word_lists,
                    timestamp_token_id=model.config.timestamp_token_id)[0]
    cues, words = make_cues(lines, tokens_by_line, aligned, duration)
    revision = args.model / "REVISION"
    result = {"schema_version": 1, "backend": "qwen3_forced_aligner", "model": ALIGNER_ID,
              "model_revision": revision.read_text().strip() if revision.is_file() else None,
              "source_file_id": request.file_id, "text": request.lyrics, "timing": "forced_alignment",
              "duration_seconds": duration, "cues": cues, "words": words,
              "warnings": ["Singing, long held notes and mismatched lyrics can affect alignment; review the result."],
              "elapsed_seconds": round(time.monotonic()-started, 3)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
