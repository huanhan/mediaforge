import argparse
from bisect import bisect_left
from difflib import SequenceMatcher
import gc
import json
import math
from pathlib import Path
import time

import numpy as np

from .models import AlignmentRequest, TranscriptionRequest
from .setup_asr import ALIGNER_ID
from .transcription_worker import assemble_result, create_transcriber


SAMPLE_RATE = 16000
ALIGNMENT_WINDOW_SECONDS = 180
TRANSCRIPTION_WINDOW_SECONDS = 120
WINDOW_PADDING_SECONDS = 30


def alignment_windows(tokens, chunks, tokenize, duration):
    """ASR locates lyric sections only; all final word times come from the aligner.

    Match the entire transcript monotonically to disambiguate repeated verses.
    Overlapping audio windows allow a lyric/word to cross a recognition boundary.
    Never substitute recognized words for the confirmed lyrics.
    """
    observed, offsets = [], [0]
    for chunk in chunks:
        if chunk["token_limit_reached"]:
            raise ValueError("Long-audio section recognition was truncated; check the lyrics or retry")
        observed.extend(word.casefold() for word in tokenize(chunk["text"]))
        offsets.append(len(observed))
    blocks = SequenceMatcher(None, [t.casefold() for t in tokens], observed, autojunk=False).get_matching_blocks()
    pairs = [(block.b+i, block.a+i) for block in blocks for i in range(block.size)]
    observed_indices = [pair[0] for pair in pairs]
    if not pairs:
        raise ValueError("Cannot locate confirmed lyrics in long audio; check the lyrics and language")
    for start, end in zip(offsets, offsets[1:]):
        matches = bisect_left(observed_indices, end) - bisect_left(observed_indices, start)
        if end > start and matches < max(min(3, end-start), (end-start)*.25):
            raise ValueError("Cannot reliably locate a long-audio lyric section; check the confirmed lyrics")
    boundaries = [0]
    for offset in offsets[1:-1]:
        if not offset:
            boundaries.append(0)
            continue
        index = bisect_left(observed_indices, offset)
        boundaries.append(pairs[index][1] if index < len(pairs) else len(tokens))
    boundaries.append(len(tokens))
    windows = []
    for chunk, start, end in zip(chunks, boundaries, boundaries[1:]):
        if end <= start:
            continue  # Instrumental windows have no confirmed words to align.
        windows.append({"start": max(0, chunk["start"]-WINDOW_PADDING_SECONDS),
                        "end": min(duration, chunk["end"]+WINDOW_PADDING_SECONDS),
                        "tokens": tokens[start:end]})
    return windows


def align_windows(audio, windows, align):
    result = []
    for window in windows:
        start, end = round(window["start"]*SAMPLE_RATE), round(window["end"]*SAMPLE_RATE)
        # Include the lyrics audible in the leading overlap. Without this context
        # the aligner can force the next verse onto the preceding verse's audio.
        # Context is aligned again, but only new words are published once.
        context = [word["text"] for word in result if word["end_time"] > start/SAMPLE_RATE]
        tokens = context + window["tokens"]
        words = align(audio[start:end], " ".join(tokens))
        if [word["text"] for word in words] != tokens:
            raise ValueError("Alignment tokens do not match the confirmed lyrics")
        for word in words[len(context):]:
            a, b = float(word["start_time"]), float(word["end_time"])
            if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < a or b > (end-start)/SAMPLE_RATE+.16:
                raise ValueError("Invalid alignment timestamps inside an audio window")
            result.append({"text": word["text"], "start_time": a+start/SAMPLE_RATE,
                           "end_time": b+start/SAMPLE_RATE})
    return result


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


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("pcm", "output", "request", "model"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    started = time.monotonic()
    request = AlignmentRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    audio = np.fromfile(args.pcm, dtype="<f4")
    duration = len(audio)/SAMPLE_RATE
    if duration < 1 or not np.all(np.isfinite(audio)) or np.max(np.abs(audio)) < 1e-5:
        raise ValueError("Alignment requires at least one second of non-silent audio")
    import torch
    from transformers import AutoProcessor, Qwen3ASRForTokenClassification
    torch.set_num_threads(args.threads)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    lines = [line.strip() for line in request.lyrics.splitlines() if line.strip()]
    tokens_by_line = [processor.split_words_for_alignment(line, request.language) for line in lines]
    if any(not tokens for tokens in tokens_by_line):
        raise ValueError("Each lyric line must contain words")
    tokens = [token for line in tokens_by_line for token in line]
    warnings = ["Singing, long held notes and mismatched lyrics can affect alignment; review the result."]
    if duration <= ALIGNMENT_WINDOW_SECONDS:
        windows = [{"start": 0, "end": duration, "tokens": tokens}]
    else:
        if args.asr_model is None:
            raise ValueError("Long-audio alignment requires the local ASR model (--asr-model)")
        transcribe = create_transcriber(args.asr_model, args.device, args.threads, request.language)
        sections = assemble_result(TranscriptionRequest(file_id=request.file_id, language=request.language,
                                   chunk_seconds=TRANSCRIPTION_WINDOW_SECONDS), audio, transcribe)
        windows = alignment_windows(tokens, sections["chunks"],
                                    lambda text: processor.split_words_for_alignment(text, request.language), duration)
        # Do not hold the ASR and alignment models on the GPU together.
        del transcribe
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        warnings.append("Long audio was aligned in overlapping windows located using ASR text. Confirmed lyrics are preserved; review section boundaries.")
    model = Qwen3ASRForTokenClassification.from_pretrained(str(args.model), dtype=dtype,
                device_map=args.device, local_files_only=True, attn_implementation="sdpa").eval()

    def align(window, transcript):
        with torch.inference_mode():
            inputs, word_lists = processor.prepare_forced_aligner_inputs(audio=window, transcript=transcript, language=request.language)
            inputs = inputs.to(model.device, model.dtype)
            logits = model(**inputs).logits
            return processor.decode_forced_alignment(logits, inputs["input_ids"], word_lists,
                        timestamp_token_id=model.config.timestamp_token_id)[0]

    aligned = align_windows(audio, windows, align)
    cues, words = make_cues(lines, tokens_by_line, aligned, duration)
    revision = args.model / "REVISION"
    result = {"schema_version": 1, "backend": "qwen3_forced_aligner", "model": ALIGNER_ID,
              "model_revision": revision.read_text().strip() if revision.is_file() else None,
              "source_file_id": request.file_id, "text": request.lyrics, "timing": "forced_alignment",
              "duration_seconds": duration, "cues": cues, "words": words,
              "warnings": warnings,
              "elapsed_seconds": round(time.monotonic()-started, 3)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
