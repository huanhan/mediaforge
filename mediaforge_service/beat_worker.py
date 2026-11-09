"""Runs independently of H3/Comfy imports. All time values reference decoded audio start."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, stft


SAMPLE_RATE = 22050


def clean_times(values, duration):
    return sorted({round(float(value), 6) for value in values
                   if np.isfinite(value) and 0 <= value < duration})


def summarize(signal, beats, downbeats, sample_rate=SAMPLE_RATE):
    """Energy and transient candidates are measurements, not beat confidence scores."""
    duration = len(signal) / sample_rate
    beats = clean_times(beats, duration)
    downbeats = clean_times(downbeats, duration)
    intervals = [{"start": a, "end": b, "bpm": round(60 / (b - a), 3)}
                 for a, b in zip(beats, beats[1:]) if b - a > 0.05]
    # The model's 20 ms grid makes single-interval BPM oscillate on steady music.
    # Summarize spans of up to eight beats, retaining every original beat for cuts.
    span = min(8, len(beats)-1)
    tempos = [60 * span / (beats[i+span] - beats[i]) for i in range(len(beats)-span)] if span > 0 else []
    block = round(sample_rate * 0.05)
    energy = []
    for offset in range(0, len(signal), block):
        samples = signal[offset:offset + block]
        energy.append({"time": round(offset / sample_rate, 6),
                       "rms": round(float(np.sqrt(np.mean(samples ** 2))), 6),
                       "peak": round(float(np.max(np.abs(samples))), 6)})
    hop = 220
    _, times, spectrum = stft(signal, fs=sample_rate, nperseg=1024,
                              noverlap=1024-hop, boundary="zeros", padded=True)
    magnitude = np.log1p(10 * np.abs(spectrum))
    flux = np.maximum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]), 0).mean(axis=0)
    scale = max(float(np.percentile(flux, 95)), float(np.max(flux)) * .1)
    onsets = []
    if scale > 1e-8:
        strength = np.clip(flux / scale, 0, 1)
        indices, _ = find_peaks(strength, height=0.25, prominence=0.12,
                                distance=max(1, round(0.08 * sample_rate / hop)))
        onsets = [{"time": round(float(times[index]), 6), "strength": round(float(strength[index]), 4)}
                  for index in indices if times[index] < duration]
    warnings = []
    if len(beats) < 2:
        warnings.append("No reliable beat sequence found; do not build a regular cut grid from this result.")
    warnings.append("Model-estimated beats require listening review. Downbeats are bar starts, not necessarily the loudest sounds. Onset strength is relative, not confidence.")
    return {
        "duration_seconds": round(duration, 6), "sample_rate": sample_rate,
        "time_origin": "decoded_audio_start", "beats_seconds": beats, "downbeats_seconds": downbeats,
        "tempo_bpm": round(float(np.median(tempos)), 3) if tempos else None,
        "tempo_intervals": intervals, "energy": energy, "onsets": onsets, "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        raise ValueError("Local Beat This! checkpoint is missing; run mediaforge-setup-beat")
    signal = np.fromfile(args.pcm, dtype="<f4")
    if len(signal) < SAMPLE_RATE or not np.all(np.isfinite(signal)):
        raise ValueError("Audio is empty, too short or contains non-finite samples")
    # Silence must not be fed into a model that may hallucinate a regular beat grid.
    if float(np.max(np.abs(signal))) < 1e-5:
        beats, downbeats = [], []
    else:
        import torch
        from beat_this.inference import Audio2Beats
        torch.set_num_threads(args.threads)
        with torch.inference_mode():
            predictor = Audio2Beats(checkpoint_path=str(args.model.resolve()), device=args.device, dbn=False)
            beats, downbeats = predictor(signal, SAMPLE_RATE)
    result = {"schema_version": 1, "backend": "beat_this", "model": args.model.name,
              "source_file_id": args.file_id, "title": args.title,
              **summarize(signal, beats, downbeats)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
