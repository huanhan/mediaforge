"""Real decoding beyond the former caps, with offline model results."""
import json
import shutil
import subprocess
import threading
import wave

import pytest

from minimaxh3_service.alignment import AlignmentRunner
from minimaxh3_service.backend import OutputArtifact
from minimaxh3_service.beat import BeatAnalysisRunner
from minimaxh3_service.config import Settings
from minimaxh3_service.models import AlignmentRequest, BeatAnalysisRequest, TranscriptionRequest
from minimaxh3_service.transcription import TranscriptionRunner


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
@pytest.mark.parametrize("runner_type,input_request,rate", [
    (BeatAnalysisRunner, BeatAnalysisRequest(file_id="long.wav"), 22050),
    (TranscriptionRunner, TranscriptionRequest(file_id="long.wav"), 16000),
    (AlignmentRunner, AlignmentRequest(file_id="long.wav", lyrics="confirmed words"), 16000),
])
def test_long_audio_is_decoded_in_full_by_default(tmp_path, runner_type, input_request, rate):
    duration = 901.25
    source = tmp_path / "long.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 1, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x80" * int(8000*duration))
    runner = runner_type(Settings(), lambda _: source)
    output = OutputArtifact("result.json", "", tmp_path / "result.json")
    observed = []

    def run(arguments, log, cancel, deadline):
        if arguments[0] == "ffmpeg":
            assert "-t" not in arguments, "Default decoding must not truncate music"
            subprocess.run(arguments, check=True, capture_output=True)
            return
        pcm = output.path.with_suffix(".f32")
        actual = pcm.stat().st_size / (4*rate)
        observed.append(actual)
        result = {"schema_version": 1, "source_file_id": input_request.file_id, "duration_seconds": actual,
                  "text": getattr(input_request, "lyrics", ""),
                  "timing": "forced_alignment" if runner_type is AlignmentRunner else "chunk_boundaries_only",
                  "cues": [{"id": "late", "start": actual-2, "end": actual-1, "text": "confirmed words"}]}
        output.path.with_suffix(".partial.json").write_text(json.dumps(result), encoding="utf-8")

    runner._run = run
    runner.process(input_request, output, threading.Event(), lambda _: None)
    assert observed == pytest.approx([duration], abs=1/rate)
    assert json.loads(output.path.read_text())["duration_seconds"] == pytest.approx(duration, abs=1/rate)
    assert runner.readiness()["max_duration_seconds"] is None
    assert not list(tmp_path.glob("*.f32")) and not list(tmp_path.glob("*.partial.json"))


@pytest.mark.parametrize("key", ["asr", "beat"])
def test_zero_disables_duration_policy_without_disabling_task_deadlines(monkeypatch, key):
    monkeypatch.setenv(f"H3_{key.upper()}_MAX_SECONDS", "0")
    assert getattr(Settings.from_env(), key+"_max_seconds") == 0
    for value in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            Settings(**{key+"_max_seconds": value})
    with pytest.raises(ValueError):
        Settings(**{key+"_timeout_seconds": 0})
