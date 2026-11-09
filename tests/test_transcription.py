import asyncio
import json
import shutil
import sys
import threading
import time
import wave

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from mediaforge_service.api import create_app
from mediaforge_service.backend import BackendError, OutputArtifact
from mediaforge_service.config import Settings
from mediaforge_service.local_backend import LocalGenerationBackend
from mediaforge_service.models import BeatAnalysisRequest, TranscriptionRequest
from mediaforge_service.tasks import GenerationCanceled
from mediaforge_service.transcription import TranscriptionRunner


class IdleRunner:
    def readiness(self):
        return {"status": "ready"}

    def release_models(self):
        pass

    def close(self):
        pass


class FakeTranscription(IdleRunner):
    def process(self, request, output, cancel, progress):
        text = "风吹过，灯还亮着。"
        output.path.write_text(json.dumps({"source_file_id": request.file_id, "title": request.title,
                                           "text": text}, ensure_ascii=False), encoding="utf-8")
        text_path = output.path.with_suffix(".txt")
        text_path.write_text(text, encoding="utf-8")
        return [output, OutputArtifact(text_path.name, output.subfolder, text_path)]


def backend_for(tmp_path, **settings):
    idle = IdleRunner()
    return LocalGenerationBackend(Settings(data_root=tmp_path, **settings), runner=idle, upscale_runner=idle,
                                  removal_runner=idle, music_runner=idle, depth_runner=idle,
                                  skeleton_runner=idle, beat_runner=idle, transcription_runner=FakeTranscription())


def test_upload_transcribe_poll_download_and_auth(tmp_path):
    backend = backend_for(tmp_path)
    with TestClient(create_app(Settings(api_key="test-key"), backend_factory=lambda _: backend)) as client:
        assert client.post("/v1/audio/transcriptions", json={"file_id": "x.wav"}).status_code == 401
        client.headers["X-API-Key"] = "test-key"
        upload = client.post("/v1/files", data={"kind": "audio"}, files={"file": ("song.m4a", b"audio", "audio/mp4")})
        file_id = upload.json()["file_id"]
        response = client.post("/v1/audio/transcriptions", json={"file_id": file_id, "title": "测试歌曲"})
        assert response.status_code == 202
        client.portal.call(backend.tasks.queue.join)
        task = client.get("/v1/tasks/" + response.json()["id"]).json()
        assert task["status"] == "succeeded" and task["kind"] == "transcription"
        assert len(task["outputs"]) == 2
        artifact = client.get(task["outputs"][0]["download_url"])
        assert artifact.headers["content-type"].startswith("application/json")
        assert artifact.json() == {"source_file_id": file_id, "title": "测试歌曲", "text": "风吹过，灯还亮着。"}
        assert client.get(task["outputs"][1]["download_url"]).text == artifact.json()["text"]
        assert client.get("/v1/models").json()["capabilities"]["transcription"]["available"] is True


@pytest.mark.parametrize("body", [{"file_id": "../x.wav"}, {"file_id": "https://x/a.wav"},
                                  {"file_id": "a.wav", "lyrics": "unsupported"},
                                  {"file_id": "a.wav", "chunk_seconds": 181},
                                  {"file_id": "a.wav", "chunk_seconds": 0},
                                  {"file_id": "a.wav", "language": "unsupported"}])
def test_invalid_requests_do_not_claim_lyric_alignment(body):
    with pytest.raises(ValidationError):
        TranscriptionRequest(**body)


def test_submission_checks_media_and_readiness(tmp_path):
    async def scenario():
        backend = backend_for(tmp_path)
        backend.assets.initialize()
        for filename, code in (("missing.wav", 404), ("image.png", 415), ("song.wav", 503)):
            if filename != "missing.wav":
                (backend.assets.input_root / filename).touch()
            backend.tasks.transcription_runner.readiness = lambda: {"status": "not_ready"}
            with pytest.raises(BackendError) as caught:
                await backend.submit_transcription(TranscriptionRequest(file_id=filename))
            assert caught.value.status_code == code
        assert not backend.tasks.records
    asyncio.run(scenario())


def test_shared_queue_releases_previous_runner_before_transcription(tmp_path):
    async def scenario():
        backend = backend_for(tmp_path)
        backend.assets.initialize()
        events = []
        backend.tasks.beat_runner.process = lambda *args: events.append("beat")
        backend.tasks.beat_runner.release_models = lambda: events.append("release")
        original = backend.tasks.transcription_runner.process

        def transcribe(*args):
            assert events[:2] == ["beat", "release"]
            events.append("transcribe")
            return original(*args)

        backend.tasks.transcription_runner.process = transcribe
        await backend.tasks.submit_beat_analysis(BeatAnalysisRequest(file_id="song.wav"))
        task = await backend.tasks.submit_transcription(TranscriptionRequest(file_id="song.wav"))
        await backend.start()
        await backend.tasks.queue.join()
        assert backend.tasks.get(task.id).status == "succeeded"
        assert events[:3] == ["beat", "release", "transcribe"]
        await backend.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_after_output", [False, True])
def test_cancellation_never_publishes_either_output(tmp_path, cancel_after_output):
    async def scenario():
        backend = backend_for(tmp_path, allow_interrupt_running=True)
        backend.assets.initialize()
        original = backend.tasks.transcription_runner.process

        def transcribe(request, output, cancel, progress):
            result = original(request, output, cancel, progress)
            cancel.set()  # Covers cancellation between runner completion and queue publication.
            return result

        backend.tasks.transcription_runner.process = transcribe
        task = await backend.tasks.submit_transcription(TranscriptionRequest(file_id="song.wav"))
        if not cancel_after_output:
            assert backend.tasks.cancel(task.id).status == "canceled"
        await backend.start()
        await backend.tasks.queue.join()
        assert backend.tasks.get(task.id).status == "canceled"
        assert backend.tasks.get(task.id).outputs == []
        assert not list(backend.assets.output_root.rglob("*.json"))
        assert not list(backend.assets.output_root.rglob("*.txt"))
        await backend.close()
    asyncio.run(scenario())


def test_runner_cancels_and_times_out_subprocesses(tmp_path):
    runner = TranscriptionRunner(Settings(), lambda _: tmp_path)
    cancel = threading.Event()
    timer = threading.Timer(.3, cancel.set)
    timer.start()
    try:
        with pytest.raises(GenerationCanceled):
            runner._run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "cancel.log", cancel, time.monotonic()+10)
    finally:
        timer.cancel()
    with pytest.raises(TimeoutError):
        runner._run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "timeout.log", threading.Event(), time.monotonic()+.1)


def test_chunking_keeps_silence_and_all_samples_without_claiming_word_timing():
    np = pytest.importorskip("numpy")
    from mediaforge_service.transcription_worker import assemble_result
    signal = np.zeros(16000*61+17, dtype=np.float32)
    signal[16000*30:] = .2
    observed_lengths = []

    def transcribe(window):
        observed_lengths.append(len(window))
        return "风吹过，灯还亮着。", "Chinese", True

    result = assemble_result(TranscriptionRequest(file_id="song.wav", chunk_seconds=30), signal, transcribe)
    assert observed_lengths == [16000*30, 16000+17]
    assert [chunk["start"] for chunk in result["chunks"]] == [0, 30, 60]
    assert result["chunks"][-1]["end"] == len(signal)/16000
    assert result["chunks"][0]["text"] == ""
    assert result["text"] == "风吹过，灯还亮着。\n风吹过，灯还亮着。"
    assert result["timing"] == "chunk_boundaries_only"
    assert any("token limit" in warning for warning in result["warnings"])
    assert any("boundary" in warning for warning in result["warnings"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_real_decode_silence_and_duration_limit_cleanup(tmp_path):
    pytest.importorskip("numpy")
    source = tmp_path / "silence.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 2)
    output = OutputArtifact("result.json", "", tmp_path / "result.json")
    request = TranscriptionRequest(file_id="silence.wav")
    runner = TranscriptionRunner(Settings(asr_model_path=tmp_path / "nonexistent-model"), lambda _: source)
    artifacts = runner.process(request, output, threading.Event(), lambda _: None)
    result = json.loads(output.path.read_text())
    assert result["duration_seconds"] == 2 and result["text"] == ""
    assert len(artifacts) == 2 and artifacts[1].path.is_file()
    limited = TranscriptionRunner(Settings(asr_max_seconds=1), lambda _: source)
    with pytest.raises(ValueError, match="exceeds"):
        limited.process(request, output, threading.Event(), lambda _: None)
    assert not output.path.exists() and not artifacts[1].path.exists()
    assert not list(tmp_path.glob("*.f32")) and not list(tmp_path.glob("*.partial.json"))


def test_failure_cleans_intermediate_files(tmp_path):
    runner = TranscriptionRunner(Settings(), lambda _: tmp_path / "bad.wav")

    def fail(arguments, log, cancel, deadline):
        (tmp_path / "result.f32").touch()
        (tmp_path / "result.partial.json").write_text("{}")
        raise RuntimeError("decode failed")

    runner._run = fail
    with pytest.raises(RuntimeError, match="decode failed"):
        runner.process(TranscriptionRequest(file_id="bad.wav"), OutputArtifact("result.json", "", tmp_path / "result.json"),
                       threading.Event(), lambda _: None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("settings", [{"asr_device": "mps"}, {"asr_max_seconds": float("nan")},
                                      {"asr_timeout_seconds": -1}, {"asr_cpu_threads": 0}])
def test_invalid_runtime_settings(settings):
    with pytest.raises(ValueError, match="H3_ASR_"):
        Settings(**settings)
