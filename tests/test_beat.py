import asyncio
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import wave

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from minimaxh3_service.api import create_app
from minimaxh3_service.backend import BackendError, OutputArtifact
from minimaxh3_service.beat import BeatAnalysisRunner
from minimaxh3_service.config import Settings
from minimaxh3_service.local_backend import LocalGenerationBackend
from minimaxh3_service.models import BeatAnalysisRequest
from minimaxh3_service.tasks import GenerationCanceled


class IdleRunner:
    def readiness(self):
        return {"status": "ready"}

    def release_models(self):
        pass

    def close(self):
        pass


class FakeBeat(IdleRunner):
    def process(self, request, output, cancel, progress):
        output.path.write_text(json.dumps({"source_file_id": request.file_id, "title": request.title,
                                           "beats_seconds": [1.25, 1.75]}), encoding="utf-8")


def backend_for(tmp_path):
    idle = IdleRunner()
    return LocalGenerationBackend(Settings(data_root=tmp_path), runner=idle, upscale_runner=idle,
                                  removal_runner=idle, music_runner=idle, depth_runner=idle,
                                  skeleton_runner=idle, beat_runner=FakeBeat())


def test_upload_analyze_poll_download_and_auth(tmp_path):
    backend = backend_for(tmp_path)
    with TestClient(create_app(Settings(api_key="test-key"), backend_factory=lambda _: backend)) as client:
        assert client.post("/v1/music/analyses", json={"file_id": "x.wav"}).status_code == 401
        client.headers["X-API-Key"] = "test-key"
        upload = client.post("/v1/files", data={"kind": "audio"}, files={"file": ("song.m4a", b"audio", "audio/mp4")})
        file_id = upload.json()["file_id"]
        response = client.post("/v1/music/analyses", json={"file_id": file_id, "title": "测试歌曲"})
        assert response.status_code == 202
        client.portal.call(backend.tasks.queue.join)
        task = client.get("/v1/tasks/" + response.json()["id"]).json()
        assert task["status"] == "succeeded" and task["kind"] == "beat_analysis"
        artifact = client.get(task["outputs"][0]["download_url"])
        assert artifact.headers["content-type"].startswith("application/json")
        assert artifact.json() == {"source_file_id": file_id, "title": "测试歌曲", "beats_seconds": [1.25, 1.75]}
        assert client.get("/v1/models").json()["capabilities"]["beat_analysis"]["available"] is True


@pytest.mark.parametrize("body", [{"file_id": "../x.wav"}, {"file_id": "https://x/a.wav"},
                                  {"file_id": "a.wav", "lyrics": "not supported"}])
def test_invalid_requests_do_not_claim_lyric_alignment(body):
    with pytest.raises(ValidationError):
        BeatAnalysisRequest(**body)


def test_submission_checks_media_and_readiness(tmp_path):
    async def scenario():
        backend = backend_for(tmp_path)
        backend.assets.initialize()
        for filename, code in (("missing.wav", 404), ("image.png", 415), ("song.wav", 503)):
            if filename != "missing.wav":
                (backend.assets.input_root / filename).touch()
            backend.tasks.beat_runner.readiness = lambda: {"status": "not_ready"}
            with pytest.raises(BackendError) as caught:
                await backend.submit_beat_analysis(BeatAnalysisRequest(file_id=filename))
            assert caught.value.status_code == code
        assert not backend.tasks.records
    asyncio.run(scenario())


def test_queued_cancellation_never_publishes_output(tmp_path):
    async def scenario():
        backend = backend_for(tmp_path)
        backend.assets.initialize()
        task = await backend.tasks.submit_beat_analysis(BeatAnalysisRequest(file_id="song.wav"))
        assert backend.tasks.cancel(task.id).status == "canceled"
        await backend.start()
        await backend.tasks.queue.join()
        assert backend.tasks.get(task.id).outputs == []
        await backend.close()
    asyncio.run(scenario())


def test_runner_cancels_and_times_out_subprocesses(tmp_path):
    runner = BeatAnalysisRunner(Settings(), lambda _: tmp_path)
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


def test_timing_preserves_leading_silence_and_reports_variable_intervals():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from minimaxh3_service.beat_worker import summarize
    signal = np.zeros(22050*4, dtype=np.float32)
    signal[22050:22050+220] = .5
    result = summarize(signal, [1, 1.5, 2, 2.75, float("nan")], [1])
    assert result["beats_seconds"] == [1, 1.5, 2, 2.75]
    assert [v["bpm"] for v in result["tempo_intervals"]] == [120, 120, 80]
    assert all(v["rms"] == 0 for v in result["energy"] if v["time"] < .95)
    assert result["onsets"] and abs(result["onsets"][0]["time"]-1) < .05


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_real_decode_silence_and_duration_limit_cleanup(tmp_path):
    pytest.importorskip("scipy")
    source = tmp_path / "silence.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 2)
    model = tmp_path / "model.ckpt"
    model.touch()  # Silent audio does not load a checkpoint or hallucinate beats.
    output = OutputArtifact("result.json", "", tmp_path / "result.json")
    request = BeatAnalysisRequest(file_id="silence.wav")
    runner = BeatAnalysisRunner(Settings(beat_model_path=model), lambda _: source)
    runner.process(request, output, threading.Event(), lambda _: None)
    result = json.loads(output.path.read_text())
    assert result["duration_seconds"] == 2 and result["beats_seconds"] == []
    assert result["tempo_bpm"] is None and result["onsets"] == []
    limited = BeatAnalysisRunner(Settings(beat_model_path=model, beat_max_seconds=1), lambda _: source)
    with pytest.raises(ValueError, match="exceeds"):
        limited.process(request, output, threading.Event(), lambda _: None)
    assert not output.path.exists()
    assert not list(tmp_path.glob("*.f32")) and not list(tmp_path.glob("*.partial.json"))


def test_failure_cleans_intermediate_files(tmp_path):
    runner = BeatAnalysisRunner(Settings(), lambda _: tmp_path / "bad.wav")
    def fail(arguments, log, cancel, deadline):
        (tmp_path / "result.f32").touch()
        raise RuntimeError("decode failed")
    runner._run = fail
    with pytest.raises(RuntimeError, match="decode failed"):
        runner.process(BeatAnalysisRequest(file_id="bad.wav"), OutputArtifact("result.json", "", tmp_path / "result.json"),
                       threading.Event(), lambda _: None)
    assert not list(tmp_path.iterdir())
