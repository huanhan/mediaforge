import asyncio
import json

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from mediaforge_service.api import create_app
from mediaforge_service.backend import BackendError, OutputArtifact
from mediaforge_service.config import Settings
from mediaforge_service.local_backend import LocalGenerationBackend
from mediaforge_service.models import AlignmentRequest


class Runner:
    def readiness(self):
        return {"status": "ready"}

    def release_models(self):
        pass

    def close(self):
        pass

    def process(self, request, output, cancel, progress):
        output.path.write_text(json.dumps({"text": request.lyrics, "timing": "forced_alignment"}), encoding="utf-8")
        txt = output.path.with_suffix(".txt")
        txt.write_text(request.lyrics, encoding="utf-8")
        return [output, OutputArtifact(txt.name, output.subfolder, txt)]


def test_alignment_api_auth_queue_and_original_lyrics(tmp_path):
    idle = Runner()
    backend = LocalGenerationBackend(Settings(data_root=tmp_path), runner=idle, upscale_runner=idle,
              removal_runner=idle, music_runner=idle, depth_runner=idle, skeleton_runner=idle,
              beat_runner=idle, transcription_runner=idle, alignment_runner=idle)
    with TestClient(create_app(Settings(api_key="test"), backend_factory=lambda _: backend)) as client:
        assert client.post("/v1/audio/alignments", json={"file_id": "a.wav", "lyrics": "人改的词"}).status_code == 401
        client.headers["X-API-Key"] = "test"
        file_id = client.post("/v1/files", data={"kind": "audio"}, files={"file": ("song.wav", b"audio", "audio/wav")}).json()["file_id"]
        response = client.post("/v1/audio/alignments", json={"file_id": file_id, "lyrics": "人改的词\n第二行"})
        assert response.status_code == 202
        client.portal.call(backend.tasks.queue.join)
        task = client.get("/v1/tasks/" + response.json()["id"]).json()
        assert task["status"] == "succeeded" and task["kind"] == "alignment"
        assert len(task["outputs"]) == 2
        assert client.get(task["outputs"][0]["download_url"]).json()["text"] == "人改的词\n第二行"
        assert client.get("/v1/models").json()["capabilities"]["alignment"]["available"] is True
        backend.tasks.alignment_runner.readiness = lambda: {"status": "not_ready"}
        assert client.post("/v1/audio/alignments", json={"file_id": file_id, "lyrics": "歌词"}).status_code == 503


def test_alignment_groups_word_times_without_changing_line_text():
    pytest.importorskip("numpy")
    from mediaforge_service.alignment_worker import make_cues
    words = [{"text": "风", "start_time": 2, "end_time": 2.2},
             {"text": "来", "start_time": 2.2, "end_time": 3},
             {"text": "了", "start_time": 4, "end_time": 5}]
    cues, result = make_cues(["风来，", "了！"], [["风", "来"], ["了"]], words, 10)
    assert cues == [{"id": "line_1", "start": 2, "end": 3, "text": "风来，"},
                    {"id": "line_2", "start": 4, "end": 5, "text": "了！"}]
    assert result[0]["start"] == 2  # Leading music is not trimmed away.
    with pytest.raises(ValueError, match="tokens"):
        make_cues(["别的词"], [["别"]], words, 10)
    with pytest.raises(ValueError, match="timestamps"):
        make_cues(["风"], [["风"]], [{"text": "风", "start_time": 3, "end_time": 2}], 10)
    with pytest.raises(ValueError, match="could not be aligned"):
        make_cues(["风"], [["风"]], [{"text": "风", "start_time": 3, "end_time": 3}], 10)


@pytest.mark.parametrize("body", [{"file_id": "../x.wav", "lyrics": "词"}, {"file_id": "x.wav", "lyrics": " "},
                                  {"file_id": "x.wav", "lyrics": "词", "language": "auto"}])
def test_alignment_rejects_invalid_requests(body):
    with pytest.raises(ValidationError):
        AlignmentRequest(**body)


def test_long_alignment_preserves_edits_repeats_and_absolute_times():
    np = pytest.importorskip("numpy")
    from mediaforge_service.alignment_worker import alignment_windows, align_windows, make_cues
    # A line crosses the ASR boundary, whose imperfect spelling must not replace it.
    lines = ["intro words here", "repeat the chorus", "new corrected words", "repeat the chorus", "final words here"]
    tokens = [word for line in lines for word in line.split()]
    chunks = [{"start": 0, "end": 120, "text": "intro words here repeat the", "token_limit_reached": False},
              {"start": 120, "end": 240, "text": "chorus new incorrect words repeat the chorus", "token_limit_reached": False},
              {"start": 240, "end": 360, "text": "final words here", "token_limit_reached": False}]
    windows = alignment_windows(tokens, chunks, str.split, 360)
    assert [word for window in windows for word in window["tokens"]] == tokens
    assert all(window["end"]-window["start"] <= 180 for window in windows)
    assert windows[1]["start"] == 90 and windows[2]["start"] == 210
    audio = np.zeros(360*16000, dtype=np.float32)
    # These mock model times intentionally differ from equal lyric distribution.
    model_times = [[(20, 21), (22, 23), (24, 25), (115, 116), (117, 118)],
                   [(25, 26), (27, 28), (29, 32), (50, 51), (52, 53), (54, 55), (110, 111), (112, 113), (114, 115)],
                   [(100, 101), (102, 103), (104, 105)]]
    calls = []

    def align(window, text):
        times = model_times[len(calls)]
        if len(calls) == 1:
            assert text.startswith("repeat the chorus"), "Leading audio overlap must include its confirmed lyric context"
        calls.append(len(window))
        return [{"text": word, "start_time": a, "end_time": b} for word, (a, b) in zip(text.split(), times)]

    aligned = align_windows(audio, windows, align)
    cues, words = make_cues(lines, [line.split() for line in lines], aligned, 360)
    assert [cue["text"] for cue in cues] == lines
    assert cues[1]["start"] == 115 and cues[1]["end"] == 122  # One subtitle crosses windows.
    assert cues[-1]["start"] == 310 and cues[-1]["end"] == 315
    assert words[7]["text"] == "corrected"
    assert calls == [150*16000, 180*16000, 150*16000]


def test_long_alignment_keeps_long_instrumental_intro_and_outro():
    pytest.importorskip("numpy")
    from mediaforge_service.alignment_worker import alignment_windows
    chunks = [{"start": i*120, "end": (i+1)*120, "text": "we sing these words" if i == 8 else "", "token_limit_reached": False}
              for i in range(12)]
    windows = alignment_windows("we sing these words".split(), chunks, str.split, 1440)
    assert windows == [{"start": 930, "end": 1110, "tokens": "we sing these words".split()}]


def test_long_alignment_rejects_unreliable_sections_and_invalid_local_times():
    np = pytest.importorskip("numpy")
    from mediaforge_service.alignment_worker import alignment_windows, align_windows
    chunk = {"start": 0, "end": 120, "text": "entirely unrelated text", "token_limit_reached": False}
    with pytest.raises(ValueError, match="Cannot locate"):
        alignment_windows(["confirmed", "lyrics"], [chunk], str.split, 120)
    with pytest.raises(ValueError, match="truncated"):
        alignment_windows(["confirmed"], [{**chunk, "token_limit_reached": True}], str.split, 120)
    with pytest.raises(ValueError, match="timestamps"):
        align_windows(np.zeros(16000), [{"start": 0, "end": 1, "tokens": ["word"]}],
                      lambda *_: [{"text": "word", "start_time": 0, "end_time": 2}])
