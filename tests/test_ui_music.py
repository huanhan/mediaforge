import json
from pathlib import Path

import pytest


def build_music(client):
    gr = pytest.importorskip("gradio")
    from mediaforge_service.ui_music import add_music_ui
    with gr.Blocks() as demo:
        add_music_ui(client)
    return demo


class MusicClient:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.downloads = []
        self.request = dict(prompt="低沉男声，摇滚", lyrics="测试歌词", duration_seconds=30,
                            seed=7, cot="full", abc=None, cfg_scale=None)

    def task(self, task_id):
        return dict(id=task_id, status="succeeded", outputs=[
            dict(filename=f"{task_id}.wav"), dict(filename=f"{task_id}.json")])

    def download(self, outputs):
        self.downloads.append(outputs)
        paths = []
        for output in outputs:
            path = self.root / output["filename"]
            if path.suffix == ".json":
                path.write_text(json.dumps({"request": self.request}), encoding="utf-8")
            else:
                path.write_bytes(b"test audio")
            paths.append(str(path))
        return paths

    def submit_music(self, payload):
        self.request = payload
        return dict(id="new-song", queue_number=1)


@pytest.mark.parametrize("cache", [
    "old-song.wav", {"task_id": "old-song", "path": "old-song.wav", "request": {"prompt": "female"}},
])
def test_refresh_never_reuses_another_songs_audio(tmp_path, cache):
    client = MusicClient(tmp_path)
    demo = build_music(client)
    refresh = next(event.fn for event in demo.fns.values() if event.fn.__name__ == "refresh")
    result = refresh("new-song", cache)
    assert client.downloads, "A different task must download its own result"
    assert Path(result[1]).name == "new-song.wav"
    assert result[2] == result[1]
    assert result[3]["task_id"] == "new-song"
    assert result[4] == client.request
    assert "new-song" in result[0]
    # A later refresh of this same task can reuse its matching audio and metadata.
    again = refresh("new-song", result[3])
    assert again[1:5] == result[1:5]
    assert len(client.downloads) == 1


def test_submit_preserves_male_prompt_and_resets_preview(tmp_path):
    client = MusicClient(tmp_path)
    demo = build_music(client)
    events = [event for event in demo.fns.values() if event.fn.__name__ in {"submit", "refresh"}]
    assert len({event.concurrency_id for event in events}) == 1
    assert all(event.concurrency_limit == 1 for event in events)
    submit = next(event.fn for event in events if event.fn.__name__ == "submit")
    result = submit("  低沉男声，摇滚  ", "测试歌词", 30, 7, "full", "", None)
    assert result[0] == "new-song"
    assert result[2:5] == (None, None, {})
    assert result[5] == client.request
    assert client.request["prompt"] == "低沉男声，摇滚"
    assert client.request["abc"] is None
