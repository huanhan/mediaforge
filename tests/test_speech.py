import asyncio
import json
import sys
import threading
import time
import wave
from pathlib import Path
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient
from minimaxh3_service.api import create_app
from minimaxh3_service.backend import OutputArtifact
from minimaxh3_service.config import Settings
from minimaxh3_service.local_backend import LocalGenerationBackend
from minimaxh3_service.speech_models import SpeechRequest
from minimaxh3_service.speech import SpeechRunner
from minimaxh3_service.speech_worker import inference_options
from minimaxh3_service.storage import LocalAssetStore
from minimaxh3_service.tasks import LocalTaskManager, GenerationCanceled

class Idle:
    def readiness(self): return {"status": "not_ready"}
    def close(self): pass
    def release_models(self): pass

@pytest.mark.parametrize("patch", [dict(text=" "),dict(text="a"*1001),dict(speed=float('inf')),
    dict(reference_file_id="../x.wav", reference_text="hello"), dict(reference_file_id="voice.wav"),
    dict(reference_text="hello"),dict(cfg_scale=float("nan")),dict(speaker="unknown"),
    dict(model="IndexTTS-2.5"),dict(emotion_mode="vector"),dict(mode="direction",instruction=" "),
    dict(language="Japanese"),dict(repetition_penalty=10)])
def test_request_rejects_invalid_or_retired_parameters(patch):
    with pytest.raises(ValidationError): SpeechRequest.model_validate({"text":"hello",**patch})


def test_clone_keeps_exact_reference_and_omits_instruction():
    request = SpeechRequest(text="hello", reference_file_id="personal.wav", reference_text="actual spoken words",
                            instruction="must not affect pure cloning")
    options = inference_options({**request.model_dump(), "reference_audio": "personal.wav"})
    assert request.mode == "clone"
    assert options["ref_audio_path"] == "personal.wav" and options["ref_text"] == "actual spoken words"
    assert "instruction" not in options
    directed = inference_options({**request.model_dump(), "reference_audio": "personal.wav",
                                  "mode": "direction", "instruction": "Speak with restrained anger"})
    assert directed["instruction"] == "Speak with restrained anger"
    assert directed["ref_audio_path"] == options["ref_audio_path"]


def test_segments_preserve_text_and_inline_events():
    from minimaxh3_service.speech_worker import split_text
    text = "你好。[叹气] 我不会认输！" * 40
    parts = split_text(text)
    assert len(parts) > 1
    assert "".join(p.replace(" ", "") for p in parts) == text.replace(" ", "")
    assert all(p.count("[") == p.count("]") for p in parts)
    assert split_text("第一句\n第二句") == ["第一句", "第二句"]

def test_segment_rejects_generation_limit_even_if_codec_drops_frames():
    from types import SimpleNamespace
    from minimaxh3_service.speech_worker import checked_audio_chunks
    class Runtime:
        def __init__(self, steps, decoded):
            self.steps, self.decoded = steps, decoded
        def iter_audio_chunks(self, inputs, *, token_observer, **kwargs):
            for _ in range(self.steps):
                token_observer(None)
            if self.decoded:
                yield SimpleNamespace(codec_frames=self.decoded)
    assert len(list(checked_audio_chunks(Runtime(2, 2), {}, request_id="test", seed=42, max_steps=3))) == 1
    with pytest.raises(RuntimeError, match="audio limit"):
        list(checked_audio_chunks(Runtime(3, 1), {}, request_id="test", seed=42, max_steps=3))
    with pytest.raises(RuntimeError, match="no speech"):
        list(checked_audio_chunks(Runtime(0, 0), {}, request_id="test", seed=42, max_steps=3))

@pytest.fixture
def speech(tmp_path):
    script=tmp_path/'worker.py'
    script.write_text('''import argparse,json,wave,math,struct,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');p.add_argument('--progress')
a,extra=p.parse_known_args();r=json.loads(Path(a.request).read_text(encoding='utf-8'))
if r['text']=='slow': time.sleep(20)
if r['text']=='fail': raise RuntimeError('failed worker')
assert '--model-dir' in extra
assert r['reference_text']
if not r['reference_file_id']: assert r['reference_audio'].endswith('Serena.wav')
with wave.open(a.output,'wb') as w:
 w.setparams((1,2,24000,0,'NONE','not compressed'))
 w.writeframes(b''.join(struct.pack('<h',int(2000*math.sin(i/20))) for i in range(24000)))
''',encoding='utf-8')
    class Runner(SpeechRunner):
        def _command(self): return [sys.executable,str(script)]
        def readiness(self): return {"status":"ready","backend":"breeze_tts"}
    settings=Settings(data_root=tmp_path,api_key="secret",allow_interrupt_running=True,tts_timeout_seconds=20)
    store=LocalAssetStore(settings.input_root,settings.output_root,1000000);store.initialize()
    return Runner(settings,store.resolve_input),store

def test_runner_generates_pcm_speed_metadata_and_cleans_failure(speech):
    runner,store=speech
    base=store.output_path('test','speech/test');output=OutputArtifact('test.wav',base.subfolder,base.path.with_suffix('.wav'))
    artifacts=runner.generate(SpeechRequest(text='你好',speed=1.25),output,threading.Event(),lambda _:None)
    with wave.open(str(output.path),'rb') as wav:
        assert wav.getframerate()==24000 and wav.getnchannels()==1
        assert 18000 < wav.getnframes() < 21000  # 1.25x postprocess; the worker emits one second
    metadata=json.loads(artifacts[1].path.read_text(encoding='utf-8'))
    assert metadata['text']=='你好' and metadata['speaker']=='Serena'
    assert metadata['model']=='Breeze-TTS-2' and metadata['parameters']['speed']==1.25
    with pytest.raises(RuntimeError): runner.generate(SpeechRequest(text='fail'),output,threading.Event(),lambda _:None)
    assert not output.path.exists() and not output.path.with_suffix('.json').exists()

def test_api_shared_queue_and_download(speech):
    runner,store=speech
    backend=LocalGenerationBackend(runner.settings,runner=Idle(),upscale_runner=Idle(),removal_runner=Idle(),speech_runner=runner)
    with TestClient(create_app(settings=runner.settings,backend_factory=lambda _:backend)) as client:
        assert client.post('/v1/audio/speech',json={'text':'你好'}).status_code==401
        client.headers['X-API-Key']='secret'
        response=client.post('/v1/audio/speech',json={'text':'你好'})
        assert response.status_code==202,response.text
        task_id=response.json()['id']
        for _ in range(100):
            state=client.get('/v1/tasks/'+task_id).json()
            if state['status'] in ['succeeded','failed']:break
            time.sleep(.05)
        assert state['status']=='succeeded',state
        assert state['kind']=='speech' and len(state['outputs'])==2
        audio=next(o for o in state['outputs'] if o['filename'].endswith('.wav'))
        assert client.get(audio['download_url']).content[:4]==b'RIFF'


def test_uploaded_voice_and_transcript_reach_breeze_instead_of_default_preset(speech, monkeypatch):
    runner, store = speech
    source = store.input_root / 'personal.wav'
    samples = b'\x10\x03\xf0\xfc' * (24000 * 2)
    with wave.open(str(source), 'wb') as wav:
        wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        wav.writeframes(samples)
    observed = []
    original_run = runner._run

    def inspect_worker(command, *args):
        if '--request' in command:
            payload = json.loads(Path(command[command.index('--request') + 1]).read_text(encoding='utf-8'))
            options = inference_options(payload)
            with wave.open(options['ref_audio_path'], 'rb') as wav:
                observed.append(wav.readframes(wav.getnframes()))
            assert options['ref_audio_path'].endswith('reference_audio.wav')
            assert 'instruction' not in options
            assert options['ref_text'] == 'personal recording transcript'
        return original_run(command, *args)

    monkeypatch.setattr(runner, '_run', inspect_worker)
    base = store.output_path('personal', 'speech/personal')
    output = OutputArtifact('personal.wav', base.subfolder, base.path.with_suffix('.wav'))
    artifacts = runner.generate(SpeechRequest(text='recorded sample', reference_file_id='personal.wav',
        reference_text='personal recording transcript'), output, threading.Event(), lambda _: None)
    assert observed == [samples]
    assert json.loads(artifacts[1].path.read_text(encoding='utf-8'))['speaker'] == 'recorded_voice'

def test_cancel_before_start_and_model_missing(tmp_path):
    async def run():
        store=LocalAssetStore(tmp_path/'in',tmp_path/'out',1000)
        manager=LocalTaskManager(Idle(),store.output_path,max_queue_size=2,allow_interrupt_running=True,speech_runner=Idle())
        task=await manager.submit_speech(SpeechRequest(text='hello'))
        assert manager.cancel(task.id).status=='canceled'
    asyncio.run(run())
    settings=Settings(data_root=tmp_path,tts_model_path=tmp_path/'missing',tts_python=tmp_path/'missing-python')
    backend=LocalGenerationBackend(settings,runner=Idle(),upscale_runner=Idle(),removal_runner=Idle())
    with TestClient(create_app(settings=settings,backend_factory=lambda _:backend)) as client:
        result=client.post('/v1/audio/speech',json={'text':'hello'})
        assert result.status_code==503 and result.json()['error']['code']=='speech_not_ready'
