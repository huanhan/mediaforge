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
    dict(reference_file_id="../x.wav", reference_text="hello"), dict(emotion_vector=[0]*7),
    dict(reference_text="hello"),dict(emotion_vector=[float("nan")]*8),dict(speaker="unknown"),
    dict(model="Qwen3-TTS-12Hz-1.7B"),dict(instruct="自然讲述"),dict(emotion_mode="text",emotion_text=" ")])
def test_request_rejects_invalid_or_unimplemented_parameters(patch):
    with pytest.raises(ValidationError): SpeechRequest.model_validate({"text":"hello",**patch})


def test_default_clone_uses_reference_delivery_without_text_emotion():
    request = SpeechRequest(text="hello", reference_file_id="personal.wav")
    options = inference_options({**request.model_dump(), "reference_audio": "personal.wav"})
    assert request.emotion_mode == "reference"
    assert not options["use_emo_text"] and options["emo_vector"] is None
    assert options["emo_audio_prompt"] is None  # IndexTTS then uses the speaker sample itself.
    assert options["spk_audio_prompt"] == "personal.wav"
    explicit = inference_options({**request.model_dump(), "reference_audio": "personal.wav",
                                  "emotion_mode": "text", "emotion_text": "angry"})
    assert explicit["use_emo_text"] and explicit["emo_text"] == "angry"

def test_native_duration_and_emotion_controls_with_cloned_voice():
    request = SpeechRequest(text="你好", reference_file_id="voice.wav", speed=1.25,
        emotion_mode="vector", emotion_vector=[.8,.2,0,0,0,0,0,.1], emotion_strength=.7,
        pause_ms=350, temperature=.6, top_p=.7, top_k=20, repetition_penalty=8,
        max_segment_tokens=80, text_normalization=False)
    options = inference_options({**request.model_dump(), "reference_audio":"local.wav"})
    assert options["spk_audio_prompt"] == "local.wav"
    assert options["duration_factor"] == .8
    assert options["emo_vector"] == [.8,.2,0,0,0,0,0,.1]
    assert options["emo_alpha"] == .7 and not options["use_emo_text"]
    assert options["interval_silence"] == 350
    assert options["temperature"] == .6 and options["top_p"] == .7 and options["top_k"] == 20
    assert options["repetition_penalty"] == 8 and options["max_text_tokens_per_segment"] == 80
    assert isinstance(options["repetition_penalty"], float)  # Transformers rejects JSON integers here.
    assert not options["text_normalization"]
    options=inference_options({**request.model_dump(),"reference_audio":"voice.wav",
        "emotion_mode":"reference","emotion_reference_audio":"angry.wav"})
    assert options["emo_audio_prompt"] == "angry.wav" and options["emo_vector"] is None
    options=inference_options({**request.model_dump(),"reference_audio":"voice.wav",
        "emotion_mode":"text","emotion_text":"愤怒"})
    assert options["use_emo_text"] and options["emo_text"] == "愤怒"

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
with wave.open(a.output,'wb') as w:
 w.setparams((1,2,24000,0,'NONE','not compressed'))
 w.writeframes(b''.join(struct.pack('<h',int(2000*math.sin(i/20))) for i in range(24000)))
''',encoding='utf-8')
    class Runner(SpeechRunner):
        def _command(self): return [sys.executable,str(script)]
        def readiness(self): return {"status":"ready","backend":"index_tts"}
    settings=Settings(data_root=tmp_path,api_key="secret",allow_interrupt_running=True,tts_timeout_seconds=20)
    store=LocalAssetStore(settings.input_root,settings.output_root,1000000);store.initialize()
    return Runner(settings,store.resolve_input),store

def test_runner_generates_pcm_speed_metadata_and_cleans_failure(speech):
    runner,store=speech
    base=store.output_path('test','speech/test');output=OutputArtifact('test.wav',base.subfolder,base.path.with_suffix('.wav'))
    artifacts=runner.generate(SpeechRequest(text='你好',speed=1.25),output,threading.Event(),lambda _:None)
    with wave.open(str(output.path),'rb') as wav:
        assert wav.getframerate()==24000 and wav.getnchannels()==1
        assert wav.getnframes() == 24000  # speed is applied by the model, never a second FFmpeg atempo pass
    metadata=json.loads(artifacts[1].path.read_text(encoding='utf-8'))
    assert metadata['text']=='你好' and metadata['speaker']=='Serena'
    assert metadata['model']=='IndexTTS-2.5' and metadata['parameters']['speed']==1.25
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


def test_uploaded_voice_reaches_index_speaker_prompt_instead_of_default_preset(speech, monkeypatch):
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
            with wave.open(options['spk_audio_prompt'], 'rb') as wav:
                observed.append(wav.readframes(wav.getnframes()))
            assert options['spk_audio_prompt'].endswith('reference_audio.wav')
            assert options['emo_audio_prompt'] is None
            assert not options['use_emo_text']
        return original_run(command, *args)

    monkeypatch.setattr(runner, '_run', inspect_worker)
    base = store.output_path('personal', 'speech/personal')
    output = OutputArtifact('personal.wav', base.subfolder, base.path.with_suffix('.wav'))
    artifacts = runner.generate(SpeechRequest(text='recorded sample', reference_file_id='personal.wav',
        emotion_mode='reference'), output, threading.Event(), lambda _: None)
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
