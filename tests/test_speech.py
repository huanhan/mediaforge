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
from minimaxh3_service.speech_worker import chunks
from minimaxh3_service.storage import LocalAssetStore
from minimaxh3_service.tasks import LocalTaskManager, GenerationCanceled

class Idle:
    def readiness(self): return {"status": "not_ready"}
    def close(self): pass
    def release_models(self): pass

@pytest.mark.parametrize("patch", [dict(text=" "),dict(text="a"*1001),dict(speed=float('inf')),
    dict(reference_file_id="../x.wav", reference_text="hello"), dict(reference_file_id="x.wav"),
    dict(reference_text="hello"),dict(reference_file_id="x.wav", reference_text="hello", instruct="angry"),dict(speaker="unknown")])
def test_request_rejects_invalid_or_unimplemented_parameters(patch):
    with pytest.raises(ValidationError): SpeechRequest.model_validate({"text":"hello",**patch})

def test_chunking_keeps_exact_text_and_bounds():
    text="你好，欢迎来到我的频道。"*40+"English words without punctuation "*20
    parts=chunks(text)
    assert ''.join(parts)==text
    assert all(0<len(p)<=100 for p in parts)

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
        def readiness(self): return {"status":"ready","backend":"qwen3_tts"}
    settings=Settings(data_root=tmp_path,api_key="secret",allow_interrupt_running=True,tts_timeout_seconds=20)
    store=LocalAssetStore(settings.input_root,settings.output_root,1000000);store.initialize()
    return Runner(settings,store.resolve_input),store

def test_runner_generates_pcm_speed_metadata_and_cleans_failure(speech):
    runner,store=speech
    base=store.output_path('test','speech/test');output=OutputArtifact('test.wav',base.subfolder,base.path.with_suffix('.wav'))
    artifacts=runner.generate(SpeechRequest(text='你好',speed=1.25),output,threading.Event(),lambda _:None)
    with wave.open(str(output.path),'rb') as wav:
        assert wav.getframerate()==24000 and wav.getnchannels()==1
        assert .7<wav.getnframes()/24000<.9
    metadata=json.loads(artifacts[1].path.read_text(encoding='utf-8'))
    assert metadata['text']=='你好' and metadata['speaker']=='Serena'
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
