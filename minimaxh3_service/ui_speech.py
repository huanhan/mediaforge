"""Optional local speech preview tab; all inference goes through the shared API queue."""
from pathlib import Path
import httpx
from .speech_models import SPEAKERS

def add_speech_ui(client):
    import gradio as gr

    def submit(text, speaker, language, speed, emotion_text, strength, pause, volume, reference, reference_text):
        try:
            result = client.submit_speech(dict(text=text, speaker=speaker, language=language, speed=speed,
                emotion_mode="text", emotion_text=emotion_text, emotion_strength=strength, pause_ms=int(pause), volume_db=volume, **({"reference_text": reference_text} if reference and reference_text else {})), reference)
            return result["id"], f"已排队 · {result['id']}", None, {}, gr.Timer(active=True)
        except (ValueError, OSError, httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc

    def refresh(task_id, cached):
        if not task_id:
            return (gr.skip(),) * 4
        try:
            task = client.task(task_id)
            cache = cached if isinstance(cached, dict) and cached.get("id") == task_id else {}
            if task["status"] == "succeeded" and not cache.get("audio"):
                files = client.download([o for o in task.get("outputs", []) if o["filename"].endswith(".wav")])
                if not files:
                    raise ValueError("没有可用的语音文件")
                cache = {"id": task_id, "audio": files[0]}
            label = {"queued":"排队中","running":"语音生成中","succeeded":"已完成","failed":"失败","canceled":"已取消"}.get(task["status"],task["status"])
            if task.get("error"):
                label += "：" + str(task["error"])
            return label, cache.get("audio"), cache, gr.Timer(active=task["status"] in {"queued", "running"})
        except (ValueError, OSError, httpx.HTTPError) as exc:
            return f"查询失败，可刷新重试：{exc}", gr.skip(), gr.skip(), gr.Timer(active=False)

    gr.Markdown("### 本地 IndexTTS 2.5\n每次生成一个镜头的口播。支持预置音色和自己的录音；生成 WAV，可下载后用于视频。人物口型同步需另行制作。")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="本镜头口播文本", lines=4, max_length=1000, value="你好，欢迎来到我的频道。今天一起了解一个有趣的知识。")
            speaker = gr.Dropdown(list(SPEAKERS), value="Serena", label="预置音色")
            language = gr.Dropdown(["Chinese", "English", "Auto", "Japanese", "Spanish", "Arabic"], value="Chinese", label="语言")
            speed = gr.Slider(.5, 2, value=1, step=.05, label="语速（模型原生控制）")
            emotion_text = gr.Textbox(label="情绪描述（预置和录音音色均生效）", value="自然清晰，像与观众交谈一样讲解。")
            strength = gr.Slider(0, 1, value=.65, step=.05, label="情绪总强度")
            pause = gr.Slider(0, 2000, value=200, step=10, label="分段间停顿（毫秒）")
            volume = gr.Slider(-12, 12, value=0, step=.5, label="输出音量（dB）")
            with gr.Accordion("录制或上传自己的声音", open=False):
                gr.Markdown("建议 10–20 秒，允许 3–30 秒，单人、清晰、无背景音乐。录音文字可选；录制音色同样支持情绪控制。视频工作室设置提供录音降噪与原声对比。麦克风需要 localhost / HTTPS。")
                reference = gr.Audio(label="参考录音", sources=["upload", "microphone"], type="filepath")
                reference_text = gr.Textbox(label="录音实际文字", lines=3)
            start = gr.Button("生成本镜头配音", variant="primary")
        with gr.Column():
            status = gr.Markdown("等待提交")
            audio = gr.Audio(label="配音试听与下载", type="filepath", interactive=False)
            update = gr.Button("刷新状态")
            stop = gr.Button("取消配音任务")
    task, cached, timer = gr.State(""), gr.State({}), gr.Timer(3, active=False)
    start.click(submit, [text,speaker,language,speed,emotion_text,strength,pause,volume,reference,reference_text], [task,status,audio,cached,timer], concurrency_id="speech-task", concurrency_limit=1)
    for event in (timer.tick,update.click):
        event(refresh,[task,cached],[status,audio,cached,timer],concurrency_id="speech-task",concurrency_limit=1)
    def cancel(task_id):
        if not task_id:
            return "没有正在执行的配音任务"
        try:
            return f"任务状态：{client.task(task_id,cancel=True)['status']}"
        except (ValueError,httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc
    stop.click(cancel,task,status)
