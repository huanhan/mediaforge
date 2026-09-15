"""Music creation, task status, audio preview and download controls."""
import httpx


def add_music_ui(client):
    import gradio as gr

    def check():
        try:
            music = client.catalog()["capabilities"].get("music", {})
            model = music.get("model", "未知模型")
            return f"音乐模型已就绪 · {model}" if music.get("available") else f"音乐模型未就绪 · {model}。请先安装本地音乐权重。"
        except (ValueError, httpx.HTTPError) as exc:
            return f"检查失败：{exc}"

    def submit(prompt, lyrics, duration, seed):
        try:
            result = client.submit_music(dict(prompt=prompt, lyrics=lyrics,
                                               duration_seconds=duration, seed=seed))
            return result["id"], f"已排队 · 队列位置 {result['queue_number']}", None, None, "", gr.Timer(active=True)
        except (ValueError, httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc

    def refresh(task_id, cached):
        if not task_id:
            return (gr.skip(),) * 5
        try:
            task = client.task(task_id)
            state = task["status"]
            labels = dict(queued="排队中", running="音乐生成中", succeeded="已完成", failed="生成失败", canceled="已取消")
            message = labels.get(state, state)
            if task.get("error"):
                message += f"：{task['error']}"
            path = cached
            if state == "succeeded" and not path:
                files = client.download(task.get("outputs", []))
                if not files:
                    raise ValueError("任务完成但没有音频文件")
                path = files[0]
            return message, path or None, path or None, path, gr.Timer(active=state in {"queued", "running"})
        except (ValueError, OSError, httpx.HTTPError) as exc:
            return f"查询失败，可刷新重试：{exc}", gr.skip(), gr.skip(), gr.skip(), gr.Timer(active=False)

    def cancel(task_id):
        if not task_id:
            raise gr.Error("请先提交任务")
        try:
            result = client.task(task_id, cancel=True)
            return f"任务状态：{result['status']}"
        except (ValueError, httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc

    gr.Markdown("本地 ACE-Step 音乐生成：描述曲风、乐器和情绪，歌词留空即可生成纯音乐；填写歌词可生成歌曲。支持 10–120 秒。")
    service = gr.Markdown("点击检查音乐服务，确认是否可用。")
    gr.Button("检查音乐服务").click(check, outputs=service)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="曲风描述（推荐英文）", lines=3, value="upbeat, cheerful acoustic pop, ukulele, bright piano, handclaps, light drums, warm bass, 118 BPM, instrumental, sunny mood")
            lyrics = gr.Textbox(label="歌词（留空生成纯音乐）", lines=10, placeholder="[Verse]\nMorning sunshine at my door\n[Chorus]\nLet the music lift us more")
            duration = gr.Slider(10, 120, value=30, step=1, label="时长（秒）")
            seed = gr.Number(value=7, minimum=0, maximum=2147483647, precision=0, label="随机种子")
            start = gr.Button("生成音乐", variant="primary")
        with gr.Column():
            status = gr.Markdown("等待提交")
            audio = gr.Audio(label="音乐试听", type="filepath", interactive=False)
            download = gr.File(label="下载 WAV", interactive=False)
            with gr.Row():
                update = gr.Button("刷新状态")
                stop = gr.Button("取消排队")
    task = gr.State("")
    cached = gr.State("")
    timer = gr.Timer(3, active=False)
    start.click(submit, [prompt, lyrics, duration, seed], [task, status, audio, download, cached, timer])
    for event in (timer.tick, update.click):
        event(refresh, [task, cached], [status, audio, download, cached, timer],
              concurrency_id="music-refresh", concurrency_limit=1)
    stop.click(cancel, task, status)
