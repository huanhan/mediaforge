# 上传音乐节奏分析 API

使用 Beat This! 1.1.0 / final0 检测逐拍位置和小节第一拍；同时返回音量包络、频谱变化起音候选和速度摘要。供 Node.js 按音乐时间规划视频。此接口不做歌词识别、歌词强制对齐或视频生成。

## 安装

项目已有 Python 3.13 / PyTorch 环境，增加 `beat` 扩展：

```powershell
uv sync --extra local --extra beat --extra dev
uv run --no-sync minimaxh3-setup-beat
```

保留部署中原有的其他 `--extra`（如 ui、pose），模型下载约 81 MB，落在 `models/beat/final0.ckpt`。代码和模型使用上游 MIT 许可。FFmpeg 必须位于 PATH。模型只在安装时下载，API 任务只读取本地权重。

默认 CPU，无需新增 GPU。仍加入视频共用队列，支持任务查询、SSE、排队取消；运行中取消遵循 `H3_ALLOW_INTERRUPT_RUNNING`。每个任务通过独立进程推理，结束后释放全部分析资源，不向视频服务注入其他音频框架。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| H3_BEAT_MODEL_PATH | models/beat/final0.ckpt | 本地模型 |
| H3_BEAT_DEVICE | cpu | cpu 或 cuda[:index] |
| H3_BEAT_CPU_THREADS | 4 | CPU 推理线程数 |
| H3_BEAT_MAX_SECONDS | 0 | 默认不限制音乐时长；可显式设为正数启用部署限额，超限失败，不截断分析 |
| H3_BEAT_TIMEOUT_SECONDS | 600 | 解码与推理总超时 |

重启 API 后，`GET /readyz` 的 `beat_analysis` 和 `GET /v1/models` 的同名 capability 提供安装状态。其就绪不要求安装 H3 视频模型。

## 调用

1. `POST /v1/files`，multipart 的 `kind=audio` 和 `file` 上传音乐，支持 M4A、MP3、WAV 等已有音频扩展。
2. `POST /v1/music/analyses`，JSON：

```json
{"file_id":"minimaxh3/20261019/song-123.m4a","title":"可选歌名"}
```

返回 202：`{"id":"任务编号","status":"queued","queue_number":1}`。

3. 用 `GET /v1/tasks/{id}` 或 `/events` 查询；成功后 `kind=beat_analysis`，从 `outputs[0].download_url` 获取 JSON。失败检查 `error`，取消使用 `DELETE /v1/tasks/{id}`。
4. 沿用 `X-API-Key` 或 Bearer 认证。

Node.js 客户端示例：

```js
const headers = apiKey ? {'X-API-Key': apiKey} : {};
const form = new FormData();
form.set('kind', 'audio');
form.set('file', new Blob([await readFile(audioPath)]), 'song.m4a');
const upload = await fetch(baseUrl + '/v1/files', {method: 'POST', headers, body: form});
if (!upload.ok) throw new Error(await upload.text());
const {file_id} = await upload.json();
const response = await fetch(baseUrl + '/v1/music/analyses', {
  method: 'POST', headers: {...headers, 'Content-Type': 'application/json'},
  body: JSON.stringify({file_id, title: '歌名'}),
});
if (!response.ok) throw new Error(await response.text());
const task = await response.json(); // 保存 task.id，轮询现有 /v1/tasks/{id}
```

## 结果约定

- `schema_version: 1`、`backend: beat_this`、`source_file_id`、`title`、`model` 标识来源。
- 所有时间单位为秒，原点是解码后原始音轨起点（`time_origin=decoded_audio_start`）。保留前奏和静音；不按片段重新从零计时。
- `beats_seconds`：逐拍时间。`downbeats_seconds`：小节第一拍，不代表最大音量或每次鼓点。
- `duration_seconds`：解码后的样本数时长，压缩格式的尾部填充可能与容器报告时长有微小差异。
- `tempo_bpm`：最多跨 8 拍的局部速度中位数，仅为摘要；不用于反推等间隔切点。无法检测时为 null。
- `tempo_intervals`：每两个相邻拍点的 `{start,end,bpm}`；模型时间格点会使单拍 BPM 波动。
- `energy`：约每 50 ms 的 `{time,rms,peak}`，线性振幅，未响度归一化。
- `onsets`：频谱变化检测出的 `{time,strength}`，相对强度 0..1；这是起音候选，不是拍点或准确率。
- `warnings`：检测结果仍需试听校准。无节奏音乐、错拍、半速/倍速识别等情况不能用平均 BPM 强行补点。

剪辑时从选中的**绝对拍点时间**计算 `round(time * fps)`，相邻帧边界相减得到片段长度；不要逐段舍入再累加。字幕时间应跟随实际人声，不吸附到拍点。

单元测试不下载模型：`python -m pytest tests/test_beat.py`。真实音乐的任务链需要已安装权重后验证。
