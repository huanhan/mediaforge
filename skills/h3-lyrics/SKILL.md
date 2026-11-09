---
name: h3-lyrics
description: 用 MiniMax H3 本地服务（http://127.0.0.1:8000）处理歌词与字幕。完整流程为：上传音频，用 Qwen3-ASR 转写出歌词草稿，用户校对后用 Qwen3 ForcedAligner 逐行对齐到音频，得到字幕时间轴与逐字时间戳。当用户要转写歌曲或语音、做歌词字幕、卡拉 OK 时间轴，或把已有歌词对齐到音频时使用。
---

# H3 歌词与字幕（转写 → 校对 → 对齐）

完整流程：**上传音频 →（无歌词时）转写出草稿 → 用户校对 → 对齐时间轴 → 取结果**。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

两个接口的分工：

- **转写**（`/v1/audio/transcriptions`）：从音频识别出文字。输出没有精确时间戳，`chunk_seconds` 只是输入分段，**分段边界不是歌词时间**。
- **对齐**（`/v1/audio/alignments`）：把已确认的歌词放到音频时间轴上，输出每行时间与逐字时间戳。

如果用户已经有正确歌词，跳过转写，直接对齐。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz"
```

确认服务为 `ready`。

## 第 1 步：上传音频

```bash
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=audio" -F "file=@/path/song.wav"
```

记下 `file_id`。

## 第 2 步：转写（没有歌词时）

接口：`POST /v1/audio/transcriptions`，JSON。

| 字段 | 必填 | 说明 |
|---|---|---|
| `file_id` | 是 | 音频的 file_id |
| `language` | 否 | `auto`、`zh`（默认）、`en`、`yue`、`ja`、`ko` |
| `chunk_seconds` | 否 | 输入分段，30–180，默认 180 |
| `title` | 否 | 歌名，只作元数据 |

```bash
curl -s -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d '{"file_id":"uploads/song.wav","language":"zh","chunk_seconds":180}'
```

轮询（见第 4 步）完成后，下载 JSON 和 TXT：

```bash
curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID" | python -c "
import sys,json
for o in json.load(sys.stdin)['outputs']: print(o['filename'], o['download_url'])"
```

**把 TXT 内容展示给用户，请用户逐行校对、修正错字和漏句。** 转写结果不能直接当作字幕使用。

## 第 3 步：对齐时间轴

用户确认歌词后，接口：`POST /v1/audio/alignments`，JSON。

| 字段 | 必填 | 说明 |
|---|---|---|
| `file_id` | 是 | 音频的 file_id |
| `lyrics` | 是 | 已确认的歌词，**每行一句**，即一条字幕 |
| `language` | 否 | `zh`（默认）、`en`、`yue` |
| `title` | 否 | 歌名，只作元数据 |

```bash
curl -s -X POST http://127.0.0.1:8000/v1/audio/alignments \
  -H "Content-Type: application/json" \
  -d '{"file_id":"uploads/song.wav","lyrics":"第一句歌词\n第二句歌词\n第三句歌词","title":"示例歌","language":"zh"}'
```

返回 202，包含 `id`、`queue_number`。

## 第 4 步：轮询任务

```bash
TASK_ID=<上一步返回的 id>
while true; do
  s=$(curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID")
  st=$(echo "$s" | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "status=$st"
  case "$st" in succeeded|failed|canceled) break;; esac
  sleep 10
done
```

## 第 5 步：取结果

对齐成功后，读取任务结果中的字幕时间轴（cues）与逐字时间戳，交给用户或用于生成字幕文件。`failed` 时读取 `error`。

## 注意

- 歌词错字或漏句会直接影响对齐结果。对齐前必须先让用户确认。
- 接口摘要写"最长 180 秒"，但字段说明写"长音频会按窗口分段，没有默认时长限制"，两者矛盾。超长音频先用短片段测试。
- 对齐只给时间，不做识别。
