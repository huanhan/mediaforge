---
name: h3-beat-analysis
description: 用 MiniMax H3 本地服务（http://127.0.0.1:8000）分析音乐的节拍与时间结构，输出 JSON 时间数据，用于视频卡点、按节拍切镜和转场。完整流程为上传音频、提交分析、轮询任务、下载结果。当用户要卡点剪辑、找节拍、分析 BPM 或按音乐节奏安排画面时使用。不做转写或歌词对齐。
---

# H3 音乐节拍分析

完整流程：**上传音频 → 提交分析 → 轮询任务 → 下载 JSON**。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz"
```

确认服务为 `ready`。

## 第 1 步：上传音频

```bash
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=audio" -F "file=@/path/bgm.wav"
```

记下 `file_id`。

## 第 2 步：提交分析

接口：`POST /v1/music/analyses`，JSON。

| 字段 | 必填 | 说明 |
|---|---|---|
| `file_id` | 是 | 音频的 file_id |
| `title` | 否 | 歌名，只作元数据，不影响检测结果 |

```bash
curl -s -X POST http://127.0.0.1:8000/v1/music/analyses \
  -H "Content-Type: application/json" \
  -d '{"file_id":"uploads/bgm.wav","title":"背景音乐"}'
```

返回 202，包含 `id`、`queue_number`。

## 第 3 步：轮询任务

```bash
TASK_ID=<上一步返回的 id>
while true; do
  s=$(curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID")
  st=$(echo "$s" | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "status=$st"
  case "$st" in succeeded|failed|canceled) break;; esac
  sleep 5
done
```

## 第 4 步：取结果

```bash
URL=$(curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID" | python -c "import sys,json;print(json.load(sys.stdin)['outputs'][0]['download_url'])")
curl -s -o beats.json "$URL"
```

读取 JSON 中的节拍时间点（秒），用于剪辑、切镜或转场的时间安排。

## 注意

- 时间点是音频内的秒数。视频起点若有偏移，需要换算。
- 没有明显节拍的音频（环境音、纯人声）结果可能不可靠。
- 本接口不做转写和歌词对齐，需要歌词时间轴请用 `h3-lyrics`。
