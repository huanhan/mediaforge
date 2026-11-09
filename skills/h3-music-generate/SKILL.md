---
name: h3-music-generate
description: 用 MiniMax H3 本地服务（http://127.0.0.1:8000）的 YuE2 生成音乐，支持纯音乐 BGM 和带歌词人声的歌曲，输出 48kHz 立体声 WAV、元数据 JSON 和可选 ABC 乐谱。完整流程为提交风格与歌词、轮询任务、下载结果。当用户要生成背景音乐、写歌、按歌词生成歌曲或做纯音乐时使用。
---

# H3 音乐生成（YuE2）

完整流程：**提交风格与歌词参数 → 轮询任务 → 下载 WAV 与元数据**。音乐生成不需要上传素材。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz"
```

确认服务为 `ready`，且音乐生成依赖可用。

## 第 1 步：准备参数

- `prompt`（必填）：风格、乐器、情绪、人声描述。
- `lyrics`：`[Verse]`、`[Chorus]` 等分段标记的歌词。**留空即生成器乐**。注意，留空也不保证完全没有人声。
- `duration_seconds`：10–500，默认 30。**这是上限，可能截断一个乐句**，需检查结果元数据的 `truncated`。
- `seed`：默认 7。
- `cot`：`full`（默认，旋律与和弦规划）、`melody`（仅旋律）、`off`（直接生成）。
- `abc`：可选，自带的 YuE2 兼容 ABC 乐谱，需 `cot` 为 `full` 或 `melody`。
- `cfg_scale`：文本引导强度，`null` 用默认。

## 第 2 步：提交

接口：`POST /v1/music/generations`，JSON。

纯音乐 BGM：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/music/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"轻柔钢琴与弦乐，温暖、适合旅行短片","duration_seconds":30,"seed":7,"cot":"full"}'
```

带歌词的歌曲：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/music/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"女声流行，清新，钢琴伴奏","lyrics":"[Verse]\n第一段歌词\n[Chorus]\n副歌歌词","duration_seconds":60}'
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
  sleep 10
done
```

## 第 4 步：取结果

成功后 `outputs` 含 WAV、元数据 JSON，规划时还会有 ABC 乐谱。全部下载：

```bash
curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID" | python -c "
import sys,json
for o in json.load(sys.stdin)['outputs']: print(o['filename'], o['download_url'])"
```

逐个下载 `download_url`。打开元数据 JSON 检查 `truncated`，若为 `true` 说明被截断，可调长 `duration_seconds` 或换 `seed` 重试。

## 注意

- 生成与其他任务共享 GPU 队列，排队较久属正常。
- 取消运行中的任务需要服务端开启 `H3_ALLOW_INTERRUPT_RUNNING=true`。
- 要配合视频卡点时，把音乐交给 `h3-beat-analysis` 找节拍。
