---
name: h3-edit-video
description: 用 MiniMax H3 本地服务（http://127.0.0.1:8000）编辑已有视频，完整流程为上传源视频和参考图、提交编辑参数、轮询任务、下载结果。支持换背景、角色替换、动作迁移、按提示词改写画面，可指定时间区间并自动分段。当用户要改视频内容、换背景、替换人物或角色、把动作迁移到另一个角色时使用。从零生成视频请用 h3-generate-video。
---

# H3 视频编辑

完整流程：**上传源视频与参考图 → 提交编辑 → 轮询任务 → 下载结果**。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz?capability=video"
```

`status` 为 `ready` 才继续。角色替换需要 Ref2V 权重，若缺失，`missing_models` 会列出来。

## 第 1 步：上传素材

```bash
# 源视频（必填）
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=video" -F "file=@/path/source.mp4"

# 参考图（角色图或背景图，可选）
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=image" -F "file=@/path/character.png"
```

记下返回的 `file_id`。**编辑接口不传 `mode`**，源视频通过 `source_video` 指定。

## 第 2 步：提交编辑

接口：`POST /v1/edits`，JSON。

关键字段：

- `source_video`（必填）：`{"file_id": "...", "start_seconds": 0, "duration_seconds": 10}`。只处理一段时填时间区间；不填 `duration_seconds` 则处理到结尾。`duration_seconds` 为 4–600。
- `prompt`（必填）：编辑指令。用 `<Video 1>` 指源视频，用 `<Picture N>` 指 `reference_images` 中第 N 张。
- `reference_images`：参考图 file_id 数组，≤9 张，顺序与提示词编号一致。
- `audio_mode`：`preserve_source`（默认，保留原音）或 `generate`。
- 分辨率：`width`+`height` 直接指定，或 `resolution`+`aspect_ratio`。都不传则沿用源视频尺寸。
- `segmentation`：`{"enabled": true, "segment_seconds": 10, "overlap_frames": 48}`，长视频建议开启。
- `seed`、`denoise`（0–1，默认 1）、`ref_image_size`（`match` 或 `max`）、`filename_prefix`。

示例（换绿幕背景）：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/edits \
  -H "Content-Type: application/json" \
  -d '{"prompt":"保持人物和动作，将背景改成纯绿色。","source_video":{"file_id":"uploads/source.mp4"},"resolution":"480p","aspect_ratio":"9:16","segmentation":{"enabled":true}}'
```

示例（角色替换 / 动作迁移）：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/edits \
  -H "Content-Type: application/json" \
  -d '{"prompt":"用 <Picture 1> 中的角色替换 <Video 1> 的主要人物，继承原动作、节奏和道具，保留原背景与镜头。","source_video":{"file_id":"uploads/source.mp4"},"reference_images":["uploads/character.png"],"width":352,"height":640,"audio_mode":"preserve_source","seed":358,"segmentation":{"enabled":true,"segment_seconds":10,"overlap_frames":48}}'
```

示例（上传背景图换背景）：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/edits \
  -H "Content-Type: application/json" \
  -d '{"prompt":"<Picture 1> 仅用作目标背景。从第一帧到最后一帧，将 <Video 1> 的背景替换为图片中的场景，保留背景构图与色彩。保留原角色、动作、道具和镜头。不要渐变或延迟切换。","source_video":{"file_id":"uploads/source.mp4"},"reference_images":["uploads/background.png"],"width":352,"height":640,"audio_mode":"preserve_source","seed":358,"segmentation":{"enabled":true,"segment_seconds":10,"overlap_frames":48}}'
```

返回 202，包含 `id`、`queue_number`、`effective_duration_seconds`、`frame_count`。

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

```bash
URL=$(curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID" | python -c "import sys,json;print(json.load(sys.stdin)['outputs'][0]['download_url'])")
curl -s -o edited.mp4 "$URL"
```

`failed` 时读取 `error`。取消任务：`curl -s -X DELETE http://127.0.0.1:8000/v1/tasks/$TASK_ID`（运行中的任务需要服务端开启 `H3_ALLOW_INTERRUPT_RUNNING=true`）。

## 注意

- 提示词里的编号必须和 `reference_images` 的顺序一致，否则模型会引用错误的素材。
- 编辑与生成、放大等任务共享 GPU 队列。
