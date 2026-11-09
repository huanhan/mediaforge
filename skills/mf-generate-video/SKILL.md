---
name: mf-generate-video
description: 用 mediaforge 本地服务（http://127.0.0.1:8000）生成视频，完整流程为上传素材、提交参数、轮询任务、下载结果。支持文生视频 t2v、首尾帧 fl2v、参考图/视频/音频 ref2v，可用自动重叠分段生成超过 15 秒的长视频。当用户要用 mediaforge 生成视频、文生视频、首尾帧过渡、参考图生视频或长视频时使用。视频编辑（换背景、换角色）请用 mf-edit-video。
---

# 视频生成

完整流程：**上传素材（如有）→ 提交生成 → 轮询任务 → 下载结果**。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz?capability=video"
```

`status` 为 `ready` 才继续。若为 `not_ready`（HTTP 503），读取 `reason` 和 `missing_runtime`、`missing_models` 告诉用户，不要提交。

## 第 1 步：上传素材（仅 fl2v / ref2v 需要）

`t2v` 不需要上传。其他模式的素材先上传，拿到 `file_id`：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=image" -F "file=@/path/first.png"
```

- 图片用 `kind=image`，视频用 `kind=video`，音频用 `kind=audio`。
- 返回的 `file_id`（如 `uploads/first.png`）用于后续请求。**不能用本地路径或外部 URL**。

## 第 2 步：提交生成

接口：`POST /v1/generations`，JSON。只接受 `mode` 为 `t2v`、`fl2v`、`ref2v`。

| mode | 用途 | 输入字段 |
|---|---|---|
| `t2v`（默认） | 文生视频 | 仅 `prompt` |
| `fl2v` | 首帧/尾帧过渡 | `first_frame` 和/或 `last_frame`（file_id） |
| `ref2v` | 参考素材驱动 | `reference_images`（≤9）、`reference_videos`（≤3，每项 `{"file_id": "..."}`）、`reference_audios`（≤3） |

关键参数：

- `prompt`（必填）：画面与声音描述，想要声音可在末尾写 `Audio: rain.`。
- `duration_seconds`：4–600，默认 5。**超过 15 秒必须开启 `segmentation`。**
- 分辨率，优先级从高到低：
  1. `width` + `height`（32 的倍数，最大 8192）
  2. `resolution`（`360p` `480p` `540p` `720p` `1080p` `1440p` `2160p`）+ `aspect_ratio`（`16:9` `9:16` `1:1` `4:3` `3:4` `21:9`）
  3. 都不传则 1344×768
- `seed`：固定种子可复现，不传则随机。
- `segmentation`：`{"enabled": true, "segment_seconds": 10, "overlap_frames": 48}`。`segment_seconds` 4–15，`overlap_frames` 5–345。
- `filename_prefix`：输出文件名前缀，相对输出目录。

示例（文生短视频）：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/generations \
  -H "Content-Type: application/json" \
  -d '{"mode":"t2v","prompt":"雨夜街道，镜头缓慢向前。Audio: rain.","duration_seconds":4,"resolution":"480p","aspect_ratio":"9:16","seed":42}'
```

示例（30 秒长视频，自动分段）：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/generations \
  -H "Content-Type: application/json" \
  -d '{"mode":"t2v","prompt":"镜头持续缓慢穿过雨夜街道。Audio: steady rain.","duration_seconds":30,"width":608,"height":352,"seed":42,"segmentation":{"enabled":true,"segment_seconds":10,"overlap_frames":48}}'
```

示例（首尾帧）：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/generations \
  -H "Content-Type: application/json" \
  -d '{"mode":"fl2v","prompt":"花朵从含苞到盛开。","first_frame":"uploads/first.png","last_frame":"uploads/last.png","duration_seconds":5}'
```

返回 202，包含 `id`、`queue_number`、`effective_duration_seconds`、`frame_count`、`segment_count`。`effective_duration_seconds` 是帧对齐后的实际时长，可能与请求值略有不同。

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
echo "$s"
```

状态：`queued` → `running` → `succeeded` / `failed` / `canceled`。`progress` 为 0–1 的估计值，可能为 `null`。

## 第 4 步：取结果

任务 `succeeded` 后，`outputs[]` 中每项的 `download_url` 可直接下载：

```bash
URL=$(curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID" | python -c "import sys,json;print(json.load(sys.stdin)['outputs'][0]['download_url'])")
curl -s -o result.mp4 "$URL"
```

- `failed` 时读取 `error` 字段告诉用户原因。
- 若需要取消：`curl -s -X DELETE http://127.0.0.1:8000/v1/tasks/$TASK_ID`。排队中的任务会立即取消，运行中的任务仅在服务端开启 `H3_ALLOW_INTERRUPT_RUNNING=true` 时可取消。

## 注意

- 生成与其他任务共享一个 GPU 队列，排队会叠加。
- 返回 429 或 503 表示本地推理后端拒绝或失败，先重新执行第 0 步检查。
- 分辨率和时长越大，耗时和显存占用越高。
