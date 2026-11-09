---
name: mf-upscale
description: 用 mediaforge 本地服务（http://127.0.0.1:8000）的 SeedVR2 对图片或整段视频做超分辨率放大，完整流程为上传素材、提交目标宽高、轮询任务、下载结果。视频保留原帧率和内嵌音频。当用户要放大、超分、提升清晰度、把图片或视频放大到指定分辨率时使用。
---

# 超分辨率放大

完整流程：**上传图片或视频 → 提交放大 → 轮询任务 → 下载结果**。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz"
```

查看 `upscale.status` 是否为 `ready`。若不是，读取 `upscale.missing_models`、`upscale.missing_dependencies` 告诉用户。

## 第 1 步：上传素材

图片用 `kind=image`，视频用 `kind=video`：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=video" -F "file=@/path/source.mp4"
```

返回的 `file_id`（如 `uploads/source.mp4`）用于下一步。

## 第 2 步：提交放大

接口：`POST /v1/upscales`，JSON。

| 字段 | 必填 | 说明 |
|---|---|---|
| `file_id` | 是 | 图片或视频的 file_id |
| `width` | 是 | 精确输出宽度，64–8192 |
| `height` | 是 | 精确输出高度，64–8192 |
| `seed` | 否 | SeedVR2 种子，用于复现 |
| `filename_prefix` | 否 | 输出文件名前缀 |

`width` 和 `height` 是精确尺寸，不是预设。源比例与目标比例不一致时会拉伸，需要先和用户确认是否保持比例。

```bash
curl -s -X POST http://127.0.0.1:8000/v1/upscales \
  -H "Content-Type: application/json" \
  -d '{"file_id":"uploads/source.mp4","width":1920,"height":1080,"seed":42}'
```

返回 202，包含 `id`、`queue_number`、`width`、`height`。

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
curl -s -o upscaled.mp4 "$URL"
```

图片输入时，输出为图片，扩展名以 `outputs[0].filename` 为准。

## 注意

- 415 表示输入格式不被支持；429/503 表示后端忙或失败，先检查 `/readyz`。
- 放大与生成、音乐等任务共享 GPU 队列。
