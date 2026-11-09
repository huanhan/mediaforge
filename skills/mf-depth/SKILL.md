---
name: mf-depth
description: 用 mediaforge 本地服务（http://127.0.0.1:8000）把视频转换为相对逆深度图（近处亮、远处暗），输出无声灰度 MP4，可选 float32 NPZ。完整流程为上传视频、提交参数、轮询任务、下载结果。当用户要视频深度图、深度估计、深度遮罩、景深效果或 Z 通道时使用。
---

# 视频深度图

完整流程：**上传视频 → 提交深度估计 → 轮询任务 → 下载结果**。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz"
```

查看 `depth` 字段是否就绪（Video Depth Anything 已安装，并能看到帧数上限）。未就绪则告诉用户，不要提交。

## 第 1 步：上传视频

```bash
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=video" -F "file=@/path/source.mp4"
```

记下 `file_id`。

## 第 2 步：提交

接口：`POST /v1/depths`，JSON。

| 字段 | 必填 | 说明 |
|---|---|---|
| `file_id` | 是 | 视频的 file_id |
| `input_size` | 否 | `252`、`392`（默认）、`518`。越大越吃显存 |
| `max_resolution` | 否 | 输出最长边，64–1280，默认 768，不会放大源视频 |
| `save_raw` | 否 | `true` 时额外返回 float32 NPZ（`depths`、`fps`），默认 `false` |

```bash
curl -s -X POST http://127.0.0.1:8000/v1/depths \
  -H "Content-Type: application/json" \
  -d '{"file_id":"uploads/source.mp4","input_size":392,"max_resolution":768,"save_raw":false}'
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

```bash
URL=$(curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID" | python -c "import sys,json;print(json.load(sys.stdin)['outputs'][0]['download_url'])")
curl -s -o depth.mp4 "$URL"
```

`save_raw=true` 时，`outputs` 中还会有 NPZ 文件，按 `filename` 后缀识别后逐个下载。

## 注意

- 输出是**相对**逆深度，只在同一段视频内可比较，不是绝对距离。
- 输出视频无声。
- `input_size` 调大需要更多显存，12GB 显卡要留意。
