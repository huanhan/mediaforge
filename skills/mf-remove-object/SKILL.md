---
name: mf-remove-object
description: 用 mediaforge 本地服务（http://127.0.0.1:8000）的 SAM2 分割与 ProPainter 修复，抠出视频中指定物体（最多 8 个）作为前景，或移除它们得到干净背景，支持绿幕或透明输出。完整流程为上传视频、框选物体、提交、轮询、下载结果。当用户要抠像、去除视频中的人或物、擦除水印或物体、分离前景与背景时使用。
---

# 视频前景抠像 / 背景修复

完整流程：**上传视频 → 指定物体框 → 提交 → 轮询任务 → 下载结果**。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz"
```

查看 `removal.status` 是否为 `ready`。`removal.missing` 为空才可用。

## 第 1 步：上传视频

```bash
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=video" -F "file=@/path/source.mp4"
```

记下 `file_id`。

## 第 2 步：指定物体

两种方式：

- **`objects`（推荐）**：最多 8 个，每个为 `{"id": "...", "box": [a,b,c,d], "frame_index": N}`。`frame_index` 是参考帧（0–899），物体在参考帧之前是透明的。`id` 只允许字母、数字、下划线、连字符，最长 40 字符。
- **`box`（旧版）**：只框选第一帧的一个物体，是 4 个数字的数组。

**`box` 的 4 个数字含义未在接口文档中写明**（是 `x1,y1,x2,y2` 还是 `x,y,w,h`，是否归一化也未说明）。正式使用前，先用一个短视频测试，确认遮罩位置正确后再批量处理。

## 第 3 步：提交

接口：`POST /v1/removals`，JSON。

其他参数：

- `output_mode`：`foreground`（默认）、`background`、`both`
- `foreground_background`：`color`（默认）或 `transparent`
- `background_color`：默认 `#00ff00`
- `max_side`：256–1920，默认 1280
- `repair_size`：256–768，默认 640
- `max_frames`：2–900，默认 600，超过会截断
- `dilation`：遮罩外扩像素，0–40，默认 8

```bash
curl -s -X POST http://127.0.0.1:8000/v1/removals \
  -H "Content-Type: application/json" \
  -d '{"file_id":"uploads/source.mp4","objects":[{"id":"person","box":[0.1,0.2,0.4,0.9],"frame_index":0},{"id":"cup","box":[0.6,0.5,0.7,0.65],"frame_index":30}],"output_mode":"both","foreground_background":"transparent","max_side":1280,"dilation":8}'
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

`output_mode=both` 时 `outputs` 含多个文件，按 `filename` 区分前景与背景：

```bash
curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID" | python -c "
import sys,json
for o in json.load(sys.stdin)['outputs']: print(o['filename'], o['download_url'])"
```

逐个下载 `download_url` 即可。

## 注意

- 物体被长时间遮挡或快速运动时，修复可能留下痕迹。
- 超过 `max_frames` 的视频会被截断，长视频需要分段。
- 框的顺序问题未经验证前，不要直接用于正式内容。
