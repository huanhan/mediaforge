---
name: mf-pose
description: 用 mediaforge 本地服务（http://127.0.0.1:8000）的 DWPose 从视频中提取骨架，输出无声骨架可视化 MP4，可指定时间段、输出帧率和分辨率。完整流程为上传视频、提交参数、轮询任务、下载结果。当用户要提取人体姿态、骨架视频、骨骼动画参考或姿态控制素材时使用。
---

# 骨架姿态提取

完整流程：**上传视频 → 提交提取 → 轮询任务 → 下载结果**。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz"
```

确认服务为 `ready`。骨架提取所需依赖（DWPose）若缺失，会体现在 `missing_*` 字段中。

## 第 1 步：上传视频

```bash
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=video" -F "file=@/path/dance.mp4"
```

记下 `file_id`。

## 第 2 步：提交

接口：`POST /v1/poses`，JSON。

| 字段 | 必填 | 说明 |
|---|---|---|
| `file_id` | 是 | 视频的 file_id |
| `output_fps` | 否 | 输出帧率，不传则沿用源视频 |
| `start_seconds` | 否 | 起始秒数，0–600，默认 0 |
| `duration_seconds` | 否 | 处理时长，不传则到结尾 |
| `resolution` | 否 | `256`、`384`（默认）、`512`、`576` |

```bash
curl -s -X POST http://127.0.0.1:8000/v1/poses \
  -H "Content-Type: application/json" \
  -d '{"file_id":"uploads/dance.mp4","start_seconds":2,"duration_seconds":10,"resolution":512}'
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
curl -s -o pose.mp4 "$URL"
```

## 注意

- 输出是骨架可视化视频，不是 3D 骨骼数据文件。
- 人物被遮挡或多人重叠时，骨架可能缺失或跳变。
- 作为控制素材使用时，可把结果交给 `mf-generate-video` 的 `ref2v` 模式。
