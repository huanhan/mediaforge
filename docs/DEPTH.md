# 本地视频转深度

`POST /v1/depths` 使用官方 Video Depth Anything Small（vits）进行时序深度估计，与 H3、超分、移除、音乐任务共用队列。独立推理子进程退出后释放显存，不需要另启模型服务。

## 安装

在项目根目录执行（需要 Git；首次安装需要网络）：

```powershell
uv sync --extra local --extra depth --extra ui --extra dev
uv run --no-sync python -m minimaxh3_service.setup_depth
```

已有环境可用 `uv pip install --python .venv/Scripts/python.exe "easydict>=1.13,<2"` 补充依赖。安装脚本固定官方源码和模型版本，不会覆盖已有权重或切换已有源码目录。

默认路径及环境变量：

| 环境变量 | 默认值 |
| --- | --- |
| `H3_DEPTH_RUNTIME_PATH` | `.runtime/Video-Depth-Anything` |
| `H3_DEPTH_MODEL_PATH` | `models/depth/video_depth_anything_vits.pth` |
| `H3_DEPTH_MAX_FRAMES` | `720` |
| `H3_DEVICE` | `cuda`（也支持 `cuda:0`、`cpu`） |

仅支持对应的 **Small 相对深度权重**，不可直接换成 Base、Large 或 metric 权重。修改后重启 API。`GET /readyz` 的 `depth` 字段及 `GET /v1/models` 的 `capabilities.depth` 可查看安装状态。总体 ready 状态仍由 H3 决定；H3 未安装不阻止单独提交深度任务。安装就绪检查不替代实际 CUDA 推理验证。

## 调用

1. 通过 `POST /v1/files` multipart 上传视频（`kind=video`，文件字段 `file`）。
2. 使用返回的 `file_id` 提交：

```http
POST /v1/depths
Content-Type: application/json

{
  "file_id": "minimaxh3/20260916/source-xxxx.mp4",
  "input_size": 392,
  "max_resolution": 768,
  "save_raw": true
}
```

返回 `202`：`{"id":"...","status":"queued","queue_number":1}`。

3. 轮询 `GET /v1/tasks/{id}` 或订阅 `/v1/tasks/{id}/events`。成功后从 `outputs[].download_url` 下载。`DELETE /v1/tasks/{id}` 可取消；运行中取消需 `H3_ALLOW_INTERRUPT_RUNNING=true`。已配置 API key 时沿用现有认证方式。

## 输出与限制

- 无声灰度 MP4：近处更亮，远处更暗。整段共用归一化范围。保留解码帧数和源名义 FPS；可变帧率源转为恒定帧率，不逐帧保留原时间戳。
- `save_raw=true` 额外返回 NPZ：`depths` 为 `[T,H,W]` float32 **相对逆深度**，越大越近，不是米数；包含 `fps`、精确帧率分子/分母、深度类型及可视化范围。
- 默认 `input_size=392`，可选 `252`、`518`；较大值增加显存消耗。`max_resolution` 限制输出长边，默认 768，最大 1280，不放大源视频，尺寸向下对齐偶数。
- 为限制内存使用，默认最多 720 帧（24 FPS 时 30 秒）。超限任务失败，不会静默截断。模型会在内存中保留序列；长视频应分段或评估内存后提高服务端上限。
- 进度为阶段性估计，推理期间可能停留在 5%。失败信息通过任务 `error` 返回；未安装返回 `503 depth_not_ready`。

官方项目：https://github.com/DepthAnything/Video-Depth-Anything
