# 本地视频骨骼提取

`POST /v1/poses` 只接收源视频，使用本地 DWPose 提取人体关键点并输出静音骨骼可视化 MP4。角色图、动作迁移和 MimicMotion 已从当前服务流程移除。

先安装本地 DWPose 资源：

```powershell
uv sync --extra local --extra pose
uv run python -m minimaxh3_service.setup_pose
```

提交前通过 `POST /v1/files` 上传视频，然后请求：

```json
{
  "file_id": "uploads/source.mp4",
  "output_fps": 12,
  "start_seconds": 0,
  "duration_seconds": 10,
  "resolution": 384
}
```

`duration_seconds` 省略或设为 `0` 时，会从 `start_seconds` 处理到原视频结尾。

接口返回任务 ID。轮询 `GET /v1/tasks/{id}`，成功时只有一个输出文件，即骨骼视频。DWPose 默认使用 CUDAExecutionProvider 在 GPU 上推理，视频读取和 MP4 编码在 CPU；服务仍使用共享任务队列。设置 `H3_DEVICE=cpu` 可切换为 CPU。

源视频应包含清晰、单人的人体姿态。动物或纯卡通角色需要动物关键点模型，当前接口会明确报告“无可用人体姿态”。
