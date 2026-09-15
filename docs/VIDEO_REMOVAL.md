# 视频人物移除与背景修复

Gradio `/ui/` 已提供“人物分离 / 背景修复”区域，HTTP 入口为 `POST /v1/removals`。
它与 H3/SeedVR2 共用 GPU 队列，切换任务时释放上一类模型。
独立命令行入口仍为 `python -m minimaxh3_service.video_remove`；独立运行不经过服务队列。

## Gradio 操作

1. 安装下述依赖和权重，重启新版服务，打开 `/ui/`。
2. 展开“人物分离 / 背景修复”，上传视频。
3. 设置参考帧（从 0 开始），点击“加载参考帧”，在图上点击目标框的两个对角。
4. 填写唯一的目标名称，点击“添加目标”；可重复添加最多 8 个目标，并撤销或清空列表。
5. 选择“透明前景”“只保留背景”或“前景和背景”。
6. 完成后从下拉框切换查看各个前景，下载每个目标的透明 MOV、遮罩、预览视频，以及合并移除后的背景。

默认输出最长边 1280，不放大源视频，宽高取 8 的倍数。最多 600 帧，可调整到 900；
超出上限报错，不静默截断。保留固定帧率及第一条可用音轨，音频转为 AAC。
背景以默认最长边 640、每段 80 帧、重叠 16 帧修复并融合，修复区域外使用源帧。
修复精度与输出尺寸分开设置，扩大输出尺寸不会增加模型补全的细节。
“移除区域扩展”只影响背景修复，单位为修复尺寸下的像素。
每个目标从各自参考帧开始跟踪，之前输出透明，不倒推之前的动作。
相互遮挡时仅输出可见部分，不补全被遮挡身体；重叠像素按模型置信度分配给一个目标。
所有目标遮罩取并集后只修复一次背景。阴影和倒影可能仍需进一步处理；不包含此前样例专用的蓝色阴影清理。
运行中取消遵循 `H3_ALLOW_INTERRUPT_RUNNING`，排队任务可直接取消。

## HTTP 接口

先通过 `POST /v1/files` 上传视频，再提交：

```json
{
  "file_id": "minimaxh3/20260915/example.mp4",
  "objects": [
    {"id": "person", "frame_index": 0, "box": [0.1, 0.1, 0.4, 0.9]},
    {"id": "dog", "frame_index": 30, "box": [0.5, 0.5, 0.8, 0.9]}
  ],
  "output_mode": "both",
  "max_side": 1280,
  "repair_size": 640,
  "max_frames": 600,
  "dilation": 8
}
```

`output_mode` 可为 `foreground`、`background` 或 `both`。
`objects` 接受 1–8 个目标，ID 不区分大小写且不能重复，只允许字母、中文、数字、下划线和短横线，最长 40 字符。
`frame_index` 须小于实际视频帧数及 `max_frames`，否则报错。
旧请求仍可传 `box: [x1,y1,x2,y2]`，相当于首帧的单个目标；`box` 与 `objects` 不能同时传。
`objects.json` 记录目标 ID、参考帧、目标框与最终下载文件名的对应关系。
各目标文件带 `object-目标ID-` 前缀，`mask.mp4` 为合并遮罩。
响应包含任务 ID 和队列位置；沿用 `/v1/tasks/{id}` 查询/取消及文件下载接口。
`/v1/models` 的 `capabilities.removal` 提供前景/背景可用状态和缺失依赖。
可用环境变量 `H3_PROPAINTER_PATH`、`H3_SAM2_CHECKPOINT`、`H3_SAM2_CONFIG` 指定安装位置。

## 安装

需要本项目 local 依赖、CUDA 和 PATH 中的 FFmpeg。另安装两个上游项目：

```powershell
git clone https://github.com/sczhou/ProPainter.git .runtime/ProPainter
git clone https://github.com/facebookresearch/sam2.git .runtime/sam2
git -C .runtime/ProPainter checkout e870e79321c31b733e2031af5aa2fb1fe3ac7eec
git -C .runtime/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
$env:SAM2_BUILD_CUDA = "0"
$env:PIP_NO_CACHE_DIR = "1"
python -m pip install --no-build-isolation --no-deps .runtime/sam2
python -m pip install --no-build-isolation hydra-core iopath imageio imageio-ffmpeg addict timm scikit-image yapf future
```

SAM 2 权重下载至 `models/sam2/sam2.1_hiera_tiny.pt`：
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

ProPainter 官方程序首次运行会下载三个权重到其 `weights/` 目录。
也可自行从 https://github.com/sczhou/ProPainter/releases/tag/v0.1.0 下载
`raft-things.pth`、`recurrent_flow_completion.pth`、`ProPainter.pth`。
ProPainter 使用 S-Lab License 1.0，默认许可为非商业用途；商业使用需联系上游授权。

## 框选人物并修复

```powershell
.venv/Scripts/python.exe -m minimaxh3_service.video_remove `
  --video input.mp4 --output outputs/remove-person-01 `
  --propainter .runtime/ProPainter `
  --sam2-checkpoint models/sam2/sam2.1_hiera_tiny.pt `
  --box 0.2 0.1 0.7 0.95
```

框坐标是首帧上归一化的 `左 上 右 下`，左上角为 `(0,0)`，右下角为 `(1,1)`。
示例坐标必须按实际人物位置修改。此独立命令行仍一次选一个对象；多目标请用 Gradio 或 HTTP 接口，不包含中途补点纠正。

输出目录必须不存在。包含：

- `background.mp4`：修复背景，保留源视频第一条可用音轨（转 AAC）。
- `masks/`：白色为选中/待移除区域，黑色为背景。
- `foreground/`：使用 SAM 2 硬遮罩的透明人物 PNG；不是精细头发 Matting。
- `source_frames/` 和 `repair/`：预览、复查及再次处理用的中间结果。
- `job.json`：帧率、尺寸、帧数与完成状态。

默认最长边 640、最多 120 帧；超限会报错，不会静默截断。
可用 `--max-side`、`--max-frames` 调整；提高限制会增加 CPU/GPU 内存。
输出使用处理尺寸，宽高取 8 的倍数。保留分数帧率；可变帧率输入需先转成固定帧率。
默认遮罩扩大 4 像素，可用 `--dilation` 调整。人物投影、倒影不一定会自动选中。

## 使用已修正的遮罩

用 `--masks path/to/masks` 替换 `--box` 和 SAM 2 参数即可跳过分割。
每帧一张 PNG，零填充文件名排序，数量及尺寸必须与处理后的视频帧完全一致。
灰度值大于等于 128 的像素作为待移除区域。
如果需要单独的 ProPainter 环境，用 `--propainter-python path/to/python.exe` 指定。

背景从未露出时，修复是推测结果。先用短片检查遮罩漂移、阴影残留和背景闪烁。

## 本机验证

已在 RTX 4070 Ti 12GB、PyTorch 2.8/CUDA 12.6 上使用 ProPainter 自带
`bmx-trees` 的前 12 帧完成真实 SAM 2 → ProPainter 推理，输出 320×176、24 fps、
12 帧，并确认输出包含音轨。样例只选择骑车人物，因此自行车仍保留。
这验证了运行链路，不代表长视频或复杂遮挡都能无痕修复。
Windows 安装未编译 SAM 2 可选 CUDA 后处理扩展，相关填孔后处理会跳过。

多目标链路另已用同一短片验证骑车人和自行车分别输出，自行车参考帧设为 6：
其前 6 帧遮罩为全透明，两个目标的遮罩不重叠，并集与背景修复输入遮罩一致。
真实推理导出了两个透明前景及预览、两个独立遮罩、合并遮罩、背景和目标清单。
