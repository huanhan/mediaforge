# 模型下载与目录

Docker 镜像和 Windows 免安装包均不包含模型权重，也不会在启动时下载模型。
用户自行下载并解压到 `models/`，保留下面的子目录。不要解压成 `models/models/`。

H3 下载地址：<https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main>

SeedVR2 下载地址：<https://huggingface.co/numz/SeedVR2_comfyUI/tree/main>

```text
models/
  diffusion_models/
    minimax_h3_fl2va_pruned_int8_convrot.safetensors
    minimax_h3_ref2va_pruned_int8_convrot.safetensors       [Ref2V 可选]
  text_encoders/
    qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
  vae/
    minimax_h3_video_vae_fp16.safetensors
    minimax_h3_audio_vae_fp32.safetensors
  loras/
    minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
    minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors [Ref2V 可选]
  SEEDVR2/
    seedvr2_ema_3b_fp8_e4m3fn.safetensors
    ema_vae_fp16.safetensors
```

仅运行 T2V/FL2V 时无需下载 Ref2V 和 SeedVR2 权重。默认开启 Turbo LoRA，
如果不下载 LoRA，需配置 `H3_ENABLE_ACCELERATION_LORA=false`。
H3 基础权重约 43 GB，FL2V Turbo LoRA 约 2 GB，Ref2V 主模型另需约 21 GB。

Windows：上述 `models/` 位于 `start.cmd` 同一目录；也可在 `.env` 中设置
`H3_MODEL_DIR=D:/AI/models` 和 `H3_UPSCALE_MODEL_DIR=D:/AI/models/SEEDVR2`。

Docker：把上述 `models/` 挂载到容器 `/models`，Compose 默认使用当前目录的
`models/`。自定义宿主目录时设置 `H3_MODELS_PATH`。

输入文件保存在 `data/input/`，输出保存在 `data/output/`，均在首次运行时创建，
不属于发行包内容。Windows 双击 `check.cmd` 或查询服务的 `/readyz` 可查看缺少的模型。

下载与使用模型须遵守相应仓库的许可；H3 许可见
<https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE>。
