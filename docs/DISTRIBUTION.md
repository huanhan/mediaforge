# 发行包使用

## Windows 免安装包

适用于 Windows 10/11 x64 和 NVIDIA GPU。包内包含 Python 3.13、CUDA 12.6 版
PyTorch 2.8、媒体编解码库、Visual C++ 运行库和固定版本推理后端。
无需安装 Python、Git、uv、FFmpeg 或 CUDA Toolkit；系统必须已有兼容 CUDA 12.6
的 NVIDIA 驱动，建议使用当前稳定版驱动。12 GB 显存建议搭配至少 64 GB 内存和充足页面文件。

1. 用 Windows 资源管理器或支持 ZIP64 的解压工具完整解压 ZIP 到可写目录。
2. 按同目录 `MODELS.md` 下载并放置模型。
3. 双击 `start.cmd`，保持控制台窗口开启。
4. 打开 <http://127.0.0.1:8000/docs>，就绪状态为 <http://127.0.0.1:8000/readyz>。

双击 `check.cmd` 检查 CUDA、运行时及模型路径。模型缺失时服务仍可启动，
对应生成接口返回未就绪；补齐模型后即可使用。关闭窗口或按 Ctrl+C 停止服务。

需要改端口或目录时，把 `.env.example` 复制为 `.env` 并编辑。
例如 `H3_PORT=8001`；路径可用 `D:/AI/models`。系统环境变量优先于 `.env`。
默认只监听本机；局域网访问可设置 `H3_HOST=0.0.0.0` 和 `H3_API_KEY`。
服务运行本身不需要联网；Swagger 文档页面的默认前端资源来自 CDN。

## Docker

Docker 方案需要 Docker Engine/Desktop 和 NVIDIA 驱动。
Linux 宿主需安装 NVIDIA Container Toolkit；Windows 使用 Docker Desktop 的 WSL2 Linux 容器与 GPU 支持。
应用依赖及 CUDA 用户态运行库均在镜像内。

导入离线镜像：

```bash
docker load -i minimaxh3-0.1.0-linux-amd64.docker.tar
```

在包含 `compose.yaml` 的目录准备 `models/` 和 `data/`，然后启动：

```bash
docker compose up -d --no-build --pull never
docker compose logs -f
docker compose down
```

默认地址为 <http://127.0.0.1:8000/docs>。可在 Compose 同目录的 `.env` 设置
`H3_MODELS_PATH`、`H3_DATA_PATH`、`H3_PUBLISHED_PORT`、`H3_API_KEY`。
`H3_BIND_ADDRESS=0.0.0.0` 可开放局域网访问。模型挂载为只读，数据挂载为可写。

不使用 Compose 时：

```bash
docker run --rm --gpus all --shm-size=8g -p 127.0.0.1:8000:8000 \
  --mount type=bind,source=/absolute/path/models,target=/models,readonly \
  --mount type=bind,source=/absolute/path/data,target=/data \
  minimaxh3:0.1.0
```

`/healthz` 检查服务存活，Docker 健康状态不代表模型已经齐全；推理就绪请检查 `/readyz`。
不要配置多个 Uvicorn worker，同一块 GPU 使用一个服务实例。

## 重新构建

以下是构建机器的要求，不适用于下载发行包的用户。

Windows 构建机器需要 Python 3.13、uv、Git、x64 Visual C++ 运行库和网络：

```powershell
python scripts/build_portable.py --runtime-source .runtime
```

省略 `--runtime-source` 会自动获取固定提交的推理后端源代码，不下载模型。
`--vc-runtime` 可以指定包含所需 x64 可再发行 DLL 的目录。
构建脚本使用 `uv.lock` 中的版本与哈希安装依赖，不拷贝开发虚拟环境。
输出位于 `dist/`，包含 ZIP、SHA-256 校验文件和已展开的发行目录。
已有同名产物时构建会退出，以免覆盖使用中的文件；可用 `--output` 指定新目录。

Docker 构建与导出：

```bash
docker build --platform linux/amd64 -t minimaxh3:0.1.0 .
docker save -o dist/minimaxh3-0.1.0-linux-amd64.docker.tar minimaxh3:0.1.0
```

Docker 构建上下文采用允许清单，只有应用源码、依赖锁和运行时导出脚本会被发送。
发行包不读取项目 `models/`、`data/`、`outputs/`、`matrix-results/` 或 `.env`。
运行时仅导出固定提交的代码与必需静态资源，不包含 Git 历史和用户生成的文件。
