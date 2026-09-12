# 自行打包与构建镜像指南

本文用于后续自己构建发行包。所有命令由你手动执行，示例使用 Windows PowerShell，默认当前版本为 `0.1.0`。

提供两种产物：

| 方式 | 产物 | 使用场景 |
| --- | --- | --- |
| Windows 免安装包 | ZIP + SHA-256 校验文件 | 解压后双击启动 |
| Docker 镜像 | 本地镜像，或 TAR + Compose 配置 | 本机部署、服务器部署、离线交付 |

两种方式均包含 Gradio 视频界面、API、Python 依赖及固定版本推理代码。API 和界面共用一个服务端口：

- 视频界面：<http://127.0.0.1:8000/ui/>
- API 文档：<http://127.0.0.1:8000/docs>
- 服务存活：<http://127.0.0.1:8000/healthz>
- 模型就绪：<http://127.0.0.1:8000/readyz>

包内不包含模型权重、个人 `.env`、输入素材或生成结果。模型目录安排见 [MODELS.md](MODELS.md)。

## 1. 构建前准备

### 工具与空间

| 项目 | Windows ZIP | Docker 镜像 |
| --- | --- | --- |
| 构建系统 | Windows x64 | 支持 Linux 容器的 Docker 主机 |
| Python | 必须使用 Python 3.13 x64 | 使用本文导出脚本时建议 Python 3.13；直接 docker build 不需要宿主 Python |
| uv / Git | 需要，且在 PATH 中 | 宿主直接构建不需要，Dockerfile 内部安装 |
| Visual C++ 运行库 | 需要脚本列出的 x64 DLL | 不需要 Windows DLL |
| Docker | 不需要 | Docker Engine 或 Docker Desktop，支持 Compose GPU 配置 |
| NVIDIA GPU | 运行及完整验证需要 | 运行需要；构建镜像本身不需要 GPU |

首次构建需要联网下载 Python、依赖和推理代码。建议至少预留 50 GB 可用空间；同时保留多个 ZIP、TAR、验证解压目录及 Docker 缓存时需要更多。模型空间另算，包大小以实际构建结果为准。

进入项目根目录（能看到 `pyproject.toml` 和 `Dockerfile`）：

```powershell
Set-Location C:\workspace\pyws\minimaxh3
python --version
uv --version
git --version
```

Windows ZIP 的 `python --version` 必须显示 `3.13.x`。如果默认 Python 不是 3.13，可将本文 `python` 命令改成 `py -3.13`，或使用现有 `.venv\Scripts\python.exe`（先确认版本）。

Docker 构建前另检查：

```powershell
docker version
docker compose version
docker info --format '{{.OSType}}'
```

最后一条应输出 `linux`。Windows Docker Desktop 需开启 Linux 容器模式。

### 依赖与版本

正常重新打包使用仓库已有的 `uv.lock`，脚本会按锁文件安装依赖，不拷贝开发虚拟环境。
Windows 脚本和 Dockerfile 均已包含 `local` 与 `ui` 扩展，因此会打入 Gradio。

只有修改了 `pyproject.toml` 的依赖或版本后，才需要先执行：

```powershell
uv lock
```

发布新版本时，同步修改：

1. `pyproject.toml` 中的 `version`。
2. `minimaxh3_service/__init__.py` 中的 `__version__`。
3. `compose.yaml` 中的镜像标签，例如 `minimaxh3:0.1.1`。

修改后运行 `uv lock`，并将下文命令里的 `0.1.0` 替换为新版本。两个构建脚本会从 `pyproject.toml` 读取产物版本，但不会替你修改 Compose 标签。

每次使用新的输出目录，例如 `dist/release-003`。失败后重试也建议换目录，脚本不会覆盖同名 ZIP、展开目录或 TAR。

## 2. 构建 Windows 免安装 ZIP

### 推荐命令：复用现有推理代码

项目已有完整 `.runtime/ComfyUI` 和 `.runtime/SeedVR2` 时，在根目录执行：

```powershell
python scripts/build_portable.py --output dist/release-003 --runtime-source .runtime
```

脚本要求两个 Git 仓库的 HEAD 与项目固定提交一致，只导出已提交的固定版本代码，不包含这些仓库中的未提交修改。

如果 `.runtime` 不存在、不完整或提交不匹配，省略该参数，让脚本自动获取固定版本：

```powershell
python scripts/build_portable.py --output dist/release-003
```

以上两条选一条，不要对同一输出目录重复执行。

脚本依次完成：下载便携 Python 3.13.7、复制运行库、安装锁定依赖、复制应用、导出推理代码、生成文件清单、压缩 ZIP 和计算 SHA-256。不会下载模型。

如果提示缺少 DLL，先按错误信息准备完整的 x64 DLL 目录，再指定路径：

```powershell
python scripts/build_portable.py --output dist/release-004 --runtime-source .runtime --vc-runtime C:\BuildTools\vc-runtime-x64
```

`--vc-runtime` 默认是 `C:\Windows\System32`。实际要求以 `scripts/build_portable.py` 的 `VC_RUNTIME_FILES` 为准，其中包含 OpenMP DLL；只安装普通 VC++ 运行库不一定补齐全部文件。

### 构建结果

```text
dist/release-003/
  minimaxh3-0.1.0-windows-x64.zip
  minimaxh3-0.1.0-windows-x64.zip.sha256
  minimaxh3-0.1.0-windows-x64/
    start.cmd
    check.cmd
    launch.py
    .env.example
    python/
    app/
    runtime/
    models/
    data/
    requirements-lock.txt
    release-manifest.json
```

交付 ZIP 和 `.zip.sha256` 即可。展开目录用于本机检查，不必再单独发一份。

### 校验与试运行

检查压缩包 SHA-256：

```powershell
Get-FileHash dist/release-003/minimaxh3-0.1.0-windows-x64.zip -Algorithm SHA256
Get-Content dist/release-003/minimaxh3-0.1.0-windows-x64.zip.sha256
```

两处哈希应一致（大小写不影响）。

自动验证需要 Windows NVIDIA GPU 和兼容驱动；它会校验文件、检查推理依赖与 CUDA，并临时启动服务验证 HTTP，结束后自动停止。验证目标目录必须尚不存在：

```powershell
python scripts/verify_portable.py dist/release-003/minimaxh3-0.1.0-windows-x64.zip --destination dist/verify-003
```

已有完整模型时，可追加外部模型目录，进一步验证模型就绪和 CUDA 运算：

```powershell
python scripts/verify_portable.py dist/release-003/minimaxh3-0.1.0-windows-x64.zip --destination dist/verify-003-models --models C:\workspace\pyws\minimaxh3\models
```

验证脚本不执行完整视频生成，也没有浏览器视觉检查。交付前再手动检查：

1. 解压 ZIP 到可写目录，按 `MODELS.md` 放好模型，或设置外部模型目录。
2. 双击 `check.cmd` 查看 CUDA、依赖和缺少的模型。
3. 双击 `start.cmd`，保持控制台开启，确认打印视频界面和 API 文档地址。
4. 打开 `/ui/`，点击“检查服务”，确认模式和 Turbo 状态；使用短时长、低分辨率生成一个视频，检查预览与下载。
5. 打开 `/docs`，确认原有 API 可用。关闭控制台或 Ctrl+C 可停止服务。

如需配置，在**解压后的包目录**执行一次：

```powershell
Copy-Item .env.example .env
notepad .env
```

已有 `.env` 时直接编辑，不重复复制覆盖。示例：

```dotenv
H3_PORT=8001
H3_MODEL_DIR=D:/AI/models
H3_UPSCALE_MODEL_DIR=D:/AI/models/SEEDVR2
H3_DATA_DIR=D:/AI/h3-data
```

修改后重启，地址变为 `http://127.0.0.1:8001/ui/` 与 `/docs`。
如果设置 `H3_API_KEY`，Gradio 用户名为 `h3`，密码为该 Key。系统环境变量优先于包目录的 `.env`。

## 3. 构建 Docker 镜像

目标镜像为 `linux/amd64`。以下两种方式按需选择。

### 方式 A：构建并导出交付包

在项目根目录执行：

```powershell
python scripts/build_docker.py --output dist/release-003
```

脚本执行 Docker 构建、检查镜像平台、导出 TAR、生成校验文件，并复制部署说明。输出：

```text
dist/release-003/
  minimaxh3-0.1.0-linux-amd64.docker.tar
  minimaxh3-0.1.0-linux-amd64.docker.tar.sha256
  minimaxh3-0.1.0-linux-amd64.docker.image.json
  compose.yaml
  DISTRIBUTION.md
  MODELS.md
  THIRD_PARTY_NOTICES.md
```

把这些文件一起交付。Dockerfile 构建时会获取固定版本推理代码，不使用宿主 `.runtime`。

镜像已经构建完，只想导出时：

```powershell
python scripts/build_docker.py --output dist/release-004 --skip-build
```

`--skip-build` 只导出本地已有 `minimaxh3:0.1.0`，不会包含尚未重新构建的源码修改。

### 方式 B：只构建本地镜像，不导出 TAR

```powershell
docker build --platform linux/amd64 --progress plain -t minimaxh3:0.1.0 .
docker image inspect minimaxh3:0.1.0 --format '{{.Os}}/{{.Architecture}}'
```

应输出 `linux/amd64`。以后需要交付 TAR 时，使用上面的 `--skip-build` 命令即可。

## 4. 运行和验证 Docker 镜像

运行机器需要 NVIDIA 驱动和 Docker GPU 支持。Linux 宿主还需 NVIDIA Container Toolkit；Windows 使用支持 GPU 的 Docker Desktop WSL2 Linux 容器环境。

### 导入并启动

接收方先进入包含 TAR 和 `compose.yaml` 的目录：

```powershell
docker load -i minimaxh3-0.1.0-linux-amd64.docker.tar
```

构建机已存在镜像时不需要再导入。确保当前目录中有 `compose.yaml`，并准备好 `models/`、`data/`。模型按 `MODELS.md` 放置。

也可以在 Compose 同目录新建或编辑 `.env`，使用现有模型目录：

```dotenv
H3_MODELS_PATH=D:/AI/models
H3_DATA_PATH=D:/AI/h3-data
H3_PUBLISHED_PORT=8000
```

Linux 路径示例为 `/srv/h3/models` 和 `/srv/h3/data`。运行前确保目录存在。

启动并查看日志：

```powershell
docker compose up -d --no-build --pull never
docker compose ps
docker compose logs -f
```

`logs -f` 中按 Ctrl+C 只退出日志跟踪，后台容器继续运行。停止服务使用：

```powershell
docker compose down
```

交付包中的 `compose.yaml` 保留了 `build: .`，但没有构建源码，所以在交付目录启动时须保留 `--no-build --pull never`，使用导入的本地镜像。

### 同端口访问与验证

默认访问 `/ui/` 和 `/docs`，只需要开放宿主的 8000 端口。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
Invoke-RestMethod http://127.0.0.1:8000/readyz
```

`/healthz` 正常只表示进程存活；缺模型时 `/readyz` 可以返回 503。补齐模型后，在界面做一次实际视频生成、预览和下载，才能验证完整推理链路。

Docker 改外部端口请设置 `H3_PUBLISHED_PORT=8001`，然后重新执行启动命令。容器内部仍为 8000，浏览器访问 `http://127.0.0.1:8001/ui/`。控制台打印的是容器内部地址，需要按宿主映射端口访问。

局域网访问可在 Compose `.env` 中设置 `H3_BIND_ADDRESS=0.0.0.0` 和 `H3_API_KEY`，浏览器使用宿主局域网 IP。不要给同一 GPU 配置多个 Uvicorn worker。

## 5. 每次改代码后重新发布

1. 保存源码；若改版本，同步上述三个版本位置；若改依赖或版本，运行 `uv lock`。
2. 选择新的输出目录，例如 `dist/release-004`。
3. Windows 重新执行 `build_portable.py`；Docker 重新执行 `build_docker.py` 或 `docker build`，不要用 `--skip-build` 代替重新构建。
4. 检查校验文件，试启动，并验证 `/ui/`、`/docs` 和一次真实生成。
5. 交付 ZIP/TAR、校验文件及配套说明。保留这一版源码、锁文件和构建产物，便于追溯。

可以先做快速代码回归（这不能替代最终包的实际验证）：

```powershell
uv run --extra local --extra ui --extra dev python -m pytest tests/test_packaging.py tests/test_ui.py tests/test_api.py
```

## 6. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 提示 Release already exists / Artifact exists | 换新的输出目录；不要覆盖正在使用的发行目录 |
| 提示 Python 版本不符 | 使用 Python 3.13 x64 执行 Windows 打包脚本 |
| 找不到 uv / Git | 安装对应工具并重新打开终端，确认 PATH 生效 |
| 提示 Missing x64 redistributable DLLs | 按报错补齐脚本要求的 DLL，必要时使用 `--vc-runtime` |
| `.runtime` 提交不匹配 | 去掉 `--runtime-source`，让脚本获取固定版本代码 |
| 锁文件过期 | 确认依赖修改后执行 `uv lock`，再重新构建 |
| Docker 无法连接 | 启动 Docker Desktop/Engine，检查是否为 Linux 容器模式 |
| Docker 下载失败 | 检查基础镜像、PyPI、PyTorch 源和 GitHub 网络，恢复后重新构建 |
| 提示 GPU 不可用 | 检查 NVIDIA 驱动、CUDA 可用性及 Docker GPU 支持 |
| `/ui/` 返回 404 | 确认使用的是重新构建的新包；检查依赖已包含 Gradio |
| `/readyz` 返回 503 | 查看缺失模型或依赖，核对挂载目录和文件名 |
| 参数不生效 | Turbo 会覆盖步数、采样器和调度器；关闭 `H3_ENABLE_ACCELERATION_LORA` 后重启服务 |
| 端口占用 | Windows 改 `H3_PORT`；Docker 改 `H3_PUBLISHED_PORT` |
| 新 Docker 源码没有生效 | 重新构建镜像，确认 Compose 标签匹配，重新创建容器；不要仅导出旧镜像 |

进一步的模型布局和使用说明见 [MODELS.md](MODELS.md)、[DISTRIBUTION.md](DISTRIBUTION.md)。
