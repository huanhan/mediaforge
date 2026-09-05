# 打包构建说明

这个项目可以打成两种包：Windows ZIP，解压后双击启动；Docker 镜像，导入后启动容器。
两种包都自带程序依赖，不包含模型权重、用户输入和生成结果。模型由用户单独下载。

## 1. 先准备好工具

打 Windows 包：使用 Windows x64，装好 Python 3.13、uv、Git 和 x64 Visual C++ 运行库。

打 Docker 镜像：装好 Python 3.13 和 Docker，先启动 Docker。Windows 上使用 Linux 容器模式。

打包时需要联网下载依赖，两种包都做的话，磁盘建议至少留 50 GB。**这些是打包电脑的要求，普通用户不用照着安装。**

下面的命令都在项目根目录执行，也就是能看到 `pyproject.toml` 的目录。
示例把新包放到 `dist/release-001/`，下次打包把 `001` 改成 `002` 即可。

## 2. 打 Windows 免安装包

```powershell
python scripts/build_portable.py --output dist/release-001
```

脚本会下载便携版 Python，安装锁定版本的依赖，准备推理代码，然后压成 ZIP。不会下载模型。
项目已有 `.runtime/` 时，可以在命令末尾加 `--runtime-source .runtime`，复用已有的固定版本代码。

完成后会看到：

```text
dist/release-001/
  minimaxh3-0.1.0-windows-x64.zip
  minimaxh3-0.1.0-windows-x64.zip.sha256
  minimaxh3-0.1.0-windows-x64/          已解压的程序目录
```

把 ZIP 和旁边的 `.sha256` 文件发给用户即可。`.sha256` 用来核对文件是否传坏。
本次 ZIP 约 3.41 GB，以实际构建结果为准。

## 3. 打 Docker 镜像

```powershell
python scripts/build_docker.py --output dist/release-001
```

脚本会构建 `minimaxh3:0.1.0` 镜像，再导出成 TAR 文件，并一起放好启动配置和使用说明。
镜像已经构建好、只想导出时，在命令末尾加 `--skip-build`。

主要文件是：

```text
dist/release-001/
  minimaxh3-0.1.0-linux-amd64.docker.tar
  minimaxh3-0.1.0-linux-amd64.docker.tar.sha256
  compose.yaml
  DISTRIBUTION.md
  MODELS.md
  THIRD_PARTY_NOTICES.md
```

把以上文件一起交付。TAR 可以离线导入，本次约 7.23 GB。

## 4. 用户拿到包后怎么用

**Windows：**解压 ZIP，按 [模型目录说明](MODELS.md) 把模型放进 `models/`，双击 `start.cmd`。
无需安装 Python、Git 或 CUDA Toolkit，电脑需要 NVIDIA 显卡和兼容的驱动。
双击 `check.cmd` 可以查看缺少哪些模型。

**Docker：**电脑需要 Docker 和 GPU 支持；Linux 宿主还需要 NVIDIA Container Toolkit。
在 TAR 和 `compose.yaml` 所在目录准备好 `models/`、`data/`，再执行：

```bash
docker load -i minimaxh3-0.1.0-linux-amd64.docker.tar
docker compose up -d --no-build --pull never
```

两种方式启动后都访问 <http://127.0.0.1:8000/docs>。
模型是否放齐，看 <http://127.0.0.1:8000/readyz>；缺模型不影响服务启动，但对应功能不能生成。
输入保存在 `data/input/`，结果保存在 `data/output/`。

## 常见问题

- 提示文件已存在：换一个新的 `--output` 目录，脚本不会覆盖旧包。
- 提示缺少 DLL：按报错补齐打包电脑的 x64 运行库，也可用 `--vc-runtime` 指定 DLL 目录。
- Docker 连不上：先启动 Docker Desktop；拉取镜像超时则检查网络或镜像加速配置。
- 要改版本号：修改 `pyproject.toml` 的 `version`，执行 `uv lock`，同时更新 `compose.yaml` 的镜像标签，再打包。

端口、模型路径和其他配置见 [使用说明](DISTRIBUTION.md)。
