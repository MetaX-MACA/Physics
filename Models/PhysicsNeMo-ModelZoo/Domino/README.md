# 沐曦 GPU 运行 PhysicsNeMo DoMINO 说明文档

本目录采用文档/容器适配模式提供 [PhysicsNeMo v2.0.0 DoMINO](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/domino) 的沐曦 GPU 使用说明。目录不包含 PhysicsNeMo 源码、DoMINO 数据集、模型权重或 MetaX Warp；使用者按下文从上游获取源码，Dockerfile 构建时固定到提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

## 一、DoMINO 简介

DoMINO 是 PhysicsNeMo 的外部空气动力学代理模型案例，用于根据 STL 几何预测车辆表面压力、壁面剪切应力及周围体场流动量。上游源码为 [PhysicsNeMo v2.0.0](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0)，固定提交为 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

上游案例使用 DrivAerML 数据集；数据集、模型权重和 Warp 不随本适配发布，须按[上游 DoMINO README](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/domino)及其数据来源说明自行获取并遵守相应条款。

本说明仅验证 DoMINO 模型安装、构造和上游单元测试，不覆盖真实数据训练、推理或评估。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的 [maca-pytorch 镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=torch&dimension=docker) 启动容器。请使用沐曦 PyTorch 基础镜像，而非 PhysicsNeMo 官方容器镜像；本文档固定使用 MACA 3.8.1.2 对应 tag：

```bash
docker run -it --name physicsnemo-domino-3.8.1.2 \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=8G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /host_dir:/workspace \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.8-py312-ubuntu24.04-amd64 \
  /bin/bash
```

> 可根据需要使用更新版本镜像，详情参见[沐曦开发者社区](https://developer.metax-tech.com)

**参数说明：**

- `--device=/dev/mxcd --device=/dev/dri`：将沐曦 GPU 设备节点传入容器。
- `--group-add video`：将容器进程加入 `video` 组，以便访问 GPU 设备。
- `--shm-size=8G`：设置共享内存大小，供数据加载和模型测试进程使用。
- `-v /host_dir:/workspace`：挂载工作目录；`/host_dir` 是主机路径，`/workspace` 是容器内路径。

### 2.2 验证环境

进入容器后，先验证沐曦 PyTorch 与 GPU 是否可用：

```bash
python -c "import torch; assert torch.cuda.is_available(); print('PyTorch version:', torch.__version__); print('CUDA OK:', torch.cuda.get_device_name(0))"
```

命令应打印 PyTorch 版本和 `cuda:0` 对应的沐曦 GPU 名称。例如：

```text
PyTorch version: <base-image-pytorch-version>
CUDA OK: <MetaX GPU name>
```

本说明的验证环境输出为 `PyTorch version: 2.8.0+metax3.8.1.0`、`CUDA OK: MetaX C500`；实际版本号和设备名称以基础镜像及硬件型号为准。若 `torch.cuda.is_available()` 为 `False`，请先检查宿主机驱动、设备节点挂载和容器启动参数，再继续安装 PhysicsNeMo。

### 2.3 配置 MACA 编译环境

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/mxgpu_llvm/lib:${CUCC_PATH}/lib:${CUCC_PATH}/lib64:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH}
export PATH=${CUCC_PATH}/tools:${CUDA_PATH}/bin:/opt/conda/bin:${PATH}
export PYTHONPATH=/root/physicsnemo/
export PYTORCH_DEFAULT_NCHW=1
export CUBLAS_WORKSPACE_CONFIG=:4096:16
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export DISABLE_PYTORCH_USE_FLASHATTN=1
export PYTORCH_ENABLE_FA_CPP_INTERFACE=1
export MCDNN_USE_DETERMINISTIC_ALGO=1
```

### 2.4 安装 PhysicsNeMo 与 DoMINO 依赖

```bash
apt update && apt install -y --no-install-recommends ca-certificates git build-essential libgl1 libx11-6 libxrender1 xz-utils
git clone https://github.com/NVIDIA/physicsnemo.git /root/physicsnemo && cd /root/physicsnemo
git checkout 1ca85d65ac2ce28ea9762910c09a954c08a37140
test "$(git rev-parse HEAD)" = "1ca85d65ac2ce28ea9762910c09a954c08a37140"
pip install --upgrade pip hatchling
pip install onnx pandas nvtx treelib numpy tqdm requests GitPython s3fs packaging timm einops h5py cftime jaxtyping termcolor hydra-core tensordict==0.10.0 omegaconf importlib-metadata gdown matplotlib pytest pydantic==2.12.3 pydantic-core==2.41.4
pip install --no-deps -e .
```

PhysicsNeMo v2.0.0 会在 `physicsnemo/__init__.py` 中导入 `warp`，因此 DoMINO 测试前必须安装与 MACA、PyTorch、Python 和 amd64 架构匹配的 MetaX Warp。请在宿主机从[沐曦资源中心](https://developer.metax-tech.com/softnova)下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，放入 `docker run` 所挂载的 `/host_dir`，容器内对应路径为 `/workspace`。压缩包内包含 `warp_lang-1.8.1-py3-none-manylinux_2_28_x86_64.whl`。在容器内执行：

```bash
export WARP_ARCHIVE=/workspace/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz
test -f "${WARP_ARCHIVE}"
mkdir -p /tmp/maca-warp
tar -xJf "${WARP_ARCHIVE}" -C /tmp/maca-warp
export WARP_WHEEL=/tmp/maca-warp/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64/warp_lang-1.8.1-py3-none-manylinux_2_28_x86_64.whl
test -f "${WARP_WHEEL}"
pip install --force-reinstall --no-deps "${WARP_WHEEL}"
python -c "import warp, physicsnemo; print('Warp version:', warp.__version__); print('PhysicsNeMo import: OK')"
```

`--no-deps` 用于避免 Warp 安装过程替换基础镜像中的沐曦定制版 PyTorch。不要使用 PyPI 的通用 `warp-lang` wheel。Warp 初始化时可能打印无法加载 NVIDIA `libcuda.so` 或 `cuGraphicsGLRegisterBuffer` 的探测提示；该提示来自 Warp 的 NVIDIA CUDA 探测逻辑，不影响后续 DoMINO 测试。

## 三、DoMINO 模型适配验证

该验证不需要数据集或权重，不修改上游文件。运行 PhysicsNeMo 上游 DoMINO 单元测试：

```bash
cd /root/physicsnemo
pytest -q test/models/domino/test_domino.py --disable-warnings --maxfail=1
```

本次验证结果为 `21 passed`。如需快速确认模型可构造，也可执行：

```bash
python - <<'PY'
from physicsnemo.models.domino import DoMINO
import torch
model = DoMINO(input_features=3, output_features_surf=1).to("cuda:0")
print("DoMINO constructed:", model.__class__.__name__)
PY
```

预期打印 `DoMINO constructed: DoMINO`。真实缓存、训练和推理命令及数据条款以[上游案例说明](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/domino)为准。

## 四、Dockerfile 构建

本目录提供可直接构建的Dockerfile。构建前需要由使用者手动准备基础镜像和 MetaX Warp 资源包；PhysicsNeMo 源码无需手动下载，Dockerfile 会在构建时从 GitHub 获取并固定到 PhysicsNeMo v2.0.0 提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

1. 在[沐曦资源中心](https://developer.metax-tech.com/softnova)获取并拉取 MACA 3.8.1.2 对应的 PyTorch 基础镜像。
2. 从同一资源中心手动下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，将其放在 Dockerfile 同级目录。该压缩包不包含在本发布包中。Dockerfile 会自动解压并安装其中的 `warp_lang-1.8.1-py3-none-manylinux_2_28_x86_64.whl`；缺少该文件时，Docker 构建会在 `COPY` 步骤失败。
3. Dockerfile 将自动安装系统与 Python 依赖，执行 `git clone https://github.com/NVIDIA/physicsnemo.git`，切换到上述固定提交，并通过 `git rev-parse HEAD` 校验源码版本。构建过程不打包数据集、模型权重或其他运行资源。

完成以上准备后，在本目录执行：

```bash
cd Domino
docker build -t physicsnemo-domino-maca:3.8.1.2 .
```

构建完成后，使用下列命令启动镜像，并按第三节执行 DoMINO 模型适配测试：

```bash
docker run --rm -it \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=8G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /host_dir:/workspace \
  physicsnemo-domino-maca:3.8.1.2 \
  /bin/bash
```

请将 `/host_dir` 替换为主机侧实际工作目录。该挂载用于访问使用者的数据和输出；Dockerfile 构建的镜像已经安装 Warp，不需要重复执行 2.4 的 Warp 安装命令。

## 五、其他

1. 开发者初次使用曦云 GPU 运行 PhysicsNeMo DoMINO 时，遵循本手册完成环境搭建和模型适配验证后，可以按照 [PhysicsNeMo DoMINO 官方说明](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/domino)开展后续数据准备、训练及推理。
2. 沐曦仅维护部分 case 的正确性，如出现运行问题，可提交 issue，亦可以在[沐曦开发者社区](https://developer.metax-tech.com/)提交 bug 反馈。
3. 了解更多沐曦开源项目，请参考[沐曦开源社区](https://github.com/metax-maca)。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对前述软件源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款与条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
