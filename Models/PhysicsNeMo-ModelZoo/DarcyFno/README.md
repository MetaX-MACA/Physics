# 沐曦 GPU 运行 PhysicsNeMo Darcy FNO 说明文档

## 一、Darcy FNO 简介

Darcy FNO 是 [PhysicsNeMo v2.0.0](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_fno) 的二维 Darcy 多孔介质流案例，使用 Fourier Neural Operator (FNO) 根据渗透率场预测流动解。训练和验证样本由上游脚本在运行时生成；本文档固定使用上游提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

本目录采用文档与容器适配方式，不包含 PhysicsNeMo 上游源码、模型权重、训练数据或 MetaX Warp。本文档对应 MACA `3.8.1.2` 和基础镜像 `cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.8-py312-ubuntu24.04-amd64`。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的 [maca-pytorch 镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=torch&dimension=docker)启动容器。请使用沐曦 PyTorch 基础镜像，而非 PhysicsNeMo 官方容器镜像；本文档固定使用 MACA 3.8.1.2 对应 tag：

```bash
docker run -it --name physicsnemo-darcy-fno-3.8.1.2 \
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

参数说明：

- `--device=/dev/mxcd --device=/dev/dri`：将沐曦 GPU 设备节点传入容器。
- `--group-add video`：将容器进程加入 `video` 组，以访问 GPU 设备。
- `--shm-size=8G`：设置共享内存大小，供训练样本生成和 PyTorch 进程使用。
- `--security-opt seccomp=unconfined`：允许 MACA 运行时所需的系统调用。
- `--ulimit memlock=-1 --ulimit stack=67108864`：设置 GPU 运行时所需的锁定内存和线程栈限制。
- `-v /host_dir:/workspace`：挂载工作目录；`/host_dir` 是主机路径，`/workspace` 是容器内路径。

### 2.2 验证环境

进入容器后，先验证沐曦 PyTorch 与 GPU 是否可用：

```bash
python -c "import torch; assert torch.cuda.is_available(); print('PyTorch version:', torch.__version__); print('CUDA OK:', torch.cuda.get_device_name(0))"
```

命令应打印 PyTorch 版本和 `cuda:0` 对应的沐曦 GPU 名称。例如：

```text
PyTorch version: 2.8.0+metax3.8.1.0
CUDA OK: MetaX C500
```

实际版本号和设备名称以基础镜像及硬件型号为准。若 `torch.cuda.is_available()` 为 `False`，请先检查宿主机驱动、设备节点挂载和容器启动参数，再继续安装 PhysicsNeMo。

### 2.3 配置 MACA 编译环境

在容器内执行：

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/mxgpu_llvm/lib:${CUCC_PATH}/lib:${CUCC_PATH}/lib64:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH}
export PATH=${CUCC_PATH}/tools:${CUDA_PATH}/bin:/opt/conda/bin:${PATH}
export PYTHONPATH=/root/physicsnemo
export PYTORCH_DEFAULT_NCHW=1
export CUBLAS_WORKSPACE_CONFIG=:4096:16
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export DISABLE_PYTORCH_USE_FLASHATTN=1
export PYTORCH_ENABLE_FA_CPP_INTERFACE=1
export MCDNN_USE_DETERMINISTIC_ALGO=1
```

### 2.4 安装 PhysicsNeMo 与 Darcy FNO 依赖

安装系统依赖并获取固定上游提交：

```bash
apt update && apt install -y --no-install-recommends \
  ca-certificates curl git build-essential libcurl4 libgl1 \
  libx11-6 libxrender1 perl vim wget xvfb xz-utils \
  && rm -rf /var/lib/apt/lists/*

git clone https://github.com/NVIDIA/physicsnemo.git /root/physicsnemo
cd /root/physicsnemo
git checkout 1ca85d65ac2ce28ea9762910c09a954c08a37140
test "$(git rev-parse HEAD)" = "1ca85d65ac2ce28ea9762910c09a954c08a37140"
```

安装 PhysicsNeMo 包本体及其基础依赖。 `--no-deps` 用于避免覆盖基础镜像中的沐曦定制版 PyTorch：

```bash
pip install --upgrade pip hatchling
pip install "onnx>=1.14.0" "pandas>=2.2.0" "nvtx>=0.2.10" "treelib>=1.2.5" \
  "numpy>=1.22.4" "tqdm>=4.60.0" "requests>=2.32.2" "GitPython>=3.1.40" "s3fs>=2023.5.0" \
  "packaging>=24.2" "timm>=1.0.22" "einops>=0.8.1" "h5py>=3.15.1" "cftime>=1.6.5" \
  "jaxtyping>=0.3.3" "termcolor>=3.2.0" "hydra-core>=1.3.2" "tensordict>=0.10.0" \
  "omegaconf>=2.3.0" "importlib-metadata>=8.7.1" "gdown>=5.0.0" "matplotlib>=3.8.0" \
  "pytest>=9.0.1"
pip install -e . --no-deps
pip install --force-reinstall --no-deps tensordict==0.10.0 pydantic==2.12.3 pydantic-core==2.41.4
```

PhysicsNeMo v2.0.0 的包初始化需要 Warp。请从沐曦资源中心自行下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，放入挂载的 `/workspace`，再执行：

```bash
export WARP_ARCHIVE=/workspace/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz
mkdir -p /tmp/maca-warp
tar -xJf "${WARP_ARCHIVE}" -C /tmp/maca-warp
pip install --force-reinstall --no-deps \
  /tmp/maca-warp/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64/warp_lang-1.8.1-py3-none-manylinux_2_28_x86_64.whl
python -c "import torch, warp, physicsnemo; assert torch.cuda.is_available(); print('Warp version:', warp.__version__); print('PhysicsNeMo version:', physicsnemo.__version__)"
```

不要使用 PyPI 的通用 `warp-lang` wheel。上游 `examples/cfd/darcy_fno/requirements.txt` 声明的 `warp-lang>=1.6.0` 由上述 MetaX Warp 1.8.1 wheel 满足。Warp 初始化时可能打印无法加载 NVIDIA `libcuda.so` 或 `cuGraphicsGLRegisterBuffer` 的探测提示；该提示来自 Warp 的 NVIDIA CUDA 探测逻辑，不代表本案例失败。

## 三、Darcy FNO 运行与验证

进入上游案例目录：

```bash
cd /root/physicsnemo/examples/cfd/darcy_fno
```

### 3.1 Darcy FNO 适配有效性验证

以下命令使用上游 `train_fno_darcy.py` 和 `config.yaml`，仅通过 Hydra 覆盖训练规模；不修改上游代码或配置文件。它生成训练和验证样本，构建 FNO，在 `cuda:0` 上完成一个伪 epoch，并将结果写入当前目录的 `outputs/`。用于验证安装和执行路径，不代表完整训练结果。

```bash
python train_fno_darcy.py \
  training.max_pseudo_epochs=1 \
  training.pseudo_epoch_sample_size=64 \
  training.batch_size=4 \
  training.rec_results_freq=1 \
  validation.sample_size=4 \
  validation.validation_pseudo_epochs=1
```

命令应无异常退出，并在 `checkpoints/` 下生成模型状态文件，在 `outputs/` 下生成本次 Hydra 运行日志。这一步只验证模型构建、GPU 前向、反向和 checkpoint 写入，不要求完成完整训练。已验证环境中该命令输出 `Training completed *yay*`，并生成 `checkpoints/FNO.0.1.mdlus`、`checkpoints/checkpoint.0.1.pt` 和 `validation_step_001.png`；文件名以运行时间和上游版本为准。

### 3.2 可选的上游 FNO 单元测试

如需只验证 PhysicsNeMo FNO 模块本身，可在源码根目录执行上游测试的 GPU 参数实例：

```bash
cd /root/physicsnemo
python -m pytest -v "test/models/fno/test_fno.py::test_fno_constructor[cuda:0]"
```

该测试不需要 Darcy 数据集或模型权重；通过表示 FNO 构造与前向适配有效。已验证命令结果为 `1 passed`。

## 四、Dockerfile 构建

本目录提供可直接构建的Dockerfile。Dockerfile 使用 MACA 3.8.1.2 `maca-pytorch` 基础镜像，获取并固定 PhysicsNeMo v2.0.0 提交，安装本文档列出的 Python 依赖和 MetaX Warp；构建时不会打包模型权重或训练数据。

构建前，请从[沐曦资源中心](https://developer.metax-tech.com/softnova)下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，放在 Dockerfile 同级目录。该文件不包含在本发布包中；缺少该文件时 Dockerfile 的 `COPY` 步骤会失败。

```bash
cd DarcyFno
docker build -t physicsnemo-darcy-fno-maca:3.8.1.2 .
```

构建完成后，使用下列命令启动镜像，并按第三节执行验证：

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
  physicsnemo-darcy-fno-maca:3.8.1.2 \
  /bin/bash
```

请将 `/host_dir` 替换为主机侧实际工作目录。Dockerfile 构建的镜像已经安装 Warp，无需重复执行 2.4 的 Warp 安装命令。

## 五、其他

1. 本文档只验证单 GPU Darcy FNO 的适配有效性，不覆盖完整训练、性能、精度、数据集或模型权重；多 GPU 使用请参阅上游 `darcy_nested_fnos` 案例。
2. 使用者须遵守 PhysicsNeMo、MetaX Warp、基础镜像和其余第三方组件各自的许可证与访问条款。
3. 更多示例和功能请参考 [PhysicsNeMo 官方文档](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html) 与 [Darcy FNO 上游说明](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_fno)。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
