# 沐曦 GPU 运行 PhysicsNeMo Datacenter 说明文档

本目录以文档与容器适配方式提供 [PhysicsNeMo v2.0.0 Datacenter](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/datacenter) 的沐曦 GPU 环境。目录不包含或分发 PhysicsNeMo 上游源码、Datacenter 数据集、模型权重或 MetaX Warp；Dockerfile 会在构建时获取固定上游提交。

## 一、Datacenter 简介

Datacenter 是 PhysicsNeMo 的数据中心热通道流场代理模型案例，使用 3D UNet 预测温度、速度和压力分布。上游源码为 [PhysicsNeMo v2.0.0](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0)，固定提交为 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

数据集须通过 [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_datacenter_cfd_dataset) 获取，并需要 NVAIE 访问权限；该数据和 OpenFOAM 配置不随本适配发布。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

从沐曦开发者社区的 [maca-pytorch 镜像页面](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=torch&dimension=docker) 获取基础镜像。请使用沐曦 PyTorch 基础镜像，而非 PhysicsNeMo 官方容器镜像；本文档固定使用 MACA 3.8.1.2 对应 tag：`cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.8-py312-ubuntu24.04-amd64`。

```bash
docker run -it --name physicsnemo-datacenter-3.8.1.2 \
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
- `--shm-size=8G`：设置共享内存大小，供数据加载和模型测试使用。
- `--security-opt seccomp=unconfined`：允许 MACA 运行时使用所需的系统调用。
- `--ulimit memlock=-1 --ulimit stack=67108864`：解除内存锁定限制并增加栈大小，避免 GPU 初始化或算子运行时受限。
- `-v /host_dir:/workspace`：挂载主机工作目录；`/host_dir` 是主机路径，`/workspace` 是容器内路径。

### 2.2 验证环境

进入容器后，先验证沐曦 PyTorch 与 GPU 是否可用：

```bash
python -c "import torch; assert torch.cuda.is_available(); print('PyTorch version:', torch.__version__); print('CUDA OK:', torch.cuda.get_device_name(0))"
```

命令应打印 PyTorch 版本和 cuda:0 对应的沐曦 GPU 名称。例如：

```text
PyTorch version: 2.8.0+metax3.8.1.0
CUDA OK: MetaX C500
```

实际版本号和设备名称以基础镜像及宿主机硬件型号为准。如果 `torch.cuda.is_available()` 为 `False`，请先检查宿主机驱动、`/dev/mxcd` 和 `/dev/dri` 设备节点、容器用户组权限以及上述容器启动参数，再继续安装 PhysicsNeMo。

### 2.3 配置 MACA 编译环境

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

### 2.4 安装 PhysicsNeMo

```bash
apt update && apt install -y --no-install-recommends ca-certificates curl git build-essential libcurl4 libgl1 libx11-6 libxrender1 perl vim wget xvfb xz-utils && rm -rf /var/lib/apt/lists/*
git clone https://github.com/NVIDIA/physicsnemo.git /root/physicsnemo
cd /root/physicsnemo
git checkout 1ca85d65ac2ce28ea9762910c09a954c08a37140
test "$(git rev-parse HEAD)" = "1ca85d65ac2ce28ea9762910c09a954c08a37140"
pip install --upgrade pip hatchling
pip install "onnx>=1.14.0" "pandas>=2.2.0" "nvtx>=0.2.10" "treelib>=1.2.5" "numpy>=1.22.4" "tqdm>=4.60.0" "requests>=2.32.2" "GitPython>=3.1.40" "s3fs>=2023.5.0" "packaging>=24.2" "timm>=1.0.22" "einops>=0.8.1" "h5py>=3.15.1" "cftime>=1.6.5" "jaxtyping>=0.3.3" "termcolor>=3.2.0" "hydra-core>=1.3.2" "tensordict>=0.10.0" "omegaconf>=2.3.0" "importlib-metadata>=8.7.1" "gdown>=5.0.0" "matplotlib>=3.8.0" "pytest>=9.0.1" vtk
pip install -e . --no-deps
pip install --force-reinstall --no-deps tensordict==0.10.0 pydantic==2.12.3 pydantic-core==2.41.4
```

`pip install -e . --no-deps` 只安装 PhysicsNeMo 本体，不覆盖基础镜像内的沐曦定制 PyTorch；其余依赖单独安装，末尾固定 `tensordict` 和 `pydantic` 版本。

PhysicsNeMo 初始化需要 Warp。请从沐曦资源中心获取 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz` 并放入 `/workspace`，再执行：

```bash
export WARP_ARCHIVE=/workspace/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz
mkdir -p /tmp/maca-warp
tar -xJf "${WARP_ARCHIVE}" -C /tmp/maca-warp
pip install --force-reinstall --no-deps /tmp/maca-warp/*/warp_lang-*.whl
python -c "import torch, warp, physicsnemo; assert torch.cuda.is_available(); print('Warp version:', warp.__version__); print('PhysicsNeMo version:', physicsnemo.__version__)"
```

预期输出包含 `Warp version: 1.8.1`、`PhysicsNeMo version: 2.0.0`。Warp 初始化可能打印无法加载 NVIDIA `libcuda.so` 或图形接口的提示；本案例验证不使用 NVIDIA CUDA 图形接口，该提示不影响后续 UNet 前向测试。

PhysicsNeMo 的部分上游测试（如 UNet 构造/前向测试）需要 `transformer_engine`。请从沐曦资源中心获取 `maca-transformerengine-2.13-py312-3.7.1.0-linux-x86_64.tar.xz` 并放入 `/workspace`，再执行：

```bash
export TE_ARCHIVE=/workspace/maca-transformerengine-2.13-py312-3.7.1.0-linux-x86_64.tar.xz
mkdir -p /tmp/maca-te
tar -xJf "${TE_ARCHIVE}" -C /tmp/maca-te
pip install --force-reinstall --no-deps /tmp/maca-te/*/wheel/transformer_engine-*.torch2.8*.whl
pip install --force-reinstall --no-deps /tmp/maca-te/*/wheel/grouped_gemm-*.torch2.8*.whl
pip install typing_inspection annotated-types onnxscript nvdlfw-inspect
python -c "import transformer_engine; print('Transformer Engine version:', transformer_engine.__version__)"
```

预期输出包含 `Transformer Engine version: 2.13.0`。安装后即可运行上游 UNet pytest 测试而不被跳过。

#### 2.4.1 数据准备（完整训练需要）

需要 NGC CLI 和 NVAIE 权限。按上游步骤下载后将 `datasets` 放入案例目录：

```bash
cd /root/physicsnemo/examples/cfd/datacenter
ngc registry resource download-version "nvidia/physicsnemo/physicsnemo_datacenter_cfd_dataset:v1"
mv physicsnemo_datacenter_cfd_dataset_vv1/datasets .
```

## 三、运行与验证

### 3.1 模型适配验证

本节只验证 PhysicsNeMo、Warp、沐曦 PyTorch 和 Datacenter 使用的 3D UNet 在 `cuda:0` 上可以构造并完成一次前向，不执行数据集下载、DALI 数据管线或完整训练。该测试使用小尺寸随机输入，仅用于确认算子和设备适配，不代表训练精度或性能：

```bash
cd /root/physicsnemo
python -c "import torch, warp, physicsnemo; from physicsnemo.models.unet import UNet; assert torch.cuda.is_available(); model=UNet(in_channels=1, out_channels=1, model_depth=2, feature_map_channels=[4,4,8,8]).to('cuda:0'); output=model(torch.randn(1,1,8,8,8,device='cuda:0')); assert output.shape == (1,1,8,8,8); print('Datacenter UNet adaptation OK:', output.shape)"
```

预期输出：

```text
Datacenter UNet adaptation OK: torch.Size([1, 1, 8, 8, 8])
```

也可以运行上游 UNet 构造/前向测试中仅使用 `cuda:0` 的实例。该测试要求基础镜像额外提供 `transformer_engine`；若未安装，pytest 会按上游测试约束跳过：

```bash
python -m pytest -v "test/models/unet/test_unet.py::test_unet_constructor_and_forward[cuda:0-with_defaults]" "test/models/unet/test_unet.py::test_unet_constructor_and_forward[cuda:0-with_custom_args]"
```

### 3.2 训练、物理信息训练与推理

完整 Datacenter 流程最低需要 80GB GPU 显存。数据准备完成后，在 `/root/physicsnemo/examples/cfd/datacenter` 执行上游入口：

```bash
python train.py
mpirun -np <#GPUs> python train.py
python train_physics_informed.py
python inference.py
```

上述命令对应上游 v2.0.0 案例；本文档不声称在无数据集或脱敏 OpenFOAM 配置下获得训练结果。

## 四、Dockerfile 构建

本目录提供可直接构建的Dockerfile。构建前需要使用者准备基础镜像和 MetaX Warp 资源包；PhysicsNeMo 源码由 Dockerfile 在构建时从 GitHub 获取，并固定到 PhysicsNeMo v2.0.0 提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

1. 拉取 MACA 3.8.1.2 对应的 PyTorch 基础镜像。
2. 从[沐曦资源中心](https://developer.metax-tech.com/softnova)下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，放在 Dockerfile 同级目录。该压缩包不包含在本发布包中。

完成准备后，在本目录执行：

```bash
docker build -t physicsnemo-datacenter-maca:3.8.1.2 .
```

构建完成后，使用下列命令启动镜像，并按第三节执行模型适配验证：

```bash
docker run --rm -it \
  --device=/dev/mxcd --device=/dev/dri --group-add video \
  --shm-size=8G --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /host_dir:/workspace \
  physicsnemo-datacenter-maca:3.8.1.2 /bin/bash
```

请将 `/host_dir` 替换为主机侧实际工作目录。Dockerfile 构建的镜像已安装 Warp，不需要重复执行 2.4 节的 Warp 安装命令。

## 五、其他

1. 开发者在初次使用曦云 GPU 运行 PhysicsNeMo Datacenter 时，遵循本手册搭建环境后，就可以按照 [PhysicsNeMo 官方文档](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html) 和 [Datacenter 案例说明](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/datacenter) 开始训练及推理。
2. 沐曦仅维护部分 case 的正确性，如出现运行问题，可提交 issue，亦可以在[开发者社区](https://developer.metax-tech.com)提交 bug 反馈。
3. 了解更多沐曦开源项目，请参考[沐曦开源社区](https://github.com/metax-maca)。

本目录的 [LICEMSE](LICEMSE) 仅适用于本目录原创 Dockerfile 和 README。PhysicsNeMo、Datacenter 案例、NGC 数据集、OpenFOAM 配置和 MetaX Warp 的许可证、版权及使用条款以各自官方来源为准。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
