# 沐曦 GPU 运行 PhysicsNeMo DarcyTransolver 说明文档

## 一、DarcyTransolver 简介

DarcyTransolver 是 [PhysicsNeMo v2.0.0](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_transolver) 的二维 Darcy 流数据驱动案例，使用 Transolver 学习渗透率到压力场的映射。本文档固定使用上游提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

本目录采用文档与容器适配方式，不包含 PhysicsNeMo 上游源码、Darcy 数据集、模型权重或 MetaX Warp。在线数据由上游 `Darcy2D` datapipe 生成；固定数据集及其使用条款请遵循[上游案例说明](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_transolver)。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的 [maca-pytorch 镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=torch&dimension=docker)启动容器。请使用沐曦 PyTorch 基础镜像，而非 PhysicsNeMo 官方容器镜像；本文档固定使用 MACA 3.8.1.2 对应 tag：

```bash
docker pull cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.8-py312-ubuntu24.04-amd64
docker run --rm -it --name darcy-transolver \
  --device=/dev/mxcd --device=/dev/dri --group-add video \
  --shm-size=8G --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /host_dir:/workspace \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.8-py312-ubuntu24.04-amd64 /bin/bash
```
> 可根据需要使用更新版本镜像，详情参见[沐曦开发者社区](https://developer.metax-tech.com)

**参数说明：**

- `--device=/dev/mxcd --device=/dev/dri`：将沐曦 GPU 设备节点传入容器。
- `--group-add video`：将容器进程加入 `video` 组，以便访问 GPU 设备。
- `--shm-size=8G`：设置共享内存大小，供数据加载和测试进程使用。
- `-v /host_dir:/workspace`：挂载工作目录；`/host_dir` 是主机路径，`/workspace` 是容器内路径。

### 2.2 验证环境

执行以下命令验证沐曦 PyTorch 与 GPU：

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

PhysicsNeMo 的部分算子及其依赖需要使用 MACA 运行时库。请在容器中设置以下环境变量；缺少相关路径可能导致 `import torch` 失败或算子不可用：

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

其中，`LD_LIBRARY_PATH` 中的 `${MACA_PATH}/ompi/lib`，以及 `PATH` 中的 `${CUCC_PATH}/bin` 与 `${CUCC_PATH}/tools` 是沐曦运行环境的重要路径，请勿省略。

### 2.4 安装 PhysicsNeMo 与 DarcyTransolver 依赖

```bash
apt update && apt install -y --no-install-recommends ca-certificates git build-essential libgl1 libx11-6 libxrender1 xz-utils
git clone https://github.com/NVIDIA/physicsnemo.git /root/physicsnemo
cd /root/physicsnemo
git checkout 1ca85d65ac2ce28ea9762910c09a954c08a37140
test "$(git rev-parse HEAD)" = "1ca85d65ac2ce28ea9762910c09a954c08a37140"
pip install --upgrade pip hatchling
grep -v '^warp-lang' examples/cfd/darcy_transolver/requirements.txt | pip install -r /dev/stdin
pip install "onnx>=1.14.0" "pandas>=2.2.0" "nvtx>=0.2.10" "treelib>=1.2.5" "numpy>=1.22.4" "tqdm>=4.60.0" "requests>=2.32.2" "GitPython>=3.1.40" "s3fs>=2023.5.0" "packaging>=24.2" "timm>=1.0.22" "einops>=0.8.1" "h5py>=3.15.1" "cftime>=1.6.5" "jaxtyping>=0.3.3" "termcolor>=3.2.0" "hydra-core>=1.3.2" "tensordict>=0.10.0" "omegaconf>=2.3.0" "importlib-metadata>=8.7.1" "matplotlib>=3.8.0"
pip install -e . --no-deps
```

PhysicsNeMo v2.0.0 会在初始化时导入 `warp`，因此测试前必须安装与 MACA、PyTorch、Python 和 amd64 架构匹配的 MetaX Warp。请从[沐曦资源中心](https://developer.metax-tech.com/softnova)下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，放入主机挂载目录 `/workspace`。上游 requirements 中的 `warp-lang` 不应从 PyPI 安装。

```bash
export WARP_ARCHIVE=/workspace/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz
mkdir -p /tmp/maca-warp
tar -xJf "${WARP_ARCHIVE}" -C /tmp/maca-warp
export WARP_WHEEL=/tmp/maca-warp/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64/warp_lang-1.8.1-py3-none-manylinux_2_28_x86_64.whl
test -f "${WARP_WHEEL}"
pip install --force-reinstall --no-deps "${WARP_WHEEL}"
python -c "import warp, physicsnemo; print('Warp version:', warp.__version__); print('PhysicsNeMo import: OK')"
```

## 三、运行与验证

本适配只验证 Transolver 在 `cuda:0` 上能够构造并完成一次二维结构化网格前向，不要求完整训练、数据集下载或 MLFlow 流程：

```bash
cd /root/physicsnemo
python -m pytest -v "test/models/transolver/test_transolver.py::test_transolver2d_forward[cuda:0]"
```

也可以执行不依赖 pytest 的最小检查：

```bash
python -c "import torch; from physicsnemo.models.transolver import Transolver; m=Transolver(structured_shape=(85,85), n_layers=8, n_hidden=64, n_head=4, functional_dim=1, out_dim=1, slice_num=32, ref=1, unified_pos=True, use_te=False).to('cuda:0'); y=m(torch.randn(1,85*85,1,device='cuda:0')); assert y.shape == (1,85*85,1); print('Transolver CUDA forward: OK')"
```

本次已验证结果为 `1 passed`，并完成 `Transolver CUDA forward: OK`。Warp 可能输出无法加载 NVIDIA CUDA 入口的探测提示，该提示不影响沐曦 GPU 前向验证。完整训练入口 `examples/cfd/darcy_transolver/train_transolver_darcy.py` 以及固定数据集入口 `train_transolver_darcy_fix.py` 请按上游文档另行运行，本适配不将其列为验收要求。

## 四、Dockerfile 构建

本目录提供可直接构建的Dockerfile。构建前请从[沐曦资源中心](https://developer.metax-tech.com/softnova)下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，并放在 Dockerfile 同级目录；该压缩包不包含在本发布内容中。

```bash
cd DarcyTransolver
docker build -t darcy-transolver-maca:3.8.1.2 .
```

构建完成后启动镜像，并按第三节执行模型适配验证：

```bash
docker run --rm -it \
  --device=/dev/mxcd --device=/dev/dri --group-add video \
  --shm-size=8G --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /host_dir:/workspace \
  darcy-transolver-maca:3.8.1.2 /bin/bash
```

## 五、其他

1. 后续数据准备、训练和推理请遵循 [PhysicsNeMo Darcy Transolver 官方说明](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_transolver)。
2. 请引用 [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) 及 [Transolver 论文](https://arxiv.org/abs/2402.02366)。
3. 沐曦仅维护本案例的模型适配验证；其他驱动、设备、数据和性能不在保证范围内。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对前述软件源代码的任何修改。使用者应遵守适用许可证、商标和数据条款；相关软件的源码、许可证全文、版权与归属声明请以官方发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
