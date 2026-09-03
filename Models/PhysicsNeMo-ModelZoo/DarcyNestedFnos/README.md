# 沐曦 GPU 运行 PhysicsNeMo Darcy Nested FNOs 说明文档

## 一、Darcy Nested FNOs 简介

Darcy Nested FNOs 是 [PhysicsNeMo v2.0.0](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_nested_fnos) 的二维达西流代理模型案例，使用嵌套 Fourier Neural Operator 分别学习全局网格和局部细化区域。本文档固定使用上游提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

本目录采用文档与容器适配方式，不包含 PhysicsNeMo 上游源码、案例数据、模型权重或 MetaX Warp。案例数据由上游脚本生成，Warp 须从沐曦资源中心获取。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的 [maca-pytorch 镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=torch&dimension=docker)启动容器。请使用沐曦 PyTorch 基础镜像，而非 PhysicsNeMo 官方容器镜像；本文档固定使用 MACA 3.8.1.2 对应 tag：

```bash
docker run -it --name physicsnemo-darcy-nested-fnos \
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
- `--shm-size=8G`：设置共享内存大小，供数据加载和训练进程使用。
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

在容器中设置 MACA 编译器、CUDA 兼容层和运行库路径：

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/mxgpu_llvm/lib:${CUCC_PATH}/lib:${CUCC_PATH}/lib64:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH}
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${CUCC_PATH}/tools:${CUDA_PATH}/bin:/opt/conda/bin:${PATH}
export PYTORCH_DEFAULT_NCHW=1
export CUBLAS_WORKSPACE_CONFIG=:4096:16
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export DISABLE_PYTORCH_USE_FLASHATTN=1
export PYTORCH_ENABLE_FA_CPP_INTERFACE=1
export MCDNN_USE_DETERMINISTIC_ALGO=1
```

### 2.4 安装 PhysicsNeMo 与案例依赖

安装系统依赖：

```bash
apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates git build-essential libgl1 libx11-6 libxrender1 \
  openmpi-bin libopenmpi-dev xz-utils \
  && rm -rf /var/lib/apt/lists/*
```

获取上游源码并锁定版本：

```bash
git clone https://github.com/NVIDIA/physicsnemo.git /workspace/physicsnemo
cd /workspace/physicsnemo
git checkout 1ca85d65ac2ce28ea9762910c09a954c08a37140
test "$(git rev-parse HEAD)" = "1ca85d65ac2ce28ea9762910c09a954c08a37140"
```

安装 PhysicsNeMo 运行依赖。 `--no-deps` 用于避免替换基础镜像中的沐曦定制版 PyTorch：

```bash
pip install --upgrade pip hatchling
pip install "onnx>=1.14.0" "pandas>=2.2.0" "nvtx>=0.2.10" "treelib>=1.2.5" \
  "numpy>=1.22.4" "tqdm>=4.60.0" "requests>=2.32.2" "GitPython>=3.1.40" "s3fs>=2023.5.0" \
  "packaging>=24.2" "timm>=1.0.22" "einops>=0.8.1" "h5py>=3.15.1" "cftime>=1.6.5" \
  "jaxtyping>=0.3.3" "termcolor>=3.2.0" "hydra-core>=1.3.2" "tensordict>=0.10.0" \
  "omegaconf>=2.3.0" "importlib-metadata>=8.7.1" "gdown>=5.0.0" "matplotlib>=3.8.0" \
  "pytest>=9.0.1"
cd /workspace/physicsnemo
pip install -e . --no-deps
pip install --force-reinstall --no-deps tensordict==0.10.0 pydantic==2.12.3 pydantic-core==2.41.4
cd /workspace/physicsnemo/examples/cfd/darcy_nested_fnos
pip install -r requirements.txt
```

PhysicsNeMo v2.0.0 导入时需要 Warp。请从[沐曦资源中心](https://developer.metax-tech.com/softnova)下载与 MACA、PyTorch、Python 和架构匹配的 Warp 包至主机工作目录，使其在容器中位于 `/workspace`。本文档使用 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`；该资源不包含在本发布内容中。

```bash
export WARP_ARCHIVE=/workspace/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz
mkdir -p /tmp/maca-warp
tar -xJf "${WARP_ARCHIVE}" -C /tmp/maca-warp
pip install --force-reinstall --no-deps /tmp/maca-warp/*/warp_lang-*.whl
rm -rf /tmp/maca-warp
export PYTHONPATH=/workspace/physicsnemo
python -c "import torch, warp, physicsnemo; assert torch.cuda.is_available(); print('Warp version:', warp.__version__); print('PhysicsNeMo version:', physicsnemo.__version__)"
```

不要以 PyPI 通用 `warp-lang` wheel 替代上述 MetaX Warp 包。Warp 初始化时可能打印无法加载 NVIDIA `libcuda.so` 的提示；该提示来自其 NVIDIA CUDA 探测逻辑，不能单独作为 MACA 环境失败的判断依据。

## 三、运行与验证

在本节所有命令前，确保已完成第二节安装，并进入案例目录：

```bash
cd /workspace/physicsnemo/examples/cfd/darcy_nested_fnos
export PYTHONPATH=/workspace/physicsnemo
```

### 3.1 生成数据

上游脚本会在当前目录生成 `data/training_data.npy`、`data/validation_data.npy` 和 `data/out_of_sample.npy`，并保存示例图。脚本默认生成 8192 个训练样本、2048 个验证样本和 2048 个测试样本，使用 CUDA 和 Warp，运行时间及磁盘空间需求应由使用者按环境评估。

```bash
python generate_nested_darcy.py
```

### 3.2 单 GPU 训练

分别训练全局网格模型 `ref0` 与局部细化模型 `ref1`：

```bash
python train_nested_darcy.py +model=ref0
python train_nested_darcy.py +model=ref1
```

上游默认配置训练 100 个 epoch，输出检查点至 `checkpoints/all/<model>/` 和 `checkpoints/best/<model>/`，MLflow 离线记录保存于当前工作目录。训练参数和数据路径定义在上游 `config.yaml`。

### 3.3 多 GPU 训练

该案例的上游实现支持 MPI 多进程训练。下列命令按上游示例使用两个进程；执行前请确认容器内可见至少两张可用 GPU，并根据实际 GPU 数量调整 `-n`：

```bash
mpirun -n 2 python train_nested_darcy.py +model=ref0
mpirun -n 2 python train_nested_darcy.py +model=ref1
```

### 3.4 评估

完成 `ref0` 和 `ref1` 训练后，运行上游评估入口：

```bash
python evaluate_nested_darcy.py
```

评估读取 `data/out_of_sample.npy` 与两个模型检查点，并在当前目录输出其结果文件和可视化图。

### 3.5 适配性验证

以下检查只验证 PhysicsNeMo、Warp、FNO 模型和案例入口能够在当前镜像中加载，不生成数据、不训练模型：

```bash
python -c "import torch, warp, physicsnemo; assert torch.cuda.is_available(); print(torch.__version__, warp.__version__, physicsnemo.__version__)"
python train_nested_darcy.py --help
python evaluate_nested_darcy.py --help
cd /workspace/physicsnemo
pytest -q test/models/fno/test_fno.py -k 'not checkpoint' --disable-warnings --maxfail=1
```

`test/models/fno/test_fno.py` 中的 checkpoint 用例需要向源码目录写入临时文件；当源码按只读方式挂载时应排除该用例。参考验证结果为 `18 passed, 8 skipped`。数据生成、训练和评估完整流程不属于本适配性检查范围。

## 四、Dockerfile 构建

本目录提供可直接构建的Dockerfile。构建前需要由使用者手动准备基础镜像和 MetaX Warp 资源包；PhysicsNeMo 源码无需手动下载，Dockerfile 会在构建时从 GitHub 获取并固定到 PhysicsNeMo v2.0.0 提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

1. 在[沐曦资源中心](https://developer.metax-tech.com/softnova)获取并拉取 MACA 3.8.1.2 对应的 PyTorch 基础镜像。
2. 从同一资源中心手动下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，将其放在 Dockerfile 同级目录。该压缩包不包含在本发布包中；缺少该文件时，Docker 构建会在 `COPY` 步骤失败。
3. Dockerfile 将自动安装系统与 Python 依赖，执行 `git clone https://github.com/NVIDIA/physicsnemo.git`，切换到上述固定提交，并通过 `git rev-parse HEAD` 校验源码版本。构建过程不打包数据集、模型权重或其他运行资源。

完成以上准备后，在本目录执行：

```bash
cd DarcyNestedFnos
docker build -t physicsnemo-darcy-nested-fnos-maca:3.8.1.2 .
```

构建完成后，使用下列命令启动镜像，并按第三节执行适配性验证：

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
  physicsnemo-darcy-nested-fnos-maca:3.8.1.2 \
  /bin/bash
```

将 `/host_dir` 替换为主机侧实际工作目录。Dockerfile 构建的镜像已经安装 Warp，不需要重复执行 2.4 的 Warp 安装命令。

## 五、其他

1. 开发者在初次使用曦云 GPU 运行 Darcy Nested FNOs 时，遵循本手册搭建环境后，可以按照 [PhysicsNeMo 官方文档](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_nested_fnos) 的教程开始数据生成、训练及评估。
2. 沐曦仅维护部分 case 的正确性，如出现运行问题，可提交 issue，亦可以在[沐曦开发者社区](https://developer.metax-tech.com)提交 bug 反馈。
3. 了解更多沐曦开源项目，请参考[沐曦开源社区](https://github.com/metax-maca)。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
