# 沐曦 GPU 运行 PhysicsNeMo v2.0.0 说明文档

[PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) 项目由第三方以 Apache-2.0 开源许可证发布。本发布包采用文档与容器适配方式提供使用说明和 Dockerfile，不包含或分发 PhysicsNeMo 上游源码、模型权重、数据集或目标代码。构建镜像时会从上游 GitHub 仓库获取指定提交；项目的源码、许可证全文、版权与归属声明及其他项目文档，请以 [PhysicsNeMo 官方仓库](https://github.com/NVIDIA/physicsnemo) 为准。

本说明对应上游 PhysicsNeMo v2.0.0（Git tag `v2.0.0`，commit `1ca85d65ac2ce28ea9762910c09a954c08a37140`）和 MACA PyTorch 基础镜像。包括环境搭建、安装、容器化构建，以及三个上游 GPU 模型定向测试。PhysicsNeMo v2.0.0 的包初始化会导入 Warp，因此第三节测试前必须手动安装与沐曦环境匹配的 MetaX Warp。

## 一、PhysicsNeMo 简介

[NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)（原名 NVIDIA Modulus）是面向 AI for Science（AI4S）和工程领域的物理 AI 深度学习框架，可用于构建、训练、微调和推理科学计算模型。其方法将物理规律与数据驱动建模结合，覆盖神经算子（Neural Operator）、图神经网络（GNN）、Transformer、物理信息神经网络（PINN）及其混合方法等建模范式。

PhysicsNeMo 提供可扩展的训练与推理流水线，支持在预测、外推和反演等场景中开发和部署融合物理知识的模型。项目包含 GPU 优化的分布式训练能力，以及 FNO、MeshGraphNet、GraphCast、Pangu、AFNO、DLWP、Diffusion UNet、Transolver 等模型和数据管道，可服务于 CFD、天气气候、分子动力学和地球科学等领域。

本说明选择三个上游 GPU 模型测试覆盖 FNO、AFNO 与 One2ManyRNN：`test_fno_constructor`、`test_afno_constructor` 和 `test_conv_rnn_one2many_constructor`。三者均通过指定 `cuda:0` 参数在沐曦 GPU 上创建模型、生成输入并执行前向形状检查；测试源文件均来自上游指定提交，本文档不修改测试源码。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的 [maca-pytorch 镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=torch&dimension=docker) 启动容器。请使用沐曦 PyTorch 基础镜像，而非 PhysicsNeMo 官方容器镜像；本文档固定使用 MACA 3.8.1.2 对应 tag：

```bash
docker run -it --name physicsnemo-3.8.1.2 \
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

使用前请将 `/host_dir` 替换为主机侧实际工作目录，并确保当前账号具有该镜像仓库的拉取权限。

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

### 2.4 安装 PhysicsNeMo

**系统依赖：**

```bash
apt update && apt install -y --no-install-recommends \
  ca-certificates curl git build-essential libcurl4 libgl1 \
  libx11-6 libxrender1 perl vim wget xvfb xz-utils \
  && rm -rf /var/lib/apt/lists/*
```

**PhysicsNeMo 代码下载：**

从上游仓库获取源码并锁定到本文档对应的 v2.0.0 commit。最后一条命令用于确认当前 HEAD 与指定提交一致：

```bash
git clone https://github.com/NVIDIA/physicsnemo.git /root/physicsnemo
cd /root/physicsnemo
git checkout 1ca85d65ac2ce28ea9762910c09a954c08a37140
test "$(git rev-parse HEAD)" = "1ca85d65ac2ce28ea9762910c09a954c08a37140"
```

**PhysicsNeMo 安装：**

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

说明：

- `pip install -e . --no-deps` 仅安装 PhysicsNeMo 包本体。依赖已单独安装，可避免安装过程替换基础镜像中的沐曦定制版 PyTorch。
- 结尾固定 `tensordict` 与 `pydantic` 版本，用于保持沐曦环境中的依赖解析一致性。
- `pytest` 用于第三节的测试运行，不会替换 PyTorch。

**MetaX Warp 安装（运行模型测试需要）：**

PhysicsNeMo v2.0.0 在 `physicsnemo/__init__.py` 中无条件导入 `warp`，因此即使测试不调用 Warp kernel，也必须先安装 Warp。请前往[沐曦资源中心](https://developer.metax-tech.com/softnova)下载与当前 MACA、PyTorch、Python 和架构匹配的 Warp 压缩包，并将其放入容器挂载目录 `/workspace`。本说明已验证的资源包为 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`；其内的 wheel 是 `warp_lang-1.8.1-py3-none-manylinux_2_28_x86_64.whl`，为 `py3` 通用 wheel，已在本文档的 Python 3.12 环境验证可安装。

```bash
export WARP_ARCHIVE=/workspace/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz
mkdir -p /tmp/maca-warp
tar -xJf "${WARP_ARCHIVE}" -C /tmp/maca-warp
pip install --force-reinstall --no-deps \
  /tmp/maca-warp/maca-warp-1.8.1-py310-3.7.0.6-linux-amd64/warp_lang-1.8.1-py3-none-manylinux_2_28_x86_64.whl
python -c "import warp, physicsnemo; print('Warp version:', warp.__version__); print('PhysicsNeMo import: OK')"
```

`--no-deps` 用于避免 Warp 安装过程替换基础镜像中的沐曦定制版 PyTorch。不要使用 PyPI 的通用 `warp-lang` wheel。Warp 初始化时可能打印其无法加载 NVIDIA `libcuda.so` 的提示；该提示来自 Warp 的 NVIDIA CUDA 探测逻辑，不影响本节 FNO、AFNO 和 One2ManyRNN 在沐曦 GPU 上运行。

**验证安装：**

```bash
python -c "import torch, physicsnemo; print('PhysicsNeMo version:', physicsnemo.__version__); print('GPU available:', torch.cuda.is_available())"
```

正常情况下会打印 PhysicsNeMo 版本及 GPU 可用状态。

## 三、运行 GPU 模型定向测试

PhysicsNeMo 上游测试目录包含覆盖模型前向、精度、训练优化、checkpoint、部署和分布式功能的大型测试套件。本节测试三个上游模型测试函数的 `cuda:0` 参数实例，三个测试均直接执行上游 v2.0.0 源码，不新增测试脚本、不修改测试代码，也不依赖外部网络、数据集、模型权重等。

| 上游测试函数                                                                | 覆盖模型与行为 | GPU 验证内容                                   |
| :-------------------------------------------------------------------------- | :------------- | :--------------------------------------------- |
| `test/models/fno/test_fno.py::test_fno_constructor[cuda:0]`               | FNO            | 覆盖 1D 至 4D FNO 构造和前向输出形状           |
| `test/models/afno/test_afno.py::test_afno_constructor[cuda:0]`            | AFNO           | 覆盖两种 AFNO 构造参数和前向输出形状           |
| `test/models/rnn/test_rnn.py::test_conv_rnn_one2many_constructor[cuda:0]` | One2ManyRNN    | 覆盖默认及自定义 2D、3D RNN 构造和前向输出形状 |

### 3.1 FNO 构造与前向测试

`test_fno_constructor` 逐项构造 1D、2D、3D 和 4D FNO，并在 `cuda:0` 生成随机输入执行前向。测试检查输出 batch size、输出通道数和空间维度是否与输入和构造参数一致，同时检查非法维度会抛出 `NotImplementedError`。

### 3.2 AFNO 构造与前向测试

`test_afno_constructor` 在 `cuda:0` 上创建两个不同配置的 AFNO，并验证各自输出形状与输入空间尺寸、输出通道数相符；此外验证无效的 `embed_dim` 与 `num_blocks` 组合会抛出 `ValueError`。

### 3.3 One2ManyRNN 构造与前向测试

`test_conv_rnn_one2many_constructor` 在 `cuda:0` 上验证默认 One2ManyRNN 的公开构造属性和前向输出形状，并使用随机配置覆盖自定义 2D 和 3D 模型。该测试不加载上游 checkpoint，也不使用预生成参考输出。

### 3.4 执行命令与预期输出

完成 Warp 安装且已设置第二节环境变量后，在 PhysicsNeMo 源码根目录执行以下三个指定测试。`[cuda:0]` 限定 pytest 仅运行 GPU 参数实例，不会执行同一测试函数的 CPU 参数实例。请勿省略 `PYTORCH_DEFAULT_NCHW=1`：它是 FNO 测试通过所需的内存布局设置。

```bash
cd /root/physicsnemo
export PYTORCH_DEFAULT_NCHW=1
python -m pytest -v \
  "test/models/fno/test_fno.py::test_fno_constructor[cuda:0]" \
  "test/models/afno/test_afno.py::test_afno_constructor[cuda:0]" \
  "test/models/rnn/test_rnn.py::test_conv_rnn_one2many_constructor[cuda:0]"
```

**预期输出：**

```text
test/models/fno/test_fno.py::test_fno_constructor[cuda:0] PASSED
test/models/afno/test_afno.py::test_afno_constructor[cuda:0] PASSED
test/models/rnn/test_rnn.py::test_conv_rnn_one2many_constructor[cuda:0] PASSED
============================== 3 passed ==============================
```

该结果表示 FNO、AFNO 和 One2ManyRNN 已在 `cuda:0` 上完成模型构造与随机输入前向检查。

更多示例以及数据、权重和使用条款请参阅 [PhysicsNeMo 官方文档](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html) 和上游 `examples/` 目录中的 README。

## 四、Dockerfile 构建

本目录提供可直接构建的 [Dockerfile](Dockerfile)，基于 MACA 3.8.1.2 `maca-pytorch` 基础镜像获取指定 PhysicsNeMo 上游提交并安装本文档列出的运行依赖。构建前，请从[沐曦资源中心](https://developer.metax-tech.com/softnova)下载与第二节一致的 MetaX Warp 资源包 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，并将其放在 Dockerfile 同级目录。该资源包由使用者自行获取，不包含在本发布包中；Dockerfile 会在构建时安装其中的 `warp_lang-1.8.1-py3-none-manylinux_2_28_x86_64.whl`，构建完成的镜像可按第三节直接运行模型测试。

在本目录执行：

```bash
docker build -t physicsnemo-maca:3.8.1.2 .
```

构建完成后，使用下列命令启动镜像；进入容器后可按第三节运行各示例：

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
  physicsnemo-maca:3.8.1.2 \
  /bin/bash
```

请将 `/host_dir` 替换为主机侧实际工作目录。该挂载用于在容器中访问使用者自行下载的 Warp 资源包及其他工作文件；使用 Dockerfile 构建的镜像已安装 Warp，无需重复执行第二节的 Warp 安装命令。

## 五、其他

1. 开发者完成本手册的环境搭建后，可按照 [PhysicsNeMo 官方文档](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html) 继续开展模型训练、推理和其他上游示例的使用。
2. PhysicsNeMo 在沐曦 GPU 上运行依赖沐曦提供的 PyTorch 基础镜像。本文档不覆盖模型权重、训练数据、单元测试、测试补丁和需要独立授权的第三方组件。
3. 本发布包仅说明本节列出的上游示例入口；如遇环境或运行问题，可通过 [沐曦开发者社区](https://developer.metax-tech.com) 提交反馈。
4. 了解更多沐曦开源项目，请参考 [沐曦开源社区](https://github.com/metax-maca)。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
