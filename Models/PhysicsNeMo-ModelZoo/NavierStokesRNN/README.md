# 沐曦 GPU 运行 PhysicsNeMo Navier Stokes RNN 说明文档

## 一、Navier Stokes RNN 简介

Navier Stokes RNN 是 [PhysicsNeMo v2.0.0](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/navier_stokes_rnn) 的二维 Navier-Stokes 流场的循环网络时序预测案例，面向涡量形式的二维流动，使用初始状态或历史时间步预测后续流场状态。本文档固定使用 PhysicsNeMo 仓库提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

本目录采用文档与容器适配方式，不包含 PhysicsNeMo 源码、二维 Navier-Stokes 时序数据和模型权重。PhysicsNeMo 案例使用 Fourier Neural Operator 论文相关数据，并在 `navier_stokes_rnn.py` 中提供下载与预处理逻辑；来源见[PhysicsNeMo 说明](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/navier_stokes_rnn)。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的[PhysicsNeMo MACA 镜像](https://developer.metax-tech.com/softnova/docker?chip_name=MXC500&package_name=physicsnemo-maca:2.0.0-3.9.0.12-torch2.8-py312-ubuntu24.04-amd64)启动容器：

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
  cr.metax-tech.com/public-ai-release/physicsnemo-maca:2.0.0-3.9.0.12-torch2.8-py312-ubuntu24.04-amd64 \
  /bin/bash
```

**参数说明：**

- `--device=/dev/mxcd --device=/dev/dri`：将沐曦 GPU 设备节点传入容器。
- `--group-add video`：将容器进程加入 `video` 组，以便访问 GPU 设备。
- `--shm-size=8G`：设置共享内存大小，供 PyTorch 和图数据加载使用。
- `--security-opt seccomp=unconfined`：允许 MACA 运行时使用所需的系统调用。
- `--ulimit memlock=-1`：取消内存锁定限制，避免 GPU 运行时申请锁页内存失败。
- `--ulimit stack=67108864`：设置线程栈大小，满足模型和数据处理过程的运行需求。
- `-v /host_dir:/workspace`：挂载工作目录；`/host_dir` 是主机路径，`/workspace` 是容器内路径。

### 2.2 验证环境

进入容器后，先验证沐曦 PyTorch 与 GPU 是否可用：

```bash
python -c "import torch; assert torch.cuda.is_available(); print('PyTorch version:', torch.__version__); print('CUDA OK:', torch.cuda.get_device_name(0))"
```

命令应打印 PyTorch 版本和 `cuda:0` 对应的沐曦 GPU 名称。例如：

```text
PyTorch version: 2.8.0+metax3.9.0.8
CUDA OK: MetaX GPU
```

本说明已验证的输出为 `PyTorch version: 2.8.0+metax3.9.0.8`、`CUDA OK: MetaX GPU`；实际版本号和设备名称以 PhysicsNeMo MACA 镜像及硬件型号为准。

如果 `torch.cuda.is_available()` 为 `False`，请先检查宿主机驱动、`/dev/mxcd` 和 `/dev/dri` 设备节点挂载、`video` 用户组权限以及容器启动参数，再继续安装 Navier Stokes RNN。

### 2.3 配置 MACA 编译环境

PhysicsNeMo MACA 镜像已配置 MACA 编译环境，并预装沐曦 PyTorch、 PhysicsNeMo 和 Warp；本案例不需要重复安装这些组件。进入容器后可验证：

```bash
python -c "import torch, physicsnemo, warp; assert torch.cuda.is_available(); print('PyTorch:', torch.__version__); print('PhysicsNeMo:', physicsnemo.__version__); print('Warp:', warp.__version__)"
```

命令应输出：

```text
PyTorch: 2.8.0+metax3.9.0.8
PhysicsNeMo: 2.0.0
Warp: 1.12.1
```

### 2.4 安装 Navier Stokes RNN 依赖

推荐使用第四节 Dockerfile 构建的镜像。镜像已包含 PhysicsNeMo 、沐曦 PyTorch 和 Warp。

本案例额外依赖：`gdown==6.5.1`、`matplotlib==3.11.2`。gdown 用于数据获取，Matplotlib 用于结果绘图。

对应的安装命令：

```bash
python -m pip install --no-cache-dir \
    "gdown==6.5.1" \
    "matplotlib==3.11.2"
```

验证依赖及案例入口（在交付镜像的默认工作目录执行）：

```bash
python -c 'import physicsnemo, torch, warp; assert physicsnemo.__version__ == "2.0.0"; print(physicsnemo.__version__, torch.__version__, warp.__version__)'
python -c 'import pathlib, runpy, sys; p=pathlib.Path("navier_stokes_rnn.py"); sys.path.insert(0, str(p.parent.resolve())); runpy.run_path(str(p))'
```

## 三、运行与验证

### 3.1 数据准备与运行命令

官方入口在数据文件不存在时自动从 Google Drive 下载并解压二维 Navier-Stokes 数据。也可在训练前手动下载并解压：

```bash
cd /root/physicsnemo/examples/cfd/navier_stokes_rnn
mkdir -p datasets
python -m gdown 'https://drive.google.com/uc?id=1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d' -O datasets/navier_stokes.zip
unzip datasets/navier_stokes.zip -d datasets
```

解压后确认 `datasets/ns_V1e-3_N5000_T50.mat` 存在。运行官方训练入口：

```bash
python navier_stokes_rnn.py max_epochs=20 model_type=one2many batch_size=8
```

官方入口会预处理数据，并在每个 epoch 后执行验证推理、生成预测图；配置位于 `conf/config_2d.yaml`。

### 3.2 训练与推理结果

使用官方数据划分的 1,000 个训练样本和 10 个测试样本，以随机种子 13、FP32 和相同初始权重分别训练 A100 与 MetaX GPU。训练运行 20 个 epoch，每个 epoch 125 次更新；模型使用 1 帧输入预测后续 16 帧。

下图对比两端各 epoch 的平均训练 MSE。两条曲线整体下降，变化趋势基本一致；以 A100 为基线，对 20 个 epoch 的平均 Loss 相对差异取平均为 6.35%。

![A100 与 MetaX GPU 训练 Loss 对比](./NavierStokesRNN-training-loss.png)

下图展示 MetaX GPU 第 20 个 epoch 的模型对一个测试样本预测的 16 帧流场，与对应参考场并列显示；预测保留了主要流场结构及其随时间的变化。

![MetaX GPU 的 16 帧流场预测与参考场](./NavierStokesRNN-postprocess.png)

## 四、Dockerfile 构建

本目录提供可直接构建的 Dockerfile。Dockerfile 使用上述 PhysicsNeMo MACA 镜像，安装第二节列出的本案例依赖，并在构建时核对 PhysicsNeMo v2.0.0 和 `navier_stokes_rnn.py` 案例入口；构建过程不打包 二维 Navier-Stokes 时序数据和模型权重。

在 `NavierStokesRNN` 目录执行：

```bash
docker build -t physicsnemo-navierstokesrnn:2.0.0-maca .
```

构建完成后，使用下列命令启动镜像，并按第三节执行模型适配验证：

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
  physicsnemo-navierstokesrnn:2.0.0-maca \
  /bin/bash
```

请将 `/host_dir` 替换为主机侧实际工作目录。该挂载用于访问使用者的数据和输出。

## 五、其他

1. 开发者在初次使用曦云 GPU 运行 Navier Stokes RNN 时，遵循本手册搭建环境后，可以按照 [PhysicsNeMo Navier Stokes RNN 官网](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/navier_stokes_rnn)的教程开展训练及推理。
2. 沐曦仅维护部分 case 的正确性，如出现运行问题，可提交 issue，亦可以在[沐曦开发者社区](https://developer.metax-tech.com)提交 bug 反馈。
3. 了解更多沐曦开源项目，请参考[沐曦开源社区](https://github.com/metax-maca)。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款与条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
