# 沐曦 GPU 运行 PhysicsNeMo AeroGraphNet 说明文档

## 一、AeroGraphNet 简介

AeroGraphNet 是 [PhysicsNeMo v2.0.0](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/aero_graph_net) 的外流场代理模型案例，面向 Ahmed body 与 DrivAerNet，预测表面压力、壁面剪切应力和阻力系数。本文档固定使用上游提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

本目录采用文档与容器适配方式，不包含 PhysicsNeMo 上游源码、AeroGraphNet 数据集、模型权重或 MetaX Warp。Ahmed body 数据须按上游案例说明申请；DrivAerNet 数据须按其[官方仓库](https://github.com/Mohamedelrefaie/DrivAerNet)获取并遵守相应条款。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的 [maca-pytorch 镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=torch&dimension=docker) 启动容器。请使用沐曦 PyTorch 基础镜像，而非 PhysicsNeMo 官方容器镜像；本文档固定使用 MACA 3.8.1.2 对应 tag：

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
  cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.8-py312-ubuntu24.04-amd64 \
  /bin/bash
```
> 可根据需要使用更新版本镜像，详情参见[沐曦开发者社区](https://developer.metax-tech.com)

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
PyTorch version: <base-image-pytorch-version>
CUDA OK: <MetaX GPU name>
```

本说明已验证的输出为 `PyTorch version: 2.8.0+metax3.8.1.0`、`CUDA OK: MetaX C500`；实际版本号和设备名称以基础镜像及硬件型号为准。

如果 `torch.cuda.is_available()` 为 `False`，请先检查宿主机驱动、`/dev/mxcd` 和 `/dev/dri` 设备节点挂载、`video` 用户组权限以及容器启动参数，再继续安装 AeroGraphNet。

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

### 2.4 安装 AeroGraphNet

推荐使用第四节 Dockerfile 构建的镜像。手动安装时，先按 PhysicsNeMo v2.0.0 文档获取并安装固定提交及其运行依赖，再安装本案例依赖：

```bash
git clone --branch v2.0.0 --depth 1 https://github.com/NVIDIA/physicsnemo.git /root/physicsnemo
cd /root/physicsnemo
git checkout 1ca85d65ac2ce28ea9762910c09a954c08a37140
pip install --no-deps -e .
pip install pyvista shapely vtk torch_geometric torch_scatter wandb
pip install s3fs cftime gitpython h5py hydra-core importlib-metadata jaxtyping nvtx omegaconf onnx tensordict timm treelib
```

从[沐曦资源中心](https://developer.metax-tech.com/softnova)下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，解压后安装其中的 `warp_lang-1.8.1-py3-none-manylinux_2_28_x86_64.whl`。不要安装 PyPI 通用 `warp-lang` 替换沐曦 Warp。

验证依赖：

```bash
python -c "import physicsnemo, warp, torch_geometric, torch_scatter, pyvista, shapely, vtk; print(physicsnemo.__version__, warp.__version__)"
```

## 三、运行与验证

本文只验证 AeroGraphNet 在沐曦 GPU 上的模型导入和前向计算，不包含真实数据训练、推理评估或 VTK 可视化。

```bash
cd /root/physicsnemo/examples/cfd/external_aerodynamics/aero_graph_net
python - <<'PY'
import torch
from torch_geometric.data import Data
from models import AeroGraphNet

edge_index = torch.tensor(
    [[0, 1, 2, 3, 1, 2, 3, 0], [1, 2, 3, 0, 0, 1, 2, 3]], device="cuda:0"
)
graph = Data(edge_index=edge_index)
node_features = torch.randn(4, 3, device="cuda:0")
edge_features = torch.randn(edge_index.shape[1], 4, device="cuda:0")
model = AeroGraphNet(
    input_dim_nodes=3,
    input_dim_edges=4,
    output_dim=4,
    processor_size=2,
    hidden_dim_node_encoder=32,
    hidden_dim_edge_encoder=32,
    hidden_dim_processor=32,
    hidden_dim_node_decoder=32,
).to("cuda:0")
with torch.no_grad():
    prediction = model(node_features, edge_features, graph)
assert prediction["graph"].shape == (4, 4)
assert prediction["c_d"].shape == (1,)
print("AeroGraphNet CUDA forward: PASSED")
PY
```

正常情况下输出 `AeroGraphNet CUDA forward: PASSED`。真实数据目录、Hydra 配置、训练和推理参数请使用[上游 AeroGraphNet README](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/aero_graph_net)。

## 四、Dockerfile 构建

本目录提供可直接构建的Dockerfile。构建前需准备 MACA 3.8.1.2 PyTorch 基础镜像和 MetaX Warp 资源包；Dockerfile 会从 GitHub 获取 PhysicsNeMo，并固定到提交 `1ca85d65ac2ce28ea9762910c09a954c08a37140`。

1. 在[沐曦资源中心](https://developer.metax-tech.com/softnova)获取 MACA 3.8.1.2 对应的 PyTorch 基础镜像。
2. 从同一资源中心下载 `maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz`，放在 Dockerfile 同级目录。

```bash
docker build -t physicsnemo-aerographnet-maca:3.8.1.2 .
```

## 五、其他

1. 开发者在初次使用曦云 GPU 运行 AeroGraphNet 时，遵循本手册搭建环境后，可以按照 [PhysicsNeMo AeroGraphNet 官网](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/aero_graph_net)的教程开展训练及推理。
2. 沐曦仅维护部分 case 的正确性，如出现运行问题，可提交 issue，亦可以在[沐曦开发者社区](https://developer.metax-tech.com)提交 bug 反馈。
3. 了解更多沐曦开源项目，请参考[沐曦开源社区](https://github.com/metax-maca)。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款与条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
