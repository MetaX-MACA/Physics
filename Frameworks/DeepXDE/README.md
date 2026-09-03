# 沐曦GPU运行DeepXDE说明文档

[DeepXDE](https://github.com/lululxvi/deepxde)项目由第三方以开源许可证发布，我方未对其源代码作任何修改。您可通过以下链接 https://github.com/lululxvi/deepxde 查阅该项目的源代码、许可证全文、版权与归属声明及其他声明文件。

## 一、DeepXDE简介

[DeepXDE](https://github.com/lululxvi/deepxde) 是一个面向科学计算的深度学习开源库，以物理信息神经网络（Physics-Informed Neural Networks，PINN）为核心，支持求解常微分方程、偏微分方程、积分微分方程及分数阶方程的正反问题。该库内置复杂几何建模、多保真度学习、自适应采样等算法，兼容 TensorFlow、PyTorch、Paddle 等多种后端，代码简洁、配置灵活，广泛用于流体力学、热传导、逆问题优化等数值模拟场景。

## 二、沐曦GPU环境配置与运行

DeepXDE 支持多种深度学习后端，沐曦平台目前提供 PyTorch 与 Paddle 两种后端环境。请根据需要选择对应的沐曦基础镜像，并通过 `DDE_BACKEND` 环境变量指定后端。

### 2.1 环境准备

使用沐曦开发者社区提供的 MACA 基础镜像启动容器，**注意需根据所选后端使用对应的基础镜像（PyTorch 或 Paddle），而非 DeepXDE 镜像**。详情参见[沐曦开发者社区](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97)。

**PyTorch 后端（torch2.6 镜像）：**
```bash
docker run -it --name test-deepxde \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=4G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /host_dir:/workspace \
  cr.metax-tech.com/public-library/maca-pytorch:<版本>-torch2.6-py310-ubuntu22.04-amd64 \
  /bin/bash
```

**Paddle 后端（paddle 镜像）：**
```bash
docker run -it --name test-deepxde \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=4G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /host_dir:/workspace \
  cr.metax-tech.com/public-library/maca-paddle:<版本>-py310-ubuntu22.04-amd64 \
  /bin/bash
```
> 可根据需要使用更新版本镜像，具体镜像地址详情参见[沐曦开发者社区](https://developer.metax-tech.com)

**参数说明：**
- `--device=/dev/mxcd --device=/dev/dri`：挂载沐曦GPU设备
- `--group-add video`：添加video组以访问GPU
- `--shm-size=4G`：设置共享内存大小
- `-v`：挂载工作目录，`host_dir` 为主机端目录，`workspace` 为容器内目录

### 2.2 验证环境

进入容器后，根据所用后端验证框架环境。

**PyTorch 镜像：**
```bash
pip list | grep torch
```
输出显示已安装沐曦定制版 torch（示例）：
```
torch                     2.6.0+metax3.7.x.x
```

**Paddle 镜像：**
```bash
pip list | grep paddle
```
输出显示已安装沐曦定制版 paddle（示例）：
```
paddlepaddle-gpu          x.x.x+metax3.7.x.x
```

### 2.3 设置后端环境变量

DeepXDE 通过环境变量 `DDE_BACKEND` 选择计算后端，请务必确保其与容器内实际安装的框架一致，否则会导致导入失败或算子不可用。

首先检查当前后端设置：
```bash
echo $DDE_BACKEND
```

若与所用镜像不一致，请根据后端类型设置：
```bash
# PyTorch 镜像后端设置
export DDE_BACKEND=pytorch

# Paddle 镜像后端设置
export DDE_BACKEND=paddle
```

### 2.4 安装DeepXDE

**DeepXDE 代码下载：**
下载 DeepXDE 源码，切换到指定 tag 版本 `v1.15.0`：
```bash
git clone git@github.com:lululxvi/deepxde.git
cd deepxde
git checkout v1.15.0
```

**DeepXDE 安装：**
运行如下命令进行安装，安装完成后通过 `pip list` 确认安装成功：
```bash
pip install -e .[all] --use-pep517 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
pip list | grep deepxde
```

## 三、运行测试

DeepXDE 内置丰富的示例，位于源码 `examples/` 目录下，按问题类型组织（如 `pinn_forward`、`pinn_inverse` 等）。可直接运行验证环境是否正常：

```bash
cd /opt/deepxde-1.15.0/examples/
python3 [example]
```

其中 `[example]` 替换为具体示例脚本路径，例如：

```bash
# 示例：求解偏微分方程正问题
cd /opt/deepxde-1.15.0/examples/pinn_forward/
python3 <example>.py
```

> 由于不同后端的部分算子在沐曦平台仍在持续适配中，个别示例在某一后端下可能出现运行问题，框架适配正在持续推进中。

## 四、常见问题与注意事项

### 4.1 后端选择

DeepXDE 通过 `DDE_BACKEND` 环境变量切换后端。运行前务必确认该变量与所用镜像一致（PyTorch 镜像用 `pytorch`，Paddle 镜像用 `paddle`），否则会导致后端加载错误或算子不可用。

### 4.2 镜像与后端对应关系

需使用沐曦的 **[PyTorch镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=pytorch)** 或 **[Paddle镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=paddle-metax)**，DeepXDE 源码在该基础镜像中编译安装。不同后端对应不同基础镜像，请勿混用。

### 4.3 依赖版本兼容性

DeepXDE 对后端框架版本有一定要求，沐曦镜像已预装适配的定制版 PyTorch / Paddle，通常无需手动调整。如需使用更新的 API，请等待官方镜像更新。

#### Dockerfile build

仓库提供了 PyTorch 与 Paddle 两套 Dockerfile，可分别构建对应后端镜像：
```bash
# PyTorch 后端
docker build -f Dockerfile_pytorch -t deepxde-pytorch-maca .

# Paddle 后端
docker build -f Dockerfile_paddle -t deepxde-paddle-maca .
```

## 五、其他

1. 开发者在初次使用曦云GPU运行 DeepXDE 时，遵循本手册搭建环境后，即可按照 [DeepXDE 官方文档](https://deepxde.readthedocs.io) 开始正反问题的求解与训练
2. DeepXDE 在曦云GPU的运行依赖沐曦提供的 PyTorch 或 Paddle 基础镜像，开发者亦可在开发者社区提供的基础镜像上自行安装沐曦提供的框架 whl 包进行测试
3. 沐曦仅维护部分 case 的正确性，如出现运行问题，可提交 issue，亦可在开发者社区提交 bug 反馈
4. 了解更多沐曦开源项目，请参考[沐曦开源社区](https://github.com/metax-maca)


---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。


Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

