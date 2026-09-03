# 沐曦 GPU 运行 CFD-GCN 说明文档

本项目由第三方以开源许可证发布，我方未对其源代码作任何修改。您可通过以下链接 https://paddlescience.readthedocs.io/en/latest/examples/cfdgcn/ 查阅本项目的源代码、许可证全文、版权与归属声明及其他声明文件。


## 一、CFD-GCN 简介

一种基于图神经网络的 CFD 计算模型，称为 CFD-GCN (Computational fluid dynamics - Graph convolution network)，该模型是一种混合的图神经网络，它将传统的图卷积网络与粗分辨率的 CFD 模拟器相结合，不仅可以大幅度加速 CFD 预测，还可以很好地泛化到新的场景，与此同时，模型的预测效果远远优于单独的粗分辨率 CFD 的模拟效果。
本案例使用 NACA0012 翼型数据集，包含训练集、测试集以及细网格和粗网格。案例代码位于 `examples/cfdgcn`，官方教程见：[PaddleScience CFDGCN 文档](https://paddlescience.readthedocs.io/en/latest/examples/cfdgcn/)。

## 二、Docker 构建与运行

- image构建
```bash
docker build \
  -f Dockerfile \
  -t paddlescience-cfdgcn .
```

- 镜像启动
```bash
docker run -it --name cfdgcn-maca \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=8G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
    paddlescience-cfdgcn \
  /bin/bash
```

参数说明：

- `--device=/dev/mxcd --device=/dev/dri`：挂载沐曦 GPU 设备。
- `--group-add video`：允许容器访问 GPU 设备。
- `--shm-size=8G`：为 PGL 图数据和 MPI 运行提供共享内存。
- `--security-opt seccomp=unconfined`：避免 MPI/SU2 运行时受到容器 seccomp 限制。
- `--ulimit memlock=-1`：允许 GPU 和 MPI 锁定内存。

Dockerfile 默认完成以下工作：

1. 使用 Metax MACA Paddle 2.6.0 基础镜像。
2. 从 Gitee `develop` 分支克隆 PaddleScience。
3. 编译安装 OpenMPI 1.10.2。
4. 安装 `pgl==2.2.6` 和 `mpi4py==3.1.4`。
5. 下载并解压 CFD-GCN 数据集、网格和 SU2 v6.2.0 预编译库。
6. 设置 `MPI_HOME`、`SU2_RUN`、`SU2_HOME`、`PATH`、`LD_LIBRARY_PATH` 和 `PYTHONPATH`。
7. 执行 Python 导入和代码编译检查。

## 三、运行 CFD-GCN

### 3.1 最小训练验证

镜像默认执行 1 个 epoch、1 个 iteration 的最小训练：

```bash
docker run --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=8G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  paddlescience-cfdgcn
```

等价命令如下：

```bash
mpirun --allow-run-as-root -np 2 python3 cfdgcn.py \
  mode=train \
  TRAIN.batch_size=1 \
  TRAIN.epochs=1 \
  TRAIN.iters_per_epoch=1 \
  TRAIN.save_freq=1 \
  TRAIN.eval_during_train=false
```

### 3.2 完整训练

完整训练使用 4 个 MPI worker：

```bash
docker run --rm -it \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=8G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  paddlescience-cfdgcn \
  sh -lc 'mpirun --allow-run-as-root -np 5 python3 cfdgcn.py TRAIN.batch_size=4'
```

### 3.3 泛化实验

使用 `NACA0012_machsplit_noshock` 数据集进行泛化实验：

```bash
docker run --rm -it \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=8G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  paddlescience-cfdgcn \
  sh -lc 'mpirun --allow-run-as-root -np 5 python3 cfdgcn.py \
    TRAIN.batch_size=4 \
    TRAIN_DATA_DIR=./data/NACA0012_machsplit_noshock/outputs_train \
    TRAIN_MESH_GRAPH_PATH=./data/NACA0012_machsplit_noshock/mesh_fine.su2 \
    EVAL_DATA_DIR=./data/NACA0012_machsplit_noshock/outputs_test \
    EVAL_MESH_GRAPH_PATH=./data/NACA0012_machsplit_noshock/mesh_fine.su2'
```

### 3.4 模型评估

指定训练得到的模型参数：

```bash
mpirun --allow-run-as-root -np 2 python3 cfdgcn.py \
  mode=eval \
  EVAL.pretrained_model_path=/path/to/epoch_500.pdparams
```

## 四、输出结果

训练输出目录由 Hydra 自动生成，通常位于：

```text
outputs_cfdgcn/YYYY-MM-DD/HH-MM-SS/
```

主要输出包括：

- `checkpoints/epoch_*.pdparams`：模型检查点。
- `train.log`：训练日志和损失。
- `result/image/cylinder` 或 `result/image/airfoil`：预测结果可视化。
- `temp_meshes`、`temp_data`：SU2 运行产生的临时文件。

### 4.1 圆柱流场可视化结果

以下结果来自 `examples/cfdgcn/result/image/cylinder`。每张图左侧为模型预测结果（`pred`），右侧为参考结果（`true`）。

#### 压力场

![圆柱流场压力预测与参考结果](img/0field.png0_field.png)

#### x 方向速度场

![圆柱流场 x 方向速度预测与参考结果](img/0field.png1_field.png)

#### y 方向速度场

![圆柱流场 y 方向速度预测与参考结果](img/0field.png2_field.png)

## 五、常见问题

### 5.1 `libmpi.so.12` 或 `libmpi_cxx.so.1` 找不到

SU2 v6.2.0 依赖 OpenMPI 1.x，不能直接使用 OpenMPI 4。Dockerfile 已在 `/opt/openmpi` 编译安装 OpenMPI 1.10.2，并设置：

```bash
export MPI_HOME=/opt/openmpi
export LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH
```

### 5.2 `pysu2` 导入失败

确认 SU2 环境变量：

```bash
export SU2_RUN=/workspace/PaddleScience/examples/cfdgcn/SU2Bin
export SU2_HOME=/workspace/PaddleScience/examples/cfdgcn/SU2Bin
export PATH=$SU2_RUN:$PATH
export PYTHONPATH=$SU2_RUN:$PYTHONPATH
```

### 5.3 MPI 进程数量不匹配

CFD-GCN 需要 1 个 master 加至少 1 个 worker。batch size 为 `N` 时，建议使用：

```bash
mpirun -np $((N + 1)) python3 cfdgcn.py TRAIN.batch_size="$N"
```

### 5.4下载数据耗时较长

CFD-GCN 数据和 SU2 资源较大，Docker 构建阶段需要下载约 1.4 GB 文件。建议配置国内网络、代理或内部制品镜像，并通过 Docker build args 覆盖：

```bash
docker build \
  --build-arg CFDGCN_DATA_URL=https://your-mirror/CFDGCN/data.zip \
  --build-arg CFDGCN_MESH_URL=https://your-mirror/CFDGCN/meshes.tar \
  --build-arg CFDGCN_SU2_URL=https://your-mirror/CFDGCN/SU2Bin.tgz \
  -f examples/cfdgcn/Dockerfile \
  -t paddlescience-cfdgcn .
```

## 六、参考资料

1. [PaddleScience CFDGCN 官方文档](https://paddlescience.readthedocs.io/en/latest/examples/cfdgcn/)
2. [PaddleScience 官方仓库](https://github.com/PaddlePaddle/PaddleScience)
3. [PaddleScience Gitee 镜像](https://gitee.com/paddlepaddle/PaddleScience)
4. [PaddleScience 多硬件支持](https://paddlescience.readthedocs.io/en/latest/multi_device/)

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.