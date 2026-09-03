# 沐曦 GPU 运行 Aneurysm 说明文档

## 一、Aneurysm 简介

Aneurysm 是 PaddleScience 中基于物理信息神经网络（PINN）的颅内动脉瘤血流建模案例。模型使用 MLP 将三维坐标 `(x, y, z)` 映射为速度和压力 `(u, v, w, p)`，并通过 Navier-Stokes 方程、入口速度、出口压力、血管壁无滑移和积分流量等约束进行无监督训练。

案例代码位于 `examples/aneurysm`，官方教程见：[PaddleScience Aneurysm 文档](https://paddlescience.readthedocs.io/en/latest/examples/aneurysm/)。

## 二、Docker 构建与运行

- image构建：

```bash
docker build -f Dockerfile -t paddlescience-aneurysm .
```
- 镜像启动
```bash
docker run -it --name aneurysm-maca \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=8G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
    paddlescience-aneurysm \
  /bin/bash
```

参数说明：

- `--device=/dev/mxcd --device=/dev/dri`：挂载沐曦 GPU 设备。
- `--group-add video`：允许容器访问 GPU 设备。
- `--shm-size=8G`：为 PGL 图数据和 MPI 运行提供共享内存。
- `--security-opt seccomp=unconfined`：避免 MPI/SU2 运行时受到容器 seccomp 限制。
- `--ulimit memlock=-1`：允许 GPU 和 MPI 锁定内存。

Dockerfile 会完成：

1. 使用 Metax MACA Paddle 2.6.0 基础镜像。
2. 从 Gitee `develop` 分支克隆 PaddleScience。
3. 安装 PaddleScience 依赖和 PaddleScience 本体。
4. 使用系统现有 GCC/G++ 编译安装 PyMesh，并安装 Open3D、PySDF、pybind11。
5. 下载并解压 Aneurysm STL 几何和评估 CSV 数据。
6. 执行 Python 编译检查和 `ppsci.run_check_mesh()`。
7. 默认执行低采样量最小训练验证。

## 三、运行 Aneurysm

### 3.1 最小训练验证

```bash
docker run --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=8G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  paddlescience-aneurysm
```

容器默认使用 1 个 epoch、1 个 iteration 和较小采样量，用于验证 STL 读取、Mesh 几何、Navier-Stokes 方程、约束、反向传播和可视化链路。

等价的容器内命令：

```bash
cd /workspace/PaddleScience/examples/aneurysm
python3 aneurysm.py \
  hydra.run.dir=outputs_aneurysm/smoke \
  TRAIN.epochs=1 \
  TRAIN.iters_per_epoch=1 \
  TRAIN.iters_integral.igc_outlet=1 \
  TRAIN.iters_integral.igc_integral=1 \
  TRAIN.batch_size.bc_inlet=64 \
  TRAIN.batch_size.bc_outlet=32 \
  TRAIN.batch_size.bc_noslip=64 \
  TRAIN.batch_size.pde=128 \
  TRAIN.integral_batch_size.igc_outlet=16 \
  TRAIN.integral_batch_size.igc_integral=16 \
  EVAL.batch_size=128
```

### 3.2 官方完整训练

官方配置为 1500 个 epoch，每个 epoch 1000 个 iteration：

```bash
python3 aneurysm.py
```

完整训练会同时进行评估和 VTK 可视化，耗时较长。建议使用较大的共享内存，并将输出目录挂载到主机保存。

### 3.3 模型评估

使用官方预训练模型评估：

```bash
python3 aneurysm.py \
  mode=eval \
  hydra.run.dir=outputs_aneurysm/eval \
  EVAL.batch_size=4096 \
  EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/aneurysm_pretrained.pdparams
```

官方文档提供的参考指标如下：

| 指标 | 参考值 |
| --- | ---: |
| `ref_u_v_w_p/loss` | `0.01488` |
| `ref_u_v_w_p/MSE.p` | `0.01412` |
| `ref_u_v_w_p/MSE.u` | `0.00021` |
| `ref_u_v_w_p/MSE.v` | `0.00024` |
| `ref_u_v_w_p/MSE.w` | `0.00032` |

### 3.4 模型导出

```bash
python3 aneurysm.py \
  mode=export \
  hydra.run.dir=outputs_aneurysm/export \
  INFER.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/aneurysm_pretrained.pdparams \
  INFER.export_path=outputs_aneurysm/export/inference/aneurysm
```

### 3.5 模型推理

```bash
python3 aneurysm.py \
  mode=infer \
  hydra.run.dir=outputs_aneurysm/infer \
  INFER.pdmodel_path=outputs_aneurysm/export/inference/aneurysm.pdmodel \
  INFER.pdiparams_path=outputs_aneurysm/export/inference/aneurysm.pdiparams \
  INFER.export_path=outputs_aneurysm/export/inference/aneurysm
```

默认推理模型和导出路径配置在 `conf/aneurysm.yaml` 的 `INFER` 节点中。

## 四、输出结果

Hydra 输出目录通常为：

```text
outputs_aneurysm/YYYY-MM-DD/HH-MM-SS/
```

主要结果包括：

- `train.log`：训练损失、评估指标和检查点日志。
- `checkpoints/epoch_*.pdparams`：训练检查点。
- `result_u_v_w_p_*.vtu`：速度和压力的 VTK 可视化结果。
- `aneurysm_pred.vtu`：推理输出的 VTK 文件。

官方预训练模型的参考指标：

| 指标 | 数值 |
| --- | ---: |
| `loss(ref_u_v_w_p)` | 0.01488 |
| `MSE.p(ref_u_v_w_p)` | 0.01412 |
| `MSE.u(ref_u_v_w_p)` | 0.00021 |
| `MSE.v(ref_u_v_w_p)` | 0.00024 |
| `MSE.w(ref_u_v_w_p)` | 0.00032 |

## 五、常见问题

### 5.1 `pymesh` 导入失败

Aneurysm 使用 `ppsci.geometry.Mesh` 读取 STL。确认以下依赖已安装：

```bash
python3 -c "import open3d, pysdf, pymesh; print('mesh dependencies: ok')"
```

如需手动安装，执行仓库提供的脚本：

```bash
cd /workspace/PaddleScience
bash install_mesh.sh
```

### 5.2 数据文件找不到

确认数据已经解压到案例目录：

```bash
cd /workspace/PaddleScience/examples/aneurysm
find stl -type f -name '*.stl'
test -f data/aneurysm_parabolicInlet_sol0.csv && echo data_ok
```

配置文件中的默认路径为：

```text
./stl/aneurysm_inlet.stl
./stl/aneurysm_outlet.stl
./stl/aneurysm_noslip.stl
./stl/aneurysm_integral.stl
./stl/aneurysm_closed.stl
./data/aneurysm_parabolicInlet_sol0.csv
```

### 5.3 GPU 或共享内存不足

降低采样量进行验证：

```bash
python3 aneurysm.py \
  TRAIN.batch_size.pde=512 \
  TRAIN.batch_size.bc_noslip=512 \
  TRAIN.integral_batch_size.igc_outlet=32 \
  TRAIN.integral_batch_size.igc_integral=32 \
  EVAL.batch_size=512
```

Docker 运行时增加共享内存：

```bash
docker run --rm --shm-size=16G ... paddlescience-aneurysm
```

## 六、参考资料

1. [PaddleScience Aneurysm 官方文档](https://paddlescience.readthedocs.io/en/latest/examples/aneurysm/)
2. [PaddleScience 安装与 Mesh 几何依赖](https://paddlescience.readthedocs.io/en/latest/install_setup/#142)
3. [NVIDIA Modulus Aneurysm 参考](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)
4. [PaddleScience Gitee 镜像](https://gitee.com/paddlepaddle/PaddleScience)

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.