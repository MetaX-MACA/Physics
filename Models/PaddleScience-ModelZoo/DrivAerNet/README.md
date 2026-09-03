# 沐曦 GPU 运行 DrivAerNet 说明文档

## 一、案例简介

DrivAerNet 是用于汽车气动设计的高保真三维汽车 CFD 数据集。本案例使用 `RegDGCNN` 从汽车表面点云直接预测阻力系数 `Cd`，属于数据驱动的监督学习任务。

代码位于 `examples/drivaernet`，官方教程见 [PaddleScience DrivAerNet 文档](https://paddlescience.readthedocs.io/en/latest/examples/drivaernet/)。官方预训练模型在测试集上的参考指标为 `R?0?5=88.22%`。

## 二、Docker 构建与运行

- image构建：

```bash
docker build -f Dockerfile -t paddlescience-drivaernet .
```

- docker运行：

```bash
docker run -it --name drivaernet-maca \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=8G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
    paddlescience-drivaernet \
  /bin/bash
```

参数说明：

- `--device=/dev/mxcd --device=/dev/dri`：挂载沐曦 GPU 设备。
- `--group-add video`：允许容器访问 GPU 设备。
- `--shm-size=8G`：为 PGL 图数据和 MPI 运行提供共享内存。
- `--security-opt seccomp=unconfined`：避免 MPI/SU2 运行时受到容器 seccomp 限制。
- `--ulimit memlock=-1`：允许 GPU 和 MPI 锁定内存。

Dockerfile 会安装 PaddleScience 和 PGL，下载 `data.tar`，执行 Python 编译检查，并默认运行 1 个 epoch、1 个 iteration 的训练验证。可通过 `DRIVAERNET_DATA_URL` 构建参数替换数据地址。

## 三、训练、评估

完整训练：

```bash
python drivaernet.py
```

评估预训练模型：

```bash
python drivaernet.py mode=eval \
  EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/DrivAerNet/CdPrediction_DrivAerNet_r2_100epochs_5k_pretrained.pdparams
```

资源有限时可降低采样点数和迭代次数：

```bash
python drivaernet.py TRAIN.num_points=512 EVAL.num_points=512 \
  TRAIN.epochs=1 TRAIN.iters_per_epoch=1 TRAIN.num_workers=0 EVAL.num_workers=0
```

训练输出默认位于 `outputs_drivaernet/运行目录/`，包含日志、检查点和评估指标。

## 四、输出结果

训练完成后，Hydra 会在 `outputs_drivaernet/运行目录/` 下生成结果目录。训练和评估结果均保存在该目录中。

```text
outputs_drivaernet/运行目录/
```

运行目录名称由 Hydra 根据运行配置自动生成。

主要输出文件：

| 路径 | 说明 |
| --- | --- |
| `train.log` | 训练和验证过程日志 |
| `checkpoints/best_model.pdparams` | 验证指标最优的模型参数 |
| `checkpoints/latest.pdparams` | 最近一次保存的模型参数 |
| `checkpoints/latest.pdopt` | 优化器状态 |
| `.hydra/config.yaml` | 本次运行的完整配置 |
| `.hydra/overrides.yaml` | 本次运行使用的参数覆盖项 |

结果指标：

| 指标 | 数值 |
| --- | ---: |
| 训练损失 | 记录训练过程中的模型损失 |
| 验证损失 | 记录验证阶段的模型损失 |
| 验证 MSE（`cd_value`） | 记录阻力系数预测误差 |

## 五、常见问题

- `FileNotFoundError`：确认当前工作目录是 `examples/drivaernet`，且 `data.tar` 已解压。
- 显存不足：降低 `TRAIN.num_points`、`EVAL.num_points`，并保持 `batch_size=1`。
- 多进程数据加载异常：将 `TRAIN.num_workers` 和 `EVAL.num_workers` 设置为 `0`。
- paddle.device.empty_cache()在Paddle 2.6.0 没有该接口，替换为paddle.device.cuda.empty_cache()

## 六、参考资料
1. [官方 DrivAerNet 文档](https://paddlescience.readthedocs.io/en/latest/examples/drivaernet/)
2. [DrivAerNet 论文代码](https://github.com/Mohamedelrefaie/DrivAerNet)。

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.