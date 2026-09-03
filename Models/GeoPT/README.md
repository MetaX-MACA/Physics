# 沐曦GPU运行GeoPT说明文档

## 一、GeoPT简介

[GeoPT](https://github.com/Physics-Scaling/GeoPT)（Scaling Physics Simulation via Lifted Geometric Pre-Training）是面向通用物理仿真的几何预训练模型。该模型通过几何预训练学习物理系统先验，并将动力学条件作为输入提示，迁移至不同的下游仿真任务。

本文档仅说明在沐曦GPU上加载GeoPT预训练权重，对AirCraft数据集进行微调和评估的方法。项目源代码、论文、下游数据集和预训练模型请分别参考[官方仓库](https://github.com/Physics-Scaling/GeoPT)、[论文页面](https://arxiv.org/abs/2602.20399)、[下游数据集页面](https://huggingface.co/datasets/GeoPT/Downstream_Physics_Simulation)和[预训练模型页面](https://huggingface.co/GeoPT/GeoPT_Pretrained_Models)。

GeoPT由第三方公开发布，源代码遵循MIT License，版权归GeoPT作者所有。本项目未对其源代码作任何修改，也不分发其源代码、目标代码、预训练权重或数据集；相关资源的使用应遵守原始发布页面及适用许可证的要求。

## 二、沐曦GPU环境配置与运行

### 2.1 环境准备

搜索使用沐曦开发者社区提供的[maca-pytorch镜像](https://developer.metax-tech.com/softnova/docker)：

```bash
docker run -it --name test-geopt \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=16G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /host_dir:/workspace \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.6-py310-ubuntu24.04-amd64 \
  /bin/bash
```

> 可根据需要使用更新版本镜像，详情参见[沐曦开发者社区](https://developer.metax-tech.com/softnova/docker)。基础镜像中的沐曦PyTorch不可使用通用PyTorch包覆盖。

**参数说明：**

* `--device=/dev/mxcd --device=/dev/dri`：挂载沐曦GPU设备。
* `--group-add video`：添加video组以访问GPU设备。
* `--shm-size=16G`：设置共享内存大小。
* `-v /host_dir:/workspace`：挂载工作目录，`/host_dir`为主机端目录。

### 2.2 验证环境

进入容器后，验证torch环境：

```bash
pip list | grep torch
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA OK:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

输出应显示沐曦定制版PyTorch，且`CUDA OK`为`True`。

### 2.3 安装GeoPT

**Git依赖：**

以下代码下载步骤依赖Git。若基础镜像未预装Git，请先安装Git后再继续。

**GeoPT代码下载：**

```bash
cd /workspace
git clone https://github.com/Physics-Scaling/GeoPT.git
cd GeoPT
git checkout 9984f359fd44b6332c03ca55ab2e659e0c6e899d
```

**GeoPT安装：**

基础镜像已经提供沐曦PyTorch。安装其余依赖时应排除上游`requirements.txt`中的`torch`项：

```bash
sed '/^torch$/d' requirements.txt > /tmp/geopt-requirements.txt
pip install -r /tmp/geopt-requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export TORCHDYNAMO_DISABLE=1
```

### 2.4 数据与预训练权重准备

下游数据集和预训练权重不随本项目分发，请在符合其使用条款的前提下，从作者指定页面下载。

AirCraft数据预处理示例：

```bash
cd /workspace/GeoPT
python data_preprocess/AirCraft_process.py \
  --hf_repo GeoPT/Downstream_Physics_Simulation \
  --hf_subdir AirCraft \
  --outdir /workspace/data/AirCraft \
  --pattern "*0.h5" \
  --seed 42
```

下载预训练权重并放置到`checkpoints/`目录：

```bash
hf download GeoPT/GeoPT_Pretrained_Models GeoPT_8layers.pt \
  --local-dir /workspace/GeoPT/checkpoints
```

## 三、GeoPT微调与推理

### 3.1 AirCraft GeoPT微调

```bash
cd /workspace/GeoPT
python run.py \
  --gpu 0 \
  --data_path /workspace/data/AirCraft \
  --loader AirCraft \
  --task GeoPT_finetune \
  --dynamics craft \
  --geotype unstructured \
  --space_dim 3 --fun_dim 11 --out_dim 6 --normalize 1 \
  --model Transolver --n_hidden 256 --n_heads 8 --n_layers 8 \
  --mlp_ratio 2 --slice_num 32 \
  --ntrain 100 --ntest 50 \
  --batch-size 1 --epochs 50 \
  --save_name craft_geopt_8layers \
  --finetune 1 --finetune_name GeoPT_8layers
```

### 3.2 模型评估

使用与训练一致的参数，并增加`--eval 1`：

```bash
python run.py \
  --gpu 0 \
  --data_path /workspace/data/AirCraft \
  --loader AirCraft \
  --task GeoPT_finetune \
  --dynamics craft \
  --geotype unstructured \
  --space_dim 3 --fun_dim 11 --out_dim 6 --normalize 1 \
  --model Transolver --n_hidden 256 --n_heads 8 --n_layers 8 \
  --mlp_ratio 2 --slice_num 32 \
  --ntrain 100 --ntest 50 \
  --batch-size 1 --epochs 50 \
  --eval 1 \
  --save_name craft_geopt_8layers \
  --vis_num 10
```

训练结果保存在`checkpoints/`、`training_logs/`和`results/<save_name>/`目录。

## 四、常见问题与注意事项

### 4.1 找不到预训练权重

请从作者指定页面获取`GeoPT_8layers.pt`，并放置到`/workspace/GeoPT/checkpoints/`。

## 五、Dockerfile构建

本目录提供[Dockerfile](Dockerfile)，可构建包含已验证GeoPT环境的镜像：

```bash
docker build -t geopt-maca .
```

## 六、其他

1. 开发者可遵循本手册完成环境验证、AirCraft数据准备、GeoPT微调和评估。
2. 沐曦仅提供本目录所述环境和适配说明；如出现运行问题，可提交Issue，亦可在开发者社区提交bug反馈。
3. 了解更多沐曦开源项目，请参考[沐曦开源社区](https://github.com/metax-maca)。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
