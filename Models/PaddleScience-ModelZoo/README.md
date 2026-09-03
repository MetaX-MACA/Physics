# PaddleScience-ModelZoo

本目录汇总了开源项目 [PaddleScience](https://github.com/PaddlePaddle/PaddleScience) 官方仓库中支持的流体相关模型/案例，供沐曦（MetaX）GPU 上的 AI for Science 场景参考与使用。

沐曦（MetaX）已完成对 PaddleScience 的适配，适配说明见本仓库的 [PaddleScience](../../Frameworks/PaddleScience/)。

---

## 一、PaddleScience 简介

[PaddleScience](https://github.com/PaddlePaddle/PaddleScience) 是一个基于深度学习框架 PaddlePaddle 开发的科学计算套件，利用深度神经网络的学习能力和 PaddlePaddle 框架的自动(高阶)微分机制，解决物理、化学、气象等领域的问题。支持物理机理驱动、数据驱动、数理融合三种求解方式，并提供了基础 API 和详尽文档供用户使用与二次开发。

### 官方资源

| 资源 | 链接 |
| :--- | :--- |
| 项目主页 | https://github.com/PaddlePaddle/PaddleScience |
| 沐曦（MetaX）适配说明 | [PaddleScience](../../Frameworks/PaddleScience/)                       |
| 官方文档 | https://paddlescience-docs.readthedocs.io/zh-cn/latest/ |
| 多硬件支持列表 | https://paddlescience-docs.readthedocs.io/zh-cn/latest/multi_device/ |
| 开源协议 | [Apache License 2.0](https://github.com/PaddlePaddle/PaddleScience/blob/develop/LICENSE) |

---

## 二、支持的模型列表

以下为 PaddleScience 仓库 `examples` 目录中与**流体**相关的模型/案例：**PaddleScience实现**指向其对应的源码目录。大部分案例都可以参考PaddleScience文档中的实现，部分复杂案例可以参考沐曦平台说明文档。

| 模型/案例名称 | 流体相关问题 | PaddleScience实现 | PaddleScience参考文档 | 沐曦说明文档 |
| :--- | :--- | :--- | :--- | :--- |
| adv | 1D 线性对流问题（CViT） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/adv | [CViT(Advection)](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/adv_cvit/) | — |
| aneurysm | 3D 颅内动脉瘤血流 | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/aneurysm | [Aneurysm](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/aneurysm/) | [ANEURYSM](./ANEURYSM/README.md) |
| bubble | 气液两相流（BubbleNet） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/bubble | [BubbleNet](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/bubble/) | — |
| cfdgcn | 求解器耦合（CFD-GCN） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/cfdgcn | [CFD-GCN](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/cfdgcn/) | [CFD-GCN](./CFD-GCN/README.md) |
| cylinder/2d_unsteady | 2D 圆柱绕流（Re100 等） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/cylinder/2d_unsteady | [Cylinder2D_unsteady](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/cylinder2d_unsteady/) | — |
| darcy | 2D 达西流 | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/darcy | [Darcy2D](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/darcy2d/) | — |
| deepcfd | 任意 2D 几何体绕流（DeepCFD） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/deepcfd | [DeepCFD](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/deepcfd/) | — |
| deephpms | 伯格斯方程（Burgers，流体动力学基础方程） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/deephpms | [DeepHPMs](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/deephpms/) | — |
| drivaernet | 汽车表面阻力预测（DrivAerNet） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/drivaernet | [DrivAerNet](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/drivaernet/) | [DrivAerNet](./DrivAerNet/README.md) |
| fsi | 流固耦合 / 涡激振动（VIV） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/fsi | [ViV](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/viv/) | — |
| ldc | 2D 方腔流（定常 / 非定常） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/ldc | [LDC2D_steady](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/ldc2d_steady/) · [LDC2D_unsteady](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/ldc2d_unsteady/) | — |
| ns | 2D 方腔浮力驱动流（N-S 方程，CViT） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/ns | [CViT(NS)](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/ns_cvit/) | — |
| nsfnet | Navier-Stokes 流场预测（NSFNet） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/nsfnet | [NSFNets](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/nsfnet/) | — |
| pipe | 2D 管道流 | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/pipe | [LabelFree-DNN-Surrogate](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/labelfree_DNN_surrogate/) | — |
| shock_wave | 2D 空气激波（可压缩流体） | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/shock_wave | [ShockWave](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/shock_wave/) | — |
| tempoGAN | 2D 湍流流场重构 | https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/tempoGAN | [tempoGAN](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/tempoGAN/) | — |

> 更多案例与更新请以 PaddleScience 官方仓库的 [`examples`](https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples) 目录为准。

---



本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
