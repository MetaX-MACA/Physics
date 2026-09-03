# PhysicsNeMo-ModelZoo

本目录汇总了开源项目 [PhysicsNeMo v2.0.0](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0) 官方仓库中支持的流体相关模型/案例，供沐曦（MetaX）GPU 上的 AI for Science 场景参考与使用。

沐曦（MetaX）已完成对 PhysicsNeMo 的适配，适配说明见本仓库的 [PhysicsNeMo](../../Frameworks/PhysicsNeMo/)。

---

## 一、PhysicsNeMo 简介

[PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0) 是一个面向科学计算和工程应用的 AI for Science 框架，支持神经算子、图神经网络、Transformer 和物理信息神经网络等方法，可用于计算流体力学、天气气候、分子动力学和地球科学等领域。

### 官方资源

| 资源                  | 链接                                                                         |
| :-------------------- | :--------------------------------------------------------------------------- |
| PhysicsNeMo 官方仓库  | https://github.com/NVIDIA/physicsnemo/tree/v2.0.0                            |
| 沐曦（MetaX）适配说明 | [PhysicsNeMo](../../Frameworks/PhysicsNeMo/)                                |
| CFD 示例目录          | https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd               |
| PhysicsNeMo 官方文档  | https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html |
| 上游许可证            | https://github.com/NVIDIA/physicsnemo/blob/v2.0.0/LICENSE.txt                    |

---

## 二、支持的模型列表

以下为 PhysicsNeMo 官方仓库 `examples/cfd` 目录中与流体相关的模型/案例，PhysicsNeMo 实现指向对应源码目录。

| 模型/案例名称 | 流体相关问题 | PhysicsNeMo 实现 | 沐曦说明文档 |
| :--- | :--- | :--- | :--- |
| Darcy FNO | 二维达西多孔介质流 | [darcy_fno](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_fno) | [DarcyFno](./DarcyFno/README.md) |
| Darcy Nested FNOs | 二维达西流（多 GPU） | [darcy_nested_fnos](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_nested_fnos) | [DarcyNestedFnos](./DarcyNestedFnos/README.md) |
| Darcy Physics Informed | 二维达西流（PINO/DeepONet） | [darcy_physics_informed](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_physics_informed) | — |
| Darcy Transolver | 二维达西多孔介质流 | [darcy_transolver](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_transolver) | [DarcyTransolver](./DarcyTransolver/README.md) |
| Datacenter | 数据中心热气流与温度分布 | [datacenter](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/datacenter) | [Datacenter](./Datacenter/README.md) |
| Aero Graph Net | 车辆外流场压力、剪切力与阻力预测 | [aero_graph_net](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/aero_graph_net) | [AeroGraphNet](./AeroGraphNet/README.md) |
| Domino | 车辆外流场表面与体积场预测 | [domino](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/domino) | [Domino](./Domino/README.md) |
| Domino NIM Finetuning | 汽车 CFD 预训练模型微调 | [domino_nim_finetuning](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/domino_nim_finetuning) | — |
| FigConvNet | 大规模三维车辆外流场预测 | [figconvnet](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/figconvnet) | — |
| MOE | 多模型外流场预测融合 | [moe](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/moe) | — |
| Transformer Models | 非规则网格外流场代理模型 | [transformer_models](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/transformer_models) | — |
| XAeroNet | 可扩展外流场表面与体积预测 | [xaeronet](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/xaeronet) | — |
| Flow Reconstruction Diffusion | 高保真流场重建与超分辨 | [flow_reconstruction_diffusion](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/flow_reconstruction_diffusion) | — |
| Gray Scott RNN | 三维 Gray-Scott 反应扩散 | [gray_scott_rnn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/gray_scott_rnn) | — |
| Lagrangian MGN | 拉格朗日流体粒子仿真 | [lagrangian_mgn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/lagrangian_mgn) | — |
| LDC PINNs | 顶盖驱动方腔流 | [ldc_pinns](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/ldc_pinns) | — |
| MHD PINO | 磁流体动力学方程 | [mhd_pino](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/mhd_pino) | — |
| Navier Stokes Dpot | 二维 Navier-Stokes 方程求解 | [navier_stokes_dpot](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/navier_stokes_dpot) | — |
| Navier Stokes RNN | 瞬态二维 Navier-Stokes 流预测 | [navier_stokes_rnn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/navier_stokes_rnn) | — |
| Stokes MGN | Stokes 流场预测 | [stokes_mgn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/stokes_mgn) | — |
| SWE Distributed GNN | 非线性浅水方程（分布式） | [swe_distributed_gnn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/swe_distributed_gnn) | — |
| SWE Nonlinear PINO | 非线性浅水方程 | [swe_nonlinear_pino](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/swe_nonlinear_pino) | — |
| Transient Conjugate Heat Transfer Tank Fill | 高压气体储罐瞬态充装传热 | [transient_conjugate_heat_transfer_tank_fill](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/transient_conjugate_heat_transfer_tank_fill) | — |
| Vortex Shedding Mesh Reduced | 瞬态涡脱落（网格降维） | [vortex_shedding_mesh_reduced](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/vortex_shedding_mesh_reduced) | — |
| Vortex Shedding MGN | 参数化几何瞬态涡脱落 | [vortex_shedding_mgn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/vortex_shedding_mgn) | — |

> 更多案例与更新请以 PhysicsNeMo v2.0.0 官方仓库的 [`examples/cfd`](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd) 目录为准。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
