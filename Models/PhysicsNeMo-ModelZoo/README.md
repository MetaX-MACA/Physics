# PhysicsNeMo-ModelZoo

本目录汇总了开源项目 [PhysicsNeMo v2.0.0](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0) 的模型与案例，覆盖计算流体力学、天气气候、工程仿真、地球科学及其他 AI for Science 场景，供沐曦（MetaX）GPU 用户参考与使用。

沐曦（MetaX）已完成对 PhysicsNeMo 的适配，适配说明见本仓库的 [PhysicsNeMo](../../Frameworks/PhysicsNeMo/)。

各案例的 A100/C500 输出与验证章节可复用[通用模板](./OUTPUT_VALIDATION_TEMPLATE.md)。

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
| PhysicsNeMo 上游许可证 | https://github.com/NVIDIA/physicsnemo/blob/v2.0.0/LICENSE.txt                    |

---

## 二、支持的模型列表

以下列出本目录中的案例。PhysicsNeMo 实现列链接到 v2.0.0 官方案例源码。

| 模型/案例名称 | 应用场景 | PhysicsNeMo 实现 | 沐曦说明文档 |
| :--- | :--- | :--- | :--- |
| Darcy FNO | 二维达西多孔介质流 | [darcy_fno](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_fno) | [DarcyFno](./DarcyFno/README.md) |
| Darcy Nested FNOs | 二维达西流（多 GPU） | [darcy_nested_fnos](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_nested_fnos) | [DarcyNestedFnos](./DarcyNestedFnos/README.md) |
| Darcy Physics Informed | 二维达西流（PINO/DeepONet） | [darcy_physics_informed](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_physics_informed) | [DarcyPhysicsInformed](./DarcyPhysicsInformed/README.md) |
| Darcy Transolver | 二维达西多孔介质流 | [darcy_transolver](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/darcy_transolver) | [DarcyTransolver](./DarcyTransolver/README.md) |
| Datacenter | 数据中心热气流与温度分布 | [datacenter](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/datacenter) | [Datacenter](./Datacenter/README.md) |
| Aero Graph Net | 车辆外流场压力、剪切力与阻力预测 | [aero_graph_net](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/aero_graph_net) | [AeroGraphNet](./AeroGraphNet/README.md) |
| Domino | 车辆外流场表面与体积场预测 | [domino](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/domino) | [Domino](./Domino/README.md) |
| FigConvNet | 大规模三维车辆外流场预测 | [figconvnet](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/figconvnet) | [FigConvNet](./FigConvNet/README.md) |
| Transformer Models | 非规则网格外流场代理模型 | [transformer_models](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/transformer_models) | [TransformerModels](./TransformerModels/README.md) |
| XAeroNet | 可扩展外流场表面与体积预测 | [xaeronet](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/external_aerodynamics/xaeronet) | [XAeroNet](./XAeroNet/README.md) |
| Flow Reconstruction Diffusion | 高保真流场重建与超分辨 | [flow_reconstruction_diffusion](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/flow_reconstruction_diffusion) | [FlowReconstructionDiffusion](./FlowReconstructionDiffusion/README.md) |
| Gray Scott RNN | 三维 Gray-Scott 反应扩散 | [gray_scott_rnn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/gray_scott_rnn) | [GrayScottRNN](./GrayScottRNN/README.md) |
| Lagrangian MGN | 拉格朗日流体粒子仿真 | [lagrangian_mgn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/lagrangian_mgn) | [LagrangianMGN](./LagrangianMGN/README.md) |
| MHD PINO | 磁流体动力学方程 | [mhd_pino](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/mhd_pino) | [MHDPINO](./MHDPINO/README.md) |
| Navier Stokes RNN | 瞬态二维 Navier-Stokes 流预测 | [navier_stokes_rnn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/navier_stokes_rnn) | [NavierStokesRNN](./NavierStokesRNN/README.md) |
| Stokes MGN | Stokes 流场预测 | [stokes_mgn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/stokes_mgn) | [StokesMGN](./StokesMGN/README.md) |
| SWE Distributed GNN | 非线性浅水方程（分布式） | [swe_distributed_gnn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/swe_distributed_gnn) | [SWEDistributedGNN](./SWEDistributedGNN/README.md) |
| SWE Nonlinear PINO | 非线性浅水方程 | [swe_nonlinear_pino](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/swe_nonlinear_pino) | [SWENonlinearPINO](./SWENonlinearPINO/README.md) |
| Vortex Shedding Mesh Reduced | 瞬态涡脱落（网格降维） | [vortex_shedding_mesh_reduced](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/vortex_shedding_mesh_reduced) | [VortexSheddingMeshReduced](./VortexSheddingMeshReduced/README.md) |
| Vortex Shedding MGN | 参数化几何瞬态涡脱落 | [vortex_shedding_mgn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/cfd/vortex_shedding_mgn) | [VortexSheddingMGN](./VortexSheddingMGN/README.md) |
| CorrDiff | 公里级天气降尺度 | [corrdiff](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/weather/corrdiff) | [CorrDiff](./CorrDiff/README.md) |
| StormCast | 对流尺度天气扩散预测 | [stormcast](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/weather/stormcast) | [StormCast](./StormCast/README.md) |
| HydroGraphNet | 洪水水深与水量预测 | [hydrographnet](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/weather/flood_modeling/hydrographnet) | [HydroGraphNet](./HydroGraphNet/README.md) |
| Automotive Crash Dynamics | 汽车碰撞结构响应预测 | [crash](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/structural_mechanics/crash) | [Crash](./Crash/README.md) |
| Deforming Plate MGN | 变形板动力学预测 | [deforming_plate](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/structural_mechanics/deforming_plate) | [DeformingPlateMgn](./DeformingPlateMgn/README.md) |
| BloodFlow 1D MGN | 一维心血管血流仿真 | [bloodflow_1d_mgn](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/healthcare/bloodflow_1d_mgn) | [Bloodflow1dMgn](./Bloodflow1dMgn/README.md) |
| Lennard-Jones MGN | 分子动力学原子受力预测 | [lennard_jones](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0/examples/molecular_dynamics/lennard_jones) | [LennardJonesMgn](./LennardJonesMgn/README.md) |

> 更多案例与更新请以 [PhysicsNeMo v2.0.0 官方仓库](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0) 为准。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
