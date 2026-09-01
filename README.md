# Physics

本项目汇集了多个可用于**沐曦（MetaX）GPU** 的**物理** 模型和应用。

---

## 📂 项目结构与说明

本文件夹以**聚合分发**方式提供多个相互独立的开源项目，各项目作为独立程序分别置于各自子目录中，我方仅将其聚合打包传输，并未将其组合为单一作品。各项目分别受其自带许可证约束，您在使用或分发任一项目前，须遵守该项目对应许可证的条款。如各项目内另附有我方提供的Patch文件或适配文件，系针对相应开源项目的修改或组合，此文件的许可证放置于对应文件夹中。

当前包含以下独立子项目：

| 子目录                                                                                                          | 项目简介                                                                                                       | 主要领域                                   |
| :-------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :----------------------------------------- |
| **[DeepCFD](https://github.com/MetaX-MACA/Physics/tree/main/DeepCFD)**                               | 基于卷积神经网络（CNN）的模型，可高效求解非均匀稳态层流问题。                                                  | 流体力学模型、流场预测                     |
| **[PaddleScience-ModelZoo](https://github.com/MetaX-MACA/Physics/tree/main/PaddleScience-ModelZoo)** | 汇总了开源项目[PaddleScience](https://github.com/PaddlePaddle/PaddleScience)官方仓库中支持的流体相关模型/案例   | PaddleScience支持的物理、化学、气象等领域  |
| **[PhysicsNeMo-ModelZoo](https://github.com/MetaX-MACA/Physics/tree/main/PhysicsNeMo-ModelZoo)**     | 汇总了开源项目[PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/v2.0.0)官方仓库中支持的流体相关模型/案例 | PhysicsNeMo支持的CFD、热传导、外流场等领域 |

> 更多模型和应用正在持续适配与添加中，敬请关注。

---

## 🚀 支持的AI4CFD应用/模型列表

以下是目前已适配沐曦GPU并可用于物理学及相关领域研究的部分模型与应用概览：

| 模型/应用                                                                                                      | 描述                                                                                       | 框架/生态                           |
| :------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- | :---------------------------------- |
| **DeepCFD**                                                                                              | 基于卷积神经网络（CNN）的模型，可高效求解非均匀稳态层流问题                                | PyTorch                             |
| **[DrivAerNet](https://github.com/MetaX-MACA/Physics/tree/main/PaddleScience-ModelZoo/DrivAerNet)** | 汽车表面阻力预测模型                                                                       | PaddleScience                       |
| **[Domino](https://github.com/MetaX-MACA/Physics/tree/main/PhysicsNeMo-ModelZoo/Domino)**                                                                                               | 车辆外流场表面与体积场预测                                                                       | PhysicsNeMo                         |
| **[AeroGraphNet](https://github.com/MetaX-MACA/Physics/tree/main/PhysicsNeMo-ModelZoo/AeroGraphNet)**                                                                                         | 车辆外流场压力、剪切力与阻力预测                                                                       | PhysicsNeMo                         |
| **[DarcyFno](https://github.com/MetaX-MACA/Physics/tree/main/PhysicsNeMo-ModelZoo/DarcyFno)** | 二维达西多孔介质流 | PhysicsNeMo |
| **[DarcyNestedFnos](https://github.com/MetaX-MACA/Physics/tree/main/PhysicsNeMo-ModelZoo/DarcyNestedFnos)** | 二维达西流（多 GPU） | PhysicsNeMo |
| **[DarcyTransolver](https://github.com/MetaX-MACA/Physics/tree/main/PhysicsNeMo-ModelZoo/DarcyTransolver)** | 二维达西多孔介质流 | PhysicsNeMo |
| **[Datacenter](https://github.com/MetaX-MACA/Physics/tree/main/PhysicsNeMo-ModelZoo/Datacenter)** | 数据中心热气流与温度分布 | PhysicsNeMo |
| **[Aneurysm](https://github.com/MetaX-MACA/Physics/tree/main/PaddleScience-ModelZoo/ANEURYSM)**     | 颅内动脉瘤压力建模                                                                         | PaddleScience、PhysicsNeMo          |
| **[CFD-GCN](https://github.com/MetaX-MACA/Physics/tree/main/PaddleScience-ModelZoo/CFD-GCN)**       | 通用流场高分辨率模型                                                                       | PaddleScience、PyTorch              |
| **FNO**                                                                                                  | 傅里叶神经算子模型                                                                         | Pytorch、PhysicsNeMo、PaddleScience |
| **KAN**                                                                                                  | 一种用可学习的样条函数替代固定激活函数的神经网络，在符号回归等任务中表现出高精度和可解释性 | PyTorch、PhysiceNeMo                |
| **DeepOnet**                                                                                             | 神经算子模型                                                                               | PhysicsNeMo、PaddleScience、DeepXDE |

---

## 🔧 环境与依赖

* **硬件**：需具备沐曦（MetaX）GPU。
* **基础软件**：Linux操作系统，沐曦GPU运行时环境。
* **主要依赖**：Python 3.8+，PyTorch/TensorFlow/PaddlePaddle/PhysiceNeMo，以及各子项目特定的依赖库（详见各子目录下的`requirements.txt`或文档）。

具体安装与使用步骤，请参阅各子项目文件夹内的独立指南（如`README.md`或`UserGuide`）。

---

## ⚠️ 合规与许可证声明

在使用本仓库中的任何资源前，请仔细阅读并遵守以下声明：

1. **独立项目**：本仓库中的每个子项目均为独立的开源项目，保留其原有的许可证和版权声明。我方仅提供聚合分发服务，不对这些项目的功能、安全性或合规性做额外担保。
2. **使用责任**：您有责任理解并遵守每个子项目自带的许可证条款。在使用或分发任何子项目时，请确保完全符合其许可证要求。
3. **修改与组合**：如子项目内包含由我方提供的Patch文件或适配文件，这些特定文件的许可证将放置于对应的子文件夹中，请在使用时一并遵循。

---

## 📝 更新日志

* **2026-08-17**：CFD-GCN、Aneurysm、DrivAerNet模型归笼到paddlescience-modelzoo。
* **2026-07-14**：初始版本，包含DeepCFD、CFD-GCN、Aneurysm、DrivAerNet模型，并提供了基础的用户指南。

---

## 🤝 贡献与反馈

欢迎通过GitHub Issues或Pull Requests提出建议、报告问题或贡献新的适配模型。让我们一起推动AI在计算流体力学领域的革新！

---

**感谢您对沐曦GPU生态及AI4CFD发展的关注与支持！**
