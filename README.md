# Physics

本项目汇集了多个可用于 **沐曦（MetaX）GPU** 的 **Physics 模型、物理仿真应用与科学计算框架**，面向物理与工程问题提供从数值模拟到数据驱动建模的模型、工具与开发环境。

项目持续跟进并支持 Physics 与 Scientific Machine Learning 领域的主流及前沿技术，覆盖**物理信息神经网络、神经算子、复杂几何建模与代理建模**等方向，并支持 **Transolver、[GeoPT](https://github.com/MetaX-MACA/Physics/tree/main/Models/GeoPT)、[DoMINO](https://github.com/MetaX-MACA/Physics/tree/main/Models/PhysicsNeMo-ModelZoo/Domino)、[AeroGraphNet](https://github.com/MetaX-MACA/Physics/tree/main/Models/PhysicsNeMo-ModelZoo/AeroGraphNet)** 等代表性模型。

---

## 📂 项目结构与说明

本仓库以**聚合分发**方式提供多个相互独立的开源项目，并按照 **Models、Simulation、Frameworks** 三类进行组织。

各项目作为独立程序分别置于对应子目录中，我方仅将其聚合打包传输，并未将其组合为单一作品。各项目分别受其自带许可证约束，您在使用或分发任一项目前，须遵守该项目对应许可证的条款。

如各项目内另附有我方提供的 Patch 文件、适配代码或构建文件，系针对相应开源项目的修改或适配，相关许可证及说明放置于对应子目录中。

当前项目结构如下：

```text
Physics/
├── Models/
│   ├── DeepCFD/
│   ├── GeoPT/
│   ├── PaddleScience-ModelZoo/
│   └── PhysicsNeMo-ModelZoo/
│
├── Simulation/
│
├── Frameworks/
│   ├── DeepXDE/
│   ├── PaddleScience/
│   ├── PhysicsNeMo/
│   ├── NeuralOperator/
│   └── PINA/
│
└── README.md
```

---

## 🚀 能力概览

本项目围绕 **Models、Simulation、Frameworks** 三类能力展开：

- **[Models](https://github.com/MetaX-MACA/Physics/tree/main/Models)**：支持 [DeepCFD](https://github.com/MetaX-MACA/Physics/tree/main/Models/DeepCFD)、[GeoPT](https://github.com/MetaX-MACA/Physics/tree/main/Models/GeoPT)、Transolver、[DoMINO](https://github.com/MetaX-MACA/Physics/tree/main/Models/PhysicsNeMo-ModelZoo/Domino)、[AeroGraphNet](https://github.com/MetaX-MACA/Physics/tree/main/Models/PaddleScience-ModelZoo/DrivAerNet) 等代表性 Physics 模型，并通过 [PhysicsNeMo-ModelZoo](https://github.com/MetaX-MACA/Physics/tree/main/Models/PhysicsNeMo-ModelZoo)、[PaddleScience-ModelZoo](https://github.com/MetaX-MACA/Physics/tree/main/Models/PaddleScience-ModelZoo) 持续扩展模型与案例覆盖。
- **Simulation**：预留 GPU 加速物理数值模拟应用目录，当前仅包含分类说明。
- **[Frameworks](https://github.com/MetaX-MACA/Physics/tree/main/Frameworks)**：支持 [PhysicsNeMo](https://github.com/MetaX-MACA/Physics/tree/main/Frameworks/PhysicsNeMo)、[PaddleScience](https://github.com/MetaX-MACA/Physics/tree/main/Frameworks/PaddleScience)、[DeepXDE](https://github.com/MetaX-MACA/Physics/tree/main/Frameworks/DeepXDE)、[NeuralOperator](https://github.com/MetaX-MACA/Physics/tree/main/Frameworks/NeuralOperator) 等主流 Physics 与 Scientific Machine Learning 开发框架。

三类能力共同覆盖从高保真物理计算、数据生成，到 Physics 建模、训练、推理与验证的完整工作链。

> 更多 Physics 模型、物理仿真应用与开发框架正在持续增加中。

---

## 🔧 环境与依赖

- **硬件**：需具备沐曦（MetaX）GPU。
- **基础软件**：Linux 操作系统，沐曦 GPU 运行时环境。
- **主要依赖**：Python 3.8+、PyTorch、PaddlePaddle、PhysicsNeMo 等，以及各子项目特定依赖库。

具体安装与使用步骤，请参阅各子项目目录中的 `README.md`、`requirements.txt` 或 User Guide。

---

## ⚠️ 合规与许可证声明

在使用本仓库中的任何资源前，请仔细阅读并遵守以下声明：

1. **独立项目**
   本仓库中的每个子项目均为独立的开源项目，保留其原有许可证和版权声明。我方仅提供聚合分发服务，不对这些项目的功能、安全性或合规性做额外担保。

2. **使用责任**
   您有责任理解并遵守每个子项目自带的许可证条款。在使用或分发任何子项目时，请确保符合其许可证要求。

3. **修改与适配**
   如子项目内包含由我方提供的 Patch 文件、适配代码、构建文件或其他修改内容，相关许可证及声明将放置于对应目录中，请在使用时一并遵循。

---

## 📝 更新日志

- **2026-09-03**：项目结构调整为 `Models / Simulation / Frameworks`，新增 GeoPT 项目。
- **2026-08-17**：CFD-GCN、Aneurysm、DrivAerNet 模型归入 PaddleScience-ModelZoo。
- **2026-07-14**：初始版本，包含 DeepCFD、CFD-GCN、Aneurysm、DrivAerNet 等模型，并提供基础用户指南。

---

## 🤝 贡献与反馈

欢迎通过 GitHub Issues 或 Pull Requests 提出建议、报告问题，或贡献新的模型适配、仿真应用、框架支持、性能优化与使用文档。

欢迎共同推动 **Physics、Scientific Machine Learning 与 GPU 加速物理计算** 在沐曦 GPU 生态中的发展。
