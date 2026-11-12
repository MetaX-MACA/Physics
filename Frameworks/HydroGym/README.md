# HydroGym JAX 后端沐曦 C500 GPU 运行说明

## 1. HydroGym 简介

HydroGym 是一个用于流体动力学主动流动控制的强化学习平台。它提供兼容 Gymnasium 的统一接口，支持多个预配置 CFD 环境，可使用 Stable-Baselines3、RLlib、PettingZoo 或自定义训练循环。

HydroGym 的 JAX 后端将 CFD 状态、观测、动作、策略网络和 PPO 更新都表达为 JAX 计算。只要 `jax.default_backend()` 为 `gpu`，并且关键数组位于 GPU，则 CFD 推进和 RL 训练均在沐曦 C500 GPU 上执行。

## 2. 构建基础镜像

在 `/root/hydroGym` 目录执行：

```bash
docker build \
  -f docker-build/Dockerfile \
  -t hydrogym-jax-base:0.9.1 \
  .
```

基础镜像不预装 MACA JAX。镜像内包含：

```text
/opt/venv       Python 虚拟环境
/opt/hydrogym   HydroGym 源码
```

## 3. 运行容器

```bash
docker run -it --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  hydrogym-jax-base:0.9.1 \
  /bin/bash -l
```

参数说明：

- `--device=/dev/mxcd` 和 `--device=/dev/dri`：挂载沐曦 GPU 设备。
- `--group-add video`：允许容器进程访问 GPU 设备。

如果宿主机设备名不同，请使用实际环境中的 MACA 设备注入方式。

## 4. 安装 MACA JAX 0.9.1

从沐曦开发者社区下载 MACA JAX 0.9.1 安装包，例如：

```text
maca-jax-0.9.1-py312-3.9.1.0-x86_64.tar.gz
```

解压后进入 wheel 目录，安装四个 wheel：

```bash
python -m pip install --no-cache-dir \
  --no-index --no-deps \
  --find-links=. \
  jax==0.9.1 \
  jaxlib==0.9.1 \
  jax-maca-pjrt==0.9.1 \
  jax-maca-plugin==0.9.1
```

安装后检查依赖：

```bash
python -m pip check
```

如缺少依赖，请按 `pip check` 输出补装。常见依赖包括：

```bash
python -m pip install \
  --proxy=http://proxy.sha-dc01.metax-tech.com:1080 \
  ml_dtypes==0.6.0 \
  numpy==2.5.4 \
  opt_einsum==3.4.0 \
  scipy==1.18.1
```

## 5. 应用 JAX 0.9.1 兼容补丁

HydroGym JAX PPO 依赖：

```text
distrax==0.1.5
tensorflow-probability==0.25.0
```

这两个包在 JAX 0.9.1 下使用了已移除的 `jax.interpreters.xla` 内部 API，需要进行修改：


### 5.1 Distrax 修改内容

目标文件：

```text
/opt/venv/lib/python3.12/site-packages/distrax/_src/utils/jittable.py
```

找到：

```python
  # Otherwise, try to make it into a tracer. If it succeeds, then it's JAX data.
  try:
    jax.interpreters.xla.abstractify(x)
    return True
  except TypeError:
    return False
```

替换为：

```python
  # Otherwise, use JAX's public Array type and tracer shape as a fallback.
  if isinstance(x, jax.Array) or hasattr(x, "shape"):
    return True
  return False
```

### 5.2 TensorFlow Probability 修改内容

目标文件：

```text
/opt/venv/lib/python3.12/site-packages/tensorflow_probability/python/internal/backend/jax/ops.py
```

找到：

```python
  jax.interpreters.xla.canonicalize_dtype_handlers[NumpyVariable] = (
      jax.interpreters.xla.canonicalize_dtype_handlers[onp.ndarray])
  jax.interpreters.xla.pytype_aval_mappings[NumpyVariable] = (
      jax.interpreters.xla.pytype_aval_mappings[onp.ndarray])
  jax.core.pytype_aval_mappings[NumpyVariable] = (
      jax.core.pytype_aval_mappings[onp.ndarray])
```

替换为：

```python
  jax.core.canonicalize_dtype_handlers[NumpyVariable] = (
      jax.core.canonicalize_dtype_handlers[onp.ndarray])
  jax.core.pytype_aval_mappings[NumpyVariable] = (
      jax.core.pytype_aval_mappings[onp.ndarray])
```

## 6. 验证 JAX GPU

```bash
python - <<'PY'
import jax

print("jax:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())

assert jax.default_backend() == "gpu"
assert "cuda" in str(jax.devices()[0]).lower()
PY
```

预期输出示例：

```text
jax: 0.9.1
backend: gpu
devices: [CudaDevice(id=0)]
```

不要设置 `JAX_PLATFORM_NAME=cpu`，否则 JAX 会强制运行在 CPU。

## 7. 运行 HydroGym JAX PPO 示例

```bash
cd /opt/hydrogym/hydrogym/examples/jax/getting_started/3_ppo

MPLBACKEND=Agg python -u run_ppo.py \
  --env kolmogorov \
  --total-timesteps 1 \
  --num-envs 1 \
  --num-steps 1 \
  --num-minibatches 1 \
  --model-save-path /tmp/ppo-maca-model.pkl \
  --plot-path /tmp/ppo-maca-reward.png
```

参数说明：

- `--total-timesteps`：总交互步数。
- `--num-envs`：并行环境数量。
- `--num-steps`：每次更新前每个环境采集的步数。
- `--num-minibatches`：PPO 更新时 batch 划分的 minibatch 数。
- `MPLBACKEND=Agg`：避免无显示环境使用交互绘图后端。

## 8. 确认 CFD 和 RL 都在 GPU

```bash
python - <<'PY'
import jax
import jax.numpy as jnp

x = jnp.ones((4, 4))
y = jax.jit(lambda value: value * 2.0)(x)

print("backend:", jax.default_backend())
print("x platform:", x.platform())
print("y platform:", y.platform())
print("x device:", x.device())
print("y device:", y.device())
PY
```

预期 `platform` 为 `gpu`，设备为 `cuda:0`。如果观测、动作、参数或梯度数组的 platform 均为 `gpu`，则 CFD 推进和 PPO 训练没有退回 CPU。


---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且我方不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

