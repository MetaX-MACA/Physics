# The 3-Clause BSD License
#
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ARG MACA_VERSION=3.8.1.2
ARG OS=ubuntu24.04
ARG ARCH=amd64
ARG PHYSICSNEMO_COMMIT=1ca85d65ac2ce28ea9762910c09a954c08a37140
ARG WARP_ARCHIVE=maca-warp-1.8.1-py310-3.7.0.6-linux-amd64.tar.xz

FROM cr.metax-tech.com/public-library/maca-pytorch:${MACA_VERSION}-torch2.8-py312-${OS}-${ARCH}

ARG PHYSICSNEMO_COMMIT
ARG WARP_ARCHIVE

SHELL ["/bin/bash", "-c"]

ENV MACA_PATH=/opt/maca \
    CUCC_PATH=/opt/maca/tools/cu-bridge \
    CUDA_PATH=/opt/maca/tools/cu-bridge \
    PYTHONPATH=/workspace/physicsnemo \
    PYTORCH_DEFAULT_NCHW=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:16 \
    PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
    DISABLE_PYTORCH_USE_FLASHATTN=1 \
    PYTORCH_ENABLE_FA_CPP_INTERFACE=1 \
    MCDNN_USE_DETERMINISTIC_ALGO=1 \
    DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=${MACA_PATH}/mxgpu_llvm/lib:${CUCC_PATH}/lib:${CUCC_PATH}/lib64:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH} \
    PATH=${CUCC_PATH}/tools:${CUDA_PATH}/bin:/opt/conda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential libgl1 libx11-6 libxrender1 \
    openmpi-bin libopenmpi-dev xz-utils \
    && rm -rf /var/lib/apt/lists/*

# The user must download this MetaX Warp archive into the Docker build context.
COPY ${WARP_ARCHIVE} /tmp/maca-warp.tar.xz
RUN mkdir -p /tmp/maca-warp && \
    tar -xJf /tmp/maca-warp.tar.xz -C /tmp/maca-warp && \
    pip install --no-deps /tmp/maca-warp/*/warp_lang-*.whl && \
    rm -rf /tmp/maca-warp /tmp/maca-warp.tar.xz

RUN git clone --branch v2.0.0 --depth 1 https://github.com/NVIDIA/physicsnemo.git /workspace/physicsnemo && \
    cd /workspace/physicsnemo && \
    git checkout "${PHYSICSNEMO_COMMIT}" && \
    test "$(git rev-parse HEAD)" = "${PHYSICSNEMO_COMMIT}"

# Install framework dependencies without replacing the MetaX PyTorch in the base image.
RUN pip install --no-cache-dir \
    onnx pandas nvtx treelib numpy tqdm requests GitPython s3fs packaging timm einops \
    h5py cftime jaxtyping termcolor hydra-core tensordict==0.10.0 omegaconf \
    importlib-metadata gdown matplotlib pytest pydantic==2.12.3 pydantic-core==2.41.4 && \
    cd /workspace/physicsnemo && pip install --no-deps -e . && \
    pip install --no-cache-dir -r examples/cfd/darcy_nested_fnos/requirements.txt

RUN python -c "import physicsnemo, torch, warp; print(physicsnemo.__version__); print(torch.__version__); print(warp.__version__)"

# Adapter smoke test: validate the FNO model without data generation or training.
RUN python - <<'PY'
import torch
from physicsnemo.models.fno import FNO

assert torch.cuda.is_available()
model = FNO(
    in_channels=1,
    out_channels=1,
    decoder_layers=1,
    decoder_layer_size=8,
    dimension=2,
    latent_channels=8,
    num_fno_layers=2,
    num_fno_modes=4,
).cuda()
with torch.no_grad():
    output = model(torch.randn(1, 1, 8, 8, device="cuda"))
assert output.shape == (1, 1, 8, 8)
print("FNO adapter smoke test passed")
PY

WORKDIR /workspace/physicsnemo/examples/cfd/darcy_nested_fnos
CMD ["/bin/bash"]
