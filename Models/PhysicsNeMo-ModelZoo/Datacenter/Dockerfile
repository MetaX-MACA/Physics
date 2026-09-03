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
ARG TE_ARCHIVE=maca-transformerengine-2.13-py312-3.7.1.0-linux-x86_64.tar.xz

FROM cr.metax-tech.com/public-library/maca-pytorch:${MACA_VERSION}-torch2.8-py312-${OS}-${ARCH}

ARG PHYSICSNEMO_COMMIT
ARG WARP_ARCHIVE
ARG TE_ARCHIVE
SHELL ["/bin/bash", "-c"]

ENV MACA_PATH=/opt/maca \
    CUCC_PATH=/opt/maca/tools/cu-bridge \
    CUDA_PATH=/opt/maca/tools/cu-bridge \
    PYTHONPATH=/root/physicsnemo \
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
    ca-certificates curl git build-essential libcurl4 libgl1 libx11-6 libxrender1 \
    perl vim wget xvfb xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch v2.0.0 --depth 1 https://github.com/NVIDIA/physicsnemo.git /workspace/physicsnemo && \
    cd /workspace/physicsnemo && \
    git checkout "${PHYSICSNEMO_COMMIT}" && \
    test "$(git rev-parse HEAD)" = "${PHYSICSNEMO_COMMIT}"

RUN pip install --upgrade pip hatchling && \
    pip install "onnx>=1.14.0" "pandas>=2.2.0" "nvtx>=0.2.10" "treelib>=1.2.5" \
    "numpy>=1.22.4" "tqdm>=4.60.0" "requests>=2.32.2" "GitPython>=3.1.40" "s3fs>=2023.5.0" \
    "packaging>=24.2" "timm>=1.0.22" "einops>=0.8.1" "h5py>=3.15.1" "cftime>=1.6.5" \
    "jaxtyping>=0.3.3" "termcolor>=3.2.0" "hydra-core>=1.3.2" "tensordict>=0.10.0" \
    "omegaconf>=2.3.0" "importlib-metadata>=8.7.1" "gdown>=5.0.0" "matplotlib>=3.8.0" \
    "pytest>=9.0.1" vtk && \
    cd /root/physicsnemo && pip install -e . --no-deps && \
    pip install --force-reinstall --no-deps tensordict==0.10.0 pydantic==2.12.3 pydantic-core==2.41.4

COPY ${WARP_ARCHIVE} /tmp/maca-warp.tar.xz
RUN mkdir -p /tmp/maca-warp && tar -xJf /tmp/maca-warp.tar.xz -C /tmp/maca-warp && \
    pip install --force-reinstall --no-deps /tmp/maca-warp/*/warp_lang-*.whl && \
    rm -rf /tmp/maca-warp /tmp/maca-warp.tar.xz

COPY ${TE_ARCHIVE} /tmp/maca-te.tar.xz
RUN mkdir -p /tmp/maca-te && tar -xJf /tmp/maca-te.tar.xz -C /tmp/maca-te && \
    pip install --force-reinstall --no-deps /tmp/maca-te/*/wheel/transformer_engine-*.torch2.8*.whl && \
    pip install --force-reinstall --no-deps /tmp/maca-te/*/wheel/grouped_gemm-*.torch2.8*.whl && \
    pip install typing_inspection annotated-types onnxscript nvdlfw-inspect && \
    rm -rf /tmp/maca-te /tmp/maca-te.tar.xz


WORKDIR /root/physicsnemo/examples/cfd/datacenter
CMD ["/bin/bash"]
