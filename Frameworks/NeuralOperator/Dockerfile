##The 3-Clause BSD License
#
## Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


FROM cr.metax-tech.com/public-library/maca-pytorch:3.7.2.1-torch2.4-py310-ubuntu22.04-amd64

LABEL description="NeuralOperator FNO environment on MetaX MACA GPU"

ARG NEURALOPERATOR_REPO=https://github.com/NeuralOperator/neuraloperator.git
ARG NEURALOPERATOR_REF=00b7d86f8d74ff0af55da53eb585fe26df9c71f0
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/workspace/neuraloperator

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone "${NEURALOPERATOR_REPO}" /workspace/neuraloperator \
    && cd /workspace/neuraloperator \
    && git checkout "${NEURALOPERATOR_REF}"

# Keep the MACA PyTorch supplied by the base image. Install the core runtime
# dependencies explicitly, then install NeuralOperator without resolving torch.
RUN /opt/conda/bin/pip3 install --upgrade pip \
    && /opt/conda/bin/pip3 install \
        "numpy>=1.25" \
        configmypy \
        h5py \
        matplotlib \
        opt-einsum \
        ruamel.yaml \
        tensorly \
        tensorly-torch \
        wandb \
        zarr \
        zencfg \
        -i "${PIP_INDEX_URL}" \
    && cd /workspace/neuraloperator \
    && /opt/conda/bin/pip3 install --no-deps -e .

# Validate imports and a small FNO forward pass during image construction.
# GPU device access is checked at container runtime because docker build does
# not normally expose /dev/mxcd or /dev/dri.
RUN /opt/conda/bin/python3 -c "import torch, neuralop; from neuralop.models import FNO; model = FNO(n_modes=(4, 4), hidden_channels=8, in_channels=1, out_channels=1); x = torch.randn(1, 1, 16, 16); y = model(x); assert y.shape == x.shape, (x.shape, y.shape); print('torch:', torch.__version__); print('neuralop:', neuralop.__version__); print('FNO CPU smoke test: ok')"

RUN mkdir -p /workspace/output

WORKDIR /workspace/neuraloperator

CMD ["/bin/bash"]
