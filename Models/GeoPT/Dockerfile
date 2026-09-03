# BSD 3-Clause License
#
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# All rights reserved.
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
#

FROM cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.6-py310-ubuntu24.04-amd64

ARG GEOPT_REPOSITORY=https://github.com/Physics-Scaling/GeoPT.git
ARG GEOPT_COMMIT=9984f359fd44b6332c03ca55ab2e659e0c6e899d

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TORCHDYNAMO_DISABLE=1 \
    HF_ENDPOINT=https://hf-mirror.com

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# The build retrieves upstream source directly. It is not part of this release repository.
RUN git clone "${GEOPT_REPOSITORY}" /workspace/GeoPT \
    && git -C /workspace/GeoPT checkout --detach "${GEOPT_COMMIT}" \
    && git -C /workspace/GeoPT rev-parse HEAD

# Keep the MetaX-enabled torch package installed by the base image intact.
RUN sed '/^torch$/d' /workspace/GeoPT/requirements.txt > /tmp/geopt-requirements.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/geopt-requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && python -c "import torch; import h5py; import scipy; import vtk; import timm; import einops; import pyvista; print('torch:', torch.__version__)"

WORKDIR /workspace/GeoPT

CMD ["/bin/bash"]
