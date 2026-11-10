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

ARG MACA_BASE_IMAGE=cr.metax-tech.com/public-library/maca-pytorch:3.8.2.6-torch2.8-py312-ubuntu24.04-amd64
FROM ${MACA_BASE_IMAGE}

ARG TRANSOLVER_ARCHIVE=https://codeload.github.com/thuml/Transolver_plus/tar.gz
ARG TRANSOLVER_COMMIT=d5a23bc734a0ebac56384cf72049a26af9673452

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:${PATH} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN mkdir -p /workspace/Transolver_plus \
    && curl --retry 3 -fL "${TRANSOLVER_ARCHIVE}/${TRANSOLVER_COMMIT}" -o /tmp/transolver-plus.tar.gz \
    && tar -xzf /tmp/transolver-plus.tar.gz --strip-components=1 -C /workspace/Transolver_plus \
    && rm /tmp/transolver-plus.tar.gz \
    && test -f /workspace/Transolver_plus/main_airplane.py \
    && test -f /workspace/Transolver_plus/LICENSE

# Install model and visualization dependencies while preserving MACA PyTorch.
RUN torch_version="$(python -c 'import torch; print(torch.__version__)')" \
    && python -m pip install \
        'numpy<2' \
        h5py==3.16.0 \
        timm==1.0.30 \
        einops==0.8.1 \
        matplotlib \
        pandas \
    && test "$(python -c 'import torch; print(torch.__version__)')" = "${torch_version}" \
    && python -c "import torch; print('Preserved PyTorch:', torch.__version__)"

COPY patch/add_maca_helpers.patch /tmp/add_maca_helpers.patch
RUN cd /workspace/Transolver_plus \
    && mkdir -p tools \
    && patch --binary -p1 < /tmp/add_maca_helpers.patch \
    && rm /tmp/add_maca_helpers.patch \
    && chmod +x tools/*.py \
    && python -c "import h5py, timm, einops, matplotlib, pandas; from models.Transolver_plus import Model; print('Transolver++ tools and imports OK')"

WORKDIR /workspace/Transolver_plus
CMD ["/bin/bash"]
