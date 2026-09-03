#The 3-Clause BSD License
#
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
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

LABEL org.opencontainers.image.title="PINA on MetaX MACA" \
      org.opencontainers.image.description="PINA scientific machine learning environment for MetaX GPUs" \
      org.opencontainers.image.source="https://github.com/mathLab/PINA"

ARG PINA_REPOSITORY=https://github.com/mathLab/PINA.git
ARG PINA_REF=60ea474a95e0aa8aa44212c094b8d92ee14b125f
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/workspace/PINA

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        python3-pip \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone "${PINA_REPOSITORY}" /workspace/PINA \
    && cd /workspace/PINA \
    && git checkout "${PINA_REF}"

# The base image supplies the MACA-patched PyTorch. Install the remaining
# runtime/tutorial dependencies explicitly, then install PINA without allowing
# pip to replace the vendor torch package.
RUN /opt/conda/bin/python3 -m pip install --upgrade pip -i "${PIP_INDEX_URL}" \
    && /opt/conda/bin/python3 -m pip install \
        "lightning==2.6.5" \
        matplotlib \
        numpy \
        scipy \
        smithers \
        tensorboard \
        "transformers==4.45.2" \
        torch-geometric \
        -i "${PIP_INDEX_URL}" \
    && cd /workspace/PINA \
    && /opt/conda/bin/python3 -m pip install --no-deps -e .

# tutorial5 is the representative CFD example documented in README.md. The
# upstream script defaults to CPU, so enable the Lightning GPU accelerator.
RUN sed -i 's/accelerator="cpu"/accelerator="gpu"/g' \
        /workspace/PINA/tutorials/tutorial5/tutorial.py \
    && mkdir -p /workspace/output

WORKDIR /workspace/PINA

CMD ["/bin/bash"]
