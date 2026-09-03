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
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FROM cr.metax-tech.com/public-ai-release/maca/paddle:2.6.0-maca.ai3.8.0.10-py310-ubuntu22.04-amd64

LABEL description="PaddleScience CFDGCN environment on Metax MACA GPU"

ARG OPENMPI_VERSION=1.10.2
ARG OPENMPI_URL=https://download.open-mpi.org/release/open-mpi/v1.10/openmpi-${OPENMPI_VERSION}.tar.gz
ARG CFDGCN_DATA_URL=https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/data.zip
ARG CFDGCN_MESH_URL=https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/meshes.tar
ARG CFDGCN_SU2_URL=https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/SU2Bin.tgz

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MPI_HOME=/opt/openmpi \
    SU2_RUN=/workspace/PaddleScience/examples/cfdgcn/SU2Bin \
    SU2_HOME=/workspace/PaddleScience/examples/cfdgcn/SU2Bin \
    PATH=/opt/openmpi/bin:/workspace/PaddleScience/examples/cfdgcn/SU2Bin:/opt/conda/bin:$PATH \
    LD_LIBRARY_PATH=/opt/openmpi/lib:/workspace/PaddleScience/examples/cfdgcn/SU2Bin \
    PYTHONPATH=/workspace/PaddleScience:/workspace/PaddleScience/examples/cfdgcn:/workspace/PaddleScience/examples/cfdgcn/SU2Bin

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        file \
        libffi-dev \
        libopenblas-dev \
        python3-pip \
        wget \
        zlib1g-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Keep the source installation flow from the existing Metax image setup.
RUN git clone https://gitee.com/paddlepaddle/PaddleScience.git /workspace/PaddleScience
RUN cd /workspace/PaddleScience && git checkout 033132e061c50147c2841400f18717998077e28d

# SU2Bin is linked against the OpenMPI 1.x ABI. Build a private MPI with C++
# bindings enabled and without RDMA components unavailable in this image.
RUN mkdir -p /tmp/openmpi-src \
    && wget -q "${OPENMPI_URL}" -O /tmp/openmpi.tar.gz \
    && tar -xzf /tmp/openmpi.tar.gz -C /tmp/openmpi-src --strip-components=1 \
    && cd /tmp/openmpi-src \
    && ./configure --prefix="${MPI_HOME}" \
        --disable-fortran \
        --disable-mpi-java \
        --disable-dlopen \
        --disable-dependency-tracking \
        --without-verbs \
        --disable-openib \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/openmpi-src /tmp/openmpi.tar.gz

# Paddle is supplied by the Metax base image. Install the remaining
# PaddleScience dependencies and PGL from the configured Python environment.
RUN /opt/conda/bin/pip3 install --upgrade pip \
    && /opt/conda/bin/pip3 install \
        pgl==2.2.6 \
        -r /workspace/PaddleScience/requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# mpi4py 3.1.4 must be compiled against the private OpenMPI installation.
RUN MPICC="${MPI_HOME}/bin/mpicc" /opt/conda/bin/pip3 install \
        --no-build-isolation mpi4py==3.1.4 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN cd /workspace/PaddleScience \
    && /opt/conda/bin/pip3 install . \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/PaddleScience/examples/cfdgcn

# Download the official CFDGCN data, meshes, and SU2 v6.2.0 binaries.
RUN wget -q "${CFDGCN_DATA_URL}" -O /tmp/data.zip \
    && python3 -m zipfile -e /tmp/data.zip /workspace/PaddleScience/examples/cfdgcn \
    && wget -q "${CFDGCN_MESH_URL}" -O /tmp/meshes.tar \
    && tar -xf /tmp/meshes.tar -C /workspace/PaddleScience/examples/cfdgcn \
    && wget -q "${CFDGCN_SU2_URL}" -O /tmp/SU2Bin.tgz \
    && tar -xzf /tmp/SU2Bin.tgz -C /workspace/PaddleScience/examples/cfdgcn \
    && rm -f /tmp/data.zip /tmp/meshes.tar /tmp/SU2Bin.tgz

RUN python3 -m py_compile cfdgcn.py utils.py su2paddle/*.py \
    && python3 -c "from mpi4py import MPI; import SU2; import pysu2; import pgl; import ppsci; assert MPI.COMM_WORLD.Get_size() == 1; print('CFDGCN MACA image imports: ok')"