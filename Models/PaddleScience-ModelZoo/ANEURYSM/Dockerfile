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


FROM cr.metax-tech.com/public-ai-release/maca/paddle:2.6.0-maca.ai3.8.0.10-py310-ubuntu22.04-amd64

LABEL description="PaddleScience Aneurysm PINN environment on Metax MACA GPU"

ARG PADDLESCIENCE_REPO=https://gitee.com/paddlepaddle/PaddleScience.git
ARG PADDLESCIENCE_REF=develop
ARG ANEURYSM_DATA_URL=https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/workspace/PaddleScience:/workspace/PaddleScience/examples/aneurysm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        file \
        git \
        libboost-dev \
        libboost-thread-dev \
        libeigen3-dev \
        libffi-dev \
        libgmp-dev \
        libgmpxx4ldbl \
        libmpfr-dev \
        libopenblas-dev \
        libtbb-dev \
        python3-dev \
        python3-pip \
        wget \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git config --global http.version HTTP/1.1 \
    && git clone --branch "${PADDLESCIENCE_REF}" \
        "${PADDLESCIENCE_REPO}" /workspace/PaddleScience
RUN cd /workspace/PaddleScience && git checkout 033132e061c50147c2841400f18717998077e28d

COPY patch/install_mesh.patch /tmp/install_mesh.patch

RUN cd /workspace/PaddleScience \
    && patch --binary -p1 < /tmp/install_mesh.patch \
    && rm -f /tmp/install_mesh.patch \
    && bash -n /workspace/PaddleScience/install_mesh.sh

# Paddle is supplied by the Metax base image. Install PaddleScience runtime
# dependencies before installing the package itself.
RUN /opt/conda/bin/pip3 install --upgrade pip \
    && /opt/conda/bin/pip3 install \
        -r /workspace/PaddleScience/requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && cd /workspace/PaddleScience \
    && /opt/conda/bin/pip3 install . \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# Mesh is required by ppsci.geometry.Mesh. The repository script builds the
# PyMesh extension and installs open3d, pysdf, and pybind11.
RUN cd /workspace/PaddleScience \
    && bash install_mesh.sh

WORKDIR /workspace/PaddleScience/examples/aneurysm

RUN wget -q "${ANEURYSM_DATA_URL}" -O /tmp/aneurysm_dataset.tar \
    && tar -xf /tmp/aneurysm_dataset.tar -C /workspace/PaddleScience/examples/aneurysm \
    && rm -f /tmp/aneurysm_dataset.tar

RUN python3 -m py_compile aneurysm.py aneurysm_flow.py \
    && python3 -c "import open3d; import pysdf; import pymesh; import ppsci; ppsci.run_check_mesh(); print('Aneurysm MACA image imports: ok')"