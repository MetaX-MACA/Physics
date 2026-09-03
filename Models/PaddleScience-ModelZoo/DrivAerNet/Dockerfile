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

LABEL description="PaddleScience DrivAerNet environment on Metax MACA GPU"

ARG PADDLESCIENCE_REPO=https://gitee.com/paddlepaddle/PaddleScience.git
ARG PADDLESCIENCE_REF=develop
ARG DRIVAERNET_DATA_URL=https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/data.tar
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 \
    PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/workspace/PaddleScience:/workspace/PaddleScience/examples/drivaernet

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates file git python3-pip wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone --branch "${PADDLESCIENCE_REF}" "${PADDLESCIENCE_REPO}" /workspace/PaddleScience
RUN cd /workspace/PaddleScience && git checkout 033132e061c50147c2841400f18717998077e28d

COPY patch/regdgcnn-empty-cache.patch /tmp/regdgcnn-empty-cache.patch

RUN cd /workspace/PaddleScience \
    && sed -i 's/\r$//' ppsci/arch/regdgcnn.py \
    && patch --binary -p1 < /tmp/regdgcnn-empty-cache.patch \
    && rm -f /tmp/regdgcnn-empty-cache.patch

RUN /opt/conda/bin/pip3 install --upgrade pip -i "${PIP_INDEX_URL}" \
    && /opt/conda/bin/pip3 install pgl==2.2.6 \
        -r /workspace/PaddleScience/requirements.txt -i "${PIP_INDEX_URL}" \
    && cd /workspace/PaddleScience \
    && /opt/conda/bin/pip3 install . -i "${PIP_INDEX_URL}"

WORKDIR /workspace/PaddleScience/examples/drivaernet
RUN wget -q "${DRIVAERNET_DATA_URL}" -O /tmp/data.tar \
    && tar -xf /tmp/data.tar -C /workspace/PaddleScience/examples/drivaernet \
    && rm -f /tmp/data.tar

RUN python3 -m py_compile drivaernet.py \
    && python3 -c "import paddle, pgl, ppsci; assert paddle.is_compiled_with_cuda(); print('DrivAerNet MACA image imports: ok')"