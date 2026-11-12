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

FROM harbor.sh.mxcr.io/cimaster/maca-pytorch:20260902-820-torch2.6-py310-ubuntu24.04-amd64

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HYDROGYM_JAX_ROOT=/opt/hydrogym \
    PATH=/opt/venv/bin:$PATH

WORKDIR /workspace

COPY ./requirements-build.txt /tmp/hydrogym-requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/venv/bin:$PATH

RUN python3.12 -m venv /opt/venv \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        -r /tmp/hydrogym-requirements.txt \
    && python -m pip check

RUN cd /opt && bash -c "git clone https://github.com/dynamicslab/hydrogym.git "

RUN echo /opt/hydrogym > /opt/venv/lib/python3.12/site-packages/hydrogym.pth \
    && python - <<'PY'
import hydrogym

print("hydrogym", hydrogym.__file__)
assert hydrogym.__file__.startswith("/opt/hydrogym")
PY

CMD ["/bin/bash", "-l"]
