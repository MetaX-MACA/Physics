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


FROM cr.metax-tech.com/public-ai-release/physicsnemo-maca:2.0.0-3.9.0.12-torch2.8-py312-ubuntu24.04-amd64

ARG PYPI_INDEX_URL=https://pypi.org/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Keep the MetaX PyTorch and PhysicsNeMo provided by the base image.
RUN unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; \
    python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" \
    "gdown" "matplotlib" "symengine==0.14.1" \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation --index-url "${PYPI_INDEX_URL}" \
    "nvidia-physicsnemo.sym==2.0.0"

RUN unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; \
    python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" pint tensorboard transforms3d

RUN python -m pip uninstall -y flash-attn-3

# The base image loads PhysicsNeMo from its source tree; expose the installed Sym module there.
RUN ln -s /opt/conda/lib/python3.12/site-packages/physicsnemo/sym /root/physicsnemo/physicsnemo/sym

WORKDIR /root/physicsnemo/examples/cfd/darcy_physics_informed
RUN python -c "import physicsnemo, torch; from physicsnemo.sym.eq.pdes.diffusion import Diffusion; from physicsnemo.sym.eq.phy_informer import PhysicsInformer; import matplotlib, gdown, symengine; print(physicsnemo.__version__, torch.__version__)" \
    && python darcy_physics_informed_fno.py --help > /dev/null \
    && python darcy_physics_informed_deeponet.py --help > /dev/null
CMD ["/bin/bash"]
