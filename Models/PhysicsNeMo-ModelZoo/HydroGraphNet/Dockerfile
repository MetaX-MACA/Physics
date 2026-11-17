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
SHELL ["/bin/bash", "-c"]
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG PYPI_INDEX_URL=https://pypi.org/simple
RUN python -m pip uninstall -y flash-attn-3
RUN python -m pip install --index-url=${PYPI_INDEX_URL} -r /root/physicsnemo/examples/weather/flood_modeling/hydrographnet/requirements.txt
WORKDIR /root/physicsnemo/examples/weather/flood_modeling/hydrographnet
RUN python -c "import physicsnemo, torch, warp, hydra, wandb, mlflow, torch_geometric, torch_scatter; from physicsnemo.models.meshgraphnet import MeshGraphKAN; assert physicsnemo.__version__ == '2.0.0'; print(torch.__version__, warp.__version__, mlflow.__version__)"
RUN python train.py --help
CMD ["/bin/bash"]
