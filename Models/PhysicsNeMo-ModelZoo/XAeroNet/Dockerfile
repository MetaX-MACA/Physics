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

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=disabled

# Reuse MetaX PyTorch, PhysicsNeMo and Warp already present in the PhysicsNeMo MACA image.
# The bundled flash-attn-3 package lacks flash_attn_3_interface.
# Remove it so the MACA Transformer Engine can be imported.
RUN python -m pip uninstall -y flash-attn-3

# The PhysicsNeMo MACA image carries the upstream PhysicsNeMo v2.0.0 source tree.
RUN python -c "import physicsnemo; version=getattr(physicsnemo, '__version__', ''); print('PhysicsNeMo', version); assert version == '2.0.0'"

RUN python -m pip install --no-cache-dir \
    "trimesh==5.2.0"

ARG PYPI_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --no-cache-dir --index-url="$PYPI_INDEX_URL" \
    "vtk==9.7.2" "pyvista==0.49.2"

# The upstream train.py imports TensorBoard even when displaying --help.
RUN python -m pip install --no-cache-dir "tensorboard==2.21.0"

WORKDIR /root/physicsnemo/examples/cfd/external_aerodynamics/xaeronet

# Verify framework versions and this example entry at build time.
RUN python -c 'import physicsnemo, torch, warp; assert physicsnemo.__version__ == "2.0.0"; print(physicsnemo.__version__, torch.__version__, warp.__version__)' && \
    python -c 'import pathlib, runpy, sys; p=pathlib.Path("surface/train.py"); sys.path.insert(0, str(p.parent.resolve())); runpy.run_path(str(p))'

CMD ["/bin/bash"]
