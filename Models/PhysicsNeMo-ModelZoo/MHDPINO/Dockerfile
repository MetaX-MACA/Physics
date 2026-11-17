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

# Install the matching v2.0.0 Sym add-on without replacing the base framework.
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation \
    "nvidia-physicsnemo.sym==2.0.0" && \
    cp -a /opt/conda/lib/python3.12/site-packages/physicsnemo/sym \
    /root/physicsnemo/physicsnemo/sym

RUN python -m pip install --no-cache-dir \
    "Cython>=0.29" \
    "chaospy>=4.3.7" \
    "mistune>=2.0,<2.1" \
    "numpoly<=1.3.4" \
    "numpy-stl>=2.16,<2.17" \
    "opencv-python>=4.8.1.78" \
    "pint>=0.19.2" \
    "symengine>=0.10.0" \
    "torch-optimizer>=0.3.0" \
    "transforms3d>=0.3.1"

RUN python -m pip install --no-cache-dir \
    "plotly==7.1.0" \
    "tensorly==0.10.0" \
    "tensorly-torch==0.5.0" \
    "mlflow==3.17.2" \
    "imageio"

ARG PYPI_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --no-cache-dir --index-url="$PYPI_INDEX_URL" \
    "vtk==9.7.2" "pyvista==0.49.2"

WORKDIR /root/physicsnemo/examples/cfd/mhd_pino

# Verify framework versions and this example entry at build time.
RUN python -c 'import physicsnemo, torch, warp; assert physicsnemo.__version__ == "2.0.0"; print(physicsnemo.__version__, torch.__version__, warp.__version__)' && \
    python -c 'import pathlib, runpy, sys; p=pathlib.Path("train_mhd.py"); sys.path.insert(0, str(p.parent.resolve())); runpy.run_path(str(p))'

CMD ["/bin/bash"]
