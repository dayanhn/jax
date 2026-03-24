#!/bin/bash

clear
# 需要激活环境
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_debug

python -m pip install -e dist/jaxlib
python -m pip install -e dist/jax_ascend910_pjrt
python -m pip install -e dist/jax_ascend910_plugin
python -m pip install -e .