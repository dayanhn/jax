#!/bin/bash

# 需要激活环境
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_debug

# 设置 Ascend 环境
echo "Setting up Ascend environment..."
source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh

echo "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"


# 编译
echo "Building jax-ascend-pjrt..."
python build/build.py build \
  --wheels=jax-ascend-pjrt \
  --editable \
  --bazel_options=--compilation_mode=dbg \
  --bazel_options=--copt=-g \
  --bazel_options=--copt=-O0 \
  --bazel_options=--strip=never \
  --bazel_options=--override_repository=xla=$(pwd)/xla \
  --local_xla_path=$(pwd)/xla 

echo "Build completed!"
