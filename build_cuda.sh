#!/bin/bash

clear
# 需要激活环境
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_uniai

export http_proxy=http://120.77.201.177:7897
export https_proxy=http://120.77.201.177:7897


# 编译 - ASCEND_TOOLKIT_HOME 已通过 source set_env.sh 设置
# build.py 会自动检测环境变量并注入 --action_env=ASCEND_TOOLKIT_HOME

python build/build.py build \
  --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt \
  --editable \
  --bazel_options=--compilation_mode=dbg \
  --bazel_options=--copt=-g \
  --bazel_options=--copt=-O0 \
  --bazel_options=--strip=never \
  --bazel_options=--override_repository=xla=$(pwd)/xla \
  --local_xla_path=$(pwd)/xla

echo "Build completed!"
