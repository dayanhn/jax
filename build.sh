#!/bin/bash

#conda init
#conda activate jax_debug

source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh

#python build/build.py build --wheels=jaxlib,jax  --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0  --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla

python build/build.py build --wheels=jax-ascend-pjrt  --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0  --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla