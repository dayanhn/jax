#!/bin/bash

clear
# 需要激活环境
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_uniai

# 找到 nanobind 的安装位置
# 通过 which python 获取 Python 可执行文件路径，然后动态获取 nanobind 目录
PYTHON_PATH=$(which python)
if [ -z "$PYTHON_PATH" ]; then
    echo "Error: python not found in PATH"
    exit 1
fi

# 使用 Python 获取 nanobind 的安装路径
nanobind_DIR=$($PYTHON_PATH -c "import nanobind; import os; print(os.path.dirname(nanobind.__file__))")

if [ -z "$nanobind_DIR" ]; then
    echo "Error: nanobind not found. Please install it with: pip install nanobind"
    exit 1
fi

export nanobind_DIR

# 创建构建目录
rm -rf build
mkdir build && cd build

# 或同时启用 CUDA
cmake .. -DCMAKE_BUILD_TYPE=Debug -DJAX_FFI_EXAMPLE_ENABLE_CUDA=ON

# 构建项目
cmake --build .

cd ..
cp build/lib_cuda_examples.so  src/jax_ffi_example/

pip install -e . --no-deps