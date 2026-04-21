# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.14)
set(CMAKE_BUILD_TYPE Debug)


# 设置工程名
project(ACLNN_EXAMPLE)

# Compile options
add_compile_options(-std=c++11)

# 设置编译选项
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC  -g -Wall")


# 设置可执行文件名（如opapi_test），并指定待运行算子文件*.cpp所在目录
add_executable(opapi_test
                test_aclnn_matmul.cpp)

# 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()
set(ASCEND_PATH "/data3/zhongzhw/Ascend8.5REL/cann-8.5.0")
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/aarch64-linux/include")
include_directories(
    ${INCLUDE_BASE_DIR}
)

# 如果手动编译并安装过ops-nn中的mat_mul_v3算子，在执行测试时，需要修改自定义算子库的路径，让opbase找不到自定义算子
# 从而使用标准算子库中的算子，否则会使用修改过的算子:
# mv  ${ASCEND_HOME_PATH}/opp/vendors  ${ASCEND_HOME_PATH}/opp/vendors_bak
target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/aarch64-linux/lib64/libascendcl.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_nn.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so)
