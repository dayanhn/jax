#!/bin/bash
clear
source  ~/Ascend8.5REL/cann-8.5.0/set_env.sh

#如果手动编译并安装过ops-nn中的mat_mul_v3算子，则这里也会使用修改过的算子而不是标准算子
rm -rf build
mkdir build && cd build
cmake ..
make

# 如果手动编译并安装过ops-nn中的mat_mul_v3算子，在执行测试时，需要修改自定义算子库的路径，让opbase找不到自定义算子
# 从而使用标准算子库中的算子，否则会使用修改过的算子:
# mv  ${ASCEND_HOME_PATH}/opp/vendors  ${ASCEND_HOME_PATH}/opp/vendors_bak
./bin/opapi_test
