/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_expand.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// 创建标量Tensor（shape为{}）
template <typename T>
int CreateScalarTensor(
    T scalarValue, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    // 标量的shape为空，size为sizeof(T)
    auto size = sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将标量值拷贝到device侧
    ret = aclrtMemcpy(*deviceAddr, size, &scalarValue, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 标量的shape和strides都为空
    *tensor = aclCreateTensor(
        nullptr, 0, dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, nullptr, 0,
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 验证broadcast算子：f32[] -> f32[784,512]{1,0}, dimensions={}
    // 输入是标量 f32[]，输出是 [784, 512]
    std::vector<int64_t> selfShape = {};  // 标量shape为空
    std::vector<int64_t> outShape = {784, 512};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* size = nullptr;
    aclTensor* out = nullptr;
    
    // 创建标量输入数据，例如值为 2.5
    float scalarValue = 2.5f;
    
    // 创建输出数据 [784, 512]，初始化为0
    std::vector<float> outHostData(784 * 512, 0.0f);
    
    // 设置broadcast的目标shape [784, 512]
    int64_t sizeValue[2] = {784, 512};
    size = aclCreateIntArray(&(sizeValue[0]), 2);
    
    // 创建self标量aclTensor
    ret = CreateScalarTensor(scalarValue, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API - aclnnExpand实现broadcast功能
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnExpand第一段接口获取workspace大小
    ret = aclnnExpandGetWorkspaceSize(self, size, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpandGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    
    // 调用aclnnExpand第二段接口执行broadcast操作
    ret = aclnnExpand(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpand failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto resultSize = GetShapeSize(outShape);
    std::vector<float> resultData(resultSize, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, resultSize * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    
    // 打印验证结果：每行显示部分元素
    LOG_PRINT("\nBroadcast result verification:\n");
    LOG_PRINT("Input shape: [], Output shape: [784, 512]\n");
    LOG_PRINT("Input scalar value: %f\n", scalarValue);
    LOG_PRINT("Expected: All elements should be %f\n\n", scalarValue);
    
    // 打印前3行和后3行作为示例
    for (int64_t row = 0; row < 784; row++) {
        if (row < 3 || row >= 781) {
            LOG_PRINT("Row %4ld: ", row);
            for (int64_t col = 0; col < 5; col++) {  // 只打印前5列
                int64_t idx = row * 512 + col;
                LOG_PRINT("%8.2f", resultData[idx]);
                if (col < 4) LOG_PRINT(", ");
            }
            LOG_PRINT(" ...\n");
        } else if (row == 3) {
            LOG_PRINT("...\n");
        }
    }
    
    // 验证broadcast正确性：检查所有元素是否等于标量值
    bool verifyPassed = true;
    int64_t errorCount = 0;
    for (int64_t i = 0; i < resultSize && errorCount < 10; i++) {
        if (std::abs(resultData[i] - scalarValue) > 1e-5) {
            if (errorCount == 0) {
                LOG_PRINT("Verification errors found:\n");
            }
            int64_t row = i / 512;
            int64_t col = i % 512;
            LOG_PRINT("  Error at [%ld, %ld]: expected %f, got %f\n", 
                     row, col, scalarValue, resultData[i]);
            errorCount++;
            verifyPassed = false;
        }
    }
    
    if (verifyPassed) {
        LOG_PRINT("\n✓ Broadcast verification PASSED! All %ld elements equal to %f\n", 
                 resultSize, scalarValue);
    } else {
        LOG_PRINT("\n✗ Broadcast verification FAILED!\n");
    }

    // 6. 释放aclTensor和aclScalar
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclDestroyIntArray(size);

    // 7. 释放device资源
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}