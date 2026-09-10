/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KV_COMPRESS_EPILOG_V2_TEST_COMMON_H
#define KV_COMPRESS_EPILOG_V2_TEST_COMMON_H

#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_kv_compress_epilog_v2.h"

namespace kcev2test {

constexpr int64_t kGroupSize16 = 16;
constexpr int64_t kGroupSize32 = 32;
constexpr int64_t kCacheRows = 8;
constexpr uint8_t kCacheFill = 0x5A;
constexpr uint16_t kBf16One = 0x3F80;  // BF16 位模式的 1.0
constexpr uint64_t kExpectWorkspaceSize = 32;

inline int64_t LayoutDataCol(int64_t d, int64_t mode)
{
    return mode == 2 ? d : d / 2;
}

// cache 行宽：RoundUp(dataCol + 2 * (d/groupSize), 32)
inline int64_t LayoutKvCacheCol(int64_t d, int64_t mode, int64_t groupSize)
{
    const int64_t concatCol = LayoutDataCol(d, mode) + 2 * (d / groupSize);
    return (concatCol + 31) / 32 * 32;
}

inline aclTensor *MakeRowMajorTensor(const std::vector<int64_t> &dims, aclDataType dtype, void *devAddr)
{
    std::vector<int64_t> strides(dims.size());
    int64_t elem = 1;
    for (int64_t i = static_cast<int64_t>(strides.size()) - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = elem;
        elem *= dims[static_cast<size_t>(i)];
    }
    return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, dims.data(),
                           dims.size(), devAddr);
}

struct CallResult {
    int32_t wsStatus = -1;      // GetWorkspaceSize 返回值
    int32_t execStatus = -1;    // aclnnKvCompressEpilogV2 返回值
    uint64_t workspaceSize = 0;
};

// 一次完整两段式调用；copyBack=true 时把 cache 拷回 hostCache
inline CallResult RunOp(const aclTensor *cache, const aclTensor *x, const aclTensor *slot, int64_t mode,
                        int64_t groupSize, bool roundScale, double xScale, void *cacheDev, size_t cacheBytes,
                        uint8_t *hostCache)
{
    CallResult result;
    aclOpExecutor *executor = nullptr;
    result.wsStatus = aclnnKvCompressEpilogV2GetWorkspaceSize(const_cast<aclTensor *>(cache), x, slot, groupSize,
                                                               mode, roundScale, xScale, &result.workspaceSize,
                                                               &executor);
    if (result.wsStatus != 0) {
        return result;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        result.execStatus = -1;
        return result;
    }
    const uint64_t wsSize = result.workspaceSize == 0 ? 1 : result.workspaceSize;
    void *workspace = nullptr;
    if (aclrtMalloc(&workspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        result.execStatus = -1;
        return result;
    }
    result.execStatus = aclnnKvCompressEpilogV2(workspace, wsSize, executor, stream);
    if (result.execStatus == 0) {
        (void)aclrtSynchronizeStream(stream);
        if (hostCache != nullptr) {
            (void)aclrtMemcpy(hostCache, cacheBytes, cacheDev, cacheBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    (void)aclrtFree(workspace);
    (void)aclrtDestroyStream(stream);
    return result;
}

// 校验命中行（dataCol/scale/padding）与未命中行保持
inline bool VerifyCache(const uint8_t *cache, int64_t cacheCol, int64_t d, int64_t mode, int64_t groupSize,
                        uint8_t dataByte, uint16_t scaleBits, const std::vector<int64_t> &hitRows,
                        const std::vector<int64_t> &keepRows)
{
    const int64_t dataCol = LayoutDataCol(d, mode);
    const int64_t scaleCol = d / groupSize;
    const int64_t concatCol = dataCol + 2 * scaleCol;
    const int64_t kvCacheCol = LayoutKvCacheCol(d, mode, groupSize);
    for (const int64_t hit : hitRows) {
        const uint8_t *row = cache + static_cast<size_t>(hit) * static_cast<size_t>(cacheCol);
        for (int64_t i = 0; i < dataCol; ++i) {
            if (row[i] != dataByte) {
                printf("  row %ld data byte %ld = 0x%02X (expect 0x%02X)\n", hit, static_cast<long>(i), row[i],
                       dataByte);
                return false;
            }
        }
        for (int64_t g = 0; g < scaleCol; ++g) {
            const uint16_t got = static_cast<uint16_t>(row[dataCol + 2 * g]) |
                                 static_cast<uint16_t>(row[dataCol + 2 * g + 1] << 8);
            if (got != scaleBits) {
                printf("  row %ld scale[%ld] = 0x%04X (expect 0x%04X)\n", hit, static_cast<long>(g), got,
                       scaleBits);
                return false;
            }
        }
        for (int64_t i = concatCol; i < kvCacheCol; ++i) {
            if (row[i] != 0) {
                printf("  row %ld padding %ld not zero\n", hit, static_cast<long>(i));
                return false;
            }
        }
    }
    for (const int64_t keep : keepRows) {
        const uint8_t *row = cache + static_cast<size_t>(keep) * static_cast<size_t>(cacheCol);
        for (int64_t i = 0; i < cacheCol; ++i) {
            if (row[i] != kCacheFill) {
                printf("  keep row %ld byte %ld modified\n", keep, static_cast<long>(i));
                return false;
            }
        }
    }
    return true;
}

}  // namespace kcev2test

#endif  // KV_COMPRESS_EPILOG_V2_TEST_COMMON_H
