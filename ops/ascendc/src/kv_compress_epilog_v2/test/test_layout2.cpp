/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// layout2（4d 分页布局）最小真值用例：aclnn 两段式调用并逐字节校验 cache 写回。
// 期望值可手工推导：mode2 data=0x78/scale=0x3B80；mode4 data=0x66/scale=0x3E80。

#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_kv_compress_epilog_v2.h"

namespace {
// 与算子 IR 注册一致的 quant_mode 取值。
constexpr int64_t MODE_MXFP8 = 2;
constexpr int64_t MODE_MXFP4 = 4;
constexpr int64_t GROUP_SIZE_32 = 32;
constexpr int64_t GROUP_SIZE_16 = 16;
constexpr int64_t CACHE_FILL = 0x5A;

struct CaseGeometry {
    int64_t blocks;
    int64_t blockSize;
    int64_t tokens;
    int64_t d;
    int64_t physicalBlocks;
    int64_t dataCol;
    int64_t scaleCol;
    int64_t tokenStride;
    int64_t cacheCol;
    int64_t blockStride;
    size_t cacheBytes;
    size_t xBytes;
};

aclTensor *MakeTensor(const std::vector<int64_t> &dims, const std::vector<int64_t> &strides,
                      aclDataType dtype, uintptr_t addr)
{
    return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                           dims.data(), dims.size(), reinterpret_cast<void *>(addr));
}

CaseGeometry MakeGeometry(int64_t mode, int64_t group)
{
    constexpr int64_t blocks = 2;
    constexpr int64_t blockSize = 4;
    constexpr int64_t tokens = 4;
    constexpr int64_t d = 512;
    constexpr int64_t physicalBlocks = 4;
    // r8 interleaved layout: token bundle = [dataCol bytes][2*scaleCol bytes].
    const int64_t dataCol = mode == MODE_MXFP8 ? d : d / 2;
    const int64_t scaleCol = group == 0 ? 0 : d / group;
    const int64_t tokenStride = dataCol + 2 * scaleCol;
    const int64_t cacheCol = ((tokenStride + 31) / 32) * 32;
    // Simulate cache[::2, :, :, :] so the logical 4D view has a padded
    // non-contiguous block stride.
    const int64_t blockStride = 2 * blockSize * cacheCol;
    return CaseGeometry{blocks, blockSize, tokens, d, physicalBlocks, dataCol, scaleCol, tokenStride,
                        cacheCol, blockStride, static_cast<size_t>(physicalBlocks * blockStride),
                        static_cast<size_t>(tokens * d * sizeof(uint16_t))};
}

// 校验命中 slot 的 token bundle 字节与 bundle 之外的哨兵保持。
bool VerifyBundles(const std::vector<uint8_t> &out, const CaseGeometry &geo,
                   uint8_t expectedData, uint16_t expectedScale)
{
    bool valuesOk = true;
    std::vector<std::pair<int64_t, int64_t> > bundles;  // [start, end) written
    for (int64_t slotId : {int64_t(0), int64_t(3), int64_t(4)}) {
        const int64_t b = slotId / geo.blockSize;
        const int64_t p = slotId % geo.blockSize;
        const int64_t tokenBase = b * geo.blockStride + p * geo.tokenStride;
        const uint8_t *value = out.data() + tokenBase;
        for (int64_t i = 0; i < geo.dataCol; ++i) {
            if (value[i] == expectedData) {
                continue;
            }
            if (i == 0) {
                printf("  slot=%ld first=0x%02X expect=0x%02X\n", slotId, value[i], expectedData);
            }
            valuesOk = false;
        }
        const uint8_t *scale = out.data() + tokenBase + geo.dataCol;
        const uint16_t got = static_cast<uint16_t>(scale[0]) | static_cast<uint16_t>(scale[1] << 8);
        if (got != expectedScale) {
            printf("  slot=%ld scale=0x%04X expect=0x%04X\n", slotId, got, expectedScale);
            valuesOk = false;
        }
        bundles.emplace_back(tokenBase, tokenBase + geo.tokenStride);
    }
    // r8 contract: nothing outside the token bundles may be written (no
    // scale padding zeroing, block tails / stride gaps stay 0x5A).
    for (size_t i = 0; i < out.size(); ++i) {
        bool inside = false;
        for (const auto &r : bundles) {
            if (i >= static_cast<size_t>(r.first) && i < static_cast<size_t>(r.second)) {
                inside = true;
                break;
            }
        }
        if (!inside && out[i] != CACHE_FILL) {
            printf("  byte %zu outside bundles modified: 0x%02X\n", i, out[i]);
            valuesOk = false;
            break;
        }
    }
    if (!valuesOk) {
        printf("  output bytes mismatch\n");
    }
    return valuesOk;
}

// 释放 acl 资源：张量、workspace 与设备内存。
void ReleaseResources(aclTensor *cache, aclTensor *x, aclTensor *slot, void *ws,
                      void *cacheDev, void *xDev, void *slotDev)
{
    if (cache != nullptr) {
        aclDestroyTensor(cache);
    }
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (slot != nullptr) {
        aclDestroyTensor(slot);
    }
    if (ws != nullptr) {
        aclrtFree(ws);
    }
    if (cacheDev != nullptr) {
        aclrtFree(cacheDev);
    }
    if (xDev != nullptr) {
        aclrtFree(xDev);
    }
    if (slotDev != nullptr) {
        aclrtFree(slotDev);
    }
}

// 两段式执行：workspace 查询 + 执行 + 结果拷回。
bool ExecuteOp(const aclTensor *cache, const aclTensor *x, const aclTensor *slot,
               const CaseGeometry &geo, int64_t mode, int64_t group, void **ws, void *cacheDev,
               uint8_t *out)
{
    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    const int32_t wsStatus =
        aclnnKvCompressEpilogV2GetWorkspaceSize(const_cast<aclTensor *>(cache), const_cast<aclTensor *>(x),
                                                const_cast<aclTensor *>(slot), group, mode, true, 1.0,
                                                geo.blockStride, &wsSize, &executor);
    if (wsStatus != 0) {
        printf("  wsStatus=%d\n", wsStatus);
        return false;
    }
    if (aclrtMalloc(ws, wsSize == 0 ? 1 : wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        return false;
    }
    const int32_t execStatus = aclnnKvCompressEpilogV2(*ws, wsSize == 0 ? 1 : wsSize, executor, stream);
    if (execStatus != 0 || aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
        aclrtMemcpy(out, geo.cacheBytes, cacheDev, geo.cacheBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        printf("  execStatus=%d\n", execStatus);
        aclrtDestroyStream(stream);
        return false;
    }
    aclrtDestroyStream(stream);
    return true;
}

int RunCase(int64_t mode, int64_t group, aclDataType dtype, uint8_t expectedData, uint16_t expectedScale)
{
    if (group == 0) {
        printf("  invalid group size 0\n");
        return 1;
    }
    const CaseGeometry geo = MakeGeometry(mode, group);
    void *cacheDev = nullptr;
    void *xDev = nullptr;
    void *slotDev = nullptr;
    void *ws = nullptr;
    aclTensor *cache = nullptr;
    aclTensor *x = nullptr;
    aclTensor *slot = nullptr;
    std::vector<uint8_t> cacheHost(geo.cacheBytes, CACHE_FILL);
    std::vector<uint8_t> out(geo.cacheBytes);
    std::vector<uint16_t> xHost(geo.tokens * geo.d, 0x3F80);
    const int32_t slots[] = {0, 3, 4, -1};
    bool ok = false;
    do {
        if (aclrtMalloc(&cacheDev, geo.cacheBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(&xDev, geo.xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(&slotDev, sizeof(slots), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            break;
        }
        if (aclrtMemcpy(cacheDev, geo.cacheBytes, cacheHost.data(), geo.cacheBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(xDev, geo.xBytes, xHost.data(), geo.xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(slotDev, sizeof(slots), slots, sizeof(slots), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            break;
        }
        cache = MakeTensor({geo.blocks, geo.blockSize, 1, geo.cacheCol},
                           {geo.blockStride, geo.cacheCol, geo.cacheCol, 1}, dtype,
                           reinterpret_cast<uintptr_t>(cacheDev));
        x = MakeTensor({geo.tokens, geo.d}, {geo.d, 1}, ACL_BF16, reinterpret_cast<uintptr_t>(xDev));
        slot = MakeTensor({geo.tokens}, {1}, ACL_INT32, reinterpret_cast<uintptr_t>(slotDev));
        if (ExecuteOp(cache, x, slot, geo, mode, group, &ws, cacheDev, out.data())) {
            ok = VerifyBundles(out, geo, expectedData, expectedScale);
        }
    } while (false);
    ReleaseResources(cache, x, slot, ws, cacheDev, xDev, slotDev);
    printf("layout2 mode=%ld group=%ld %s\n", mode, group, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
}  // namespace

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        return 1;
    }
    int ret = RunCase(MODE_MXFP8, GROUP_SIZE_32, ACL_FLOAT8_E4M3FN, 0x78, 0x3B80);
    ret |= RunCase(MODE_MXFP4, GROUP_SIZE_32, ACL_UINT8, 0x66, 0x3E80);
    ret |= RunCase(MODE_MXFP4, GROUP_SIZE_16, ACL_UINT8, 0x66, 0x3E80);
    aclrtResetDevice(0);
    aclFinalize();
    return ret;
}
