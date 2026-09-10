/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// KvCompressEpilogV2 Kernel UT：aclnn 两段式执行并逐字节校验 cache 写回。
//
// 覆盖：mode 2/4、groupSize 16/32、短尾 chunk 与多 vreg，
// slot_mapping 含合法下标、-1 跳过与越界下标（保持 cache 不变）。
// x 取全 1.0 的 BF16 常量，期望字节可手工推导：
//   mode 2: data = 0x78 (E4M3FN 的 256.0 / E5M2 的 32768.0)
//           scale = 0x3B80 (E4M3FN, BF16 的 2^-8) 或 0x3800 (E5M2, BF16 的 2^-15)
//   mode 4: data = 0x66 (两个 E2M1 code 6), scale = 0x3E80 (BF16 的 2^-2)
//
// 构建运行（见同目录 CMakeLists.txt）：
//   cd <repo>/ops/ascendc/src/kv_compress_epilog_v2/test && mkdir -p build && cd build
//   cmake .. && make && ./test_kernel
// 前置：source <ascend-toolkit>/set_env.sh 与 vendor set_env.bash。

#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "kcev2_test_common.h"

namespace {

using namespace kcev2test;

int gPass = 0;
int gFail = 0;

struct KernelCase {
    const char *name;
    int64_t d;
    int64_t mode;
    int64_t groupSize;
    aclDataType cacheDtype;
    uint8_t dataByte;    // 期望量化数据字节
    uint16_t scaleBits;  // 期望 BF16 scale 位模式
};

void RunKernelCase(const KernelCase &tc)
{
    constexpr int64_t tokenNum = 4;
    const int64_t cacheCol = LayoutKvCacheCol(tc.d, tc.mode, tc.groupSize);
    const size_t cacheBytes = static_cast<size_t>(kCacheRows * cacheCol);
    const size_t xBytes = static_cast<size_t>(tokenNum * tc.d) * sizeof(uint16_t);

    // slot：row0 -> 1（命中），row1 -> -1（跳过），row2 -> 3（命中），row3 -> N（越界跳过）
    const std::vector<int32_t> slotHost = {1, -1, 3, static_cast<int32_t>(kCacheRows)};
    const std::vector<uint8_t> cacheHost(cacheBytes, kCacheFill);
    const std::vector<uint16_t> xHost(static_cast<size_t>(tokenNum * tc.d), kBf16One);

    void *cacheDev = nullptr;
    void *xDev = nullptr;
    void *slotDev = nullptr;
    std::vector<uint8_t> outHost(cacheBytes);
    bool ok = false;
    do {
        if (aclrtMalloc(&cacheDev, cacheBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(&slotDev, slotHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            printf("[FAIL] %s: malloc device memory\n", tc.name);
            break;
        }
        if (aclrtMemcpy(cacheDev, cacheBytes, cacheHost.data(), cacheBytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
                ACL_SUCCESS ||
            aclrtMemcpy(xDev, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(slotDev, slotHost.size() * sizeof(int32_t), slotHost.data(),
                        slotHost.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            printf("[FAIL] %s: copy inputs\n", tc.name);
            break;
        }
        aclTensor *cache = MakeRowMajorTensor({kCacheRows, cacheCol}, tc.cacheDtype, cacheDev);
        aclTensor *x = MakeRowMajorTensor({tokenNum, tc.d}, ACL_BF16, xDev);
        aclTensor *slot = MakeRowMajorTensor({tokenNum}, ACL_INT32, slotDev);
        const CallResult result = RunOp(cache, x, slot, tc.mode, tc.groupSize, true, 1.0, cacheDev, cacheBytes,
                                        outHost.data());
        (void)aclDestroyTensor(cache);
        (void)aclDestroyTensor(x);
        (void)aclDestroyTensor(slot);
        if (result.wsStatus != 0 || result.execStatus != 0) {
            printf("[FAIL] %s: wsStatus=%d execStatus=%d\n", tc.name, result.wsStatus, result.execStatus);
            break;
        }
        // 命中行：slot 1/3；保持行：0/2（未命中）、-1 行与越界行无对应 cache 行，归入 slot 未命中集合
        if (!VerifyCache(outHost.data(), cacheCol, tc.d, tc.mode, tc.groupSize,
                         tc.dataByte, tc.scaleBits, {1, 3}, {0, 2})) {
            printf("[FAIL] %s: cache bytes mismatch\n", tc.name);
            break;
        }
        ok = true;
    } while (false);

    if (cacheDev != nullptr) { (void)aclrtFree(cacheDev); }
    if (xDev != nullptr) { (void)aclrtFree(xDev); }
    if (slotDev != nullptr) { (void)aclrtFree(slotDev); }
    if (ok) {
        ++gPass;
        printf("[PASS] %s: d=%ld, mode=%ld, groupSize=%ld, cacheCol=%ld\n",
               tc.name, tc.d, tc.mode, tc.groupSize, cacheCol);
    } else {
        ++gFail;
    }
}

}  // namespace

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        printf("[FAIL] acl init\n");
        return -1;
    }

    const KernelCase cases[] = {
        {"kernel mxfp8 group32 d=128 (short vreg)", 128, 2, kGroupSize32,
         ACL_FLOAT8_E4M3FN, 0x78, 0x3B80},
        {"kernel mxfp8 group32 d=512 (multi vreg)", 512, 2, kGroupSize32,
         ACL_FLOAT8_E4M3FN, 0x78, 0x3B80},
        {"kernel mxfp8 group32 d=512 e5m2", 512, 2, kGroupSize32,
         ACL_FLOAT8_E5M2, 0x78, 0x3800},
        {"kernel mxfp4 group32 d=128 (short vreg, F2 shape)", 128, 4, kGroupSize32,
         ACL_UINT8, 0x66, 0x3E80},
        {"kernel mxfp4 group32 d=512 (multi vreg)", 512, 4, kGroupSize32,
         ACL_UINT8, 0x66, 0x3E80},
        {"kernel mxfp4 group16 d=16 (quarter chunk)", 16, 4, kGroupSize16,
         ACL_UINT8, 0x66, 0x3E80},
        {"kernel mxfp4 group16 d=32 (half chunk)", 32, 4, kGroupSize16,
         ACL_UINT8, 0x66, 0x3E80},
        {"kernel mxfp4 group16 d=48 (tail chunk)", 48, 4, kGroupSize16,
         ACL_UINT8, 0x66, 0x3E80},
        {"kernel mxfp4 group16 d=64 (single chunk)", 64, 4, kGroupSize16,
         ACL_UINT8, 0x66, 0x3E80},
        {"kernel mxfp4 group16 d=96 (full plus tail chunk)", 96, 4, kGroupSize16,
         ACL_UINT8, 0x66, 0x3E80},
        {"kernel mxfp4 group16 d=128 (multi chunk)", 128, 4, kGroupSize16,
         ACL_UINT8, 0x66, 0x3E80},
        {"kernel mxfp4 group16 d=512 (multi chunk)", 512, 4, kGroupSize16,
         ACL_UINT8, 0x66, 0x3E80},
    };
    for (const auto &tc : cases) {
        RunKernelCase(tc);
    }

    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    printf("kernel UT: %d passed, %d failed\n", gPass, gFail);
    return gFail == 0 ? 0 : -1;
}
