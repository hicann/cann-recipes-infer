/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// KvCompressEpilogV2 Host UT：通过 aclnn GetWorkspaceSize 驱动 host tiling 的
// 正例与负例校验（tiling 的 rank/dtype/布局/属性校验在 GetWorkspaceSize 阶段执行）。
//
// 正例断言：GetWorkspaceSize 返回 0 且 workspace 大小为固定值 32。
// 负例断言：非法 rank、dtype、d%groupSize、cache 行宽、非法 mode、非法属性组合均返回非 0。
//
// 构建运行（见同目录 CMakeLists.txt）：
//   cd <repo>/ops/ascendc/src/kv_compress_epilog_v2/test && mkdir -p build && cd build
//   cmake .. && make && ./test_host_tiling
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

struct HostCase {
    const char *name;
    int64_t d;
    int64_t tokenNum;
    int64_t cacheCol;       // 实际 cache 行宽（-1 表示取合法值 kvCacheCol）
    int64_t mode;
    int64_t groupSize;
    double xScale;
    aclDataType cacheDtype;
    aclDataType xDtype;
    aclDataType slotDtype;
    int64_t cacheRank;      // cache 的 rank（默认 2）
    int64_t slotLen;        // slot_mapping 长度（-1 表示与 tokenNum 一致）
    bool expectOk;
};

void RunHostCase(const HostCase &tc)
{
    const int64_t d = tc.d;
    const int64_t kvCacheCol = LayoutKvCacheCol(d, tc.mode, tc.groupSize);
    const int64_t cacheCol = tc.cacheCol >= 0 ? tc.cacheCol : kvCacheCol;
    const size_t cacheBytes = static_cast<size_t>(kCacheRows * cacheCol);
    const size_t xBytes = static_cast<size_t>(tc.tokenNum * d) * sizeof(uint16_t);
    const size_t slotBytes = static_cast<size_t>(tc.tokenNum) * sizeof(int32_t);

    void *cacheDev = nullptr;
    void *xDev = nullptr;
    void *slotDev = nullptr;
    (void)aclrtMalloc(&cacheDev, cacheBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMalloc(&slotDev, slotBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    const int64_t slotLen = tc.slotLen >= 0 ? tc.slotLen : tc.tokenNum;
    const std::vector<int64_t> cacheDims = tc.cacheRank == 1
        ? std::vector<int64_t>{kCacheRows * cacheCol}
        : std::vector<int64_t>{kCacheRows, cacheCol};
    aclTensor *cache = MakeRowMajorTensor(cacheDims, tc.cacheDtype, cacheDev);
    aclTensor *x = MakeRowMajorTensor({tc.tokenNum, d}, tc.xDtype, xDev);
    aclTensor *slot = MakeRowMajorTensor({slotLen}, tc.slotDtype, slotDev);

    const CallResult result = RunOp(cache, x, slot, tc.mode, tc.groupSize, true, tc.xScale, nullptr, 0, nullptr);
    const bool ok = tc.expectOk ? (result.wsStatus == 0 && result.workspaceSize == kExpectWorkspaceSize)
                                : (result.wsStatus != 0);
    if (ok) {
        ++gPass;
        printf("[PASS] %s\n", tc.name);
    } else {
        ++gFail;
        printf("[FAIL] %s: wsStatus=%d, workspaceSize=%lu\n", tc.name, result.wsStatus,
               static_cast<unsigned long>(result.workspaceSize));
    }

    (void)aclDestroyTensor(cache);
    (void)aclDestroyTensor(x);
    (void)aclDestroyTensor(slot);
    (void)aclrtFree(cacheDev);
    (void)aclrtFree(xDev);
    (void)aclrtFree(slotDev);
}

}  // namespace

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        printf("[FAIL] acl init\n");
        return -1;
    }

    // ---- 正例：合法 shape/dtype/属性，mode 2 与 mode 4 ----
    const HostCase positives[] = {
        {"positive mxfp8 group32 d=512 e4m3fn", 512, 4, -1, 2, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16,
         ACL_INT32, 2, -1, true},
        {"positive mxfp8 group32 d=128 e5m2", 128, 4, -1, 2, kGroupSize32, 1.0, ACL_FLOAT8_E5M2, ACL_BF16, ACL_INT32, 2,
         -1, true},
        {"positive mxfp4 group32 d=128 uint8", 128, 4, -1, 4, kGroupSize32, 1.0, ACL_UINT8, ACL_BF16, ACL_INT32, 2, -1,
         true},
        {"positive mxfp4 group16 d=128 uint8", 128, 4, -1, 4, kGroupSize16, 1.0, ACL_UINT8, ACL_BF16,
         ACL_INT32, 2, -1, true},
        {"positive mxfp4 group16 d=512 uint8 wide cache", 512, 4,
         LayoutKvCacheCol(512, 4, kGroupSize16) + 64, 4, kGroupSize16, 1.0,
         ACL_UINT8, ACL_BF16, ACL_INT32, 2, -1, true},
        {"positive slot int64", 512, 4, -1, 2, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT64, 2, -1,
         true},
    };
    for (const auto &tc : positives) {
        RunHostCase(tc);
    }

    // ---- 负例：host tiling 校验必须拒绝 ----
    const HostCase negatives[] = {
        {"negative cache rank=1", 512, 4, -1, 2, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT32, 1, -1,
         false},
        {"negative x dtype fp16", 512, 4, -1, 2, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, ACL_INT32, 2,
         -1, false},
        {"negative slot dtype int16", 512, 4, -1, 2, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT16, 2,
         -1, false},
        {"negative slot length mismatch", 512, 4, -1, 2, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT32,
         2, 5, false},
        {"negative mode2 d not divisible by 32", 100, 4, -1, 2, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16,
         ACL_INT32, 2, -1, false},
        {"negative mode4 group16 d not divisible by 16", 72, 4, -1, 4, kGroupSize16, 1.0, ACL_UINT8,
         ACL_BF16, ACL_INT32, 2, -1, false},
        {"negative d exceeds 8192", 8224, 4, -1, 2, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT32, 2,
         -1, false},
        {"negative cache col too narrow", 512, 4, LayoutKvCacheCol(512, 2, kGroupSize32) - 1, 2, kGroupSize32, 1.0,
         ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT32, 2, -1, false},
        {"negative illegal quant_mode=3", 512, 4, -1, 3, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT32,
         2, -1, false},
        {"negative mode2 with uint8 cache", 512, 4, -1, 2, kGroupSize32, 1.0, ACL_UINT8, ACL_BF16, ACL_INT32, 2,
         -1, false},
        {"negative mode4 with fp8 cache", 512, 4, -1, 4, kGroupSize32, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT32,
         2, -1, false},
        {"negative x_scale=2.0", 512, 4, -1, 2, kGroupSize32, 2.0, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT32, 2, -1,
         false},
        {"negative group_size=64", 512, 4, -1, 2, 64, 1.0, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_INT32, 2, -1, false},
        {"negative mxfp8 group_size=16", 512, 4, -1, 2, kGroupSize16, 1.0, ACL_FLOAT8_E4M3FN,
         ACL_BF16, ACL_INT32, 2, -1, false},
    };
    for (const auto &tc : negatives) {
        RunHostCase(tc);
    }

    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    printf("host tiling UT: %d passed, %d failed\n", gPass, gFail);
    return gFail == 0 ? 0 : -1;
}
