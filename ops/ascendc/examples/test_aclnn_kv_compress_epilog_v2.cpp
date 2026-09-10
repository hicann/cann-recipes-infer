// This program is free software, you can redistribute it and/or modify it.
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// KvCompressEpilogV2 ACLNN 两段式调用最小样例。
//
// 覆盖 quant_mode=2（MXFP8，cache 为 float8_e4m3fn）与 quant_mode=4（packed
// MXFP4，cache 为 uint8）两种模式，slot_mapping 含合法下标、-1 跳过与越界下标。
// x 取全 1.0 的 BF16 常量，因此每组的量化结果可手工推导：
//   mode 2: data = 0x78 (E4M3FN 的 256.0), scale = 0x3B80 (BF16 的 2^-8)
//   mode 4: data = 0x66 (两个 E2M1 code 6), scale = 0x3E80 (BF16 的 2^-2)
//
// 前置条件：已编译安装自定义算子 run 包，并 source CANN 与 vendor 环境：
//   source <ascend-toolkit>/set_env.sh
//   source <ascend-toolkit>/opp/vendors/customize/bin/set_env.bash
//
// 编译运行：
//   g++ -std=c++17 test_aclnn_kv_compress_epilog_v2.cpp -o test_aclnn_kv_compress_epilog_v2 \
//     -I${ASCEND_HOME_PATH}/include \
//     -I$(echo ${ASCEND_CUSTOM_OPP_PATH} | cut -d: -f1)/op_api/include \
//     -L${ASCEND_HOME_PATH}/lib64 -lascendcl -lnnopbase \
//     -L$(echo ${ASCEND_CUSTOM_OPP_PATH} | cut -d: -f1)/op_api/lib -lcust_opapi \
//     -Wl,-rpath,${ASCEND_HOME_PATH}/lib64:$(echo ${ASCEND_CUSTOM_OPP_PATH} | cut -d: -f1)/op_api/lib
//   ./test_aclnn_kv_compress_epilog_v2

#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_kv_compress_epilog_v2.h"

namespace {

constexpr int64_t kTokenNum = 4;
constexpr int64_t kCacheRows = 8;
constexpr uint8_t kCacheFill = 0x5A;
constexpr uint16_t kBf16One = 0x3F80;  // BF16 位模式的 1.0

struct CaseSpec {
  const char *name;
  int64_t d;
  int64_t mode;         // GE/ACLNN 层 int：2=MXFP8, 4=packed MXFP4
  int64_t groupSize;
  aclDataType cacheDtype;
  uint8_t dataByte;     // 期望量化数据字节
  uint16_t scaleBits;   // 期望 BF16 scale 位模式（小端两字节）
  int64_t wideBytes;    // cache 行宽在 kvCacheCol 基础上的加宽字节数
};

int64_t LayoutDataCol(int64_t d, int64_t mode) { return mode == 2 ? d : d / 2; }

// cache 行宽：RoundUp(dataCol + 2 * (d/groupSize), 32)
int64_t LayoutKvCacheCol(int64_t d, int64_t mode, int64_t groupSize) {
  const int64_t concatCol = LayoutDataCol(d, mode) + 2 * (d / groupSize);
  return (concatCol + 31) / 32 * 32;
}

aclTensor *MakeRowMajorTensor(const std::vector<int64_t> &dims, aclDataType dtype, void *devAddr) {
  std::vector<int64_t> strides(dims.size());
  int64_t elem = 1;
  for (int64_t i = static_cast<int64_t>(strides.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = elem;
    elem *= dims[static_cast<size_t>(i)];
  }
  return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                         dims.data(), dims.size(), devAddr);
}

#define CHECK(cond, msg)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      printf("[FAIL] %s: %s\n", spec.name, msg);             \
      return -1;                                             \
    }                                                        \
  } while (0)

// 一次完整的两段式调用与结果校验，返回 0 表示通过
int RunCase(const CaseSpec &spec) {
  const int64_t dataCol = LayoutDataCol(spec.d, spec.mode);
  const int64_t scaleCol = spec.d / spec.groupSize;
  const int64_t concatCol = dataCol + 2 * scaleCol;
  const int64_t kvCacheCol = LayoutKvCacheCol(spec.d, spec.mode, spec.groupSize);
  const int64_t cacheCol = kvCacheCol + spec.wideBytes;
  const size_t cacheBytes = static_cast<size_t>(kCacheRows * cacheCol);
  const size_t xBytes = static_cast<size_t>(kTokenNum * spec.d) * sizeof(uint16_t);
  const std::vector<int32_t> slotHost = {1, -1, 3, static_cast<int32_t>(kCacheRows)};

  std::vector<uint8_t> cacheHost(cacheBytes, kCacheFill);
  std::vector<uint16_t> xHost(static_cast<size_t>(kTokenNum * spec.d), kBf16One);

  void *cacheDev = nullptr;
  void *xDev = nullptr;
  void *slotDev = nullptr;
  aclrtStream stream = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  void *workspace = nullptr;
  uint64_t wsSize = 0;
  aclTensor *cacheTensor = nullptr;
  aclTensor *xTensor = nullptr;
  aclTensor *slotTensor = nullptr;
  int32_t ret = -1;

  do {
    CHECK(aclrtMalloc(&cacheDev, cacheBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc cache");
    CHECK(aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc x");
    CHECK(aclrtMalloc(&slotDev, slotHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
          "malloc slot");
    CHECK(aclrtMemcpy(cacheDev, cacheBytes, cacheHost.data(), cacheBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
          ACL_SUCCESS, "copy cache");
    CHECK(aclrtMemcpy(xDev, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, "copy x");
    CHECK(aclrtMemcpy(slotDev, slotHost.size() * sizeof(int32_t), slotHost.data(),
                      slotHost.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
          "copy slot");
    CHECK(aclrtCreateStream(&stream) == ACL_SUCCESS, "create stream");

    cacheTensor = MakeRowMajorTensor({kCacheRows, cacheCol}, spec.cacheDtype, cacheDev);
    xTensor = MakeRowMajorTensor({kTokenNum, spec.d}, ACL_BF16, xDev);
    slotTensor = MakeRowMajorTensor({kTokenNum}, ACL_INT32, slotDev);
    CHECK(cacheTensor != nullptr && xTensor != nullptr && slotTensor != nullptr, "create tensors");

    // 第一段：GetWorkspaceSize，参数顺序与算子 IR 一致（cache, x, slot_mapping, 4 个属性）
    CHECK(aclnnKvCompressEpilogV2GetWorkspaceSize(cacheTensor, xTensor, slotTensor, spec.groupSize, spec.mode,
                                                  true, 1.0, &workspaceSize, &executor) == 0,
          "GetWorkspaceSize");
    wsSize = workspaceSize == 0 ? 1 : workspaceSize;
    CHECK(aclrtMalloc(&workspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc workspace");

    // 第二段：执行（原地更新 cache）
    CHECK(aclnnKvCompressEpilogV2(workspace, wsSize, executor, stream) == 0, "aclnnKvCompressEpilogV2");
    CHECK(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "sync stream");
    CHECK(aclrtMemcpy(cacheHost.data(), cacheBytes, cacheDev, cacheBytes, ACL_MEMCPY_DEVICE_TO_HOST) ==
          ACL_SUCCESS, "copy back");

    // ---- 校验：slot=1 与 slot=3 命中行 ----
    for (const int64_t hit : {int64_t(1), int64_t(3)}) {
      const uint8_t *row = cacheHost.data() + static_cast<size_t>(hit) * static_cast<size_t>(cacheCol);
      bool dataOk = true;
      for (int64_t i = 0; i < dataCol; ++i) {
        dataOk = dataOk && row[i] == spec.dataByte;
      }
      CHECK(dataOk, "quantized data bytes");
      bool scaleOk = true;
      for (int64_t g = 0; g < scaleCol; ++g) {
        const uint16_t got = static_cast<uint16_t>(row[dataCol + 2 * g]) |
                             static_cast<uint16_t>(row[dataCol + 2 * g + 1] << 8);
        scaleOk = scaleOk && got == spec.scaleBits;
      }
      CHECK(scaleOk, "bf16 scale bytes");
      bool padOk = true;
      for (int64_t i = concatCol; i < kvCacheCol; ++i) {
        padOk = padOk && row[i] == 0;
      }
      CHECK(padOk, "padding zeroed");
      bool wideOk = true;
      for (int64_t i = kvCacheCol; i < cacheCol; ++i) {
        wideOk = wideOk && row[i] == kCacheFill;
      }
      CHECK(wideOk, "wide tail preserved");
    }
    // ---- 校验：未命中行与 slot=-1/越界行保持原值 ----
    for (const int64_t keep : {int64_t(0), int64_t(2)}) {
      const uint8_t *row = cacheHost.data() + static_cast<size_t>(keep) * static_cast<size_t>(cacheCol);
      bool keepOk = true;
      for (int64_t i = 0; i < cacheCol; ++i) {
        keepOk = keepOk && row[i] == kCacheFill;
      }
      CHECK(keepOk, "untouched row preserved");
    }
    ret = 0;
  } while (false);

  if (cacheTensor != nullptr) { aclDestroyTensor(cacheTensor); }
  if (xTensor != nullptr) { aclDestroyTensor(xTensor); }
  if (slotTensor != nullptr) { aclDestroyTensor(slotTensor); }
  if (workspace != nullptr) { (void)aclrtFree(workspace); }
  if (stream != nullptr) { (void)aclrtDestroyStream(stream); }
  if (cacheDev != nullptr) { (void)aclrtFree(cacheDev); }
  if (xDev != nullptr) { (void)aclrtFree(xDev); }
  if (slotDev != nullptr) { (void)aclrtFree(slotDev); }
  if (ret == 0) {
    printf("[PASS] %s: d=%ld, mode=%ld, groupSize=%ld, cacheCol=%ld (kvCacheCol=%ld)\n",
           spec.name, spec.d, spec.mode, spec.groupSize, cacheCol, kvCacheCol);
  }
  return ret;
}

}  // namespace

int main() {
  if (aclInit(nullptr) != ACL_SUCCESS) {
    printf("[FAIL] aclInit\n");
    return -1;
  }
  if (aclrtSetDevice(0) != ACL_SUCCESS) {
    printf("[FAIL] aclrtSetDevice\n");
    return -1;
  }
  // mode 2：MXFP8（E4M3FN cache），加宽行 64 字节验证尾部保持
  const CaseSpec mxfp8 = {"aclnn mxfp8_bf16 group32", 512, 2, 32,
                          ACL_FLOAT8_E4M3FN, 0x78, 0x3B80, 64};
  // mode 4：packed MXFP4（UINT8 cache），d=128 恰好一行 vreg
  const CaseSpec mxfp4 = {"aclnn mxfp4_bf16 group32", 128, 4, 32,
                          ACL_UINT8, 0x66, 0x3E80, 0};
  const CaseSpec mxfp4Group16 = {"aclnn mxfp4_bf16 group16", 512, 4, 16,
                                 ACL_UINT8, 0x66, 0x3E80, 32};
  int ret = RunCase(mxfp8);
  ret |= RunCase(mxfp4);
  ret |= RunCase(mxfp4Group16);
  (void)aclrtResetDevice(0);
  (void)aclFinalize();
  if (ret == 0) {
    printf("all aclnn cases passed\n");
  }
  return ret;
}
