// This program is free software, you can redistribute it and/or modify it.
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// KvCompressEpilogV2 GEIR 图模式调用最小样例。
//
// 使用 GE 原生构图接口创建 KvCompressEpilogV2 节点：连接 cache/x/slot_mapping
// 三个输入并设置 quant_group_size/quant_mode/round_scale/x_scale 四个属性，经
// ge::Session 在线构图执行。x 取全 1.0 的 BF16 常量，期望结果与 ACLNN 样例
// 一致（可手工推导）：
//   mode 2: data = 0x78 (E4M3FN 的 256.0), scale = 0x3B80 (BF16 的 2^-8)
//   mode 4: data = 0x66 (两个 E2M1 code 6), scale = 0x3E80 (BF16 的 2^-2)
//
// 说明：算子 IR 定义来自 vendor 包生成的 kv_compress_epilog_v2_proto.h
// （op_proto/inc 下），Data 节点由本文件内 REG_OP 注册。
//
// 前置条件：已编译安装自定义算子 run 包，并 source CANN 与 vendor 环境：
//   source <ascend-toolkit>/set_env.sh
//   source <ascend-toolkit>/opp/vendors/customize/bin/set_env.bash
//
// 编译运行：
//   g++ -std=c++17 test_geir_kv_compress_epilog_v2.cpp -o test_geir_kv_compress_epilog_v2 \
//     -I${ASCEND_HOME_PATH}/include \
//     -I$(echo ${ASCEND_CUSTOM_OPP_PATH} | cut -d: -f1)/op_proto/inc \
//     -L${ASCEND_HOME_PATH}/lib64 -lge_runner -lgraph -lgraph_base -lascendcl -lrt \
//     -Wl,-rpath,${ASCEND_HOME_PATH}/lib64
//   ./test_geir_kv_compress_epilog_v2

#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

#include "ge/ge_api.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "graph/tensor.h"
#include "kv_compress_epilog_v2_proto.h"

// Data 节点的 IR 注册（进程内注册表默认不包含 Data，需在构图程序内注册）
namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)
}  // namespace ge

namespace {

constexpr int64_t kTokenNum = 4;
constexpr int64_t kCacheRows = 8;
constexpr uint8_t kCacheFill = 0x5A;
constexpr uint16_t kBf16One = 0x3F80;  // BF16 位模式的 1.0

struct CaseSpec {
  const char *name;
  int64_t d;
  int64_t mode;        // GE 层 int：2=MXFP8, 4=packed MXFP4
  int64_t groupSize;
  ge::DataType cacheDtype;
  uint8_t dataByte;    // 期望量化数据字节
  uint16_t scaleBits;  // 期望 BF16 scale 位模式（小端两字节）
};

int64_t LayoutDataCol(int64_t d, int64_t mode) { return mode == 2 ? d : d / 2; }

int64_t LayoutKvCacheCol(int64_t d, int64_t mode, int64_t groupSize) {
  const int64_t concatCol = LayoutDataCol(d, mode) + 2 * (d / groupSize);
  return (concatCol + 31) / 32 * 32;
}

#define CHECK(cond, msg)                         \
  do {                                           \
    if (!(cond)) {                               \
      printf("[FAIL] %s: %s\n", spec.name, msg); \
      return -1;                                 \
    }                                            \
  } while (0)

ge::Tensor MakeHostTensor(const ge::TensorDesc &desc, const uint8_t *data, size_t size) {
  ge::Tensor tensor(desc);
  tensor.SetData(data, size);
  return tensor;
}

// 构图 + 执行 + 校验，返回 0 表示通过
int RunCase(const CaseSpec &spec, ge::Session &session, uint32_t graphId) {
  const int64_t dataCol = LayoutDataCol(spec.d, spec.mode);
  const int64_t scaleCol = spec.d / spec.groupSize;
  const int64_t concatCol = dataCol + 2 * scaleCol;
  const int64_t kvCacheCol = LayoutKvCacheCol(spec.d, spec.mode, spec.groupSize);
  const int64_t cacheCol = kvCacheCol;
  const size_t cacheBytes = static_cast<size_t>(kCacheRows * cacheCol);
  const size_t xBytes = static_cast<size_t>(kTokenNum * spec.d) * sizeof(uint16_t);

  // ---- 输入数据：cache 填充 0x5A，x 全 1.0，slot 含合法/-1/越界下标 ----
  std::vector<uint8_t> cacheHost(cacheBytes, kCacheFill);
  std::vector<uint16_t> xHost(static_cast<size_t>(kTokenNum * spec.d), kBf16One);
  const std::vector<int32_t> slotHost = {1, -1, 3, static_cast<int32_t>(kCacheRows)};

  // ---- 构图：Data 输入节点 + KvCompressEpilogV2 节点 ----
  const ge::TensorDesc cacheDesc(ge::Shape({kCacheRows, cacheCol}), ge::FORMAT_ND, spec.cacheDtype);
  const ge::TensorDesc xDesc(ge::Shape({kTokenNum, spec.d}), ge::FORMAT_ND, ge::DT_BF16);
  const ge::TensorDesc slotDesc(ge::Shape({kTokenNum}), ge::FORMAT_ND, ge::DT_INT32);

  auto cacheData = ge::op::Data("cache").set_attr_index(0);
  cacheData.update_input_desc_x(cacheDesc);
  cacheData.update_output_desc_y(cacheDesc);
  auto xData = ge::op::Data("x").set_attr_index(1);
  xData.update_input_desc_x(xDesc);
  xData.update_output_desc_y(xDesc);
  auto slotData = ge::op::Data("slot_mapping").set_attr_index(2);
  slotData.update_input_desc_x(slotDesc);
  slotData.update_output_desc_y(slotDesc);

  // ---- KvCompressEpilogV2 节点：3 输入 + 4 属性（GE 层 quant_mode 为 int） ----
  auto kcev2 = ge::op::KvCompressEpilogV2("kcev2");
  kcev2.set_input_cache(cacheData);
  kcev2.set_input_x(xData);
  kcev2.set_input_slot_mapping(slotData);
  kcev2.update_input_desc_cache(cacheDesc);
  kcev2.update_input_desc_x(xDesc);
  kcev2.update_input_desc_slot_mapping(slotDesc);
  kcev2.update_output_desc_cache(cacheDesc);
  kcev2.set_attr_quant_group_size(spec.groupSize);
  kcev2.set_attr_quant_mode(spec.mode);
  kcev2.set_attr_round_scale(true);
  kcev2.set_attr_x_scale(1.0f);

  ge::Graph graph("kv_compress_epilog_v2_graph");
  graph.SetInputs({cacheData, xData, slotData});
  graph.SetOutputs({kcev2});
  const std::map<ge::AscendString, ge::AscendString> graphOptions;
  CHECK(session.AddGraph(graphId, graph, graphOptions) == ge::SUCCESS, "AddGraph");

  // ---- 输入 Tensor（host 内存）----
  auto cacheInDesc = cacheDesc;
  cacheInDesc.SetPlacement(ge::kPlacementHost);
  auto xInDesc = xDesc;
  xInDesc.SetPlacement(ge::kPlacementHost);
  auto slotInDesc = slotDesc;
  slotInDesc.SetPlacement(ge::kPlacementHost);
  std::vector<ge::Tensor> inputs = {
      MakeHostTensor(cacheInDesc, cacheHost.data(), cacheBytes),
      MakeHostTensor(xInDesc, reinterpret_cast<const uint8_t *>(xHost.data()), xBytes),
      MakeHostTensor(slotInDesc, reinterpret_cast<const uint8_t *>(slotHost.data()),
                     slotHost.size() * sizeof(int32_t))};

  // ---- 执行 ----
  std::vector<ge::Tensor> outputs;
  CHECK(session.RunGraph(graphId, inputs, outputs) == ge::SUCCESS, "RunGraph");
  CHECK(outputs.size() == 1 && outputs[0].GetSize() == cacheBytes, "output size");
  const uint8_t *out = outputs[0].GetData();

  // ---- 校验：slot=1 与 slot=3 命中行 ----
  for (const int64_t hit : {int64_t(1), int64_t(3)}) {
    const uint8_t *row = out + static_cast<size_t>(hit) * static_cast<size_t>(cacheCol);
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
  }
  // ---- 校验：未命中行与 slot=-1/越界行保持原值 ----
  for (const int64_t keep : {int64_t(0), int64_t(2)}) {
    const uint8_t *row = out + static_cast<size_t>(keep) * static_cast<size_t>(cacheCol);
    bool keepOk = true;
    for (int64_t i = 0; i < cacheCol; ++i) {
      keepOk = keepOk && row[i] == kCacheFill;
    }
    CHECK(keepOk, "untouched row preserved");
  }
  printf("[PASS] %s: d=%ld, mode=%ld, groupSize=%ld, cacheCol=%ld (kvCacheCol=%ld)\n",
         spec.name, spec.d, spec.mode, spec.groupSize, cacheCol, kvCacheCol);
  return 0;
}

}  // namespace

int main() {
  // GE 全局初始化（含设备选择），等价于独立 aclInit 流程
  std::map<ge::AscendString, ge::AscendString> globalOptions;
  globalOptions["ge.exec.deviceId"] = "0";
  globalOptions["ge.graphRunMode"] = "1";
  if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
    printf("[FAIL] GEInitialize\n");
    return -1;
  }

  std::map<ge::AscendString, ge::AscendString> sessionOptions;
  sessionOptions["ge.exec.deviceId"] = "0";
  ge::Session session(sessionOptions);
  // mode 2：MXFP8（E4M3FN cache）
  const CaseSpec mxfp8 = {"geir mxfp8_bf16 group32", 512, 2, 32,
                          ge::DT_FLOAT8_E4M3FN, 0x78, 0x3B80};
  // mode 4：packed MXFP4（UINT8 cache），d=128 恰好一行 vreg
  const CaseSpec mxfp4 = {"geir mxfp4_bf16 group32", 128, 4, 32,
                          ge::DT_UINT8, 0x66, 0x3E80};
  const CaseSpec mxfp4Group16 = {"geir mxfp4_bf16 group16", 512, 4, 16,
                                 ge::DT_UINT8, 0x66, 0x3E80};
  int ret = RunCase(mxfp8, session, 1);
  ret |= RunCase(mxfp4, session, 2);
  ret |= RunCase(mxfp4Group16, session, 3);
  (void)ge::GEFinalize();
  if (ret == 0) {
    printf("all geir cases passed\n");
  }
  return ret;
}
