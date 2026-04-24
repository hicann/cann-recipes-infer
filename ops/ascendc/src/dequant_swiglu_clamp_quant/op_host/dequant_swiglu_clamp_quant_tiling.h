/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_swiglu_clamp_quant_tiling.h
 * \brief dequant_swiglu_quant 算子 ClampQuant tiling 实现统一头文件
 *        合并原有的 DequantSwigluQuantBaseTilingData 和 DequantSwigluClampQuantTilingData
 */

#ifndef DEQUANT_SWIGLU_CLAMP_QUANT_TILING_H
#define DEQUANT_SWIGLU_CLAMP_QUANT_TILING_H

#include <vector>
#include <iostream>
#include <sstream>
#include <set>
#include <map>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

#ifndef OP_CHECK_IF
#define OP_CHECK_IF(COND, LOG_FUNC, EXPR) OPS_CHECK(COND, LOG_FUNC, EXPR)
#endif

namespace optiling
{

// ============================================================
// 常量定义
// ============================================================

// Group 模式 TilingKey 基础值
constexpr uint64_t TILING_KEY_HAS_GROUP = 100000000;
constexpr uint64_t TILING_KEY_NO_GROUP = 200000000;
constexpr uint64_t TILING_KEY_CUT_GROUP = 10000000;

// 基础模式 TilingKey（继承自 swi_glu_tiling.h）
// Static: 10000~14111
// Dynamic: 30001~30013

// 输入索引
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_SCALE_INDEX = 1;
constexpr int64_t ACTIVATION_SCALE_INDEX = 2;
constexpr int64_t BIAS_INDEX = 3;
constexpr int64_t QUANT_SCALE_INDEX = 4;
constexpr int64_t QUANT_OFFSET_INDEX = 5;
constexpr int64_t INPUT_GROUP_INDEX = 6;

// 属性索引
constexpr int64_t ATTR_ACTIVATE_LEFT_INDEX = 0;
constexpr int64_t ATTR_QUANT_MODE_INDEX = 1;
constexpr int64_t SWIGLU_MODE_INDEX = 5;
constexpr int64_t CLAMP_LIMIT_INDEX = 6;
constexpr int64_t GLU_ALPHA_INDEX = 7;
constexpr int64_t GLU_BIAS_INDEX = 8;

// 尺寸常量
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BLOCK_ELEM = BLOCK_SIZE / static_cast<int64_t>(sizeof(float));
constexpr uint64_t WORKSPACE_SIZE = 32;
constexpr uint64_t TILING_KEY_QS_DTYPE = 100;
constexpr uint64_t TILING_KEY_BIAS_DTYPE = 1000;
constexpr int64_t UB_RESERVE = 1024;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t PERFORMANCE_H_2048 = 2048;
constexpr int64_t PERFORMANCE_H_4096 = 4096;
constexpr int64_t PERFORMANCE_CORE_NUM = 36;
constexpr int64_t PERFORMANCE_UB_FACTOR = static_cast<int64_t>(4096) * 4;
constexpr int64_t CUT_GROUP_LARGE_THAN_64 = 64;
constexpr int64_t CUT_GROUP_LARGE_THAN_32 = 32;
constexpr int64_t EACH_GROUP_TOKEN_LESS_THAN = 16;
constexpr int64_t DIM_SIZE_2 = 2;
constexpr uint32_t USER_WORKSPACE = 16777216; // 16 * 1024 * 1024

// Quant scale dtype 编码
constexpr int QUANT_SCALE_DTYPE_BF16 = 2;
constexpr int QUANT_SCALE_DTYPE_FP32 = 0;
constexpr int QUANT_SCALE_DTYPE_FP16 = 1;

// Bias dtype 编码
constexpr int BIAS_DTYPE_BF16 = 0;
constexpr int BIAS_DTYPE_FP16 = 1;
constexpr int BIAS_DTYPE_FP32 = 2;
constexpr int BIAS_DTYPE_INT32 = 3;

// 默认值
constexpr float CLAMP_LIMIT_DEFAULT = 7.0;
constexpr float GLU_ALPHA_DEFAULT = 1.702;
constexpr float GLU_BIAS_DEFAULT = 1.0;

// UB 常量（基础模式）
constexpr uint32_t UB_RESERVED_BUFF = 0;
constexpr uint32_t PACK_UINT_IN_CACHE_512B = 512;
constexpr uint32_t ALIGN_UINT_IN_CACHE_32B = 32;
constexpr uint32_t ALIGN_UINT_IN_CACHE_64B = 64;
constexpr uint32_t ALIGN_TYPE_INT32 = 8;
constexpr uint32_t DEFAULT_BUFFER_NUM = 2;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;
constexpr uint32_t MAX_BLOCK_LEN = 2097120;
constexpr uint32_t MAX_CORE_NUMBER = 64;
constexpr uint32_t PERFORMANCE_COL_LEN = 1536;
constexpr uint32_t PERFORMANCE_ROW_LEN = 128;
constexpr uint32_t MIN_CORE = 12;
constexpr uint16_t DISCONTINE_COPY_MAX_BLOCKCNT = 4095;
constexpr uint16_t DISCONTINE_COPY_MAX_BLOCKLEN = 65535;
constexpr uint16_t DISCONTINE_COPY_MAX_STRIDE = 65535;

static const uint32_t DYNAMIC_BF16_TBUF_NUM_HALF = 11;
static const uint32_t DYNAMIC_BF16_INT16_TBUF_NUM_HALF = 6;
static const uint32_t STATIC_BF16_TBUF_NUM_HALF = 12;
static const uint32_t STATIC_BF16_INT16_TBUF_NUM_HALF = 7;
static const uint32_t DYNAMIC_INT16_TBUF_NUM_HALF = 2;

const int64_t STATIC_FLOAT16_X = 10000;
const int64_t STATIC_BFLOAT16_X = 10001;
const int64_t STATIC_FLOAT16_XD = 10002;
const int64_t STATIC_BFLOAT16_XD = 10003;
const int64_t STATIC_INT_X_INT_BIAS_QUANT_ONE = 10004;
const int64_t STATIC_INT_X_INT_BIAS_QUANT_D = 10005;
const int64_t STATIC_INT_X_FLOAT16_BIAS_QUANT_ONE = 10006;
const int64_t STATIC_INT_X_FLOAT16_BIAS_QUANT_D = 10007;
const int64_t STATIC_INT_X_FLOAT32_BIAS_QUANT_ONE = 10008;
const int64_t STATIC_INT_X_FLOAT32_BIAS_QUANT_D = 10009;
const int64_t STATIC_INT_X_BFLOAT16_BIAS_QUANT_ONE = 10010;
const int64_t STATIC_INT_X_BFLOAT16_BIAS_QUANT_D = 10011;

const int64_t DYNAMIC_FLOAT16_X = 30009;
const int64_t DYNAMIC_BFLOAT16_X = 30011;
const int64_t DYNAMIC_FLOAT16_XD = 30010;
const int64_t DYNAMIC_BFLOAT16_XD = 30012;
const int64_t DYNAMIC_INT_X_INT_BIAS_QUANT_ONE = 30001;
const int64_t DYNAMIC_INT_X_INT_BIAS_QUANT_D = 30005;
const int64_t DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_ONE = 30003;
const int64_t DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_D = 30007;
const int64_t DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_ONE = 30002;
const int64_t DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_D = 30006;
const int64_t DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_ONE = 30004;
const int64_t DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_D = 30008;

static const size_t INDEX_IN_WEIGHT_SCALE = 1;
static const size_t INDEX_IN_ACTIVATE_SCALE = 2;
static const size_t INDEX_IN_BIAS = 3;
static const size_t INDEX_IN_QUANT_SCALE = 4;
static const size_t INDEX_IN_QUANT_OFFSET = 5;
static const size_t NUMBER_OF_INPUT_SIZE = 10;

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_BF16};
static const std::map<std::string, int64_t> SUPPORT_QUANT_MODE = {{ "dynamic", 1}, {"static", 0}};

const int64_t DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_D_PERFORMANCE = 30013;

// ============================================================
// 统一 TilingData 结构体定义
// ============================================================

BEGIN_TILING_DATA_DEF(DequantSwigluClampQuantTilingData)
  // ===== Group 模式字段 =====
  TILING_DATA_FIELD_DEF(int64_t, inDimx);           // 输入行数（token数）
  TILING_DATA_FIELD_DEF(int64_t, inDimy);           // 输入尾轴长度（2H）
  TILING_DATA_FIELD_DEF(int64_t, outDimy);          // 输出尾轴长度（H）
  TILING_DATA_FIELD_DEF(int64_t, UbFactorDimx);     // UB内一次处理的行数
  TILING_DATA_FIELD_DEF(int64_t, UbFactorDimy);     // UB内一次处理的尾轴元素数
  TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);      // 实际使用的核数
  TILING_DATA_FIELD_DEF(int64_t, maxCoreNum);       // 最大可用核数
  TILING_DATA_FIELD_DEF(int64_t, inGroupNum);       // group数量（MoE场景）
  TILING_DATA_FIELD_DEF(int64_t, hasBias);          // 是否有bias输入
  TILING_DATA_FIELD_DEF(int64_t, quantMode);        // 量化模式：0=static, 1=dynamic
  TILING_DATA_FIELD_DEF(int64_t, actRight);         // SwiGLU激活与门控排布：1=右半部为激活
  TILING_DATA_FIELD_DEF(int64_t, quantScaleDtype);  // quant_scale数据类型编码
  TILING_DATA_FIELD_DEF(int64_t, groupIndexDtype);  // group_index数据类型编码
  TILING_DATA_FIELD_DEF(int64_t, needSmoothScale);  // 是否需要smooth scale
  TILING_DATA_FIELD_DEF(int64_t, biasDtype);        // bias数据类型编码
  TILING_DATA_FIELD_DEF(int64_t, speGroupType);     // group_index是否为2维
  TILING_DATA_FIELD_DEF(int64_t, activationScaleIsEmpty); // activation_scale是否为空
  TILING_DATA_FIELD_DEF(int64_t, quantIsOne);       // kernel侧计算时quant尾轴是否为单个元素
  // SwiGLU扩展参数（GPT-OSS场景）
  TILING_DATA_FIELD_DEF(int64_t, swigluMode);       // SwiGLU模式：0=左右排布, 1=奇偶排布
  TILING_DATA_FIELD_DEF(float, clampLimit);         // clamp限制值
  TILING_DATA_FIELD_DEF(float, gluAlpha);           // GLU alpha参数
  TILING_DATA_FIELD_DEF(float, gluBias);            // GLU bias参数

  // ===== 基础模式字段 =====
  TILING_DATA_FIELD_DEF(uint32_t, is32BAligned);
  TILING_DATA_FIELD_DEF(uint32_t, isDoubleBuffer);
  TILING_DATA_FIELD_DEF(uint64_t, rowLen);          // 行长度（对应 inDimx）
  TILING_DATA_FIELD_DEF(uint64_t, colLen);          // 列长度（对应 outDimy）
  TILING_DATA_FIELD_DEF(uint32_t, baseRowLen);      // 基础行切分（对应 UbFactorDimx）
  TILING_DATA_FIELD_DEF(uint32_t, baseColLen);      // 基础列切分（对应 UbFactorDimy）
  TILING_DATA_FIELD_DEF(uint32_t, activateLeft);    // 激活位置（actRight的反向）
  TILING_DATA_FIELD_DEF(uint32_t, biasIsEmpty);
  TILING_DATA_FIELD_DEF(uint32_t, quantScaleIsEmpty);
  TILING_DATA_FIELD_DEF(uint32_t, activateScaleIsEmpty);
  TILING_DATA_FIELD_DEF(uint64_t, swiColLen);
  TILING_DATA_FIELD_DEF(uint64_t, perRowLen);
  TILING_DATA_FIELD_DEF(uint64_t, modRowLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DequantSwigluClampQuant, DequantSwigluClampQuantTilingData)

// ============================================================
// 辅助函数
// ============================================================

template <typename T>
inline auto AlignUp(T num, T rnd) -> decltype(num)
{
  return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T>
inline auto AlignDown(T num, T rnd) -> decltype(num)
{
  return ((((rnd) == 0) || ((num) < (rnd))) ? 0 : ((num) / (rnd) * (rnd)));
}

template <typename T>
inline auto DivCeil(T num, T div) -> decltype(num)
{
  return (((div) == 0) ? 0 : (((num) + (div) - 1) / (div)));
}

inline bool GetLengthByType(int32_t dtype, uint32_t& dsize)
{
  switch (dtype) {
    case ge::DT_FLOAT16:
    case ge::DT_INT16:
    case ge::DT_UINT16:
    case ge::DT_BF16:
      dsize = sizeof(int16_t);
      return true;
    case ge::DT_FLOAT:
    case ge::DT_INT32:
    case ge::DT_UINT32:
      dsize = sizeof(int32_t);
      return true;
    case ge::DT_DOUBLE:
    case ge::DT_INT64:
    case ge::DT_UINT64:
      dsize = sizeof(int64_t);
      return true;
    default:
      return false;
  }
}

// ============================================================
// 编译时信息结构
// ============================================================

struct DequantSwigluClampQuantCompileInfo {
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
};

// ============================================================
// 基础模式 Tiling 优选参数结构
// ============================================================

struct GluSingleTilingOptParam {
  uint32_t maxTileLen = 0;
  uint32_t optBaseRowLen = 0;
  uint32_t optBaseColLen = 0;
  uint64_t optTotalTileNum = 0;
  uint64_t optBaseSize = 0;
  uint64_t optBaseTileNum = 0;
  uint32_t totalUsedCoreNum = 0;
  uint64_t tileNumPerCore = 0;
};

// ============================================================
// 统一 Tiling 类定义（不继承 TilingBaseClass）
// ============================================================

class DequantSwigluClampQuantTiling {
public:
  explicit DequantSwigluClampQuantTiling(gert::TilingContext* context);
  ~DequantSwigluClampQuantTiling() = default;

  ge::graphStatus DoTiling();  // 主入口

private:
  // ===== 策略判断 =====
  bool IsGroupMode();
  bool IsRegbaseSoc();

  // ===== 公共方法 =====
  ge::graphStatus GetPlatformInfo();
  void DumpTilingInfo();

  // ===== Group 模式分支 =====
  ge::graphStatus DoTilingGroupMode();
  ge::graphStatus GetAttrGroupMode();
  ge::graphStatus GetShapeAttrsInfoGroupMode();
  ge::graphStatus DoOpTilingGroupMode();
  uint64_t GetTilingKeyGroupMode() const;
  ge::graphStatus GetWorkspaceSizeGroupMode();
  ge::graphStatus PostTilingGroupMode();

  // Group 模式专用方法
  static bool CheckOptionalShapeExisting(const gert::StorageShape* storageShape);
  ge::graphStatus CheckXAndGroupIndexDtype();
  ge::graphStatus CheckBias();
  ge::graphStatus CheckWeightScale();
  ge::graphStatus CheckActivationScale();
  ge::graphStatus CheckForDequant();
  ge::graphStatus CheckForQuant();
  ge::graphStatus CheckForDynamicQuant();
  ge::graphStatus CheckForStaticQuant();
  ge::graphStatus CheckQuantScaleDtype();
  ge::graphStatus CheckStaticQuantShape(const int64_t quantInputIdx, int64_t& colLen);
  ge::graphStatus CheckIllegalParam();
  void CountTilingKey();
  ge::graphStatus CountMaxDim(int64_t& ubFactorDimx);
  ge::graphStatus CheckScaleShapeWithDim(const int64_t scaleInputIdx, const int64_t expectDim);
  bool IsPerformanceAndGroupIndexBrach();
  ge::graphStatus GetShapeAttrsInfoGroupModeInner();

  // ===== 基础模式分支 =====
  ge::graphStatus DoTilingBaseMode();
  ge::graphStatus GetShapeAttrsInfoBaseMode();
  ge::graphStatus DoOpTilingBaseMode();
  uint64_t GetTilingKeyBaseMode() const;
  ge::graphStatus GetWorkspaceSizeBaseMode();
  ge::graphStatus PostTilingBaseMode();

  // 基础模式专用方法
  void ShowTilingData();
  ge::graphStatus checkInputShape(gert::TilingContext* context, ge::DataType xDataType);
  ge::graphStatus checkWeightBiasActivate(gert::TilingContext* context);
  ge::graphStatus SetTotalShape(gert::TilingContext* cont, const gert::Shape& inShape);
  bool SetAttr(const gert::RuntimeAttrs* attrs);
  bool CalcTiling(const uint32_t totalCores, const uint64_t ubSize, const platform_ascendc::SocVersion socVersion_);
  bool CalcOptTiling(const uint64_t ubSize, const int32_t dtype, GluSingleTilingOptParam& optTiling);
  bool CalcUbMaxTileLen(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam& optTiling);
  bool GetBufferNumAndDataLenPerUB(uint64_t ubSize, int32_t dtype, uint64_t& dataLenPerUB);
  bool CalcOptBaseShape(GluSingleTilingOptParam& optTiling, int32_t dtype);
  uint32_t getBaseColLenUpBound(GluSingleTilingOptParam& optTiling);
  void SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam& optTiling);
  int64_t getTilingKeyDynamic(const int32_t inputDtype, const ge::DataType biasType, const int64_t scaleSize) const;
  bool isPerformanceBranch();
  int64_t getTilingKeyStatic(const int32_t inputDtype, const ge::DataType biasType, const int64_t scaleSize) const;
  ge::graphStatus GetShapeAttrsInfoBaseModeInner();

  // ===== 成员变量 =====
  gert::TilingContext* context_;
  bool isGroupMode_;
  uint64_t tilingKey_;
  uint64_t workspaceSize_;

  // 公共平台信息
  uint64_t coreNum_;
  uint64_t ubSize_;
  platform_ascendc::SocVersion socVersion_;

  // Group 模式成员变量
  int64_t groupNum_;
  bool hasGroupIndex_;
  bool speGroupType_;
  bool hasBias_;
  bool hasWeightScale_;
  bool hasActivationScale_;
  bool hasQuantScale_;
  bool hasQuantOffset_;
  int64_t quantMode_;
  int64_t actRight_;
  int64_t swigluMode_;
  float clampLimit_;
  float gluAlpha_;
  float gluBias_;
  int64_t maxPreCore_;
  int64_t inDimx_;
  int64_t inDimy_;
  int64_t outDimy_;

  // 基础模式成员变量
  uint32_t inputDTypeLen_;
  uint32_t totalCore_;
  uint32_t totalAvailableCore_;
  uint32_t totalUsedCoreNum_;
  uint32_t baseRowLen_;
  uint32_t baseColLen_;
  uint32_t maxTileLen_;
  uint32_t ubMinBlockLen_;
  uint32_t cacheLineLen_;
  uint32_t alignPackLen_;
  uint64_t rowLen_;
  uint64_t colLen_;
  uint64_t optTotalTileNum_;
  uint64_t optBaseSize_;
  uint64_t optBaseTileNum_;
  uint64_t quantScaleShapeSize_;
  ge::DataType xInputDataType_;
  ge::DataType biasDataType_;
  bool isPerfBranch_;

  // TilingData
  DequantSwigluClampQuantTilingData tilingData_;
};

// ============================================================
// 全局函数声明
// ============================================================

ge::graphStatus TilingForDequantSwigluQuant(gert::TilingContext* context);
ge::graphStatus TilingPrepareForDequantSwigluQuant(gert::TilingParseContext* context);

}  // namespace optiling

#endif  // DEQUANT_SWIGLU_CLAMP_QUANT_TILING_H