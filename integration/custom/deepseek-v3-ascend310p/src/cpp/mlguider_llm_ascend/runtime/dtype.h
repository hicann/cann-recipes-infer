/**
 * Copyright (c) 2025 QINGMAO INTELLIGENCE TECHNOLOGY (BEIJING) CO., LTD. and Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "acl/acl.h"
#include "hccl/hccl_types.h"
#include <cmath>
#include <cstddef> // for size_t
#include <cstdint>
#include <cstring>
#include <string>

enum class Device
{
    CPU,
    ASCEND,
    CUDA,
    UNKNOWN
};

struct DType
{
    enum class Type
    {
        FLOAT32,
        FLOAT16,
        BFLOAT16,
        INT32,
        UINT32,
        INT64,
        UINT64,
        INT8,
        UINT8,
        BOOL,
        UNKNOWN
    };

    Type type;
    aclDataType aclType;
    HcclDataType hcclType;
    size_t size;

    static const DType FLOAT32;
    static const DType FLOAT16;
    static const DType BFLOAT16;
    static const DType INT32;
    static const DType UINT32;
    static const DType INT64;
    static const DType UINT64;
    static const DType INT8;
    static const DType UINT8;
    static const DType BOOL;
    static const DType UNKNOWN;

    bool operator==(const DType& other) const
    {
        return type == other.type && size == other.size;
    }

    template <typename T>
    static constexpr DType FromType();

    static DType FromString(std::string inputDtype)
    {
        if (inputDtype == "F32")
            return FLOAT32;
        else if (inputDtype == "F16")
            return FLOAT16;
        else if (inputDtype == "I8")
            return INT8;
        else
            return UNKNOWN;
    }

    std::string ToString() const
    {
        switch (type)
        {
            case Type::FLOAT32: return "float32";
            case Type::FLOAT16: return "float16";
            case Type::BFLOAT16: return "bfloat16";
            case Type::INT32: return "int32";
            case Type::UINT32: return "uint32";
            case Type::INT64: return "int64";
            case Type::UINT64: return "uint64";
            case Type::INT8: return "int8";
            case Type::UINT8: return "uint8";
            case Type::BOOL: return "bool";
            default: return "unknown";
        }
    }
};


inline constexpr DType DType::FLOAT32 = {DType::Type::FLOAT32, aclDataType::ACL_FLOAT, HCCL_DATA_TYPE_FP32, sizeof(float)};
inline constexpr DType DType::FLOAT16 = {DType::Type::FLOAT16, aclDataType::ACL_FLOAT16, HCCL_DATA_TYPE_FP16, 2};
inline constexpr DType DType::BFLOAT16 = {DType::Type::BFLOAT16, aclDataType::ACL_BF16, HCCL_DATA_TYPE_BFP16, 2};
inline constexpr DType DType::INT32 = {DType::Type::INT32, aclDataType::ACL_INT32, HCCL_DATA_TYPE_INT32, sizeof(int32_t)};
inline constexpr DType DType::UINT32 = {DType::Type::UINT32, aclDataType::ACL_UINT32, HCCL_DATA_TYPE_UINT32, sizeof(uint32_t)};
inline constexpr DType DType::INT64 = {DType::Type::INT64, aclDataType::ACL_INT64, HCCL_DATA_TYPE_INT64, sizeof(int64_t)};
inline constexpr DType DType::UINT64 = {DType::Type::UINT64, aclDataType::ACL_UINT64, HCCL_DATA_TYPE_UINT64, sizeof(uint64_t)};
inline constexpr DType DType::INT8 = {DType::Type::INT8, aclDataType::ACL_INT8, HCCL_DATA_TYPE_INT8, sizeof(int8_t)};
inline constexpr DType DType::UINT8 = {DType::Type::UINT8, aclDataType::ACL_UINT8, HCCL_DATA_TYPE_UINT8, sizeof(uint8_t)};
inline constexpr DType DType::BOOL = {DType::Type::BOOL, aclDataType::ACL_BOOL, HCCL_DATA_TYPE_INT8, sizeof(bool)};
inline constexpr DType DType::UNKNOWN = {DType::Type::UNKNOWN, aclDataType::ACL_DT_UNDEFINED, HCCL_DATA_TYPE_RESERVED, 0};

template <>
constexpr DType DType::FromType<float>()
{
    return DType::FLOAT32;
}

template <>
constexpr DType DType::FromType<int32_t>()
{
    return DType::INT32;
}

template <>
constexpr DType DType::FromType<uint32_t>()
{
    return DType::UINT32;
}

template <>
constexpr DType DType::FromType<int64_t>()
{
    return DType::INT64;
}

template <>
constexpr DType DType::FromType<uint64_t>()
{
    return DType::UINT64;
}

template <>
constexpr DType DType::FromType<int8_t>()
{
    return DType::INT8;
}

template <>
constexpr DType DType::FromType<uint8_t>()
{
    return DType::UINT8;
}

template <>
constexpr DType DType::FromType<bool>()
{
    return DType::BOOL;
}

template <typename T>
constexpr DType DType::FromType()
{
    return DType::UNKNOWN;
}

static uint32_t AsUint(const float x)
{
    return *(uint32_t*) &x;
}

static float AsFloat(const uint32_t x)
{
    return *(float*) &x;
}

static uint16_t FloatToHalf(const float x)
{ // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5,
  // +-5.9604645E-8, 3.311 digits
    const uint32_t b = AsUint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const uint32_t e = (b & 0x7F800000) >> 23;  // exponent
    const uint32_t m = b & 0x007FFFFF;          // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal
                                                // indicator flag - initial rounding
    return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13)
        | ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1)
        | (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}


/**
 * @brief Convert half-precision float (FLOAT16) to single-precision float (FLOAT32).
 * HalfToFloat implementation:
 * Sign bit:      MSB indicates the sign.
 * Exponent bits: Middle 5 bits, bias = 15.
 * Mantissa bits: Lower 10 bits.
 * Special cases handled:
 * Denormals:  exponent == 0 and mantissa != 0.
 * Zero:       exponent and mantissa both == 0.
 * Infinity / NaN: exponent == 31; mantissa == 0 ⇒ Inf, mantissa != 0 ⇒ NaN.
*/
static float HalfToFloat(uint16_t half)
{
    uint16_t sign = (half & 0x8000) >> 15;
    uint16_t exponent = (half & 0x7C00) >> 10;
    uint16_t mantissa = half & 0x03FF;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            return sign ? -0.0f : 0.0f;
        }
        else
        {
            float value = mantissa / 1024.0f; // 1024 = 2^10
            return sign ? -value * std::pow(2, -14) : value * std::pow(2, -14); // ± mantissa * 2^(-24)
        }
    }

    if (exponent == 31) // Inf or NaN
    {
        if (mantissa == 0)
        {
            return sign ? -INFINITY : INFINITY; // + inf
        }
        else
        {
            return NAN; // NaN
        }
    }

    float value = (1.0f + mantissa / 1024.0f) * std::pow(2, exponent - 15);
    return sign ? -value : value;
}

/**
 * @brief Convert BFLOAT16 to single-precision float (FLOAT32).
 * BfloatToFloat implementation:
 * BFLOAT16 is a truncated version of FLOAT32, keeping only the upper 16 bits.
 * Shift those 16 bits to the upper half of a FLOAT32 and fill the lower 16 bits with zeros.
 * Use std::memcpy to reinterpret the bit pattern as a float.
*/
static float BfloatToFloat(uint16_t bfloat)
{
    uint32_t floatBits = static_cast<uint32_t>(bfloat) << 16;
    float result;
    std::memcpy(&result, &floatBits, sizeof(float));
    return result;
}

struct DataFormat
{
    enum class Format
    {
        ND,
        NCHW,
        NHWC,
        FRACTAL_NZ,
        UNKNOWN
    };

    Format format;
    aclFormat aclfmt; // ACL format

    static const DataFormat ND;
    static const DataFormat NCHW;
    static const DataFormat NHWC;
    static const DataFormat UNKNOWN;
    static const DataFormat FRACTAL_NZ;

    bool operator==(const DataFormat& other) const
    {
        return format == other.format && aclfmt == other.aclfmt;
    }

    bool operator!=(const DataFormat& other) const
    {
        return !(*this == other);
    }

    std::string ToString() const
    {
        switch (format)
        {
            case Format::ND: return "ND";
            case Format::NCHW: return "NCHW";
            case Format::NHWC: return "NHWC";
            case Format::FRACTAL_NZ: return "FRACTAL_NZ";
            default: return "UNKNOWN";
        }
    }
};

inline constexpr DataFormat DataFormat::ND = {DataFormat::Format::ND, aclFormat::ACL_FORMAT_ND};
inline constexpr DataFormat DataFormat::FRACTAL_NZ = {DataFormat::Format::ND, aclFormat::ACL_FORMAT_FRACTAL_NZ};
inline constexpr DataFormat DataFormat::NCHW = {DataFormat::Format::NCHW, ACL_FORMAT_NCHW};
inline constexpr DataFormat DataFormat::NHWC = {DataFormat::Format::NHWC, ACL_FORMAT_NHWC};
inline constexpr DataFormat DataFormat::UNKNOWN = {DataFormat::Format::UNKNOWN, ACL_FORMAT_UNDEFINED};
