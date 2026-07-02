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
#include "quant_op.h"
#include "acl_utils.h"
#include "aclnn_mi_batch_dequantize.h"
#include "aclnn_mi_batch_quant.h"
#include "aclnn_mi_group_quant.h"
#include "aclnn_de_quantize.h"
#include "aclnnop/aclnn_quantize.h"
#include "logger.h"

void QuantizeFP16ToInt8(aclrtStream& stream, void *input, void *weight, void *output, int m, int n)
{
    std::vector<int64_t> ashape = {m, n};
    std::vector<int64_t> bshape = {n};
    std::vector<int64_t> cshape = {m, n};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_INT8;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;

    int32_t axis = 1;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), input);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), weight);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnQuantizeGetWorkspaceSize(aclaten, aclbten, nullptr, ACL_INT8, axis, aclcten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnQuantizeGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnQuantize(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnQuantize failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("QuantizeFP16ToInt8 op success");
}

void GroupQuantizeFP16ToInt8(aclrtStream& stream, void *input, void *weight, void *groupList, void *output, int m, int n, int g)
{
    std::vector<int64_t> ashape = {m, n};
    std::vector<int64_t> bshape = {g, n};
    std::vector<int64_t> cshape = {g};
    std::vector<int64_t> dshape = {m, n};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_INT32;
    aclDataType dType = ACL_INT8;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), input);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), weight);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), groupList);
    aclTensor* acldten = aclCreateTensor(dshape.data(), dshape.size(), dType, nullptr, 0, dformat, dshape.data(), dshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIGroupQuantGetWorkspaceSize(aclaten, aclbten, aclcten, m, n, g, acldten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMIGroupQuantGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMIGroupQuant(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMIGroupQuant failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclDestroyTensor(acldten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("GroupQuantizeFP16ToInt8 op success");
}

void BatchQuantizeFP16ToInt8(aclrtStream &stream, void *input, void *weight, void *output, int m, int n, int bs)
{
    std::vector<int64_t> ashape = {bs, m, n};
    std::vector<int64_t> bshape = {bs, n};
    std::vector<int64_t> cshape = {bs, m, n};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_INT8;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), input);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), weight);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIBatchQuantGetWorkspaceSize(aclaten, aclbten, aclcten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMIBatchQuantGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    ret = aclrtMalloc(&workspaceAddr, workspaceSize+36, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnMIBatchQuant(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMIBatchQuant failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclrtFree(workspaceAddr);

    LOG_DEBUG("QuantizeFP16ToInt8 op success");
}

void DequantizeInt32ToFp16(aclrtStream& stream, void* input, void* weight, void* output, int m, int n)
{
    std::vector<int64_t> ashape = {m, n};
    std::vector<int64_t> bshape = {n};
    std::vector<int64_t> cshape = {m, n};

    aclDataType aType = ACL_INT32;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), input);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), weight);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnDeQuantizeGetWorkspaceSize(aclaten, aclbten, aclcten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMiMatmulDequantGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnDeQuantize(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMiMatmulDequant failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclrtFree(workspaceAddr);

    LOG_DEBUG("DequantizeInt32ToFp16 op success");
}

void BatchDequantizeInt32ToFp16(aclrtStream& stream, void *input, void *weight, void* output, int m, int n, int bs)
{
    std::vector<int64_t> ashape = {bs, m, n};
    std::vector<int64_t> bshape = {bs, n};
    std::vector<int64_t> cshape = {bs, m, n};

    aclDataType aType = ACL_INT32;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), input);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), weight);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIBatchDequantizeGetWorkspaceSize(aclaten, aclbten, aclcten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMIBatchDequantizeGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize+36, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMIBatchDequantize(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMIBatchDequantize failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("QuantizeFP16ToInt8 op success");
}
