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
#include "per_token_per_channel_dequant_op.h"
#include "aclnn_mi_per_token_per_channel_de_quant_i32.h"
#include "aclnn_mi_per_token_per_channel_batch_de_quant_i32.h"
#include "aclnn_mi_per_token_per_channel_group_de_quant_i32.h"
#include "logger.h"

void PerTokenPerChannelDequant(aclrtStream& stream, void* qx, void* wscale, void* xscale, void* outputPtr, int64_t num_tokens, int64_t dimension)
{
    std::vector<int64_t> ashape = {num_tokens, dimension};
    std::vector<int64_t> bshape = {num_tokens};
    std::vector<int64_t> cshape = {dimension};
    std::vector<int64_t> dshape = {num_tokens, dimension};

    aclDataType aType = ACL_INT32;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_FLOAT;
    aclDataType dType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), qx);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), xscale);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), wscale);
    aclTensor* acldten = aclCreateTensor(dshape.data(), dshape.size(), dType, nullptr, 0, dformat, dshape.data(), dshape.size(), outputPtr);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIPerTokenPerChannelDeQuantI32GetWorkspaceSize(aclaten, aclbten, aclcten, num_tokens, dimension, acldten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMIPerTokenQuantI8GetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMIPerTokenPerChannelDeQuantI32(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMIPerTokenQuantI8 failed");
    }

    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclDestroyTensor(acldten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("aclnnMIPerTokenQuantI8 op success");
}


void PerTokenPerChannelBatchDeQuantI32(aclrtStream& stream, void* qx, void* wscale, void* xscale, void* outputPtr, int64_t num_tokens, int64_t dimension, int64_t batchSize)
{
    std::vector<int64_t> qxShape{batchSize, num_tokens, dimension};
    std::vector<int64_t> wscaleShape{batchSize, num_tokens};
    std::vector<int64_t> xscaleShape{batchSize, dimension};
    std::vector<int64_t> x_outShape{batchSize, num_tokens, dimension};

    aclDataType aType = ACL_INT32;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_FLOAT;
    aclDataType dType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(qxShape.data(), qxShape.size(), aType, nullptr, 0, aformat, qxShape.data(), qxShape.size(), qx);
    aclTensor* aclbten = aclCreateTensor(wscaleShape.data(), wscaleShape.size(), bType, nullptr, 0, bformat, wscaleShape.data(), wscaleShape.size(), xscale);
    aclTensor* aclcten = aclCreateTensor(xscaleShape.data(), xscaleShape.size(), cType, nullptr, 0, cformat, xscaleShape.data(), xscaleShape.size(), wscale);
    aclTensor* acldten = aclCreateTensor(x_outShape.data(), x_outShape.size(), dType, nullptr, 0, dformat, x_outShape.data(), x_outShape.size(), outputPtr);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMiPerTokenPerChannelBatchDeQuantI32GetWorkspaceSize(aclaten, aclbten, aclcten, num_tokens, dimension, acldten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMiPerTokenPerChannelBatchDeQuantI32GetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMiPerTokenPerChannelBatchDeQuantI32(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMiPerTokenPerChannelBatchDeQuantI32 failed");
    }

    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclDestroyTensor(acldten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("aclnnMiPerTokenPerChannelBatchDeQuantI32 op success");
}


void PerTokenPerChannelGroupDeQuantI32(aclrtStream& stream, void* qx, void* wscale, void* xscale, void* groupList, void* outputPtr, int64_t num_tokens, int64_t dimension, int64_t cur_rank_experts)
{
    std::vector<int64_t> qxShape{num_tokens, dimension};
    std::vector<int64_t> wscaleShape{cur_rank_experts, dimension};
    std::vector<int64_t> xscaleShape{num_tokens, 1};
    std::vector<int64_t> groupListShape{cur_rank_experts};
    std::vector<int64_t> x_outShape{num_tokens, dimension};

    aclDataType aType = ACL_INT32;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_FLOAT;
    aclDataType dType = ACL_INT32;
    aclDataType eType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;
    aclFormat eformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(qxShape.data(), qxShape.size(), aType, nullptr, 0, aformat, qxShape.data(), qxShape.size(), qx);
    aclTensor* aclbten = aclCreateTensor(wscaleShape.data(), wscaleShape.size(), bType, nullptr, 0, bformat, wscaleShape.data(), wscaleShape.size(), wscale);
    aclTensor* aclcten = aclCreateTensor(xscaleShape.data(), xscaleShape.size(), cType, nullptr, 0, cformat, xscaleShape.data(), xscaleShape.size(), xscale);
    aclTensor* acldten = aclCreateTensor(groupListShape.data(), groupListShape.size(), dType, nullptr, 0, dformat, groupListShape.data(), groupListShape.size(), groupList);
    aclTensor* acleten = aclCreateTensor(x_outShape.data(), x_outShape.size(), eType, nullptr, 0, eformat, x_outShape.data(), x_outShape.size(), outputPtr);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIPerTokenPerChannelGroupDeQuantI32GetWorkspaceSize(aclaten, aclcten, aclbten, acldten, num_tokens, dimension, cur_rank_experts, acleten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMIPerTokenPerChannelGroupDeQuantI32GetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMIPerTokenPerChannelGroupDeQuantI32(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMIPerTokenPerChannelGroupDeQuantI32 failed");
    }

    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclDestroyTensor(acldten);
    aclDestroyTensor(acleten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("aclnnMIPerTokenPerChannelGroupDeQuantI32 op success");
}
