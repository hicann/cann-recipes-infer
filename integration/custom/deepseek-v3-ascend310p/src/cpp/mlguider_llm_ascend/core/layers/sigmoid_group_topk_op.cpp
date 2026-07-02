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
#include "sigmoid_group_topk_op.h"
#include "acl_utils.h"
#ifdef PLATFORM_910A
#else
#include "aclnn_sigmoid_group_top_kv2.h"
#endif
#include "logger.h"

void DeepSeekV3SigmoidGroupTopkV2(aclrtStream& stream, void* input,
    void* beScoreCorrectionBias, void* topkExpertIdsPerToken, void* topkExpertScoresPerToken, void* encodedTokenIds,
    int m, int n, int64_t nGroup, int64_t topkGroup, int64_t topk, int64_t curRankBeginExpertId,
    int64_t curRankEndExpertId, float routingScalingFactor)
{
    std::vector<int64_t> ashape = {m, n};
    std::vector<int64_t> bshape = {n};
    std::vector<int64_t> cshape = {m, topk};
    std::vector<int64_t> dshape = {m, topk};
    std::vector<int64_t> eshape = {m, topk};

    aclDataType aType = ACL_FLOAT;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_INT32;
    aclDataType dType = ACL_FLOAT;
    aclDataType eType = ACL_INT32;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;
    aclFormat eformat = ACL_FORMAT_ND;

    aclTensor* aclaten
        = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), input);
    aclTensor* aclbten
        = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), beScoreCorrectionBias);
    aclTensor* aclcten
        = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), topkExpertIdsPerToken);
    aclTensor* acldten
        = aclCreateTensor(dshape.data(), dshape.size(), dType, nullptr, 0, dformat, dshape.data(), dshape.size(), topkExpertScoresPerToken);
    aclTensor* acleten
        = aclCreateTensor(eshape.data(), eshape.size(), eType, nullptr, 0, eformat, eshape.data(), eshape.size(), encodedTokenIds);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnSigmoidGroupTopKV2GetWorkspaceSize(aclaten, aclbten, nGroup, topkGroup,
        routingScalingFactor, topk, curRankBeginExpertId, curRankEndExpertId, aclcten,
        acldten, acleten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnSigmoidGroupTopKV2GetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnSigmoidGroupTopKV2(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnSigmoidGroupTopKV2 failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclDestroyTensor(acldten);
    aclDestroyTensor(acleten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("SigmoidGroupTopK success");
}
