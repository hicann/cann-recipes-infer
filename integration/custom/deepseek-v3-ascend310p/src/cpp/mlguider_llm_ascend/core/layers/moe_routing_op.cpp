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
#include "moe_routing_op.h"
#ifdef PLATFORM_910A
#else
#include "aclnn_mi_moe_routing.h"
#endif
#include "logger.h"

void MoeRoutingV1(aclrtStream& stream, void* gatherTokens, void* topkWeights, void* des2src, void* tokensPerExpert, void* output, int m, int n, int topk, int expertNum)
{
    if (m % topk != 0)
    {
        LOG_ERROR("m %% topk != 0");
        throw std::runtime_error("m %% topk != 0");
    }

    if (m < topk)
    {
        LOG_ERROR("m < topk");
        throw std::runtime_error("m < topk");
    }

    int tokenNum = m / topk;

    std::vector<int64_t> ashape = {m, n};
    std::vector<int64_t> bshape = {tokenNum, topk};
    std::vector<int64_t> cshape = {m};
    std::vector<int64_t> dshape = {expertNum};
    std::vector<int64_t> eshape = {tokenNum, n};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_INT32;
    aclDataType dType = ACL_INT32;
    aclDataType eType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;
    aclFormat eformat = ACL_FORMAT_ND;

    aclTensor* aclaten
        = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), gatherTokens);
    aclTensor* aclbten
        = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), topkWeights);
    aclTensor* aclcten
        = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), des2src);
    aclTensor* acldten
        = aclCreateTensor(dshape.data(), dshape.size(), dType, nullptr, 0, dformat, dshape.data(), dshape.size(), tokensPerExpert);
    aclTensor* acleten
        = aclCreateTensor(eshape.data(), eshape.size(), eType, nullptr, 0, eformat, eshape.data(), eshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIMoeRoutingGetWorkspaceSize(
        aclaten, aclbten, aclcten, acldten, expertNum, acleten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMIMoeRoutingGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMIMoeRouting(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMIMoeRouting failed");
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

    LOG_DEBUG("MoeRouting success");
}
