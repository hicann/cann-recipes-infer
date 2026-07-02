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

#include "expert_token_dispatcher_op.h"
#include "acl_utils.h"
#include "aclnn_mi_expert_token_dispatcher.h"
#include "logger.h"
constexpr int CORE_NUM_310P3 = 8; // core num of 310P3
constexpr int DEFAULT_SYNCALL_NEED_SIZE = 20; // sync all need 8 cores

void ExpertTokenDispatcher(aclrtStream& stream, void* hiddenStates, void* topkExpertIdsPerToken, void* encodedTokenIds, void* gatherTokens,
    void* tokensPerExpert, void* des2src, int m, int n, int topk, int64_t curRankExpertNum)
{
    std::vector<int64_t> ashape = {m*n};
    std::vector<int64_t> bshape = {m*topk};
    std::vector<int64_t> cshape = {m*topk};
    std::vector<int64_t> dshape = {curRankExpertNum};
    std::vector<int64_t> eshape = {m*n*topk};
    std::vector<int64_t> fshape = {m*topk};

    int64_t dictLength = m * topk;

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_INT32;
    aclDataType cType = ACL_INT32;
    aclDataType dType = ACL_FLOAT16;
    aclDataType eType = ACL_FLOAT;
    aclDataType fType = ACL_FLOAT;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;
    aclFormat eformat = ACL_FORMAT_ND;
    aclFormat fformat = ACL_FORMAT_ND;

    aclTensor* aclaten
        = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), hiddenStates);
    aclTensor* aclbten
        = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), topkExpertIdsPerToken);
    aclTensor* aclcten
        = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), encodedTokenIds);
    aclTensor* acldten
        = aclCreateTensor(dshape.data(), dshape.size(), dType, nullptr, 0, dformat, dshape.data(), dshape.size(), gatherTokens);
    aclTensor* acleten
        = aclCreateTensor(eshape.data(), eshape.size(), eType, nullptr, 0, eformat, eshape.data(), eshape.size(), tokensPerExpert);
    aclTensor* aclften
        = aclCreateTensor(fshape.data(), fshape.size(), fType, nullptr, 0, fformat, fshape.data(), fshape.size(), des2src);

    // output sync tensor
    int64_t syncLength = DEFAULT_SYNCALL_NEED_SIZE * CORE_NUM_310P3;
    void* g = nullptr;
    auto ret = aclrtMalloc(&g, syncLength * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemsetAsync(g, syncLength * sizeof(int32_t), 0, syncLength * sizeof(int32_t), stream);
    std::vector<int64_t> gshape = {syncLength};
    aclTensor* aclgten
        = aclCreateTensor(gshape.data(), gshape.size(), ACL_INT32, nullptr, 0, ACL_FORMAT_ND, gshape.data(), gshape.size(), g);

    size_t workspaceSize;
    aclOpExecutor* handle;
    ret = aclnnMiExpertTokenDispatcherGetWorkspaceSize(aclbten, aclcten,
        aclaten, dictLength, curRankExpertNum, m, topk, n, acleten, acldten,
        aclften, aclgten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMiExpertTokenDispatcherGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMiExpertTokenDispatcher(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMiExpertTokenDispatcher failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclDestroyTensor(acldten);
    aclDestroyTensor(acleten);
    aclDestroyTensor(aclften);
    aclDestroyTensor(aclgten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("ExpertTokenDispatcher success");
}
