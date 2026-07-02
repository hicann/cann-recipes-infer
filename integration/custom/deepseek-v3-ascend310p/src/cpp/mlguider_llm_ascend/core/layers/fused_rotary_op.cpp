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
#include "fused_rotary_op.h"
#include "acl_utils.h"
#include "aclnn_mi_bias_rotary_split.h"
#include "logger.h"

#ifdef PLATFORM_910A
#else
#include "aclnn_deep_seek_v3_apply_ro_pev2.h"
#include "aclnn_mi_bias_rotary_split_general.h"
#endif

void DeepSeekV3ApplyRoPeV2Inplace(
    aclrtStream& stream, void* query, void* key, void* positionIds, void* cosCache, void* sinCache, int m, int n, int bs, int v)
{
    std::vector<int64_t> ashape = {m, bs, n};
    std::vector<int64_t> bshape = {m, n};
    std::vector<int64_t> cshape = {m};
    std::vector<int64_t> dshape = {v, n / 2};
    std::vector<int64_t> eshape = {v, n / 2};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT16;
    aclDataType cType = ACL_INT32;
    aclDataType dType = ACL_FLOAT;
    aclDataType eType = ACL_FLOAT;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;
    aclFormat eformat = ACL_FORMAT_ND;

    aclTensor* aclaten
        = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), query);
    aclTensor* aclbten
        = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), key);
    aclTensor* aclcten
        = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), positionIds);
    aclTensor* acldten
        = aclCreateTensor(dshape.data(), dshape.size(), dType, nullptr, 0, dformat, dshape.data(), dshape.size(), cosCache);
    aclTensor* acleten
        = aclCreateTensor(eshape.data(), eshape.size(), eType, nullptr, 0, eformat, eshape.data(), eshape.size(), sinCache);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnDeepSeekV3ApplyRoPEV2GetWorkspaceSize(
        aclaten, aclbten, acldten, acleten, aclcten, aclaten, aclbten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnDeepSeekV3ApplyRoPEV2GetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnDeepSeekV3ApplyRoPEV2(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnDeepSeekV3ApplyRoPEV2 failed");
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

    LOG_DEBUG("DeepSeekV3ApplyRoPeV2Inplace success");
}
