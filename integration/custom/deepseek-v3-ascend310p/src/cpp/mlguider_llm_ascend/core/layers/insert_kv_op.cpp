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
#include <cmath>
#include "insert_kv_op.h"
#include "logger.h"
#ifdef PLATFORM_910A
#else
#include "aclnn_mi_deepseek_v3_insert_cache_v2.h"
#include "aclnn_mi_deepseek_v3_insert_cache_v3.h"
#endif
#include "acl_utils.h"

void DeepSeekV3InsertCacheV3(aclrtStream& stream, void* compressedKv, void* keyPe, void* cCache, void* kPeCache, void* slotMapping, int m, int n1, int n2, int pn, int pl)
{

    std::vector<int64_t> ashape = {m, n1};
    std::vector<int64_t> bshape = {m, n2};
    std::vector<int64_t> cshape = {pn, n2 / 16, pl, 16};
    std::vector<int64_t> dshape = {pn, n1 / 16, pl, 16};
    int eshapeSize = m % pl == 0 ? m : (m / pl + 1) * pl;
    std::vector<int64_t> eshape = {eshapeSize};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT16;
    aclDataType cType = ACL_FLOAT16;
    aclDataType dType = ACL_FLOAT16;
    aclDataType eType = ACL_INT32;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;
    aclFormat eformat = ACL_FORMAT_ND;

    aclTensor* aclaten
        = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), compressedKv);
    aclTensor* aclbten
        = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), keyPe);
    aclTensor* aclcten
        = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), cCache);
    aclTensor* acldten
        = aclCreateTensor(dshape.data(), dshape.size(), dType, nullptr, 0, dformat, dshape.data(), dshape.size(), kPeCache);
    aclTensor* acleten
        = aclCreateTensor(eshape.data(), eshape.size(), eType, nullptr, 0, eformat, eshape.data(), eshape.size(), slotMapping);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIDeepseekV3InsertCacheV3GetWorkspaceSize(
        aclbten, aclaten, acleten, acldten, aclcten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMIDeepseekV3InsertCacheV3GetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMIDeepseekV3InsertCacheV3(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMIDeepseekV3InsertCacheV3 failed");
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

    LOG_DEBUG("ResRmsNorm success");
}
