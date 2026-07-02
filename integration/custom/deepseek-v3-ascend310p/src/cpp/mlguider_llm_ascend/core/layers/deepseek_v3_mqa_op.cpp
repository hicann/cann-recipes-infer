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

#include "deepseek_v3_mqa_op.h"
#include "acl_utils.h"
#include "aclnn_mi_deep_seek_v3_mqa.h"
#include "aclnn_mi_deep_seek_v3_mqa_tnd.h"
#include "logger.h"
#include "transpose_op.h"
#define ENABLE_BFA

void DeepSeekV3MQA(aclrtStream& stream, void* queryNope, void* queryRope, void* cCache, void* kPeCache, void* pageIds, void* cuSeqlensQuery, void* seqlensKv, void* outputPtr, int m,
    int n1, int n2, int bs, int mpn, int pl, int sn, int pn, float softmaxScale, bool isChunkPrefill)
{
    std::vector<int64_t> ashape = {m, bs, n1};
    std::vector<int64_t> bshape = {m, bs, n2};
    std::vector<int64_t> cshape = {mpn, n1 / 16, pl, 16};
    std::vector<int64_t> dshape = {mpn, n2 / 16, pl, 16};
    std::vector<int64_t> eshape = {sn, pn};
    std::vector<int64_t> fshape = {sn + 1};
    std::vector<int64_t> gshape = {sn};
    std::vector<int64_t> hshape = {m, bs, n1};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT16;
    aclDataType cType = ACL_FLOAT16;
    aclDataType dType = ACL_FLOAT16;
    aclDataType eType = ACL_INT32;
    aclDataType fType = ACL_INT32;
    aclDataType gType = ACL_INT32;
    aclDataType hType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;
    aclFormat eformat = ACL_FORMAT_ND;
    aclFormat fformat = ACL_FORMAT_ND;
    aclFormat gformat = ACL_FORMAT_ND;
    aclFormat hformat = ACL_FORMAT_ND;

    aclTensor* aclaten
        = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), queryNope);
    aclTensor* aclbten
        = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), queryRope);
    aclTensor* aclcten
        = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), cCache);
    aclTensor* acldten
        = aclCreateTensor(dshape.data(), dshape.size(), dType, nullptr, 0, dformat, dshape.data(), dshape.size(), kPeCache);
    aclTensor* acleten
        = aclCreateTensor(eshape.data(), eshape.size(), eType, nullptr, 0, eformat, eshape.data(), eshape.size(), pageIds);
    aclTensor* aclften
        = aclCreateTensor(fshape.data(), fshape.size(), fType, nullptr, 0, fformat, fshape.data(), fshape.size(), cuSeqlensQuery);
    aclTensor* aclgten
        = aclCreateTensor(gshape.data(), gshape.size(), gType, nullptr, 0, gformat, gshape.data(), gshape.size(), seqlensKv);
    aclTensor* aclhten
        = aclCreateTensor(hshape.data(), hshape.size(), hType, nullptr, 0, hformat, hshape.data(), hshape.size(), outputPtr);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIDeepSeekV3MQATndGetWorkspaceSize(aclaten, aclbten, aclcten, acldten, acleten,
        aclften, aclgten, softmaxScale, isChunkPrefill, aclhten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMIDeepSeekV3MQATndGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMIDeepSeekV3MQATnd(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMIDeepSeekV3MQATnd failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclDestroyTensor(acldten);
    aclDestroyTensor(acleten);
    aclDestroyTensor(aclften);
    aclDestroyTensor(aclgten);
    aclDestroyTensor(aclhten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("RmsNormSplit success");
}
