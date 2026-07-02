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
#include "per_token_quant_op.h"
#include "aclnn_mi_per_token_quant_i8.h"
#include "logger.h"

void PerTokenQuant(aclrtStream& stream, void* input, void* qOut, void* qScale, int numTokens, int dimension)
{
    std::vector<int64_t> ashape = {numTokens, dimension};
    std::vector<int64_t> bshape = {numTokens, dimension};
    std::vector<int64_t> cshape = {numTokens};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_INT8;
    aclDataType cType = ACL_FLOAT;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), input);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), qOut);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), qScale);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIPerTokenQuantI8GetWorkspaceSize(aclaten, numTokens, dimension, 127, aclbten, aclcten, &workspaceSize, &handle);

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
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnMIPerTokenQuantI8(workspaceAddr, workspaceSize, handle, stream);

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
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("aclnnMIPerTokenQuantI8 op success");
}
