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
#include "scaled_softmax.h"
#include "acl_utils.h"
#include "acl/acl.h"
#include "aclnn_scaled_softmax.h"
#include "aclnnop/aclnn_softmax.h"
#include "scaled_op.h"
#include "logger.h"

void ScaledSoftmax(aclrtStream& stream, void* input, void* output, int64_t M, int64_t N, double scale)
{
    std::vector<int64_t> inputshape = {M, N};
    std::vector<int64_t> outputshape = {M, N};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;

    aclTensor* aclaten
        = aclCreateTensor(inputshape.data(), inputshape.size(), aType, nullptr, 0, aformat, inputshape.data(), inputshape.size(), input);
    aclTensor* aclbten
        = aclCreateTensor(outputshape.data(), outputshape.size(), bType, nullptr, 0, bformat, outputshape.data(), outputshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnScaledSoftmaxGetWorkspaceSize(aclaten, static_cast<float>(scale), aclbten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnScaledSoftmaxGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnScaledSoftmax(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnScaledSoftmax failed");
    }

    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("ScaledSoftmax success");
}

void Softmax(aclrtStream& stream, void* input, void* output, int64_t M, int64_t N)
{
    std::vector<int64_t> inputshape = {M, N};
    std::vector<int64_t> outputshape = {M, N};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;

    aclTensor* aclaten
        = aclCreateTensor(inputshape.data(), inputshape.size(), aType, nullptr, 0, aformat, inputshape.data(), inputshape.size(), input);
    aclTensor* aclbten
        = aclCreateTensor(outputshape.data(), outputshape.size(), bType, nullptr, 0, bformat, outputshape.data(), outputshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnSoftmaxGetWorkspaceSize(aclaten, 1, aclbten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnSoftmaxGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnSoftmax(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnSoftmax failed");
    }

    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("Softmax success");
}

void FusedScaledSoftmax(aclrtStream& stream, void* input, void* scalePtr, void* scaleOutPtr, void* output, int64_t M, int64_t N)
{
}
