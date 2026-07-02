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
#include "acl/acl.h"
#include "acl_utils.h"
#include "aclnnop/aclnn_multinomial.h"
#include "multinomial.h"
#include "logger.h"

void Multinomial(aclrtStream& stream, void* input, void* output, int64_t m, int64_t n, int64_t numsamples, bool replacement, int64_t seed, int64_t offset)
{
    std::vector<int64_t> inputshape = {m, n};
    std::vector<int64_t> outputshape = {m, numsamples};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_INT64;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;

    aclTensor* aclaten
        = aclCreateTensor(inputshape.data(), inputshape.size(), aType, nullptr, 0, aformat, inputshape.data(), inputshape.size(), input);
    aclTensor* aclbten
        = aclCreateTensor(outputshape.data(), outputshape.size(), bType, nullptr, 0, bformat, outputshape.data(), outputshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMultinomialGetWorkspaceSize(aclaten, numsamples, replacement, seed, offset, aclbten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMultinomialGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMultinomial(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMultinomial failed");
    }

    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("Multinomial success");
}
