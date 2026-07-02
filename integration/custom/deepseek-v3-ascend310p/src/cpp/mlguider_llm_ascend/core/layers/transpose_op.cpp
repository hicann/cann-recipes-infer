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
#include "transpose_op.h"
#include "aclnnop/aclnn_permute.h"
#include "logger.h"
#include "acl_utils.h"

void TransposeFP16(aclrtStream& stream, void* a, void* b, std::vector<int64_t> ashape, std::vector<int64_t> dims)
{
    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;

    aclTensor* aclaten =  aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), a);
    aclIntArray* aclDims = aclCreateIntArray(dims.data(), dims.size());
    std::vector<int64_t> bshape;
    int bSize = 1;
    for (int i = 0; i < dims.size(); i++)
    {
        bshape.push_back(ashape[dims[i]]);
        bSize *= ashape[dims[i]];
    }

    aclTensor* aclbten =  aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), b);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnPermuteGetWorkspaceSize(aclaten, aclDims, aclbten, &workspaceSize, &handle);
    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("Get aclnnPermuteGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnPermute(workspaceAddr, workspaceSize, handle, stream);
    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnPermute failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("Transpose success");
}

void TransposeI8(aclrtStream& stream, void* a, void* b, std::vector<int64_t> ashape, std::vector<int64_t> dims)
{
    aclDataType aType = ACL_INT8;
    aclDataType bType = ACL_INT8;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;

    aclTensor* aclaten =  aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), a);
    aclIntArray* aclDims = aclCreateIntArray(dims.data(), dims.size());
    std::vector<int64_t> bshape;
    int bSize = 1;
    for (int i = 0; i < dims.size(); i++)
    {
        bshape.push_back(ashape[dims[i]]);
        bSize *= ashape[dims[i]];
    }

    aclTensor* aclbten =  aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), b);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnPermuteGetWorkspaceSize(aclaten, aclDims, aclbten, &workspaceSize, &handle);
    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("Get aclnnPermuteGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
     ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnPermute(workspaceAddr, workspaceSize, handle, stream);
    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnPermute failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("Transpose success");
}
