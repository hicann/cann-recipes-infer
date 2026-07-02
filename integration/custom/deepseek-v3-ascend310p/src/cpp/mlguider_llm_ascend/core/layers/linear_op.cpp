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
#include "linear_op.h"
#include "acl_utils.h"
#include "aclnn_linear.h"
#include "aclnn_mi_matmul.h"
#include "aclnnop/aclnn_matmul.h"
#include "logger.h"
#include "transpose_op.h"

void LinearI8(aclrtStream& stream, void* a, void* b, void** c, int32_t m, int32_t n, int32_t k)
{
    std::vector<int64_t> ashape = {m, k};
    std::vector<int64_t> bshape = {n, k};
    std::vector<int64_t> cshape = {m, n};

    aclDataType aType = ACL_INT8;
    aclDataType bType = ACL_INT8;
    aclDataType cType = ACL_INT32;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_FRACTAL_NZ;
    aclFormat cformat = ACL_FORMAT_ND;

    auto ret = aclrtMalloc(c, m * n * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), a);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), b);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), *c);

    size_t workspaceSize;
    aclOpExecutor* handle;
    ret = aclnnMiMatmulGetWorkspaceSize(aclaten, aclbten, aclcten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMiMatmulGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnMiMatmul(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMiMatmul failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    aclrtFree(workspaceAddr);

    LOG_DEBUG("Linear op success");
}

void LinearI8(aclrtStream& stream, void* a, void* b, void* c, int32_t m, int32_t n, int32_t k)
{
    std::vector<int64_t> ashape = {m, k};
    std::vector<int64_t> bshape = {n, k};
    std::vector<int64_t> cshape = {m, n};

    aclDataType aType = ACL_INT8;
    aclDataType bType = ACL_INT8;
    aclDataType cType = ACL_INT32;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_FRACTAL_NZ;
    aclFormat cformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), a);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), b);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), c);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMiMatmulGetWorkspaceSize(aclaten, aclbten, aclcten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMiMatmulGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMiMatmul(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMiMatmul failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("Linear op success");
}

void LinearFP16(aclrtStream& stream, void* a, void* b, void* c, int32_t m, int32_t n, int32_t k)
{
    std::vector<int64_t> ashape = {m, k};
    std::vector<int64_t> bshape = {n, k};
    std::vector<int64_t> cshape = {m, n};

    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT16;
    aclDataType cType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_FRACTAL_NZ;
    aclFormat cformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), a);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), b);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), c);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMiMatmulGetWorkspaceSize(aclaten, aclbten, aclcten, &workspaceSize, &handle);
    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMiMatmulGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMiMatmul(workspaceAddr, workspaceSize, handle, stream);
    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMiMatmul failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("Linear op success");
}

void LinearFP32(aclrtStream& stream, void* a, void* b, void* c, int32_t m, int32_t n, int32_t k)
{
    std::vector<int64_t> ashape = {m, k};
    std::vector<int64_t> bshape = {n, k};
    std::vector<int64_t> cshape = {m, n};

    aclDataType aType = ACL_FLOAT;
    aclDataType bType = ACL_FLOAT;
    aclDataType cType = ACL_FLOAT;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_ND;
    aclFormat cformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), a);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), b);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), c);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnLinearGetWorkspaceSize(aclaten, aclbten, aclcten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMatmulGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnLinear(workspaceAddr, workspaceSize, handle, stream);
    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMatmul failed");
    }

    CHECK_MIACL_ERROR;
    aclDestroyTensor(aclaten);
    aclDestroyTensor(aclbten);
    aclDestroyTensor(aclcten);
    if(workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    LOG_DEBUG("Linear op success");
}
