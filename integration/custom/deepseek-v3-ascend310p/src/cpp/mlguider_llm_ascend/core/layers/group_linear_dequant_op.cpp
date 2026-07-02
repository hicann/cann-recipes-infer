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
#include "group_linear_dequant_op.h"

#include "acl_utils.h"
#include "aclnn_mi_group_linear_de_quant.h"
#include "logger.h"
#include "transpose_op.h"

void GroupLinearDequant(
    aclrtStream& stream, void* input, void* weight, void* groupList, void* dequantScale, void* output, int m, int n, int k, int g)
{

    std::vector<int64_t> ashape = {m, k};
    std::vector<int64_t> bshape = {g, n, k};
    std::vector<int64_t> cshape = {g};
    std::vector<int64_t> dshape = {g, n};
    std::vector<int64_t> eshape = {m, n};

    aclDataType aType = ACL_INT8;
    aclDataType bType = ACL_INT8;
    aclDataType cType = ACL_INT32;
    aclDataType dType = ACL_FLOAT;
    aclDataType eType = ACL_FLOAT16;

    aclFormat aformat = ACL_FORMAT_ND;
    aclFormat bformat = ACL_FORMAT_FRACTAL_NZ;
    aclFormat cformat = ACL_FORMAT_ND;
    aclFormat dformat = ACL_FORMAT_ND;
    aclFormat eformat = ACL_FORMAT_ND;

    aclTensor* aclaten = aclCreateTensor(ashape.data(), ashape.size(), aType, nullptr, 0, aformat, ashape.data(), ashape.size(), input);
    aclTensor* aclbten = aclCreateTensor(bshape.data(), bshape.size(), bType, nullptr, 0, bformat, bshape.data(), bshape.size(), weight);
    aclTensor* aclcten = aclCreateTensor(cshape.data(), cshape.size(), cType, nullptr, 0, cformat, cshape.data(), cshape.size(), groupList);
    aclTensor* acldten = aclCreateTensor(dshape.data(), dshape.size(), dType, nullptr, 0, dformat, dshape.data(), dshape.size(), dequantScale);
    aclTensor* acleten = aclCreateTensor(eshape.data(), eshape.size(), eType, nullptr, 0, eformat, eshape.data(), eshape.size(), output);

    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnMIGroupLinearDeQuantGetWorkspaceSize(
        aclaten, aclbten, aclcten, acldten, acleten, &workspaceSize, &handle);

    if (ret != ACL_SUCCESS)
    {
        (void) aclrtDestroyStream(stream);
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        throw std::runtime_error("Get aclnnMIGroupLinearDeQuantGetWorkspaceSize failed");
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnMIGroupLinearDeQuant(workspaceAddr, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS)
    {
        const char* tmpErrMsg = nullptr;
        tmpErrMsg = aclGetRecentErrMsg();
        if (tmpErrMsg != nullptr)
        {
            LOG_ERROR("%s", tmpErrMsg);
        }
        (void) aclrtDestroyStream(stream);
        throw std::runtime_error("aclnnMIGroupLinearDeQuant failed");
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

    LOG_DEBUG("GroupLinearDeQuant op success");
}
