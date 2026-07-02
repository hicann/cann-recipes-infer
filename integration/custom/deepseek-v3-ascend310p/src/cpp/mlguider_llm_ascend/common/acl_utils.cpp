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

#include "acl_utils.h"
#include "logger.h"

bool CreateStream(aclrtStream &stream)
{
    if (aclrtCreateStream(&stream) != ACL_SUCCESS)
    {
        LOG_ERROR("Create stream failed");
        return false;
    }
    LOG_INFO("Create stream success.");
    return true;
}

bool DestroyStream(aclrtStream &stream)
{
    if (aclrtDestroyStream(stream) != ACL_SUCCESS)
    {
        LOG_ERROR("Destroy stream failed.");
        return false;
    }
    LOG_INFO("Destroy stream success.");
    return true;
}

bool SetDevice(int32_t deviceId)
{
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS)
    {
        LOG_ERROR("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    LOG_INFO("Set device[%d] success", deviceId);
    return true;
}

bool ResetDevice(int32_t deviceId)
{
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS)
    {
        LOG_ERROR("Reset device %d failed", deviceId);
        return false;
    }
    LOG_INFO("Reset Device[%d] success", deviceId);
    return true;
}
