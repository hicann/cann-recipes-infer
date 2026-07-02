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

#pragma once
#include "acl/acl.h"
#include <cstdint>
#include <iostream>

#ifdef ENABLE_CHECK_MIACL_ERROR
#define CHECK_MIACL_ERROR                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (aclrtSynchronizeStream(stream) != ACL_ERROR_NONE)                                                          \
        {                                                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;                                                     \
            (void) aclrtDestroyStream(stream);                                                                         \
            const char* tmpErrMsg = aclGetRecentErrMsg();                                                            \
            if (tmpErrMsg != nullptr)                                                                                   \
            {                                                                                                          \
                printf(" ERROR Message : %s \n", tmpErrMsg);                                                         \
            }                                                                                                          \
            LOG_ERROR("Sync ops failed");                                                                           \
        }                                                                                                              \
    } while (0)
#else
#define CHECK_MIACL_ERROR                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#endif

#define CHECK_ACL(x)                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        aclError __ret = x;                                                                                            \
        if (__ret != ACL_ERROR_NONE)                                                                                   \
        {                                                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl;                            \
        }                                                                                                              \
    } while (0);


bool CreateStream(aclrtStream& stream);
bool DestroyStream(aclrtStream& stream);
bool SetDevice(int32_t deviceId);
bool ResetDevice(int32_t deviceId);
