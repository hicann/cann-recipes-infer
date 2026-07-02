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
#include "string_utils.h"

#include <cstdarg>
#include <stdexcept>
#include <string>
#include <string_view>

std::string Vformat(const char* format, va_list args)
{
    if (!format)
    {
        return {};
    }

    char stack_buf[128];
    va_list args_copy;
    va_copy(args_copy, args);

    const int size = vsnprintf(stack_buf, sizeof(stack_buf), format, args_copy);
    va_end(args_copy);

    if (size < 0)
    {
        throw std::runtime_error("Format string error");
    }

    if (size > 1000000) // Default vaule
    {
        throw std::length_error("Formatted string too large");
    }

    if (size < sizeof(stack_buf))
    {
        return std::string(stack_buf, size);
    }

    std::string stringBuf(size, char{});
    va_list args_copy2;
    va_copy(args_copy2, args);

    if (std::vsnprintf(&stringBuf[0], size + 1, format, args_copy2) < 0)
    {
        va_end(args_copy2);
        throw std::runtime_error("String formatting failed");
    }

    va_end(args_copy2);
    return stringBuf;
}

std::string fmtstr(const char* format, ...)
{
    if (!format)
    {
        return {};
    }

    va_list args;
    va_start(args, format);

    struct va_list_guard
    {
        va_list& args;

        va_list_guard(va_list& a)
            : args(a)
        {
        }

        ~va_list_guard()
        {
            va_end(args);
        }
    } guard(args);

    return Vformat(format, args);
}
