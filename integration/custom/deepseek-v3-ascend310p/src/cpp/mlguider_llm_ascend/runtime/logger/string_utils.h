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
#include <cstdarg>
#include <string>
#include <string_view>

std::string Vformat(const char* format, va_list args);

inline std::string fmtstr(const std::string& s)
{
    return s;
}

inline std::string fmtstr(std::string&& s)
{
    return std::move(s);
}
inline std::string fmtstr(std::string_view s)
{
    return std::string(s);
}

#if defined(__GNUC__) || defined(__clang__)
[[gnu::format(printf, 1, 2)]]
#endif
std::string fmtstr(const char* format, ...);

inline std::string fmtstr(std::string_view format, ...)
{
    va_list args;
    va_start(args, format);
    auto guard = [&args]() { va_end(args); };
    std::string result = Vformat(format.data(), args);
    guard();
    return result;
}
