/**
 * Adapt from https://github.com/prakhar1989/progress-cpp/blob/master/include/progresscpp/ProgressBar.hpp
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

#include <chrono>
#include <iostream>

namespace progresscpp
{
class ProgressBar
{
private:
    unsigned int ticks = 0;

    const unsigned int totalTicks;
    const unsigned int barWidth;
    const char completeChar = '=';
    const char incompleteChar = ' ';
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

public:
    ProgressBar(unsigned int total, unsigned int width, char complete, char incomplete)
        : totalTicks{total}
        , barWidth{width}
        , completeChar{complete}
        , incompleteChar{incomplete}
    {
    }

    ProgressBar(unsigned int total, unsigned int width)
        : totalTicks{total}
        , barWidth{width}
    {
    }

    unsigned int operator++()
    {
        return ++ticks;
    }

    void Display() const
    {
        float progress = static_cast<float> (ticks) / totalTicks;
        int pos = static_cast<int>(barWidth * progress);

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();

        std::cout << "[";

        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos)
            {
                std::cout << completeChar;
            }
            else if (i == pos)
            {
                std::cout << ">";
            }
            else{
                std::cout << incompleteChar;
            }
        }
        std::cout << "] " << int(progress * 100.0) << "% " << float(timeElapsed) / 1000.0 << "s\r"; // convert ms to s
        std::cout.flush();
    }

    void Done() const
    {
        Display();
        std::cout << std::endl;
    }
};
} // namespace progresscpp
