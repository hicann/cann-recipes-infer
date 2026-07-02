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
#include <cmath>

inline float FindCorrectionDim(float numRotations, int dim, float base, int maxSeqLen)
{
    return dim * std::log(maxSeqLen / (numRotations * 2 * M_PI)) / (2 * std::log(base)); // 2 * M_PI
}

inline std::pair<int, int> FindCorrectionRange(float low_rot, float high_rot, int dim, float base, int maxSeqLen)
{
    int low = static_cast<int>(std::floor(FindCorrectionDim(low_rot, dim, base, maxSeqLen)));
    int high = static_cast<int>(std::ceil(FindCorrectionDim(high_rot, dim, base, maxSeqLen)));
    return {std::max(low, 0), std::min(high, dim - 1)};
}

inline std::vector<float> LinearRampFactor(float min, float max, int dim)
{
    std::vector<float> ramp_func(dim);
    if (min == max)
    {
        max += 0.001f;
    }
    for (int i = 0; i < dim; ++i)
    {
        ramp_func[i] = std::clamp((i - min) / static_cast<float>(max - min), 0.0f, 1.0f);
    }
    return ramp_func;
}

inline float YarnGetMscale(float scale, float mscale)
{
    if (scale <= 1.0f)
    {
        return 1.0f;
    }
    return 0.1f * mscale * std::log(scale) + 1.0f;
}
