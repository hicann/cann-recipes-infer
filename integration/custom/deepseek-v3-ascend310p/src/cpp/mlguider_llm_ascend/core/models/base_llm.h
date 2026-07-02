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
#include "config.h"
#include "hccl/hccl.h"
#include "input_meta.h"
#include "sample_op.h"
#include "transpose_op.h"

struct WeightData
{
    void* dataPtr = nullptr;

    void* data()
    {
        return dataPtr;
    }
};

struct WeightMap
{
    std::map<std::string, WeightData> weight;

    void LoadFromFile(const std::string& path);
    WeightData& operator[](const std::string& key);

    void Load(aclrtStream stream, int32_t layerId);
};


class BaseLLM
{
public:
    virtual bool Inference(InputContext& inputContext, OutputContext& outputContext,
            std::vector<void*> &kcache, std::vector<void*> &vcache,
            int maxPageNum, int pageLen,
            int batchSize, int pageIdLen,
            int tokenNum,
            aclrtStream& stream) = 0;

protected:
    WeightMap weight;
};
