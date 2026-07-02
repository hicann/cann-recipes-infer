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
#include "base_llm.h"
#include "config.h"
#include "input_meta.h"
class TextGenerator
{
public:
    TextGenerator(LLMConfig& config);

    std::tuple<std::vector<int>, std::vector<float>, float> Inference(InputMeta& inputData);

public:
    LLMConfig& mConfig;
    std::unique_ptr<BaseLLM> model;
    std::vector<void*> mKcache;
    std::vector<void*> mVcache;

};
