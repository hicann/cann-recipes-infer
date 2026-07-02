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
#include "nlohmann/json.hpp"
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

struct ServiceConfig
{
    std::string ip;
    int32_t port;
};

struct CacheConfig
{
    int32_t blockSize;
    int32_t numGpuBlocks;
    int32_t numCpuBlocks = 0;
};

struct SchedulerConfig
{
    int32_t maxNumBatchedTokens; // max token number for a batch
    int32_t maxModelLen = 4096;  // max len for prefill
};

struct ParallelConfig
{
    uint32_t rank;
    uint32_t worldSize;
    uint32_t nodeSize;
    uint32_t tensorParallelSize;
    uint32_t pipelineParallelSize;
    uint32_t expertParallelSize;
    uint32_t dataParallelSize;
};

struct ModelConfig
{
    std::string modelType; // Model type (llama/baichuan/qwen2 etc.)
    std::string modelPath; // Model weights path
    nlohmann::json modelParams;
    bool enableOffloading;
    int32_t offloadingLayerBlockSize; // if enableOffloading is true，offloadingLayerBlockSize is enable, for scheduling layers

    void Init(const std::string& modelDir)
    {
        modelPath = modelDir;
        modelParams = nlohmann::json::parse(std::ifstream(modelDir + "config.json"));
        modelType = modelParams["model_type"];
    }
};

struct LLMConfig
{
    std::vector<int32_t> devices;

    ServiceConfig serviceConfig;
    ModelConfig modelConfig;
    CacheConfig cacheConfig;
    SchedulerConfig schedulerConfig;
    ParallelConfig parallelConfig;

    void ParseCommandLine(int argc, char* argv[]);
    void SetWorldSize(int32_t worldSize);
    void SetRankId(int32_t rankId);

    std::string Serialize() const;
    static LLMConfig Deserialize(const std::string& jsonStr);

    friend std::ostream& operator<<(std::ostream& os, const LLMConfig& config);
    friend std::ostream& operator<<(std::ostream& os, const ServiceConfig& config);
    friend std::ostream& operator<<(std::ostream& os, const ModelConfig& config);
    friend std::ostream& operator<<(std::ostream& os, const CacheConfig& config);
    friend std::ostream& operator<<(std::ostream& os, const SchedulerConfig& config);
    friend std::ostream& operator<<(std::ostream& os, const ParallelConfig& config);
};

std::ostream& operator<<(std::ostream& os, const ServiceConfig& config);
std::ostream& operator<<(std::ostream& os, const ModelConfig& config);
std::ostream& operator<<(std::ostream& os, const CacheConfig& config);
std::ostream& operator<<(std::ostream& os, const SchedulerConfig& config);
std::ostream& operator<<(std::ostream& os, const ParallelConfig& config);
std::ostream& operator<<(std::ostream& os, const LLMConfig& config);
