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

#include "config.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <sstream>
#include <stdexcept>

ABSL_FLAG(std::string, ip, "127.0.0.1", "LLM server ip");
ABSL_FLAG(std::string, port, "8088", "LLM server port");
ABSL_FLAG(std::string, model_path, "", "Path to the model directory");
ABSL_FLAG(std::string, config_file, "config.json", "Path to the config file");
ABSL_FLAG(std::int32_t, max_num_batched_tokens, 4096, "the max number token for a batch");
ABSL_FLAG(std::int32_t, block_size, 16, "kv cache block size");
ABSL_FLAG(std::int32_t, num_gpu_blocks, 1024, "kvcache block number");
ABSL_FLAG(std::int32_t, world_size, 1, "world size");
ABSL_FLAG(uint16_t, tp, 1, "tp_size");
ABSL_FLAG(uint16_t, pp, 1, "pp_size");
ABSL_FLAG(uint16_t, ep, 1, "ep_size");
ABSL_FLAG(std::int32_t, offloading_layer_block_size, -1, "default: -1, meanings disable offlading");
ABSL_FLAG(uint16_t, dp, 1, "dp_size");

std::vector<int32_t> ParsePorts(const std::string& port_str)
{
    std::vector<int32_t> ports;
    std::istringstream iss(port_str);
    std::string port;

    while (std::getline(iss, port, ','))
    {
        try
        {
            int32_t portNum = std::stoi(port);
            ports.push_back(portNum);
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Invalid port number: " << port << std::endl;
            throw;
        }
        catch (const std::out_of_range& e)
        {
            std::cerr << "Port number out of range: " << port << std::endl;
            throw;
        }
    }

    return ports;
}

void LLMConfig::ParseCommandLine(int argc, char* argv[])
{
    absl::ParseCommandLine(argc, argv);

    // model init
    std::string modelPath = absl::GetFlag(FLAGS_model_path);
    if (!modelPath.empty() && modelPath.back() != '/')
    {
        modelPath += '/';
    }
    modelConfig.Init(modelPath);
    int32_t layerBlockSize = absl::GetFlag(FLAGS_offloading_layer_block_size);
    if (layerBlockSize > 0)
    {
        modelConfig.enableOffloading = true;
        modelConfig.offloadingLayerBlockSize = layerBlockSize;
    }
    else
    {
        modelConfig.enableOffloading = false;
        modelConfig.offloadingLayerBlockSize = -1;
    }

    // init schedule config
    schedulerConfig.maxNumBatchedTokens = absl::GetFlag(FLAGS_max_num_batched_tokens);

    // init cache config
    cacheConfig.blockSize = absl::GetFlag(FLAGS_block_size);
    cacheConfig.numGpuBlocks = absl::GetFlag(FLAGS_num_gpu_blocks);

    // init parallel config
    parallelConfig.worldSize = absl::GetFlag(FLAGS_world_size);
    parallelConfig.nodeSize = parallelConfig.worldSize;
    parallelConfig.tensorParallelSize = absl::GetFlag(FLAGS_tp);
    parallelConfig.pipelineParallelSize = absl::GetFlag(FLAGS_pp);
    parallelConfig.expertParallelSize = absl::GetFlag(FLAGS_ep);
    parallelConfig.dataParallelSize = absl::GetFlag(FLAGS_dp);

    // init serve config
    serviceConfig.ip = absl::GetFlag(FLAGS_ip);
    serviceConfig.port = ParsePorts(absl::GetFlag(FLAGS_port))[parallelConfig.rank / (parallelConfig.worldSize / parallelConfig.dataParallelSize)];

    devices.clear();
    if (const char* envP = std::getenv("ASCEND_VISIBLE_DEVICES"))
    {
        if (!envP)
        {
            throw std::runtime_error("env var ASCEND_VISIBLE_DEVICES not existing");
        }
        std::string envStr(envP);
        std::stringstream ss(envStr);
        std::string id;
        while (std::getline(ss, id, ','))
        {
            if (!id.empty())
            {
                devices.push_back(std::stoi(id));
            }
        }
    }
    else
    {
        throw std::runtime_error("env var ASCEND_VISIBLE_DEVICES not existing");
    }
}

// Member function: serialize entire LLMConfig to JSON string
std::string LLMConfig::Serialize() const
{
    nlohmann::json j;

    // Devices
    j["devices"] = devices;

    // ServiceConfig
    j["serviceConfig"]["ip"] = serviceConfig.ip;
    j["serviceConfig"]["port"] = serviceConfig.port;

    // CacheConfig
    j["cacheConfig"]["blockSize"] = cacheConfig.blockSize;
    j["cacheConfig"]["numGpuBlocks"] = cacheConfig.numGpuBlocks;
    j["cacheConfig"]["numCpuBlocks"] = cacheConfig.numCpuBlocks;

    // SchedulerConfig
    j["schedulerConfig"]["maxNumBatchedTokens"] = schedulerConfig.maxNumBatchedTokens;
    j["schedulerConfig"]["maxModelLen"] = schedulerConfig.maxModelLen;

    // ParallelConfig
    j["parallelConfig"]["rank"] = parallelConfig.rank;
    j["parallelConfig"]["worldSize"] = parallelConfig.worldSize;
    j["parallelConfig"]["nodeSize"] = parallelConfig.nodeSize;
    j["parallelConfig"]["tensorParallelSize"] = parallelConfig.tensorParallelSize;
    j["parallelConfig"]["pipelineParallelSize"] = parallelConfig.pipelineParallelSize;
    j["parallelConfig"]["expertParallelSize"] = parallelConfig.expertParallelSize;

    // ModelConfig
    j["modelConfig"]["modelType"] = modelConfig.modelType;
    j["modelConfig"]["modelPath"] = modelConfig.modelPath;
    j["modelConfig"]["modelParams"] = modelConfig.modelParams;

    return j.dump();
}

// Static or free function: deserialize JSON string into LLMConfig
LLMConfig LLMConfig::Deserialize(const std::string& jsonStr)
{
    nlohmann::json j = nlohmann::json::parse(jsonStr);
    LLMConfig config;

    // Devices
    config.devices = j.value("devices", std::vector<int32_t>{});

    // ServiceConfig
    config.serviceConfig.ip = j["serviceConfig"].value("ip", std::string(""));
    config.serviceConfig.port = j["serviceConfig"].value("port", 0);

    // CacheConfig
    config.cacheConfig.blockSize = j["cacheConfig"].value("blockSize", 0);
    config.cacheConfig.numGpuBlocks = j["cacheConfig"].value("numGpuBlocks", 0);
    config.cacheConfig.numCpuBlocks = j["cacheConfig"].value("numCpuBlocks", 0);

    // SchedulerConfig
    config.schedulerConfig.maxNumBatchedTokens = j["schedulerConfig"].value("maxNumBatchedTokens", 0);
    config.schedulerConfig.maxModelLen = j["schedulerConfig"].value("maxModelLen", 4096); // default value

    // ParallelConfig
    config.parallelConfig.rank = j["parallelConfig"].value("rank", 0);
    config.parallelConfig.worldSize = j["parallelConfig"].value("worldSize", 1);
    config.parallelConfig.nodeSize = j["parallelConfig"].value("nodeSize", 1);
    config.parallelConfig.tensorParallelSize = j["parallelConfig"].value("tensorParallelSize", 1);
    config.parallelConfig.pipelineParallelSize = j["parallelConfig"].value("pipelineParallelSize", 1);
    config.parallelConfig.expertParallelSize = j["parallelConfig"].value("expertParallelSize", 1);

    // ModelConfig
    config.modelConfig.modelType = j["modelConfig"].value("modelType", std::string(""));
    config.modelConfig.modelPath = j["modelConfig"].value("modelPath", std::string(""));
    config.modelConfig.modelParams = j["modelConfig"].value("modelParams", nlohmann::json::object());

    return config;
}

void LLMConfig::SetWorldSize(int32_t worldSize)
{
    this->parallelConfig.worldSize = worldSize;
    this->parallelConfig.nodeSize = worldSize;
}

void LLMConfig::SetRankId(int32_t rankId)
{
    parallelConfig.rank = rankId;
}

std::ostream& operator<<(std::ostream& os, const ServiceConfig& config)
{
    os << "{ip='" << config.ip << "', port=" << config.port << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const ModelConfig& config)
{
    os << "{type='" << config.modelType << "', path='" << config.modelPath << "'}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const CacheConfig& config)
{
    os << "{blockSize=" << config.blockSize << ", gpuBlocks=" << config.numGpuBlocks
       << ", cpuBlocks=" << config.numCpuBlocks << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const SchedulerConfig& config)
{
    os << "{batchTokens=" << config.maxNumBatchedTokens << ", modelLen=" << config.maxModelLen << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const ParallelConfig& config)
{
    os << "{rank=" << config.rank << ", world=" << config.worldSize << ", tp=" << config.tensorParallelSize
       << ", pp=" << config.pipelineParallelSize << ", ep=" << config.expertParallelSize << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const LLMConfig& config)
{
    os << "LLMConfig(\n  devices: [";
    const auto& devs = config.devices;
    for (size_t i = 0; i < devs.size(); ++i)
    {
        if (i > 0)
            os << ",";
        os << devs[i];
    }
    os << "],\n"
       << "  service: " << config.serviceConfig << ",\n"
       << "  model: " << config.modelConfig << ",\n"
       << "  cache: " << config.cacheConfig << ",\n"
       << "  scheduler: " << config.schedulerConfig << ",\n"
       << "  parallel: " << config.parallelConfig << "\n)";
    return os;
}
