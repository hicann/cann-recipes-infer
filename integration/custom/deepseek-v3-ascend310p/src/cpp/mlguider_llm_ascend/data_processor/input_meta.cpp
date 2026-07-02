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
#include "input_meta.h"
#include "logger.h"
#include <unordered_set>

InputMeta::InputMeta(std::string stage, json data)
    : stage(stage)
{
    id = data["id"].get<std::vector<std::string>>();
    batchSize = id.size();
    inputIds = data["input_ids"].get<std::vector<int32_t>>();
    batchTokens = inputIds.size();

    positionIds = data["position_ids"].get<std::vector<int32_t>>();
    pageIds = data["page_id_arr"].get<std::vector<int32_t>>();
    slotMapping = data["slot_mapping"].get<std::vector<int32_t>>();

    seqlensQuery = data["seqlens_query"].get<std::vector<int32_t>>();
    cuSeqlensQuery = data["cu_seqlens_query"].get<std::vector<int32_t>>();
    seqlensKV = data["seqlens_kv"].get<std::vector<int32_t>>();

    maxSeqlensQuery = data["max_seqlens_query"].get<int32_t>();
    maxSeqlensKV = data["max_seqlens_kv"].get<int32_t>();
    maxNumBlocksPerSeq = data["max_num_blocks_per_seq"].get<int32_t>();

    if (stage != "prefill" && stage != "generation")
    {
        throw std::invalid_argument("Invalid stage: " + stage);
    }
}

template <typename T>
void PrintVector(std::string name, std::vector<T>& vec)
{
    std::cout << name << ": [";
    for (auto& id : vec)
    {
        std::cout << id << " ";
    }
    std::cout << "]" << std::endl;
}

std::string InputMeta::Serialize() const
{
    nlohmann::json j;
    j["stage"] = stage;
    j["batchSize"] = batchSize;
    j["batchTokens"] = batchTokens;
    j["inputIds"] = inputIds;
    j["positionIds"] = positionIds;
    j["page_ids"] = pageIds;
    j["slotMapping"] = slotMapping;
    j["seqlens_query"] = seqlensQuery;
    j["cu_seqlens_query"] = cuSeqlensQuery;
    j["seqlens_kv"] = seqlensKV;
    j["max_seqlens_query"] = maxSeqlensQuery;
    j["max_seqlens_kv"] = maxSeqlensKV;
    j["max_num_blocks_per_seq"] = maxNumBlocksPerSeq;
    return j.dump();
}

void InputMeta::Deserialize(const std::string& data)
{
    auto j = nlohmann::json::parse(data);
    stage = j["stage"];
    batchSize = j["batchSize"];
    batchTokens = j["batchTokens"];
    inputIds = j["inputIds"].get<std::vector<int32_t>>();
    positionIds = j["positionIds"].get<std::vector<int32_t>>();
    pageIds = j["page_ids"].get<std::vector<int32_t>>();
    slotMapping = j["slotMapping"].get<std::vector<int32_t>>();
    seqlensQuery = j["seqlens_query"].get<std::vector<int32_t>>();
    cuSeqlensQuery = j["cu_seqlens_query"].get<std::vector<int32_t>>();
    seqlensKV = j["seqlens_kv"].get<std::vector<int32_t>>();
    maxSeqlensQuery = j["max_seqlens_query"];
    maxSeqlensKV = j["max_seqlens_kv"];
    maxNumBlocksPerSeq = j["max_num_blocks_per_seq"];
}
