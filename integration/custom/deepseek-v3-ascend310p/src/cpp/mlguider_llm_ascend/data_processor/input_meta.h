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
#include "nlohmann/json.hpp"
#include "tensor.h"
#include <string>
#include <vector>

using json = nlohmann::json;

class InputMeta
{
public:
    // base info
    std::string stage;   // stage: prefill or generation
    int32_t batchSize;   // batch size
    int32_t batchTokens; // token for a batch

    // about input meta info
    std::vector<std::string> id;         // batch id
    std::vector<int32_t> inputIds;       // input tokens
    std::vector<int32_t> positionIds;    // position ID
    std::vector<int32_t> pageIds;        // kv cache block ID
    std::vector<int32_t> slotMapping;    // slot in KV Cache
    std::vector<int32_t> seqlensQuery;   // len for each seq
    std::vector<int32_t> cuSeqlensQuery; // cuLen for seq
    std::vector<int32_t> seqlensKV;      // KV len for each seq
    std::vector<uint64_t> batchTokensPerRank;
    int32_t maxSeqlensQuery;             // max seqlen for a batch
    int32_t maxSeqlensKV;                // max kv len for a batch
    int32_t maxNumBlocksPerSeq;          // max number of block for a batch

    [[deprecated("Use cuSeqlensQuery/seqlensKV instead")]] std::vector<int64_t> curlens;
    [[deprecated("Use seqlensKV/seqlensKV instead")]] std::vector<int64_t> seqlenSet;

    InputMeta() = default;
    [[deprecated]]InputMeta(std::string stage, json data);

    std::string Serialize() const;
    void Deserialize(const std::string& data);

private:
    template <typename T>
    static void PrintVector(const std::string& name, const std::vector<T>& vec);
};

struct InputContext
{
    bool isPrefill;
    int32_t batchSize = 0;
    int32_t batchTokens = 0;

    void* inputIds=nullptr;
    void* positionIds=nullptr;
    void* pageIds=nullptr;
    void* slotMapping=nullptr;
    void* cuSeqlensQuery=nullptr;
    void* seqlensKV=nullptr;

    std::vector<int32_t> cuSeqlensQueryHost;
    std::vector<int> inputIdsHost;
    std::vector<int32_t> pageIdsInsertHost;
    std::vector<int32_t> pageOffsetHost;
    std::vector<int32_t> cuNumPageTokenHost;
    std::vector<uint64_t> batchTokensPerRankHost;
    ParallelConfig parallelConfig;

    [[deprecated("This parameter willl be canceled.")]]std::vector<int64_t> seqIdxSet;
    [[deprecated("This parameter willl be canceled.")]]std::vector<int64_t> seqlenSet;
    [[deprecated("This parameter willl be canceled.")]]std::vector<int64_t> curlenDict;
    [[deprecated("This parameter willl be canceled.")]]std::vector<int32_t> seqlensKVHost;
};

struct OutputContext
{
    std::vector<int32_t> outputIds;
    std::vector<float> outputValues;
};
