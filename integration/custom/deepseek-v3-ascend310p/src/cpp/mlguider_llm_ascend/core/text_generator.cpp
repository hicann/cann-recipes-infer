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
#include "text_generator.h"
#include "acl_utils.h"
#include "logger.h"
#include "model_utils.h"
#include "mpi.h"
#include <chrono>

TextGenerator::TextGenerator(LLMConfig& config)
    : mConfig(config)
{
    uint32_t rank = config.parallelConfig.rank;
    uint32_t nodeSize = config.parallelConfig.nodeSize;

    int32_t deviceId = config.devices[rank % nodeSize];
    SetDevice(deviceId);
    LOG_INFO("Local device: %d", deviceId);

    model = std::move(CreateLLMModel(config));
    LOG_INFO("Load model success.");

    const nlohmann::json& modelParams = config.modelConfig.modelParams;
    uint32_t hiddenLayers = modelParams["num_hidden_layers"].get<uint>();
    int32_t kHeadDim = modelParams["qk_rope_head_dim"].get<int>();
    int32_t vHeadDim = modelParams["kv_lora_rank"].get<int>();

    for (int i = 0; i < 61; i++)
    { // 61 is the max number of layers
        size_t kSize = config.cacheConfig.numGpuBlocks * (kHeadDim / 16) * config.cacheConfig.blockSize * 16 * 2;
        size_t vSize = config.cacheConfig.numGpuBlocks * (vHeadDim / 16) * config.cacheConfig.blockSize * 16 * 2;

        void* kPtr = nullptr;
        aclError ret = aclrtMalloc(&kPtr, kSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
        {
            for (int j = 0; j < i; j++)
            {
                aclrtFree(mKcache[j]);
                aclrtFree(mVcache[j]);
            }
            throw std::runtime_error("Failed to allocate K Cache");
        }
        mKcache.emplace_back(kPtr);

        void* vPtr = nullptr;
        ret = aclrtMalloc(&vPtr, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
        {
            aclrtFree(kPtr);
            for (int j = 0; j < i; j++)
            {
                aclrtFree(mKcache[j]);
                aclrtFree(mVcache[j]);
            }
            throw std::runtime_error("Failed to allocate V Cache");
        }
        mVcache.emplace_back(vPtr);
    }
    LOG_INFO("kv cache allocate success");

    ResetDevice(deviceId);
}

std::tuple<std::vector<int>, std::vector<float>, float> TextGenerator::Inference(InputMeta& inputData)
{
    uint32_t rank = mConfig.parallelConfig.rank;
    uint32_t nodeSize = mConfig.parallelConfig.nodeSize;

    int32_t deviceId = mConfig.devices[rank % nodeSize];
    SetDevice(deviceId);
    LOG_INFO("Local device: %d", deviceId);
    aclrtStream stream;
    CreateStream(stream);

    bool isPrefill = inputData.stage == "prefill";
    int32_t batchSize = inputData.batchSize;
    int32_t batchTokens = inputData.batchTokens;

    InputContext inputContext;
    inputContext.isPrefill = isPrefill;
    inputContext.batchSize = batchSize;
    inputContext.batchTokens = batchTokens;

    aclrtMalloc(&inputContext.inputIds, batchTokens * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(inputContext.inputIds, batchTokens * sizeof(int32_t), inputData.inputIds.data(),
        batchTokens * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMalloc(&inputContext.positionIds, batchTokens * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(inputContext.positionIds, batchTokens * sizeof(int32_t), inputData.positionIds.data(),
        batchTokens * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMalloc(
        &inputContext.pageIds, batchSize * inputData.maxNumBlocksPerSeq * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(inputContext.pageIds, batchSize * inputData.maxNumBlocksPerSeq * sizeof(int32_t),
        inputData.pageIds.data(), batchSize * inputData.maxNumBlocksPerSeq * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMalloc(&inputContext.slotMapping, batchTokens * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(inputContext.slotMapping, batchTokens * sizeof(int32_t), inputData.slotMapping.data(),
        batchTokens * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    inputContext.batchTokensPerRankHost = inputData.batchTokensPerRank;

    aclrtMalloc(&inputContext.cuSeqlensQuery, (batchSize + 1) * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST); // cumulative: batchSize + 1
    aclrtMemcpy(inputContext.cuSeqlensQuery, (batchSize + 1) * sizeof(int32_t), inputData.cuSeqlensQuery.data(),
        (batchSize + 1) * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE); // cumulative: batchSize + 1

    aclrtMalloc(&inputContext.seqlensKV, batchSize * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(
        inputContext.seqlensKV, batchSize * sizeof(int32_t), inputData.seqlensKV.data(), batchSize * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    inputContext.cuSeqlensQueryHost = inputData.cuSeqlensQuery;
    inputContext.parallelConfig = mConfig.parallelConfig;
    inputContext.inputIdsHost = std::vector<int>(inputData.inputIds.begin(), inputData.inputIds.end());

    if (inputData.stage == "prefill")
    {
        inputContext.curlenDict = std::vector<long int>(batchSize, 0); // init batchSize "0" in prefill stage
    }
    else if (inputData.stage == "generation")
    {
        inputContext.curlenDict = std::vector<long int>(inputData.positionIds.begin(), inputData.positionIds.end());
    }
    else if (inputData.stage == "mix")
    {
        inputContext.curlenDict = std::vector<long int>(inputData.seqlensKV.begin(), inputData.seqlensKV.end());
    }
    else
    {
        LOG_ERROR("Invalid stage: %s", inputData.stage.c_str());
        DestroyStream(stream);
        ResetDevice(deviceId);
        return {};
    }
    inputContext.seqlenSet = std::vector<long int>(inputData.seqlensQuery.begin(), inputData.seqlensQuery.end());

    OutputContext outputContext;

    aclrtSynchronizeStream(stream);
    auto start = std::chrono::high_resolution_clock::now();
    bool success = model->Inference(inputContext, outputContext, mKcache, mVcache, 128, 256, batchSize,
        inputData.maxNumBlocksPerSeq, batchTokens, stream);
    aclrtSynchronizeStream(stream);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;

    if (rank == 0)
    {
        std::cout << "Time:" << duration.count() / 1000.0 << std::endl; // convert us to ms
    }

    if (!success)
    {
        LOG_ERROR("Run Inference Failed");
        DestroyStream(stream);
        ResetDevice(deviceId);
        return {};
    }

    const char* msg = isPrefill ? "Run prompt success" : "Run Forward success";
    LOG_INFO("%s", msg);

    DestroyStream(stream);
    ResetDevice(deviceId);

    return {outputContext.outputIds, outputContext.outputValues, duration.count() / 1000.0};
}
