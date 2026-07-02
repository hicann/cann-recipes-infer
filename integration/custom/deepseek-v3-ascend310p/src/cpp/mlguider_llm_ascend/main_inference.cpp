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
#include "acl/acl.h"
#include "config.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "input_meta.h"
#include "logger.h"
#include "runtime/comm/comm_domain_manager.h"
#include "text_generator.h"
#include "zmq.hpp"
#include <atomic>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <mpi.h>
#include <nlohmann/json.hpp>  // need 3rd nlohmann/json
#include <sstream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <sched.h>


ABSL_FLAG(std::string, tokenIds, "", "JSON string of token IDs (List[List[int]])");
ABSL_FLAG(int32_t, maxTokens, 100, "Maximum number of tokens");


std::vector<std::vector<int>> ParseTokenIds(const std::string& jsonStr)
{
    if (jsonStr.empty())
    {
        return {};
    }

    try
    {
        auto j = nlohmann::json::parse(jsonStr);
        return j.get<std::vector<std::vector<int>>>();
    } catch (const std::exception& e)
    {
        std::cout << "Failed to parse token_ids: " << std::endl;
        return {};
    }
}

float ComputeMean(const std::vector<float>& timeCost)
{
    if (timeCost.empty())
    {
        return 0.0f;
    }
    float sum = std::accumulate(timeCost.begin(), timeCost.end(), 0.0f);
    return sum / timeCost.size();
}

float ComputeMedian(std::vector<float> timeCost)
{
    if (timeCost.empty())
    {
        return 0.0f;
    }

    std::sort(timeCost.begin(), timeCost.end());

    size_t n = timeCost.size();
    if (n % 2 == 1) { // wheather odd
        return timeCost[n / 2]; // if odd get the median
    } else {
        return (timeCost[n / 2 - 1] + timeCost[n / 2]) / 2.0f; // if even get the average of the two middle
    }
}

float computePercentile(std::vector<float>& timeCost, float percentile) {
    if (timeCost.empty())
    {
        throw std::invalid_argument("Input vector cannot be empty");
    }

    if (percentile < 0 || percentile > 100) // check if percentile is in range
    {
        throw std::invalid_argument("Percentile must be between 0 and 100");
    }

    std::sort(timeCost.begin(), timeCost.end());

    float index = percentile / 100.0f * (timeCost.size() - 1);

    size_t lower = static_cast<size_t>(index);
    float fraction = index - lower;

    if (lower + 1 >= timeCost.size()) {
        return timeCost.back();
    }
    return timeCost[lower] + fraction * (timeCost[lower + 1] - timeCost[lower]);
}

void printInputData(const InputMeta& inputData)
{
    std::cout << "stage: " << inputData.stage << std::endl;
    std::cout << "batchSize: " << inputData.batchSize << std::endl;
    std::cout << "batchTokens: " << inputData.batchTokens << std::endl;

    auto printVector = [](const auto& vec, const std::string& name)
    {
        std::cout << name << ": [";
        for (size_t i = 0; i < vec.size(); ++i)
        {
            if (i != 0)
            {
                std::cout << ", ";
            }
            std::cout << vec[i];
        }
        std::cout << "]" << std::endl;
    };

    printVector(inputData.id, "id");
    printVector(inputData.inputIds, "inputIds");
    printVector(inputData.positionIds, "positionIds");
    printVector(inputData.pageIds, "pageIds");
    printVector(inputData.slotMapping, "slotMapping");
    printVector(inputData.seqlensQuery, "seqlensQuery");
    printVector(inputData.cuSeqlensQuery, "cuSeqlensQuery");
    printVector(inputData.seqlensKV, "seqlensKV");
    printVector(inputData.batchTokensPerRank, "batchTokensPerRank");

    std::cout << "maxSeqlensQuery: " << inputData.maxSeqlensQuery << std::endl;
    std::cout << "maxSeqlensKV: " << inputData.maxSeqlensKV << std::endl;
    std::cout << "maxNumBlocksPerSeq: " << inputData.maxNumBlocksPerSeq << std::endl;
}

std::vector<std::vector<int>> Process(LLMConfig& config, TextGenerator& generator, std::vector<std::vector<int32_t>> batchInputIds, int maxToken)
{
    // prefill

#ifdef MSPROF
    batchInputIds =
    {
        {0, 128803, 5356, 32917, 1662, 5512, 6328, 3465, 769, 127295, 17856, 11663, 83086, 28, 5512, 61, 15891, 3031, 14993, 28, 12249, 11, 6248, 12270, 1137, 361, 26495, 8824, 855, 295, 2910, 2310, 294, 3737, 14, 477, 1117, 1234, 3737, 12721, 304, 1660, 915, 1099, 201, 361, 2910, 14993, 603, 361, 46324, 769, 127295, 17856, 11663, 10425, 19, 16, 18, 14, 223, 20, 16, 18, 14, 223, 21, 16, 18, 3031, 223, 18, 16, 23, 682, 361, 14584, 201, 361, 46324, 769, 127295, 17856, 11663, 10425, 19, 16, 18, 14, 223, 20, 16, 26, 14, 223, 21, 16, 18, 14, 223, 22, 16, 18, 14, 223, 23, 16, 18, 14, 223, 20, 16, 18, 3031, 223, 18, 16, 21, 682, 361, 11485, 201, 361, 16786, 128804}
    };
#endif

    std::vector<std::vector<int>> outputIds;

    InputMeta inputData;
    inputData.stage = "prefill";
    inputData.batchSize = batchInputIds.size();
    int maxNumBlocksPerSeq = 0;
    inputData.cuSeqlensQuery.push_back(0);
    inputData.batchTokens = 0;
    inputData.maxSeqlensQuery = -1;
    for(int i = 0; i < inputData.batchSize; i++)
    {
        outputIds.push_back({});
    }
    std::vector<int> isEnd(inputData.batchSize, 0);
    std::vector<int> promoptLen;
    inputData.maxSeqlensKV = 0;
    int pageNumPerSeq = config.cacheConfig.numGpuBlocks / inputData.batchSize;
    if (pageNumPerSeq == 0)
    {
        throw std::invalid_argument(
            "Invalid configuration: 'numGpuBlocks / batchSize' must be positive. "
            "Got numGpuBlocks=" + std::to_string(config.cacheConfig.numGpuBlocks) +
            ", batchSize=" + std::to_string(inputData.batchSize)
        );
    }
    int seqId = 0;
    int maxSeqLen = -1;
    for(auto inputIds : batchInputIds)
    {
        int seqLen = inputIds.size();
        promoptLen.push_back(seqLen);
        maxSeqLen = maxSeqLen > seqLen ? maxSeqLen : seqLen;
    }

    inputData.maxNumBlocksPerSeq =  maxSeqLen / generator.mConfig.cacheConfig.blockSize + 1;

    for(auto inputIds : batchInputIds)
    {
        int seqLen = inputIds.size();

        inputData.batchTokens += seqLen;

        inputData.inputIds.insert(
            inputData.inputIds.end(),
            inputIds.begin(),
            inputIds.end()
        );

        std::vector<int> positionId(inputIds.size());
        std::iota(positionId.begin(), positionId.end(), 0);
        inputData.positionIds.insert(
            inputData.positionIds.end(),
            positionId.begin(),
            positionId.end()
        );

        int firstPageId = 0;
        for(int i = 0; i < inputData.maxNumBlocksPerSeq; i++)
        {
            if (i < (seqLen / generator.mConfig.cacheConfig.blockSize + 1))
            {
                inputData.pageIds.push_back(seqId*pageNumPerSeq + i);
            }
            else
            {
                inputData.pageIds.push_back(0);
            }
            if(i == 0)
            {
                firstPageId = seqId*pageNumPerSeq;
            }
        }

        for(int i = 0; i < seqLen; i++)
        {
            inputData.slotMapping.push_back(
                (firstPageId + i / generator.mConfig.cacheConfig.blockSize) * generator.mConfig.cacheConfig.blockSize +
                    i % generator.mConfig.cacheConfig.blockSize
            );
        }

        inputData.cuSeqlensQuery.push_back(
            inputData.cuSeqlensQuery[inputData.cuSeqlensQuery.size() - 1] + seqLen
        );
        inputData.seqlensQuery.push_back(seqLen);

        inputData.seqlensKV.push_back(0);
        inputData.maxSeqlensQuery = inputData.maxSeqlensQuery > seqLen ? inputData.maxSeqlensQuery : seqLen;
        seqId++;
    }

    if(config.parallelConfig.rank == 0)
    {
        std::cout << "inputData.batchTokens: " << inputData.batchTokens << std::endl;
    }

    int WARM_UP = 0; // the number for warm up
    for(int i = 0; i < WARM_UP; i++)
    {
        if(config.parallelConfig.rank == 0)
        {
            std::cout << "[WARM UP] " << i << "  iter...." << std::endl;
        }
        generator.Inference(inputData);
    }

    auto outputs = generator.Inference(inputData);

    if(config.parallelConfig.rank == 0)
    {
        float_t time = std::get<2>(outputs);
        std::cout << "[PREFILL] TIME: " << time << " ms" << std::endl;
    }

    std::vector<int> outIds = std::get<0>(outputs);

    for(int i = 0; i < inputData.batchSize; i++)
    {
        if (isEnd[i] == 1)
        {
            continue;
        }

        isEnd[i] = outIds[i] == 1 ? 1 : 0;
    }

    for(int i = 0; i < inputData.batchSize; i++)
    {
        if (isEnd[i] == 1)
        {
            continue;
        }
        outputIds[i].push_back(outIds[i]);
    }

    std::vector<int> seqKVLen = inputData.seqlensQuery;
    std::vector<float> timeCost;
    float time;

    int minPromoptLen = std::numeric_limits<int>::max();
    for (int len : promoptLen)
    {
        if (len < minPromoptLen)
        {
            minPromoptLen = len;
        }
    }

    for(int t = 1; t < maxToken; t++)
    {
        int seqId = 0;
        InputMeta inputData;
        inputData.stage = "generation";
        inputData.batchSize = batchInputIds.size();
        inputData.cuSeqlensQuery.push_back(0);
        inputData.batchTokens = 0;
        inputData.maxSeqlensQuery = -1;
        inputData.maxSeqlensKV = 0;
        maxSeqLen++;
        inputData.maxNumBlocksPerSeq =  maxSeqLen / generator.mConfig.cacheConfig.blockSize + 1 ;

        for(int j = 0; j < inputData.batchSize; j++)
        {
            int seqLen = 1;

            inputData.batchTokens += seqLen;

            inputData.inputIds.push_back(outIds[j]);
            inputData.positionIds.push_back(promoptLen[j] + t - 1);

            for(int i = 0; i < inputData.maxNumBlocksPerSeq; i++)
            {
                if(i < (promoptLen[j] + t) / generator.mConfig.cacheConfig.blockSize + 1)
                {
                    inputData.pageIds.push_back(seqId*pageNumPerSeq + i);
                }
                else
                {
                    inputData.pageIds.push_back(0);
                }
            }

            inputData.slotMapping.push_back(seqId*pageNumPerSeq* generator.mConfig.cacheConfig.blockSize + inputData.positionIds[inputData.positionIds.size()-1]);

            inputData.cuSeqlensQuery.push_back(
                inputData.cuSeqlensQuery[inputData.cuSeqlensQuery.size() - 1] + seqLen
            );
            inputData.seqlensQuery.push_back(seqLen);

            inputData.seqlensKV.push_back(promoptLen[j] + t - 1);
            inputData.maxSeqlensQuery = inputData.maxSeqlensQuery > seqLen ? inputData.maxSeqlensQuery : seqLen;
            inputData.maxSeqlensKV = inputData.maxSeqlensKV > inputData.seqlensKV[inputData.seqlensKV.size() - 1] ?
                                    inputData.maxSeqlensKV : inputData.seqlensKV[inputData.seqlensKV.size() - 1];

            seqId++;
        }

        if(config.parallelConfig.rank == 0)
        {
            std::cout << "inputData.batchTokens: " << inputData.batchTokens << " The " << t << " th token" << std::endl;
        }

        seqKVLen = inputData.seqlensKV;

        outputs = generator.Inference(inputData);

        outIds = std::get<0>(outputs);
        time = std::get<2>(outputs); //  get time cost

        if(t + minPromoptLen > 2048) // we calaute time cost when context greater than 2048
        {
            timeCost.push_back(time);
        }

        for(int i = 0; i < inputData.batchSize; i++)
        {
            if (isEnd[i] == 1)
            {
                continue;
            }

            isEnd[i] = outIds[i] == 1 ? 1 : 0;
        }

        int numFinishedSeq = 0;
        for(int i = 0; i < inputData.batchSize; i++)
        {
            if (isEnd[i] == 1)
            {
                numFinishedSeq++;
                continue;
            }

            outputIds[i].push_back(outIds[i]);
        }

        if(maxToken < 2148)// judge whether the generation is finished
        {
            if(numFinishedSeq == inputData.batchSize)
            {
                break;
            }
        }

    }

    // print time cost on rank 0
    if (config.parallelConfig.rank == 0 && maxToken > 2048)
    {
        // only static the performance after 2048 tokens
        std::cout << "======================<<<<<Time cost : " << 2048 << " ~ " << (minPromoptLen + maxToken) << " >>>>>======================" << std::endl;
        std::cout << "Mean time: " << ComputeMean(timeCost) << " ms" << std::endl;
        std::cout << "Percentile 10 time: " << computePercentile(timeCost, 10) << " ms" << std::endl;
        std::cout << "Percentile 20 time: " << computePercentile(timeCost, 20) << " ms" << std::endl;
        std::cout << "Percentile 30 time: " << computePercentile(timeCost, 30) << " ms" << std::endl;
        std::cout << "Percentile 50 time: " << computePercentile(timeCost, 50) << " ms" << std::endl;
        std::cout << "Percentile 70 time: " << computePercentile(timeCost, 70) << " ms" << std::endl;
        std::cout << "Percentile 90 time: " << computePercentile(timeCost, 90) << " ms" << std::endl;
        std::cout << "Percentile 95 time: " << computePercentile(timeCost, 95) << " ms" << std::endl;
        std::cout << "Percentile 99 time: " << computePercentile(timeCost, 99) << " ms" << std::endl;
        std::cout << "Percentile 100 time: " << computePercentile(timeCost, 100) << " ms" << std::endl;
    }

    return outputIds;
}

std::atomic<bool> exit_flag(false);

void SignalHandle(int signum)
{
    exit_flag.store(true);
}

int32_t main(int argc, char* argv[])
{

    absl::ParseCommandLine(argc, argv);

    const std::string tokenIdsJson = absl::GetFlag(FLAGS_tokenIds);
    const int32_t maxTokens = absl::GetFlag(FLAGS_maxTokens);

    std::vector<std::vector<int>> token_ids = ParseTokenIds(tokenIdsJson);

    MPI_Init(&argc, &argv);
    int worldSize, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    signal(SIGINT, SignalHandle);
    signal(SIGTERM, SignalHandle);

    if (aclInit(NULL) != ACL_SUCCESS)
    {
        LOG_ERROR("acl init failed.");
    }

    LLMConfig config;
    config.SetRankId(rank);
    config.ParseCommandLine(argc, argv);
    config.SetWorldSize(worldSize);
    if(rank == 0)
    {
        std::cout << config << std::endl;
        std::cout << config.modelConfig.modelParams.dump(4) << std::endl; // print model params
    }

    int64_t deviceId = config.devices[rank % config.parallelConfig.nodeSize];
    aclrtSetDevice(deviceId);

    CommDomainManager::Initialize(config.parallelConfig);

    TextGenerator generator(config);
    MPI_Barrier(MPI_COMM_WORLD);

    int maxToken = maxTokens;
    std::vector<std::vector<int32_t>> batchInputIds = token_ids;
    std::vector<std::vector<int>> outputIds = Process(config, generator, batchInputIds, maxToken);

    if(rank == 0)
    {
        nlohmann::json j;
        for (const auto& inner : outputIds)
        {
            j.push_back(inner);
        }
        std::ofstream outfile("outputIds.txt");
        outfile << j.dump();
    }

    CommDomainManager::Finalize();

    aclrtResetDevice(deviceId);

    if (aclFinalize() != ACL_SUCCESS)
    {
        throw std::runtime_error("acl finalize failed.");
    }

    MPI_Finalize();

    return 0;
}
